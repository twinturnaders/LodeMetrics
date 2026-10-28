import torch
import open_clip
import requests
import numpy as np
import logging
from PIL import Image
from io import BytesIO
from db import LodeDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisionEngine:
    def __init__(self, db: LodeDB, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.db = db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model ({model_name}) on {self.device.upper()}...")

        # Load model and preprocessing pipeline
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()  # Set to evaluation mode
        logger.info("✅ CLIP model loaded successfully.")

    def _download_image(self, url: str) -> Image.Image | None:
        """Downloads an image from a URL, handling errors gracefully."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None

    def process_batch(self, batch_size: int = 50):
        """Fetches unprocessed photos, embeds them, and saves to DB."""
        while True:
            photos = self.db.get_photos_without_embeddings(limit=batch_size)
            if not photos:
                logger.info("🎉 All photos have been processed! No more embeddings needed.")
                break

            logger.info(f"Processing batch of {len(photos)} photos...")
            valid_images = []
            valid_photo_ids = []

            # 1. Download and preprocess
            for photo in photos:
                img = self._download_image(photo['url'])
                if img:
                    valid_images.append(self.preprocess(img))
                    valid_photo_ids.append(photo['id'])

            if not valid_images:
                logger.warning("No valid images in this batch. Skipping.")
                continue

            # 2. Stack into a single tensor for batched inference (much faster)
            image_tensor = torch.stack(valid_images).to(self.device)

            # 3. Generate embeddings
            with torch.no_grad():  # Disable gradient calculation for speed/memory
                image_features = self.model.encode_image(image_tensor)
                # Normalize features to unit length (required for cosine similarity)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Convert to numpy CPU array
                embeddings = image_features.cpu().numpy().astype(np.float32)

            # 4. Update database with new embeddings
            for photo_id, embedding in zip(valid_photo_ids, embeddings):
                self.db.update_photo_embedding(photo_id, embedding)

            logger.info(f"✅ Saved {len(valid_photo_ids)} embeddings to DB.")

    def search_similar(self, input_image_path: str, top_k: int = 5) -> list:
        """Finds the top_k most visually similar items in the database."""
        logger.info(f"Searching for visual comps for: {input_image_path}")

        # 1. Embed the query image
        try:
            query_img = Image.open(input_image_path).convert("RGB")
            query_tensor = self.preprocess(query_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                query_feature = self.model.encode_image(query_tensor)
                query_feature /= query_feature.norm(dim=-1, keepdim=True)
                query_vector = query_feature.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to process query image: {e}")
            return []

        # 2. Fetch ALL embeddings AND listing details in a single JOIN query
        with self.db.get_conn() as conn:
            rows = conn.execute("""
                                SELECT p.id as photo_id,
                                       p.listing_id,
                                       p.embedding,
                                       l.title,
                                       l.sold_price,
                                       l.url
                                FROM photos p
                                         LEFT JOIN listings l ON p.listing_id = l.id
                                WHERE p.embedding IS NOT NULL
                                """).fetchall()

        if not rows:
            logger.warning("No embeddings found in database. Run process_batch() first.")
            return []

        # 3. Calculate cosine similarity using vectorized numpy
        db_vectors = np.array([np.frombuffer(row['embedding'], dtype=np.float32) for row in rows])

        # Dot product of normalized vectors = cosine similarity
        similarities = np.dot(db_vectors, query_vector)

        # 4. Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = rows[idx]
            results.append({
                "photo_id": row['photo_id'],
                "listing_id": row['listing_id'],
                "title": row['title'] or "Unknown",
                "sold_price": row['sold_price'] or 0,
                "url": row['url'] or "",
                "similarity_score": float(similarities[idx])
            })

        return results