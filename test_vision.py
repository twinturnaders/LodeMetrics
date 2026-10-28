import logging
from db import LodeDB
from vision_engine import VisionEngine
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    db = LodeDB()
    engine = VisionEngine(db)

    # 1. Process a small batch to prove it works
    print("\n--- STEP 1: Generating Embeddings ---")
    engine.process_batch(batch_size=50)

    # 2. Test Similarity Search
    print("\n--- STEP 2: Testing Visual Search ---")
    # Let's grab a real image URL from our DB to use as a test query
    with db.get_conn() as conn:
        test_photo = conn.execute("""
                                  SELECT p.url, l.title, l.sold_price
                                  FROM photos p
                                           JOIN listings l ON p.listing_id = l.id
                                  WHERE p.embedding IS NOT NULL LIMIT 1
                                  """).fetchone()

    if test_photo:
        print(f"Using this item as the visual query: '{test_photo['title']}' (${test_photo['sold_price']})")

        # Download it locally to act as our "user uploaded photo"
        test_img_path = "test_query_image.jpg"
        with open(test_img_path, "wb") as f:
            f.write(requests.get(test_photo['url']).content)

        # Run the search
        results = engine.search_similar(test_img_path, top_k=5)

        print("\n🏆 Top 5 Visual Comps:")
        for i, res in enumerate(results, 1):
            print(f"{i}. Score: {res['similarity_score']:.4f} | ${res['sold_price']} | {res['title'][:50]}...")
            print(f"   URL: {res['url']}")
    else:
        print("No embedded photos found to test with.")