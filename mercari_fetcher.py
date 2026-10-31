import asyncio
import logging
import random
from datetime import datetime
from playwright.async_api import async_playwright
from db import LodeDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MercariFetcher:
    def __init__(self, db: LodeDB, vision_engine=None):
        self.db = db
        self.vision_engine = vision_engine  # Optional: pass VisionEngine to auto-embed
        self.latest_items = None
        self.waiting_for_payload = True

    def _calculate_dom(self, listed_at: str | int, sold_at: str | int) -> int:
        try:
            if isinstance(listed_at, (int, float)) and isinstance(sold_at, (int, float)):
                listed = datetime.fromtimestamp(listed_at)
                sold = datetime.fromtimestamp(sold_at)
            else:
                listed = datetime.fromisoformat(str(listed_at).replace('Z', '+00:00'))
                sold = datetime.fromisoformat(str(sold_at).replace('Z', '+00:00'))
            return (sold - listed).days
        except Exception:
            return 0

    def _safe_string(self, value) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get('name') or value.get('value') or str(value)
        return str(value)

    async def search_sold(self, keyword: str, max_pages: int = 3):
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=['--disable-blink-features=AutomationControlled']
            )
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            async def handle_response(response):
                if "/v1/api" in response.url and response.status == 200 and self.waiting_for_payload:
                    try:
                        data = await response.json()
                        search_data = data.get("data", {}).get("search", {})
                        items = search_data.get("itemsList", [])
                        if not items:
                            items = search_data.get("items", [])
                        if items:
                            self.latest_items = items
                            self.waiting_for_payload = False
                            logger.info(f"🎯 Captured {len(items)} items from payload!")
                    except Exception:
                        pass

            page.on("response", handle_response)

            processed_ids = set()
            total_saved = 0

            for page_num in range(max_pages):
                offset = page_num * 100
                encoded_keyword = keyword.replace(' ', '%20')
                url = f"https://www.mercari.com/search/?keyword={encoded_keyword}&itemStatuses=3&offset={offset}"

                logger.info(f"Navigating to page {page_num + 1} (offset {offset})...")
                self.latest_items = None
                self.waiting_for_payload = True

                try:
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    for _ in range(30):
                        if not self.waiting_for_payload:
                            break
                        await page.wait_for_timeout(500)

                    if self.waiting_for_payload:
                        logger.warning(f"⚠️ Timeout waiting for payload on page {page_num + 1}.")
                        break

                    saved_count = 0
                    for item in self.latest_items:
                        item_id = item.get("id")
                        if item_id in processed_ids:
                            continue
                        self._process_item(item)
                        processed_ids.add(item_id)
                        saved_count += 1

                    logger.info(f"✅ Saved {saved_count} items to DB from page {page_num + 1}.")
                    total_saved += saved_count

                    if page_num < max_pages - 1:
                        await page.wait_for_timeout(random.uniform(2500, 4500))
                except Exception as e:
                    logger.error(f" Page navigation failed: {e}")
                    break

            await browser.close()
            logger.info(f"Mercari scraping complete. Total items saved: {total_saved}")

            # AUTO-EMBED TRIGGER
            if self.vision_engine:
                logger.info("🧠 Auto-triggering Vision Engine to embed new photos...")
                # process_batch runs in a synchronous loop, which is fine here
                self.vision_engine.process_batch(batch_size=50)
                logger.info("✅ All new photos embedded and ready for search!")

    def _process_item(self, item: dict):
        try:
            raw_price = item.get("price")
            sold_price = raw_price if isinstance(raw_price, (int, float)) else (
                raw_price.get("value") if isinstance(raw_price, dict) else 0)
            # Mercari US API often returns price in cents.
            if isinstance(sold_price, (int, float)) and sold_price > 100:
                sold_price = sold_price / 100.0

            category_name = self._safe_string(item.get("itemCategory", {}).get("name") or item.get("categoryName"))
            brand_name = self._safe_string(item.get("brand", {}).get("name") or item.get("brandName"))
            condition_name = self._safe_string(item.get("itemCondition", {}).get("name") or item.get("condition"))
            size_name = self._safe_string(item.get("itemSize"))

            listing_data = {
                "platform": "mercari",
                "platform_id": str(item.get("id")),
                "title": self._safe_string(item.get("name")),
                "description": self._safe_string(item.get("description", "")),
                "category": category_name,
                "brand": brand_name,
                "size": size_name,
                "condition": condition_name,
                "original_price": raw_price if isinstance(raw_price, (int, float)) and raw_price <= 100 else (
                    raw_price / 100.0 if isinstance(raw_price, (int, float)) else None),
                "sold_price": sold_price,
                "listed_at": item.get("created"),
                "sold_at": item.get("soldAt"),
                "url": f"https://www.mercari.com/us/item/{item.get('id')}/"
            }

            if listing_data['listed_at'] and listing_data['sold_at']:
                listing_data['dom'] = self._calculate_dom(listing_data['listed_at'], listing_data['sold_at'])
            else:
                listing_data['dom'] = 0

            local_id = self.db.insert_listing(listing_data)

            photos = item.get("photos", [])
            for photo in photos:
                photo_url = photo.get("imageUrl") or photo.get("url") or photo.get("thumbnailUrl")
                if photo_url:
                    self.db.insert_photo(listing_id=local_id, url=photo_url)
        except Exception as e:
            logger.error(f"Failed to process item {item.get('id')}: {e}")


if __name__ == "__main__":
    # For CLI usage, we can optionally load the vision engine
    # To keep CLI fast, we'll leave it un-embedded by default,
    # but you can uncomment the next 3 lines to auto-embed via CLI.
    db = LodeDB()
    # from vision_engine import VisionEngine
    # engine = VisionEngine(db)
    fetcher = MercariFetcher(db)  # Pass 'engine' here if you want CLI auto-embed
    logger.info("Starting Mercari fetch...")
    asyncio.run(fetcher.search_sold("vintage leather jacket", max_pages=1))
    logger.info("Done.")