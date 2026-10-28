import json
import os

dump_dir = "mercari_dumps"


def find_mercari_items(obj, path="root"):
    """Recursively search for a list of dicts that look like Mercari items."""
    if isinstance(obj, dict):
        # Check if this dict looks like a Mercari item
        if "id" in obj and str(obj["id"]).startswith("m") and "name" in obj and "price" in obj:
            return path, obj

        # Recurse into dict values
        for key, value in obj.items():
            result = find_mercari_items(value, f"{path}['{key}']")
            if result:
                return result

    elif isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], dict):
            # Check if the first item in the list looks like a Mercari item
            first_item = obj[0]
            if "id" in first_item and str(first_item["id"]).startswith("m") and "name" in first_item:
                return path, obj

        # Recurse into list items
        for i, item in enumerate(obj):
            result = find_mercari_items(item, f"{path}[{i}]")
            if result:
                return result

    return None


print(f"🔍 Recursively scanning {dump_dir} for Mercari item shapes...\n")

found = False
for filename in sorted(os.listdir(dump_dir)):
    if filename.endswith(".json"):
        filepath = os.path.join(dump_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                result = find_mercari_items(data)

                if result:
                    path, sample = result
                    print(f"🎯 JACKPOT! Found Mercari items in: {filename}")
                    print(f"   ➔ Exact path to the data: {path}")

                    # If it's a single item, tell them it's a single item
                    if isinstance(sample, dict):
                        print(f"   ➔ Found a single item. Sample:\n{json.dumps(sample, indent=2)[:400]}...")
                    else:
                        print(
                            f"   ➔ Found a list of {len(sample)} items. First item sample:\n{json.dumps(sample[0], indent=2)[:400]}...")

                    found = True
                    break  # Stop after finding the first match

            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

if not found:
    print("❌ No Mercari items found in any of the dumped files.")
    print("💡 This usually means the browser didn't make the item-fetching request before the script closed.")