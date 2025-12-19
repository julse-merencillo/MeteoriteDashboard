import pandas as pd
import requests
import time
import re
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fix_names_and_fetch_ids():
    input_file = "Meteorite_Landings_Final.csv"
    
    print(f"📂 Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("❌ File not found.")
        return

    # --- STEP 1: CLEAN THE NAMES ---
    print("🧹 Cleaning Names (Removing '**' artifacts)...")
    
    # Check how many contain asterisks
    dirty_count = df['name'].astype(str).str.contains(r'\*').sum()
    print(f"   Found {dirty_count} names with asterisks.")
    
    if dirty_count > 0:
        # Remove * and strip whitespace
        df['name'] = df['name'].astype(str).str.replace('*', '', regex=False).str.strip()
        print("✅ Names cleaned.")
    
    # Save the cleaned names immediately just in case
    df.to_csv(input_file, index=False)

    # --- STEP 2: RE-SCAN RECENT PAGES ---
    # The missing meteorites are mostly 2016-2025, which are on the FIRST 50 PAGES.
    # We just need to re-scan these now that the names match.
    
    missing_mask = (df['id'] == 0) | (df['id'].isna())
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print("🎉 No missing IDs! Data is perfect.")
        return

    print(f"🔍 Searching for {missing_count} missing IDs (Scannning Pages 0-60)...")

    base_url = "https://www.lpi.usra.edu/meteor/metbull.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    name_id_map = {}
    
    # Scan first 60 pages (30,000 records). This covers the last ~10-15 years easily.
    for page in range(0, 61):
        print(f"   ... Scanning Page {page}...", end=" ")
        
        params = {
            'sea': '*', 'sfor': 'names', 'srt': 'year', 'dir': 'desc',
            'lrec': '500', 'page': str(page)
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=45, verify=False)
            
            # Extract IDs and Names
            pattern = r'code=(\d+)[^>]*>(.*?)</a>'
            matches = re.findall(pattern, response.text, re.IGNORECASE)
            
            if matches:
                count = 0
                for code, name_html in matches:
                    clean_name = re.sub(r'<[^>]+>', '', name_html).replace('&nbsp;', ' ').strip()
                    name_id_map[clean_name] = int(code)
                    count += 1
                print(f"✅ Indexed {count} meteorites.")
            else:
                print("⚠️ No links.")
            
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error: {e}")

    # --- STEP 3: APPLY MAP ---
    print("🔄 Applying new IDs...")
    
    name_id_map_lower = {k.lower(): v for k, v in name_id_map.items()}

    def fill_id(row):
        if pd.notna(row['id']) and row['id'] != 0:
            return row['id']
        
        name = str(row['name']).strip()
        # Try exact match
        if name in name_id_map: return name_id_map[name]
        # Try lowercase match
        if name.lower() in name_id_map_lower: return name_id_map_lower[name.lower()]
        
        return 0

    df['id'] = df.apply(fill_id, axis=1)
    
    final_missing = ((df['id'] == 0) | (df['id'].isna())).sum()
    print(f"✨ Success! Remaining missing: {final_missing}")
    
    df.to_csv("Meteorite_Landings_Final.csv", index=False)
    print("💾 Saved to Meteorite_Landings_Final.csv")

if __name__ == "__main__":
    fix_names_and_fetch_ids()