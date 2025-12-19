import pandas as pd
import requests
import time
import re
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fill_missing_ids():
    input_file = "Meteorite_Landings_Ready.csv"
    output_file = "Meteorite_Landings_Final.csv"
    
    print(f"📂 Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("❌ File not found. Please run the cleaning script first.")
        return

    # Create 'id' column if missing
    if 'id' not in df.columns:
        df['id'] = 0
    
    # Check how many are missing (0 or NaN)
    missing_mask = (df['id'] == 0) | (df['id'].isna())
    missing_count = missing_mask.sum()
    print(f"🔍 Found {missing_count} meteorites missing IDs.")

    if missing_count == 0:
        print("✅ No missing IDs found! You are good to go.")
        return

    # --- DICTIONARY BUILDER ---
    print("🚀 Building ID Map from Meteoritical Bulletin...")
    print("   Strategy: Scanning newest 12,500 records (Standard View).")

    base_url = "https://www.lpi.usra.edu/meteor/metbull.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    name_id_map = {}
    
    # Scan 25 pages (500 * 25 = 12,500 records)
    # This covers the last ~12 years of discoveries
    for page in range(0, 25):
        print(f"   ... Scanning Page {page}...", end=" ")
        
        params = {
            'sea': '*',           # Search All
            'sfor': 'names',      
            'srt': 'year',        # Sort by Year
            'dir': 'desc',        # Newest first
            'lrec': '500',        
            'page': str(page)     
            # REMOVED: 'pnt': 'Normal table' (This was stripping the links!)
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=45, verify=False)
            
            # REGEX EXPLANATION:
            # 1. code=(\d+)   -> Capture the ID number
            # 2. [^>]*>       -> Skip remaining tag attributes
            # 3. (.*?)        -> Capture the Meteorite Name
            # 4. </a>         -> Stop at the closing tag
            # We use re.IGNORECASE just in case
            pattern = r'code=(\d+)[^>]*>(.*?)</a>'
            
            matches = re.findall(pattern, response.text, re.IGNORECASE)
            
            if not matches:
                print("⚠️ No links found (Page might be empty).")
            else:
                new_found = 0
                for code, name_html in matches:
                    # Clean name: sometimes it has <b> or &nbsp; tags
                    clean_name = re.sub(r'<[^>]+>', '', name_html) # Remove tags
                    clean_name = clean_name.replace('&nbsp;', ' ').strip()
                    
                    name_id_map[clean_name] = int(code)
                    new_found += 1
                
                print(f"✅ Indexed {new_found} meteorites.")
            
            time.sleep(1) # Be polite

        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
            break

    print(f"📚 Dictionary built! Contains {len(name_id_map)} Name->ID pairs.")

    # --- APPLY MAP TO DATAFRAME ---
    print("🔄 Filling in missing IDs...")
    
    # Convert map keys to lower case for fuzzy matching safety
    name_id_map_lower = {k.lower(): v for k, v in name_id_map.items()}

    def fill_id(row):
        # If ID exists, keep it
        if pd.notna(row['id']) and row['id'] != 0:
            return row['id']
        
        # Look up Name
        name = str(row['name']).strip()
        
        # Try exact match
        if name in name_id_map:
            return name_id_map[name]
        
        # Try case-insensitive match
        if name.lower() in name_id_map_lower:
            return name_id_map_lower[name.lower()]
        
        return 0 

    df['id'] = df.apply(fill_id, axis=1)
    
    # Final check
    remaining = ((df['id'] == 0) | (df['id'].isna())).sum()
    filled = missing_count - remaining
    
    print(f"✨ Successfully filled {filled} IDs.")
    print(f"⚠️ Still missing: {remaining} (Older historical data).")
    
    df.to_csv(output_file, index=False)
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    fill_missing_ids()