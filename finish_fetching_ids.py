import pandas as pd
import requests
import time
import re
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def finish_filling_ids():
    input_file = "Meteorite_Landings_Final.csv" # Load the partially filled one
    
    print(f"📂 Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("❌ File not found.")
        return

    # Check missing
    missing_mask = (df['id'] == 0) | (df['id'].isna())
    missing_count = missing_mask.sum()
    print(f"🔍 Starting with {missing_count} missing IDs.")

    if missing_count == 0:
        print("✅ No missing IDs! Visualization ready.")
        return

    # --- DICTIONARY BUILDER ---
    print("🚀 Resuming ID Scan (Going Deeper)...")
    print("   Strategy: Scanning up to 50,000 records (Pages 0-100).")
    print("   Stop Condition: When we reach years older than 2012.")

    base_url = "https://www.lpi.usra.edu/meteor/metbull.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    name_id_map = {}
    
    # We scan up to page 100. This covers ~50,000 records.
    # The loop will likely break early when it hits the year 2012.
    for page in range(0, 101):
        print(f"   ... Scanning Page {page}...", end=" ")
        
        params = {
            'sea': '*',           
            'sfor': 'names',      
            'srt': 'year',        
            'dir': 'desc',        
            'lrec': '500',        
            'page': str(page)     
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=45, verify=False)
            
            # Regex to capture ID, Name, and importantly, YEAR (to know when to stop)
            # Table row structure: <tr>...code=123>Name</a>...<td>2024</td>...</tr>
            # But searching specifically for the ID link is safer for the map.
            # We will grab ID and Name first.
            
            pattern = r'code=(\d+)[^>]*>(.*?)</a>'
            matches = re.findall(pattern, response.text, re.IGNORECASE)
            
            if not matches:
                print("⚠️ No links found.")
            else:
                new_found = 0
                for code, name_html in matches:
                    clean_name = re.sub(r'<[^>]+>', '', name_html).replace('&nbsp;', ' ').strip()
                    name_id_map[clean_name] = int(code)
                    new_found += 1
                
                print(f"✅ Indexed {new_found} items.", end=" ")

                # --- AUTO STOP LOGIC ---
                # Check the text of the response for the Years present on this page
                # We look for 4-digit years in table cells
                years_on_page = re.findall(r'<td>(\d{4})</td>', response.text)
                if years_on_page:
                    min_year = min([int(y) for y in years_on_page])
                    print(f"(Oldest on page: {min_year})")
                    
                    if min_year < 2012:
                        print(f"   🛑 Reached historical data (Year {min_year}). Stopping scan.")
                        break
                else:
                    print("(Year check skipped)")

            # --- INTERMEDIATE SAVE (Every 10 pages) ---
            if page % 10 == 0 and page > 0:
                print("   💾 Saving intermediate progress...")
                _apply_and_save(df, name_id_map, input_file)
            
            time.sleep(1) 

        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
            # Don't break, just retry next page
            time.sleep(2)

    # --- FINAL APPLY ---
    print("\n📚 Scan complete. Applying IDs to dataset...")
    _apply_and_save(df, name_id_map, input_file)

def _apply_and_save(df, name_id_map, filename):
    # Helper function to map and save
    name_id_map_lower = {k.lower(): v for k, v in name_id_map.items()}

    def fill_id(row):
        if pd.notna(row['id']) and row['id'] != 0:
            return row['id']
        name = str(row['name']).strip()
        if name in name_id_map:
            return name_id_map[name]
        if name.lower() in name_id_map_lower:
            return name_id_map_lower[name.lower()]
        return 0

    df['id'] = df.apply(fill_id, axis=1)
    
    remaining = ((df['id'] == 0) | (df['id'].isna())).sum()
    print(f"   ✨ Current missing: {remaining}")
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    finish_filling_ids()