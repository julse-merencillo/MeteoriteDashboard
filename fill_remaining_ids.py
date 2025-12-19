import pandas as pd
import requests
import time
import re
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fill_remaining_ids():
    input_file = "Meteorite_Landings_Final.csv" # Load your current progress
    
    print(f"📂 Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("❌ File not found.")
        return

    # 1. DIAGNOSE: Where are the missing IDs coming from?
    if 'id' not in df.columns: df['id'] = 0
    missing_mask = (df['id'] == 0) | (df['id'].isna())
    missing_count = missing_mask.sum()
    
    print(f"🔍 Starting with {missing_count} missing IDs.")
    
    if missing_count == 0:
        print("✅ No missing IDs! You are done.")
        return

    # Show the user which years are missing data
    # (This confirms if we just need to go deeper into history)
    if 'year_int' in df.columns:
        missing_years = df[missing_mask]['year_int'].value_counts().head(5)
        print("   Most missing IDs are from years:", missing_years.to_dict())

    # --- DICTIONARY BUILDER ---
    print("\n🚀 Scanning Deep History (Pages 100-180)...")
    
    base_url = "https://www.lpi.usra.edu/meteor/metbull.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    name_id_map = {}
    
    # We scan Page 100 to 180 (Covering the remaining ~40,000 older records)
    for page in range(100, 181):
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
            
            # Extract ID and Name
            pattern = r'code=(\d+)[^>]*>(.*?)</a>'
            matches = re.findall(pattern, response.text, re.IGNORECASE)
            
            if not matches:
                print("⚠️ No links (Page might be empty/done).")
                # If we hit 3 empty pages in a row, we could stop, but let's just push through.
            else:
                new_found = 0
                for code, name_html in matches:
                    clean_name = re.sub(r'<[^>]+>', '', name_html).replace('&nbsp;', ' ').strip()
                    name_id_map[clean_name] = int(code)
                    new_found += 1
                
                print(f"✅ Indexed {new_found} items.", end=" ")
                
                # Check what year we are currently looking at (for sanity)
                years_on_page = re.findall(r'<td>(\d{4})</td>', response.text)
                if years_on_page:
                    min_year = min([int(y) for y in years_on_page])
                    print(f"(Reached Year: {min_year})")
                else:
                    print("")

            # Intermediate Save every 10 pages
            if page % 10 == 0:
                _apply_and_save(df, name_id_map, input_file)
            
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
            time.sleep(2)

    # --- FINAL APPLY ---
    print("\n📚 Deep Scan complete. Finalizing dataset...")
    _apply_and_save(df, name_id_map, input_file)

def _apply_and_save(df, name_id_map, filename):
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
    print(f"   💾 Saving... Remaining missing: {remaining}")
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    fill_remaining_ids()