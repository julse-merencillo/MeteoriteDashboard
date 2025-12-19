import pandas as pd
import requests
import io
import numpy as np
import urllib3
import re
import time

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_latest_meteorites():
    print("🚀 Connecting to Meteoritical Bulletin Database...")
    print("   Strategy: Paginating 500 records/page (Newest -> Oldest).")

    base_url = "https://www.lpi.usra.edu/meteor/metbull.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_dfs = []
    page = 0
    records_per_page = 500
    target_stop_year = 2012
    keep_scraping = True

    while keep_scraping:
        print(f"   ... Scraping Page {page}...", end=" ")
        
        params = {
            'sea': '*',           
            'sfor': 'names',      
            'srt': 'year',        
            'dir': 'desc',        
            'lrec': str(records_per_page),
            'page': str(page),    
            'pnt': 'Normal table',
            'map': 'ge'           
        }

        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=45, verify=False)
            
            # 1. Parse Table: Look for "Mass" to find the data table
            try:
                tables = pd.read_html(io.StringIO(response.text), match="Mass")
            except ValueError:
                print("❌ No table found (End of data?).")
                break
            
            df_chunk = tables[0]
            
            # 2. DYNAMIC COLUMN RENAMING (The Fix)
            # Normalize all columns to lowercase and strip whitespace
            df_chunk.columns = df_chunk.columns.astype(str).str.lower().str.strip()
            
            # Map based on keywords
            col_map = {}
            for col in df_chunk.columns:
                if 'name' in col and 'type' not in col: col_map[col] = 'name'
                elif 'class' in col or 'type' in col: col_map[col] = 'recclass'
                elif 'mass' in col: col_map[col] = 'mass (g)'
                elif 'year' in col or 'date' in col: col_map[col] = 'year'
                elif 'fall' in col: col_map[col] = 'fall'
                elif 'co-ord' in col or 'loc' in col: col_map[col] = 'GeoLocation'
            
            df_chunk = df_chunk.rename(columns=col_map)
            
            # Check if critical columns exist
            if 'year' not in df_chunk.columns:
                print(f"⚠️ 'Year' column missing. Columns found: {df_chunk.columns.tolist()}")
                # Skip this chunk if unusable
                page += 1
                continue

            # 3. Check Years
            df_chunk['temp_year'] = pd.to_numeric(df_chunk['year'], errors='coerce').fillna(0)
            
            min_year = df_chunk['temp_year'].min()
            max_year = df_chunk['temp_year'].max()
            count = len(df_chunk)
            
            print(f"✅ Got {count} rows (Years: {int(min_year)} - {int(max_year)})")
            
            all_dfs.append(df_chunk)

            # STOP LOGIC
            if count == 0:
                keep_scraping = False
            elif max_year < target_stop_year and max_year > 0:
                print(f"   🛑 Reached historical data ({int(max_year)} < {target_stop_year}). Stopping.")
                keep_scraping = False
            
            page += 1
            time.sleep(1) 

        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
            break

    # --- PROCESS ---
    if not all_dfs:
        print("❌ No data collected.")
        return

    print("🔄 Processing collected data...")
    df_new = pd.concat(all_dfs, ignore_index=True)

    # 1. Coordinate Handling
    if 'GeoLocation' not in df_new.columns:
        df_new['GeoLocation'] = np.nan

    # 2. Mass Cleaning
    def clean_mass(val):
        if pd.isna(val) or val == '': return 0
        val_str = str(val).lower().replace(',', '').strip()
        match = re.search(r"(\d*\.?\d+)", val_str)
        if match:
            number = float(match.group(1))
            if 'kg' in val_str: return number * 1000
            if 'mg' in val_str: return number / 1000
            if 'ton' in val_str: return number * 1000000
            return number
        return 0

    df_new['mass (g)'] = df_new['mass (g)'].apply(clean_mass)
    df_new = df_new[df_new['mass (g)'] > 0]
    df_new['mass_log'] = np.log10(df_new['mass (g)'])

    # 3. Clean Year & Filter
    df_new['year_int'] = pd.to_numeric(df_new['year'], errors='coerce').fillna(0).astype(int)
    # Strict filter
    df_new = df_new[df_new['year_int'] > 2012]

    # 4. Categories
    def get_category(c):
        c = str(c).lower()
        if 'iron' in c or 'mesosiderite' in c or 'pallasite' in c: return 'Iron / Stony-Iron'
        elif 'chondrite' in c: return 'Stony (Chondrite)'
        elif 'achondrite' in c or 'martian' in c or 'lunar' in c: return 'Stony (Achondrite)'
        else: return 'Other / Unknown'

    if 'recclass' in df_new.columns:
        df_new['category_broad'] = df_new['recclass'].apply(get_category)
    else:
        df_new['category_broad'] = "Unknown"

    # 5. Parse Coordinates
    def parse_coord(coord_str, idx):
        if pd.isna(coord_str): return np.nan
        parts = str(coord_str).split()
        if len(parts) < 2: return np.nan
        try:
            val = parts[idx]
            num = float(re.findall(r"[\d\.]+", val)[0])
            direction = val[-1]
            if idx == 0 and direction == 'S': num *= -1
            if idx == 1 and direction == 'W': num *= -1
            return num
        except:
            return np.nan

    df_new['reclat'] = df_new['GeoLocation'].apply(lambda x: parse_coord(x, 0))
    df_new['reclong'] = df_new['GeoLocation'].apply(lambda x: parse_coord(x, 1))

    # --- MERGE ---
    try:
        df_base = pd.read_csv("Meteorite_Landings_Cleaned.csv")
        print(f"📚 Loaded Historical Base: {len(df_base)} records.")
    except:
        print("⚠️ Base file not found. Creating new database.")
        df_base = pd.DataFrame()

    df_final = pd.concat([df_base, df_new], ignore_index=True)
    
    # Deduplicate (Priority: Base dataset often has better coordinates)
    # We sort so that entries with valid lat/lon come first
    df_final = df_final.sort_values(by=['name', 'reclat'], na_position='last')
    df_final = df_final.drop_duplicates(subset=['name'], keep='first')

    output_file = "Meteorite_Landings_Updated.csv"
    df_final.to_csv(output_file, index=False)
    
    print(f"🎉 Success! Database updated.")
    print(f"   - Total Count: {len(df_final)}")
    print(f"   - Saved to: {output_file}")

if __name__ == "__main__":
    get_latest_meteorites()