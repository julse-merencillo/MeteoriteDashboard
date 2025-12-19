import pandas as pd

def diagnose_missing():
    try:
        df = pd.read_csv("Meteorite_Landings_Final.csv")
    except:
        print("❌ File not found.")
        return

    # Filter for missing IDs
    if 'id' not in df.columns: df['id'] = 0
    missing = df[(df['id'] == 0) | (df['id'].isna())]
    
    print(f"🔍 Total Missing IDs: {len(missing)}")
    
    if len(missing) > 0:
        print("\n--- SAMPLE OF MISSING ENTRIES ---")
        # Show Name, Year, and Mass to identify them
        print(missing[['name', 'year_int', 'mass (g)']].head(20).to_string(index=False))
        
        print("\n--- YEAR DISTRIBUTION ---")
        print(missing['year_int'].value_counts().sort_index().tail(10))

if __name__ == "__main__":
    diagnose_missing()