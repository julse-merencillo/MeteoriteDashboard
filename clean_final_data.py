import pandas as pd
import numpy as np

def clean_and_finalize():
    input_file = "Meteorite_Landings_Updated.csv"
    output_file = "Meteorite_Landings_Ready.csv"

    print(f"🧹 Loading {input_file}...")
    try:
        # Read CSV (treat string 'NULL' as actual NaN)
        df = pd.read_csv(input_file, na_values=['NULL', 'null', ''])
    except FileNotFoundError:
        print("❌ File not found. Run the update script first.")
        return

    print(f"   Raw Shape: {df.shape}")

    # 1. REMOVE GHOST ROWS
    # Drop rows where Name is missing
    df = df.dropna(subset=['name'])

    # 2. FIX 'FALL' COLUMN
    # Logic: 'Y', 'Yc', 'Yes', 'Fell' -> 'Fell'
    #        Everything else (N, Found, Blank) -> 'Found'
    def clean_fall(val):
        s = str(val).lower().strip()
        if s in ['yc', 'yp', 'y', 'yes', 'fell']:
            return 'Fell'
        else:
            return 'Found'

    df['fall'] = df['fall'].apply(clean_fall)
    print("   ✅ Fixed 'Fall' status (Yc -> Fell)")

    # 3. SELECT ONLY USEFUL COLUMNS
    # We drop 'metbull', 'antarctic', 'notes', etc.
    desired_cols = [
        'name', 'id', 'recclass', 'mass (g)', 'fall', 
        'year', 'reclat', 'reclong', 'year_int', 
        'mass_log', 'category_broad'
    ]
    
    # Only keep columns that actually exist in our dataframe
    cols_to_keep = [c for c in desired_cols if c in df.columns]
    df = df[cols_to_keep]

    # 4. HANDLE COORDINATES
    # Force them to be numeric. Errors become NaN.
    df['reclat'] = pd.to_numeric(df['reclat'], errors='coerce')
    df['reclong'] = pd.to_numeric(df['reclong'], errors='coerce')

    # 5. FINAL CLEANUP
    # Log Mass check
    df['mass (g)'] = pd.to_numeric(df['mass (g)'], errors='coerce').fillna(0)
    df = df[df['mass (g)'] > 0]
    df['mass_log'] = np.log10(df['mass (g)'])

    # Year check
    df['year_int'] = pd.to_numeric(df['year_int'], errors='coerce').fillna(0).astype(int)
    # Filter valid years (e.g., 860 to 2026)
    df = df[(df['year_int'] >= 860) & (df['year_int'] <= 2026)]

    # 6. DEDUPLICATE
    # Sort so that entries with valid coordinates come FIRST
    df = df.sort_values(by=['name', 'reclat'], na_position='last')
    df = df.drop_duplicates(subset=['name'], keep='first')

    # Save
    df.to_csv(output_file, index=False)
    
    print("✨ Cleaning Complete!")
    print(f"   - Final Rows: {len(df)}")
    print(f"   - Rows with Coords (Map): {df['reclat'].notna().sum()}")
    print(f"   - Rows without Coords (Analysis): {df['reclat'].isna().sum()}")
    print(f"   - Saved to: {output_file}")

if __name__ == "__main__":
    clean_and_finalize()