import pandas as pd
import numpy as np
import os

# --- Configuration ---
RAW_DATA_FILE = "Meteorite_Landings.csv"
CLEANED_DATA_FILE = "Meteorite_Landings_Cleaned.csv"
# ---------------------

def clean_meteorite_data(input_file):
    """
    Loads the raw NASA meteorite data, cleans it by dropping invalid rows,
    and adds processed columns for app functionality without renaming originals.
    """
    print(f"Starting data cleaning for '{input_file}'...")

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Raw data file '{input_file}' not found.")
        print("Please download it and place it in the same folder as this script.")
        return None

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    print(f"Loaded {df.shape[0]} raw records.")
    
    # --- Data Cleaning Steps ---
    
    # 1. Drop rows where essential data is missing
    original_count = df.shape[0]
    df.dropna(subset=['reclat', 'reclong', 'mass (g)', 'year'], inplace=True)
    print(f"Dropped {original_count - df.shape[0]} rows with missing essential data.")

    # 2. Filter out invalid coordinates (0, 0 is a common placeholder)
    original_count = df.shape[0]
    df = df[(df['reclat'] != 0) | (df['reclong'] != 0)]
    print(f"Dropped {original_count - df.shape[0]} rows with (0,0) coordinates.")

    # 3. Process and filter years (THE FIX IS HERE)
    
    # Convert 'year' to numeric (in case any are strings), dropping errors
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df.dropna(subset=['year'], inplace=True) # Drop any rows that failed conversion
    
    # Convert to integer (whole number) *as a new column*
    # This also fixes the ".0" problem
    df['year_int'] = df['year'].astype(int)
    
    # Filter for realistic years
    original_count = df.shape[0]
    df = df[(df['year_int'] >= 860) & (df['year_int'] <= 2024)]
    print(f"Dropped {original_count - df.shape[0]} rows with unrealistic years (outside 860-2024).")

    # 4. Process mass
    # Convert mass to numeric, coercing errors
    df['mass (g)'] = pd.to_numeric(df['mass (g)'], errors='coerce')
    
    original_count = df.shape[0]
    df.dropna(subset=['mass (g)'], inplace=True)
    print(f"Dropped {original_count - df.shape[0]} rows with invalid mass values.")
    
    # Create a log-transformed mass *as a new column*
    df['mass_log'] = np.log10(df['mass (g)'] + 1) # Add 1 to avoid log(0)

    print(f"Cleaning complete. {df.shape[0]} valid records remaining.")
    return df

# --- Main script execution ---
if __name__ == "__main__":
    cleaned_df = clean_meteorite_data(RAW_DATA_FILE)
    
    if cleaned_df is not None:
        # Save the cleaned data to a new CSV file
        try:
            cleaned_df.to_csv(CLEANED_DATA_FILE, index=False)
            print(f"Successfully saved cleaned data to '{CLEANED_DATA_FILE}'")
            print("\nFinal columns in cleaned file:")
            print(list(cleaned_df.columns))
        except Exception as e:
            print(f"Error saving cleaned file: {e}")