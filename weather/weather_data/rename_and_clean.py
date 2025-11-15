import os
import pandas as pd

# -------------------------------------------------------
# 1. Region → cities mapping
# -------------------------------------------------------
REGIONS = {
    'Etelä-Karjala': ['Lappeenranta'],
    'Etelä-Pohjanmaa': ['Seinäjoki'],
    'Etelä-Savo': ['Mikkeli'],
    'Kanta-Häme': ['Hämeenlinna'],
    'Keski-Suomi': ['Jyväskylä'],
    'Lappi': ['Rovaniemi'],
    'Pirkanmaa': ['Tampere'],
    'Pohjanmaa': ['Vaasa'],
    'Pohjois-Karjala': ['Joensuu'],
    'Pohjois-Pohjanmaa': ['Oulu'],
    'Pohjois-Savo': ['Kuopio'],
    'Päijät-Häme': ['Lahti'],
    'Satakunta': ['Pori'],
    'Uusimaa': ['Vantaa', 'Helsinki'],
    'Varsinais-Suomi': ['Turku']
}

# Flatten: city → region
CITY_TO_REGION = {
    city.lower(): region
    for region, cities in REGIONS.items()
    for city in cities
}

# -------------------------------------------------------
# 2. Detect region by substring search
# -------------------------------------------------------
def detect_region(filename):
    name = filename.lower()
    for city, region in CITY_TO_REGION.items():
        if city in name:
            return region
    return None  # no match found

# -------------------------------------------------------
# 3. Process files
# -------------------------------------------------------
def process_files(directory="."):
    for file in os.listdir(directory):
        if not file.endswith(".csv"):
            continue

        region = detect_region(file)
        if region is None:
            print(f"⚠ No region match for: {file}")
            continue

        old_path = os.path.join(directory, file)

        # Load CSV
        df = pd.read_csv(old_path)

        # Drop first column if it's the station name
        first_col = df.columns[0]
        if "observation" in first_col.lower():
            df = df.drop(columns=[first_col])

        # Merge date and time
        if {"Year", "Month", "Day", "Time [UTC]"}.issubset(df.columns):
            df["timestamp"] = (
                df["Year"].astype(str)
                + "-"
                + df["Month"].astype(str).str.zfill(2)
                + "-"
                + df["Day"].astype(str).str.zfill(2)
                + " "
                + df["Time [UTC]"]
            )
            df = df.drop(columns=["Year", "Month", "Day", "Time [UTC]"])

        # Save file as region.csv
        new_filename = f"{region}.csv"
        new_path = os.path.join(directory, new_filename)
        df.to_csv(new_path, index=False)

        print(f"✔ {file} → {new_filename}")

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    process_files(".")
