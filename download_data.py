import requests
import os
from tqdm import tqdm

# Creates folders 
os.makedirs("data/labels", exist_ok=True)
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/patches", exist_ok=True)

print("Folder structure created ✓")
print("data/")
print("  ├── labels/   ← crater coordinates (CSV)")
print("  ├── images/   ← raw Mars imagery (.tif)")
print("  └── patches/  ← small tiles we'll feed to the model")

# Robbins & Hynek 2012 - Mars Crater Database
# Hosted on Planetary Science Journal's supplementary data
url = "https://astropedia.astrogeology.usgs.gov/download/Mars/Research/Robbins_et_al_2012_Mars_Crater_Database/mars_crater_robbins_2012.csv"

output_path = "data/labels/robbins_craters.csv"

print("\nDownloading Robbins crater database...")

response = requests.get(url, stream=True)
total = int(response.headers.get('content-length', 0))

with open(output_path, 'wb') as f, tqdm(
    total=total, unit='B', unit_scale=True
) as bar:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        bar.update(len(chunk))

print(f"Downloaded to {output_path} ✓")