import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import os
import requests

# Paths to the project data folder and saves results in the result folder
DATA_PATH = os.path.join("..", "..", "data", "integrated_311_yelp.csv")
SAVE_DIR = os.path.join("..", "..", "results")

# URL for ZIP code boundaries
GEOJSON_URL = "https://raw.githubusercontent.com/opendataphilly/open-geo-data/master/" "philadelphia-neighborhoods/philadelphia-neighborhoods.geojson"

def load_data():
    # Loads the integrated dataset
    if os.path.exists(DATA_PATH):
        print(f"Loading data from: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    else:
        print(f"Error: Could not find {DATA_PATH}.")
        return None

def plot_hotspots(df):
# Generates hexbin plots and folium heatmaps to identify complaint hotspots by neighborhood.
    # makes sure the results directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    lat_col = 'lat_311'
    lon_col = 'lon_311'
    zip_col = 'zipcode_311' 

    # Hexbin Plot
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(df[lon_col], df[lat_col], gridsize=50, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, label='Number of 311 Complaints')
    plt.title('311 Complaint Hotspots (Hexbin Density Analysis)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    hexbin_path = os.path.join(SAVE_DIR, 'hotspot_hexbin.png')
    plt.savefig(hexbin_path)
    plt.show() 

    #  Finds zip codes and groups complaints by them
    zip_stats = df.dropna(subset=[zip_col, lat_col, lon_col]).groupby(zip_col).agg({
        lat_col: 'mean',
        lon_col: 'mean',
        zip_col: 'count'
    }).rename(columns={zip_col: 'count'}).reset_index()

    # Creates the map
    map_center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=11)
    
    # This sections adds lines around the neighborhoods of the zipcodes
    folium.GeoJson(
        GEOJSON_URL,
        name="ZIP Code Boundaries",
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0.1
        }
    ).add_to(m)

    # Adds the heat map layer
    heat_data = df[[lat_col, lon_col]].dropna().values.tolist()
    HeatMap(heat_data, radius=15, blur=20, max_zoom=1).add_to(m)

    # Adds the labels to the map
    for _, row in zip_stats.iterrows():
        folium.map.Marker(
            [row[lat_col], row[lon_col]],
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f"""
                    <div style="
                        display: inline-block;
                        background-color: white; 
                        border: 2px solid black; 
                        border-radius: 5px; 
                        padding: 2px 8px; 
                        font-family: Arial; 
                        font-size: 12px; 
                        font-weight: bold;
                        color: black;
                        text-align: center;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                    ">
                        {int(row['count'])}
                    </div>"""
            )
        ).add_to(m)
    
    # creates html graph
    html_path = os.path.join(SAVE_DIR, 'complaint_heatmap.html')
    m.save(html_path)
    print(f"Interactive heatmap with boundaries saved to: {html_path}")

if __name__ == "__main__":
    df_integrated = load_data()
    if df_integrated is not None:
        plot_hotspots(df_integrated)
        print("Hotspot analysis complete!")