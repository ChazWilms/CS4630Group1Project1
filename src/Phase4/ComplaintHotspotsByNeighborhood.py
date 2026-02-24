import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import os

# Paths to the project data folder and saves results in the result folder
DATA_PATH = os.path.join("..", "..", "data", "integrated_311_yelp.csv")
SAVE_DIR = os.path.join("..", "..", "results")


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

    # Hexbin Plot
    plt.figure(figsize=(10, 8))

    # Using the updated column
    hb = plt.hexbin(df[lon_col], df[lat_col], gridsize=50, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, label='Number of 311 Complaints')
    plt.title('311 Complaint Hotspots (Hexbin Density Analysis)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # imaage
    hexbin_path = os.path.join(SAVE_DIR, 'hotspot_hexbin.png')
    plt.savefig(hexbin_path)
    print(f"Static hexbin saved to: {hexbin_path}")
    plt.show() 

    # Heatmap
    map_center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=12)
    
    # Prepare lat and lon terms for the HeatMap
    heat_data = df[[lat_col, lon_col]].dropna().values.tolist()
    
    # adds heatmap layer 
    HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)
    
    # creates html graph
    html_path = os.path.join(SAVE_DIR, 'complaint_heatmap.html')
    m.save(html_path)
    print(f"Interactive heatmap saved to: {html_path}")

# main block 
if __name__ == "__main__":
    df_integrated = load_data()
    if df_integrated is not None:
        plot_hotspots(df_integrated)
        print("Hotspot analysis complete!")