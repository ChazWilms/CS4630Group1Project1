import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# paths for data input and result output 
DATA_PATH = "../../data/integrated_311_yelp.csv"
SAVE_DIR = "../../results"

print("--- STARTING ANALYSIS ---")

# Data verification and loading 
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Cannot find {os.path.abspath(DATA_PATH)}")
else:
    # will load the integrated 311 yelp data set
    df = pd.read_csv(DATA_PATH)
    print(f"Data Loaded: {len(df)} rows.")

    # puts results in the ree=sult folder
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # uses zip codes of _311
    zip_col = 'zipcode_311'
    
    # compares volume verse complaints 
    density_df = df.groupby(zip_col).agg(
        complaint_count=('service_request_id', 'count'),
        business_count=('business_id', 'nunique')
    ).reset_index()

    # Analyisis and correlationn
    correlation = density_df['business_count'].corr(density_df['complaint_count'])
    print(f"Correlation: {correlation:.2f}")

    # prints visual
    plt.figure(figsize=(10, 6))
    sns.regplot(data=density_df, x='business_count', y='complaint_count')
    plt.title('Business Density vs. 311 Complaints')
    
    save_path = os.path.join(SAVE_DIR, 'business_vs_complaints.png')
    plt.savefig(save_path)
    print(f"SUCCESS: Graph saved to {save_path}")
    plt.show()
