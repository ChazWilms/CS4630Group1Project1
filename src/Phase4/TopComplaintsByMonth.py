import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_monthly_trends(df):
    # finds the month name
    df['requested_datetime'] = pd.to_datetime(df['requested_datetime'])
    df['month'] = df['requested_datetime'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Groups by month and counts the number of complaints in each month
    monthly_counts = df['month'].value_counts().reindex(month_order).fillna(0)
    
    # Creates the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, hue=monthly_counts.index, palette='magma', legend=False)
    plt.title('Total 311 Complaints by Month')
    plt.xticks(rotation=45)
    
    # saves into results folder
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    plt.savefig(os.path.join(SAVE_DIR, 'monthly_trends.png'))
    print("General trend graph saved.")
    plt.show()

def plot_top_5_per_month(df):
    # Finds the top 5 specific complaint types for EACH individual month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # makes sure date is correct and month
    df['requested_datetime'] = pd.to_datetime(df['requested_datetime'])
    df['month'] = df['requested_datetime'].dt.month_name()

    # Loops through each month to find local top 5
    monthly_top_data = []
    for month in month_order:
        month_filter = df[df['month'] == month]
        if not month_filter.empty:
            top_5 = month_filter['service_name'].value_counts().nlargest(5).reset_index()
            top_5['month'] = month
            monthly_top_data.append(top_5)

    # Combine into one dataframe for graphing
    plot_df = pd.concat(monthly_top_data)
    plot_df.columns = ['service_name', 'count', 'month']

    # Creates the bar graph
    plt.figure(figsize=(15, 8))
    sns.barplot(data=plot_df, x='month', y='count', hue='service_name', palette='tab20')
    
    # Adds labels and moves the legend outside
    plt.title('Top 5 Complaint Types Specific to Each Month')
    plt.legend(title='Complaint Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # saves into results folder
    plt.savefig(os.path.join(SAVE_DIR, 'top_5_per_month.png'))
    print("Specific monthly top 5 graph saved.")
    plt.show()

# main block
if __name__ == "__main__":
    df_integrated = load_data()
    if df_integrated is not None:
        # First graph
        plot_monthly_trends(df_integrated)
        
        # Second graph
        plot_top_5_per_month(df_integrated)
        
        print("Both monthly analyses complete!")