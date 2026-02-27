import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Path Setup
DATA_PATH = os.path.join("..", "..", "data", "integrated_311_yelp.csv")
SAVE_DIR = os.path.join("..", "..", "results")

# loads integrated data set
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    print(f"Error: Could not find {DATA_PATH}")
    return None

# looks at yelp complaint type by business categories
def compare_categories(df):
    print("Analyzing patterns by business category...")
    
    # Define groups
    target_categories = ['Restaurant', 'Retail', 'Service']
    df['business_group'] = 'Other'
    
    cat_col = 'categories_normalized'

    # finds rows using key words
    for cat in target_categories:
        mask = df[cat_col].str.contains(cat, case=False, na=False)
        df.loc[mask, 'business_group'] = cat

    # takes top 5 complaints
    top_complaints = df['service_name'].value_counts().nlargest(5).index
    filtered_df = df[df['service_name'].isin(top_complaints)]

    # creates graph
    plt.figure(figsize=(12, 7))
    sns.countplot(data=filtered_df, x='service_name', hue='business_group', palette='viridis')
    plt.title('Top 311 Complaints by Business Category')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # saves into results folder
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    plt.savefig(os.path.join(SAVE_DIR, 'category_comparison.png'))
    print("Graph saved to results/category_comparison.png")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        compare_categories(df)