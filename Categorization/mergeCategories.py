import pandas as pd

# Load the main messages file
df_messages = pd.read_csv('clustered_messages.csv')

# Load the cluster-to-category mapping file
df_mapping = pd.read_csv('cluster_category_mapping.csv')

# Merge: keep all messages and pull in category
df_merged = pd.merge(df_messages, df_mapping, on='cluster', how='left')

print(df_merged.head())

df_merged.to_csv('merged_messages_with_categories.csv', index=False)

category_counts = df_merged['category'].value_counts()

print("=== Message Counts per Category ===")
print(category_counts)
