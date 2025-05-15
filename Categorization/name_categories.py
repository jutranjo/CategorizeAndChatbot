import pandas as pd

# Load clustered messages
df = pd.read_csv('clustered_messages.csv')

# Check unique clusters
clusters = sorted(df['cluster'].unique())
print(f"Found {len(clusters)} clusters: {clusters}")

# Prepare category mapping dictionary
category_mapping = {}

# Loop through each cluster
for cluster_id in clusters:
    print(f"\n=== Cluster {cluster_id} ===")
    # Show 10 random messages from this cluster
    samples = df[df['cluster'] == cluster_id].sample(n=min(10, len(df[df['cluster'] == cluster_id])), random_state=42)
    for i, message in enumerate(samples['message']):
        print(f"{i + 1}. {message}")

    # Ask for manual category assignment
    category = input(f"\nEnter a category name for cluster {cluster_id}: ")
    category_mapping[cluster_id] = category

# Save mapping to file
mapping_df = pd.DataFrame(list(category_mapping.items()), columns=['cluster', 'category'])
mapping_df.to_csv('cluster_category_mapping.csv', index=False)

print("\nSaved cluster-category mapping to 'cluster_category_mapping.csv'")
