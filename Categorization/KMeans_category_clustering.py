import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


df = pd.read_csv('LLM-DataScientist-Task_Data.csv')
messages = df['message'].tolist()


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(messages, show_progress_bar=True)

#K clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df['cluster'] = kmeans.fit_predict(embeddings)

df.to_csv('clustered_messages.csv', index=False)

print(df[['message', 'cluster']].head())