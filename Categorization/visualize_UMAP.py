import pandas as pd
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv('LLM-DataScientist-Task_Data.csv')
messages = df['message'].tolist()

# Get embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(messages, show_progress_bar=True)

# Reduce dimensions to 2D
reducer = umap.UMAP(n_components=3,n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_3d = reducer.fit_transform(embeddings)

# Plot

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=5, alpha=0.7)

ax.set_title('3D UMAP Projection of Message Embeddings')
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
ax.set_zlabel('UMAP-3')
plt.show()
