import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Section 0: Setup

sentences = [
    "I love machine learning.",
    "I enjoy studying artificial intelligence.",
    "The weather is sunny today.",
    "It's raining heavily outside.",
    "Cats are great pets.",
    "Dogs are loyal and friendly.",
    "我喜歡機器學習",
    "Me gusta estudiar inteligencia artificial.",
    "J'aime les cuisines française.",
    "Ich liebe das Wetter heute.",
]

models = {
    "MiniLM": "all-MiniLM-L6-v2",
    "MPNet": "all-mpnet-base-v2",
    "BGE-Base": "BAAI/bge-base-en-v1.5",
    "E5-Base": "intfloat/e5-base-v2",
}

# models = {
#     "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
#     "MPNet": "sentence-transformers/all-mpnet-base-v2",
#     "BGE-Base": "BAAI/bge-base-en-v1.5",
#     "E5-Base": "intfloat/e5-base-v2",
#     "BGE-M3": "BAAI/bge-m3",
#     "Instructor-Base": "hkunlp/instructor-base",
#     "GTR-T5-Base": "sentence-transformers/gtr-t5-base",
#     "LaBSE": "sentence-transformers/LaBSE",
#     "Paraphrase-MiniLM": "sentence-transformers/paraphrase-MiniLM-L6-v2",
# }

embeddings_per_model = {}

# Section 1: Generate Embeddings

for name, model_id in models.items():
    model = SentenceTransformer(model_id)
    embeddings = model.encode(sentences, normalize_embeddings=True)
    embeddings_per_model[name] = embeddings

# Section 2: Visualize with UMAP

reducer_umap = umap.UMAP(n_components=2, random_state=42)
fig_umap, axes_umap = plt.subplots(1, len(models), figsize=(5 * len(models), 5))

for i, (name, embeddings) in enumerate(embeddings_per_model.items()):
    embedding_2d = reducer_umap.fit_transform(embeddings)
    axes_umap[i].scatter(embedding_2d[:, 0], embedding_2d[:, 1])
    for j, txt in enumerate(sentences):
        axes_umap[i].annotate(f"{j+1}", (embedding_2d[j, 0], embedding_2d[j, 1]))
    axes_umap[i].set_title(f"{name}\n(UMAP)")

plt.tight_layout()
plt.suptitle("UMAP Visualization of Sentence Embeddings", fontsize=16, y=1.05)
plt.show()

# Section 3: Visualize with PCA

reducer_pca = PCA(n_components=2)
fig_pca, axes_pca = plt.subplots(1, len(models), figsize=(5 * len(models), 5))

for i, (name, embeddings) in enumerate(embeddings_per_model.items()):
    embedding_2d = reducer_pca.fit_transform(embeddings)
    axes_pca[i].scatter(embedding_2d[:, 0], embedding_2d[:, 1])
    for j, txt in enumerate(sentences):
        axes_pca[i].annotate(f"{j+1}", (embedding_2d[j, 0], embedding_2d[j, 1]))
    axes_pca[i].set_title(f"{name}\n(PCA)")

plt.tight_layout()
plt.suptitle("PCA Visualization of Sentence Embeddings", fontsize=16, y=1.05)
plt.show()

# Section 4: Cosine Similarity Matrix (Print to CMD)
print("\n==============================")
print("==> Starting Cosine Similarity Matrices")
print("==============================")

for name, embeddings in embeddings_per_model.items():
    print(f"\n==== Cosine Similarity Matrix for {name} ====")
    sim_matrix = cosine_similarity(embeddings)
    df = pd.DataFrame(sim_matrix, index=range(1, len(sentences)+1), columns=range(1, len(sentences)+1))
    print(df.round(2))


# Section 5: Nearest Neighbors by Cosine Similarity 
print("\n==============================")
print("==> Starting Nearest Neighbor Detection")
print("==============================")

for name, embeddings in embeddings_per_model.items():
    print(f"\n==== Nearest Neighbors for {name} ====")
    sim = cosine_similarity(embeddings)
    for i, row in enumerate(sim):
        sorted_indices = row.argsort()[::-1]
        top_match = sorted_indices[1]
        print(f"Sentence {i+1} -> Most similar: Sentence {top_match+1} (score: {row[top_match]:.3f})")
