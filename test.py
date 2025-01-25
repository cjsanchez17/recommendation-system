import faiss
index = faiss.read_index("model_files/music_embeddings.index")
print("FAISS index loaded successfully!")