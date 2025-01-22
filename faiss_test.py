import faiss
import numpy as np
import torch

# Load FAISS index
index = faiss.read_index("model_files/music_embeddings.index")

# Load a sample vector
tag_vector = torch.load("model_files/tag_vector.pt")
query_vector = tag_vector[0].numpy().astype('float32')

# Search the index
D, I = index.search(np.array([query_vector]), 5)

# Convert index to int64
I = I.astype(np.int64)
print("Top 5 indices:", I[0])

for idx in I[0]:
    print(f"Processing index {int(idx)}")
    
    # Correct way to initialize reconstruction vector
    vector = np.zeros(index.d, dtype=np.float32)
    
    # Perform reconstruction
    index.reconstruct(int(idx), vector)
    print(f"Reconstructed vector: {vector[:5]}...")  # Print a preview of the vector
