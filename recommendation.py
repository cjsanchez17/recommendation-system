import faiss
import numpy as np
import torch
from gensim.models import KeyedVectors
import fasttext

# Load model files
index = faiss.read_index("model_files/music_embeddings.index")
tag_list = np.load("model_files/tag_list.npy", allow_pickle=True)
tag_vector = torch.load("model_files/tag_vector.pt")

# Load FastText models
fasttext_vectors = KeyedVectors.load("model_files/wiki_news_300d_1M.bin")
music_vocab_model = fasttext.load_model("model_files/music_vocab_embeddings.bin")

def get_combined_embedding(word):
    if word in tag_list:
        return tag_vector[tag_list.tolist().index(word)]
    elif word in music_vocab_model:
        return torch.tensor(music_vocab_model[word])
    elif word in fasttext_vectors:
        return torch.tensor(fasttext_vectors[word])
    else:
        return None


def new_query_recommendation(user_query, topk=10):
    token_list = [i.strip() for i in user_query.split()]
    query_embeddings = [get_combined_embedding(token) for token in token_list]
    query_embeddings = [emb for emb in query_embeddings if emb is not None]

    if not query_embeddings:
        return []

    # Stack and normalize embeddings
    query_vector = torch.stack(query_embeddings)
    query_vector = torch.nn.functional.normalize(query_vector)
    query_vector = query_vector.mean(0, True) if query_vector.size(0) > 1 else query_vector

    # Perform FAISS search and ensure index type is int64
    _, indices = index.search(query_vector.numpy().astype(np.float32), topk)
    indices = indices.astype(np.int64)

    results = []
    for i in indices[0]:
        try:
            idx = int(i)
            if idx < 0 or idx >= len(tag_list):
                print(f"Skipping out-of-range index: {idx}")
                continue

            print(f"Processing index: {idx} (type: {type(idx)})")

            # Correct vector allocation for FAISS reconstruction
            vector = np.zeros(index.d, dtype=np.float32)
            index.reconstruct(idx, vector)

            score = float(np.dot(vector, query_vector.numpy().flatten()))

            results.append({"entity": tag_list[idx], "score": score})

        except Exception as e:
            print(f"Error processing index {idx}: {e}")

    return results
