import torch
import pickle
from datasets import load_dataset

# Load the MSD dataset to map track IDs to YouTube URLs
msd_dataset = load_dataset("seungheondoh/multimodal_msd", split="train")
id2url = {i["msd_track_id"]: i["youtube_url"] for i in msd_dataset}

# Save YouTube URL mapping
with open("model_files/id2url.pkl", "wb") as f:
    pickle.dump(id2url, f)

# Load track data
mwe_dataset = load_dataset("seungheondoh/multimodal_msd", split="train")
track_ids = [i["token"] for i in mwe_dataset['track']]
track_list = [i["content"] for i in mwe_dataset['track']]
track_vector = torch.tensor([i["vector"] for i in mwe_dataset['track']])

# Normalize the track vectors before saving
track_vector = torch.nn.functional.normalize(track_vector)

# Save track-related data
torch.save(track_vector, "model_files/track_vector.pt")
with open("model_files/track_list.pkl", "wb") as f:
    pickle.dump(track_list, f)
with open("model_files/track_ids.pkl", "wb") as f:
    pickle.dump(track_ids, f)
