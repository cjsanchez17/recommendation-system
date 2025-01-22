#!/bin/bash

# Create the model_files directory if it doesn't exist
mkdir -p model_files

# Google Drive file IDs extracted from shared links
declare -A FILES
FILES["music_embeddings.index"]="1p6RaIppCROD3rtuLpCjpmN7_WVvOtEkd"
FILES["tag_list.npy"]="1jfAcCWIgdxDn-S8hdhHboUSPbXs26iLD"
FILES["tag_vector.pt"]="1F2NCubdL3PMu9HGSCDmGpzz_Ny50KFtU"
FILES["track_list.pkl"]="1kp9hjTqyo3rY7IUQeUHBPdtYp_f98C3v"
FILES["track_ids.pkl"]="1IJIzRlmlxkODSmPJe1Hy91KbQnj-x_51"
FILES["id2url.pkl"]="1BuFqX_0gJpUDZBhXb_j9DLIUTB9SzT30"

# Function to download files from Google Drive
download_from_gdrive() {
    file_id=$1
    file_name=$2
    echo "Downloading $file_name from Google Drive..."
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${file_id}" -O "model_files/${file_name}" || {
        echo "Error downloading ${file_name}"
        exit 1
    }
}

# Download each file
for file in "${!FILES[@]}"; do
    download_from_gdrive "${FILES[$file]}" "$file"
done

echo "All model files downloaded successfully!"

# Start the FastAPI application
uvicorn api:app --host 0.0.0.0 --port $PORT
