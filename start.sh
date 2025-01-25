#!/bin/bash

# Create the model_files directory if it doesn't exist
mkdir -p model_files

# Google Drive file IDs extracted from shared links
declare -A FILES
FILES["music_embeddings.index"]="1NBCRqrJXGyXnH9gogOiEqOtIC2bIizJu"
FILES["tag_list.npy"]="1jfAcCWIgdxDn-S8hdhHboUSPbXs26iLD"
FILES["tag_vector.pt"]="1F2NCubdL3PMu9HGSCDmGpzz_Ny50KFtU"
FILES["track_list.pkl"]="1kp9hjTqyo3rY7IUQeUHBPdtYp_f98C3v"
FILES["track_ids.pkl"]="1IJIzRlmlxkODSmPJe1Hy91KbQnj-x_51"
FILES["id2url.pkl"]="1BuFqX_0gJpUDZBhXb_j9DLIUTB9SzT30"

# Function to download files from Google Drive with proper handling
download_from_gdrive() {
    file_id=$1
    file_name=$2
    output_path="model_files/${file_name}"

    echo "Downloading ${file_name} from Google Drive..."

    # Download the file and follow redirects
    wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate \
        "https://drive.google.com/uc?export=download&id=${file_id}" -O- \
        | grep -o 'confirm=[^&]*' | head -1 | sed 's/confirm=//' > confirm.txt

    confirm_code=$(<confirm.txt)

    wget --load-cookies cookies.txt "https://drive.google.com/uc?export=download&confirm=${confirm_code}&id=${file_id}" -O "${output_path}"

    # Cleanup temporary files
    rm -f cookies.txt confirm.txt

    # Verify successful download
    if [[ ! -f "${output_path}" || ! -s "${output_path}" ]]; then
        echo "Error downloading ${file_name}, file may be empty or missing."
        exit 1
    fi

    echo "${file_name} downloaded successfully."
}

# Download each file
for file in "${!FILES[@]}"; do
    download_from_gdrive "${FILES[$file]}" "$file"
done

echo "All model files downloaded successfully!"

# Ensure PORT environment variable is set
if [ -z "$PORT" ]; then
  echo "PORT environment variable not set. Defaulting to 8000."
  export PORT=8000
fi

# Start the FastAPI application with error handling
echo "Starting the FastAPI server on port $PORT..."
uvicorn api:app --host 0.0.0.0 --port "$PORT" --reload || {
    echo "Error starting FastAPI server"
    exit 1
}
