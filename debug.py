import requests
import json

# Load Hugging Face API key from your API keys file
def load_api_keys(file_path):
    api_keys = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            api_keys[key.strip()] = value.strip("'")  # Removes single quotes if present
    return api_keys

# Load the API keys
api_keys = load_api_keys('API Keys')

# Check if HuggingFace_Key is in API keys
if 'HuggingFace_Key' not in api_keys:
    raise KeyError("HuggingFace_Key is missing in the API Keys file")

huggingface_token = api_keys['HuggingFace_Key']

# Function to fetch and print metadata for a few Hugging Face models
def test_huggingface_metadata(limit=5):
    headers = {
        'Authorization': f'Bearer {huggingface_token}'
    }

    url = f"https://huggingface.co/api/models?limit={limit}&offset=0"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        models = response.json()
        print(f"Fetched {len(models)} models from Hugging Face:")
        for model in models:
            print(f"Model ID: {model.get('id')}")
            print(f"  Description: {model.get('description', 'No description available')}")
            print(f"  Downloads: {model.get('downloads', 'N/A')}")
            print(f"  Likes: {model.get('likes', 'N/A')}")
            print(f"  License: {model.get('license', 'N/A')}")
            print("-" * 40)
    else:
        print(f"Error fetching Hugging Face metadata: {response.status_code}")

# Run the test
test_huggingface_metadata(limit=5)
