import json

# Function to count the number of entries in the JSON file
def count_entries_in_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return len(data)
    except json.JSONDecodeError:
        print(f"Error: Could not parse {file_path}")
        return 0
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return 0

# Count entries in GitHub metadata
github_count = count_entries_in_json('github_metadata.json')
print(f"Number of repositories in GitHub metadata: {github_count}")

# Count entries in Hugging Face metadata
huggingface_count = count_entries_in_json('huggingface_metadata.json')
print(f"Number of models in Hugging Face metadata: {huggingface_count}")
