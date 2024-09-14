import os
import requests
import json
import time

# Function to load API keys from the file, removing any surrounding quotes
def load_api_keys(file_path):
    api_keys = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            api_keys[key.strip()] = value.strip("'")  # Removes single quotes if present
    return api_keys

# Load the API keys
api_keys = load_api_keys('API Keys')

# Check if the keys are loaded correctly
if 'HuggingFace_Key' not in api_keys:
    raise KeyError("HuggingFace_Key is missing in the API Keys file")
if 'GITHUB_API_KEY' not in api_keys:
    raise KeyError("GITHUB_API_KEY is missing in the API Keys file")

huggingface_token = api_keys['HuggingFace_Key']
github_token = api_keys['GITHUB_API_KEY']

# Function to fetch GitHub metadata with pagination (up to 50,000)
def fetch_github_metadata(limit=50000, output_file='github_metadata.json'):
    headers = {
        'Authorization': f'token {github_token}'
    }

    # Load existing metadata to avoid duplicates
    existing_repos = []
    existing_repo_ids = set()  # Using a set for faster lookup of IDs
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_repos = json.load(f)
            existing_repo_ids = {repo['id'] for repo in existing_repos}

    # Initialize the list with existing repos
    repo_metadata = existing_repos.copy()

    per_page = 100  # GitHub API limit per request
    stars_thresholds = [">5000", "1000..5000", "500..1000", "100..500", "50..100", "10..50", "0..10"]
    total_fetched = 0  # Track how many repos have been fetched

    for stars_range in stars_thresholds:
        for page in range(1, 11):  # 10 pages per range (1000 repos max)
            if total_fetched >= limit:  # Stop if we've hit the limit
                break
            url = f"https://api.github.com/search/repositories?q=stars:{stars_range}&sort=stars&order=desc&per_page={per_page}&page={page}"
            response = requests.get(url, headers=headers)

            if response.status_code == 403:  # Rate limit exceeded
                rate_limit_url = "https://api.github.com/rate_limit"
                rate_limit_response = requests.get(rate_limit_url, headers=headers)
                rate_limit_data = rate_limit_response.json()

                reset_time = rate_limit_data['rate']['reset']
                current_time = time.time()
                sleep_duration = reset_time - current_time

                if sleep_duration > 0:
                    print(f"Rate limit exceeded. Sleeping for {sleep_duration / 60:.2f} minutes.")
                    time.sleep(sleep_duration)
                continue

            elif response.status_code == 422:  # Handle the 422 error
                print("Error fetching GitHub metadata: 422 - possibly too many requests or invalid query.")
                break

            elif response.status_code == 200:
                repos = response.json().get('items', [])
                if not repos:  # Handle empty response
                    break

                for repo in repos:
                    if repo['id'] not in existing_repo_ids:
                        readme_url = f"https://api.github.com/repos/{repo['full_name']}/readme"
                        languages_url = f"https://api.github.com/repos/{repo['full_name']}/languages"
                        contributors_url = repo.get('contributors_url')

                        # Fetch README content
                        readme_response = requests.get(readme_url, headers=headers)
                        readme_content = readme_response.json().get('content') if readme_response.status_code == 200 else "README not available"

                        # Fetch languages
                        languages_response = requests.get(languages_url, headers=headers)
                        languages = languages_response.json() if languages_response.status_code == 200 else {}

                        # Fetch contributors
                        contributors_response = requests.get(contributors_url, headers=headers)
                        contributors = [contributor['login'] for contributor in contributors_response.json()] if contributors_response.status_code == 200 else []

                        metadata = {
                            'id': repo.get('id'),
                            'name': repo.get('name'),
                            'full_name': repo.get('full_name'),
                            'html_url': repo.get('html_url'),
                            'description': repo.get('description'),
                            'stargazers_count': repo.get('stargazers_count'),
                            'language': repo.get('language'),
                            'contributors': contributors,
                            'languages': languages,
                            'readme': readme_content
                        }
                        repo_metadata.append(metadata)
                        existing_repo_ids.add(repo['id'])  # Add to set of fetched IDs
                        total_fetched += 1

            else:
                print(f"Error fetching GitHub metadata: {response.status_code}")
                break

            # Save progress to JSON after each page
            with open(output_file, 'w') as gh_file:
                json.dump(repo_metadata, gh_file, indent=4)

    return repo_metadata

# Function to fetch Hugging Face metadata with pagination (up to 50,000)
def fetch_huggingface_metadata_and_model_card(limit=50000, output_file='huggingface_metadata.json'):
    headers = {
        'Authorization': f'Bearer {huggingface_token}'
    }

    # Load existing metadata to avoid duplicates
    existing_models = []
    existing_model_ids = set()  # Using a set for faster lookup of IDs
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_models = json.load(f)
            existing_model_ids = {model['id'] for model in existing_models}

    # Initialize the list with existing models
    model_metadata = existing_models.copy()

    per_page = 1000  # Hugging Face API limit per request
    total_pages = (limit + per_page - 1) // per_page

    for page in range(total_pages):
        url = f"https://huggingface.co/api/models?limit={per_page}&offset={page * per_page}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            models = response.json()
            if not models:  # Handle empty response
                break

            for model in models:
                # Check if the model is new
                if model['id'] not in existing_model_ids:
                    model_id = model.get('id')
                    model_card_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
                    model_card_response = requests.get(model_card_url)
                    model_card = model_card_response.text if model_card_response.status_code == 200 else "No model card available."

                    # Metadata structure
                    metadata = {
                        'id': model_id,
                        'name': model_id,
                        'description': model_card[:200] + "..." if len(model_card) > 200 else model_card,
                        'license': model.get('license', 'No license information available'),
                        'likes': model.get('likes'),
                        'downloads': model.get('downloads'),
                        'pipeline_tag': model.get('pipeline_tag'),
                        'model_card': model_card
                    }
                    model_metadata.append(metadata)
                    existing_model_ids.add(model['id'])  # Add to set of fetched IDs

            # Handle rate limit
            remaining_requests = int(response.headers.get('X-RateLimit-Remaining', 1))
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            current_time = time.time()
            sleep_duration = reset_time - current_time

            # Ensure we only sleep for positive durations
            if sleep_duration > 0:
                print(f"Rate limit reached. Sleeping for {sleep_duration} seconds.")
                time.sleep(sleep_duration)

        else:
            print(f"Error fetching Hugging Face metadata on page {page}: {response.status_code}")
            break

        # Save progress to JSON after each page
        with open(output_file, 'w') as hf_file:
            json.dump(model_metadata, hf_file, indent=4)

    return model_metadata

# Fetch GitHub and Hugging Face metadata and save to files
github_repos = fetch_github_metadata(limit=500000)
huggingface_repos = fetch_huggingface_metadata_and_model_card(limit=500000)

print(f"Number of repositories in GitHub metadata: {len(github_repos)}")
print(f"Number of models in Hugging Face metadata: {len(huggingface_repos)}")
print("Metadata saved to 'huggingface_metadata.json' and 'github_metadata.json'.")
