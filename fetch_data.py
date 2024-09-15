import os
import requests
import json
import time
from datetime import datetime
import threading

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

# Function to fetch GitHub metadata with adjusted query
def fetch_github_metadata(limit=500000, output_file='github_metadata.json'):
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
    total_fetched = 0  # Track how many repos have been fetched

    # Adjusted query parameters
    # Remove star thresholds to get more repositories
    # Use different sorting criteria: e.g., sort by recently updated repositories
    query = "created:>=2008-01-01"  # Fetch repositories created since 2008
    sort = "updated"  # Sort by last updated date
    order = "desc"  # Most recently updated first

    for page in range(1, 101):  # Adjust the range as needed
        if total_fetched >= limit:
            break
        url = f"https://api.github.com/search/repositories?q={query}&sort={sort}&order={order}&per_page={per_page}&page={page}"
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

        elif response.status_code == 422:
            print("Error fetching GitHub metadata: 422 - possibly invalid query.")
            break

        elif response.status_code == 200:
            repos = response.json().get('items', [])
            if not repos:
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

                    # Time series data capturing
                    metadata = {
                        'id': repo.get('id'),
                        'name': repo.get('name'),
                        'full_name': repo.get('full_name'),
                        'html_url': repo.get('html_url'),
                        'description': repo.get('description'),
                        'stargazers_count': repo.get('stargazers_count'),
                        'forks_count': repo.get('forks_count'),
                        'language': repo.get('language'),
                        'contributors': contributors,
                        'languages': languages,
                        'readme': readme_content,
                        'created_at': repo.get('created_at'),
                        'updated_at': repo.get('updated_at'),
                        'pushed_at': repo.get('pushed_at'),
                        'time_series_data': {
                            'stars': [(datetime.now().strftime("%Y-%m-%d"), repo.get('stargazers_count'))],
                            'forks': [(datetime.now().strftime("%Y-%m-%d"), repo.get('forks_count'))]
                        }
                    }
                    repo_metadata.append(metadata)
                    existing_repo_ids.add(repo['id'])
                    total_fetched += 1

        else:
            print(f"Error fetching GitHub metadata: {response.status_code}")
            break

        # Save progress to JSON after each page
        with open(output_file, 'w') as gh_file:
            json.dump(repo_metadata, gh_file, indent=4)
        print(f"Fetched {total_fetched} repositories so far.")

    return repo_metadata

# Function to fetch Hugging Face metadata with specific filters
def fetch_huggingface_metadata_and_model_card(limit=500000, output_file='huggingface_metadata.json'):
    headers = {
        'Authorization': f'Bearer {huggingface_token}'
    }

    # Load existing metadata to avoid duplicates
    existing_models = []
    existing_model_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_models = json.load(f)
            existing_model_ids = {model['id'] for model in existing_models}

    # Initialize the list with existing models
    model_metadata = existing_models.copy()

    per_page = 1000
    total_pages = (limit + per_page - 1) // per_page
    total_fetched = len(existing_model_ids)

    # Define filters for Hugging Face models
    filters = {
        'pipeline_tag': ['text-generation', 'question-answering', 'translation'],
        # Add more filters as needed
    }

    for page in range(total_pages):
        if total_fetched >= limit:
            break

        # Build filter query string
        filter_query = "&".join([f"pipeline_tag={tag}" for tag in filters['pipeline_tag']])

        url = f"https://huggingface.co/api/models?{filter_query}&limit={per_page}&offset={page * per_page}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            models = response.json()
            if not models:
                break

            for model in models:
                model_id = model.get('id')

                if model_id not in existing_model_ids:
                    # Check if the model has a valid model card
                    model_card_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
                    model_card_response = requests.get(model_card_url)

                    if model_card_response.status_code == 200:
                        model_card = model_card_response.text

                        # Time series data capturing
                        metadata = {
                            'id': model_id,
                            'name': model_id,
                            'description': model_card[:200] + "..." if len(model_card) > 200 else model_card,
                            'license': model.get('license', 'No license information available'),
                            'likes': model.get('likes'),
                            'downloads': model.get('downloads'),
                            'pipeline_tag': model.get('pipeline_tag'),
                            'model_card': model_card,
                            'time_series_data': {
                                'likes': [(datetime.now().strftime("%Y-%m-%d"), model.get('likes'))],
                                'downloads': [(datetime.now().strftime("%Y-%m-%d"), model.get('downloads'))]
                            }
                        }
                        model_metadata.append(metadata)
                        existing_model_ids.add(model_id)
                        total_fetched += 1
                    else:
                        print(f"Model {model_id} skipped: No model card found")
                else:
                    # Update time series data for existing models
                    for existing_model in model_metadata:
                        if existing_model['id'] == model_id:
                            existing_model['time_series_data']['likes'].append(
                                (datetime.now().strftime("%Y-%m-%d"), model.get('likes'))
                            )
                            existing_model['time_series_data']['downloads'].append(
                                (datetime.now().strftime("%Y-%m-%d"), model.get('downloads'))
                            )
                            break

            # Handle rate limit (if applicable)
            remaining_requests = int(response.headers.get('X-RateLimit-Remaining', 1))
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            current_time = time.time()
            sleep_duration = reset_time - current_time

            if sleep_duration > 0:
                print(f"Rate limit reached. Sleeping for {sleep_duration} seconds.")
                time.sleep(sleep_duration)

            # Save progress to JSON after each page
            with open(output_file, 'w') as hf_file:
                json.dump(model_metadata, hf_file, indent=4)
            print(f"Fetched {total_fetched} models so far.")

        else:
            print(f"Error fetching Hugging Face metadata on page {page}: {response.status_code}")
            break

    return model_metadata

# Threading to run both GitHub and Hugging Face data fetching simultaneously
def run_parallel_fetching():
    github_thread = threading.Thread(target=fetch_github_metadata, args=(500000,))
    huggingface_thread = threading.Thread(target=fetch_huggingface_metadata_and_model_card, args=(500000,))

    # Start both threads
    github_thread.start()
    huggingface_thread.start()

    # Wait for both threads to finish
    github_thread.join()
    huggingface_thread.join()

    print("Fetching complete for both GitHub and Hugging Face.")

# Run the fetching in parallel
run_parallel_fetching()
