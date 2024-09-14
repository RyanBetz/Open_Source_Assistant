import json
import base64  # Add to handle Base64 decoding

# Function to load repository data from a file (JSON format)
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to decode Base64 content
def decode_base64_content(content, encoding):
    if encoding == "base64":
        return base64.b64decode(content).decode('utf-8', errors='ignore')
    return content

# Function to search GitHub and Hugging Face repositories
def search_repositories(query, github_repos, huggingface_repos):
    """
    Searches for repositories in both GitHub and Hugging Face based on the query.
    """
    results = []
    query = query.lower()

    # Search GitHub repositories
    for repo in github_repos:
        repo_name = repo.get('name', '').lower()
        repo_description = repo.get('description', '').lower()

        # Check if the repository has a README in Base64 encoding
        if 'readme' in repo and isinstance(repo['readme'], dict):
            readme_content = decode_base64_content(repo['readme'].get('content', ''), repo['readme'].get('encoding', ''))
            readme_content = readme_content.lower()
        else:
            readme_content = ""

        # Search in the repository name, description, and README content
        if query in repo_name or query in repo_description or query in readme_content:
            repo_info = f"Repo Name: {repo['name']}, Description: {repo['description']}, Stars: {repo.get('stargazers_count', 'N/A')}, Language: {repo.get('language', 'N/A')}"
            results.append(repo_info)

    # Search Hugging Face repositories
    for repo in huggingface_repos:
        repo_name = repo.get('id', '').lower()  # Hugging Face uses 'id' as the name
        repo_description = repo.get('tags', [])  # Hugging Face uses 'tags' for descriptions

        if query in repo_name or any(query in tag.lower() for tag in repo_description):
            repo_info = f"Model Name: {repo['id']}, Tags: {', '.join(repo.get('tags', []))}, Likes: {repo.get('likes', 'N/A')}, Downloads: {repo.get('downloads', 'N/A')}"
            results.append(repo_info)

    return results
