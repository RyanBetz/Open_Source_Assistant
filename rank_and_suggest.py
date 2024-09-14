import faiss
import numpy as np
import json
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
import base64  # To handle Base64 decoding
import os

# Load OpenAI API key from environment or a file
def load_openai_api_key():
    return os.getenv("OPENAI_API_KEY") or load_api_key_from_file()

def load_api_key_from_file():
    try:
        with open("API Keys", "r") as f:
            for line in f:
                if "OPENAI_API_KEY" in line:
                    return line.split("=")[1].strip()
    except FileNotFoundError:
        raise ValueError("API Keys file not found")
    return None

# Initialize OpenAI embeddings with the API key
api_key = load_openai_api_key()
if not api_key:
    raise ValueError("OpenAI API key not found. Set the `OPENAI_API_KEY` environment variable or add it to the 'API Keys' file.")

embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

# Define the metadata_weights globally
metadata_weights = {
    'stars': 0.4,  # 40% importance to stars
    'forks': 0.3,  # 30% importance to forks
    'issues': 0.2,  # 20% importance to the number of open issues
    'updated': 0.1  # 10% importance to how recently the repo was updated
}

# Function to load data from a JSON file
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to parse {file_path}.")
        return []

# Function to decode Base64 content
def decode_base64_content(content, encoding):
    if encoding == "base64":
        return base64.b64decode(content).decode('utf-8', errors='ignore')
    return content

# Function to create embeddings for repository data, accounting for Base64 encoding
def create_embeddings(repositories):
    descriptions = []
    for repo in repositories:
        if repo.get('description'):
            descriptions.append(repo['description'])
        elif repo.get('readme'):
            readme_content = repo['readme']
            if isinstance(readme_content, dict) and readme_content.get('content') and readme_content.get('encoding'):
                # Decode Base64 README content if encoded
                readme_content = decode_base64_content(readme_content['content'], readme_content['encoding'])
            descriptions.append(readme_content)

    if descriptions:
        return embedding_model.embed_documents(descriptions)
    else:
        print("Error: No descriptions or README content available to create embeddings.")
        return []

# Function to create a FAISS index
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])  # Size of embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.array(embeddings))  # Ensure embeddings are a NumPy array
    return index

# Function to score how recently the repository was updated
def get_recent_update_score(updated_at):
    try:
        last_updated = datetime.strptime(updated_at, '%Y-%m-%dT%H:%M:%SZ')
        time_diff = datetime.now() - last_updated
        return 1 / (1 + time_diff.days)  # More recent = higher score
    except Exception as e:
        return 0

# Function to rank repositories based on embedding similarity and metadata
def rank_repositories(repositories, top_k_indices, embedding_scores, metadata_weights):
    ranked_repos = []
    for i in range(len(top_k_indices)):
        repo = repositories[top_k_indices[i]]
        metadata_score = (
                metadata_weights['stars'] * repo.get('stargazers_count', 0) +
                metadata_weights['forks'] * repo.get('forks_count', 0) +
                metadata_weights['issues'] * (1 / (1 + repo.get('open_issues_count', 1))) +
                metadata_weights['updated'] * get_recent_update_score(repo.get('updated_at'))
        )

        # Combine embedding similarity and metadata score
        total_score = embedding_scores[i] + metadata_score
        ranked_repos.append((repo, total_score))

    # Sort repositories by total score in descending order
    ranked_repos.sort(key=lambda x: x[1], reverse=True)

    return [repo for repo, _ in ranked_repos]  # Return ranked repositories

# Function to update metadata weights based on feedback
def update_metadata_weights(feedback, metadata_weights):
    if feedback == 'good':
        metadata_weights['stars'] += 0.05  # Increase importance of stars
        metadata_weights['forks'] += 0.05  # Increase importance of forks
        metadata_weights['issues'] -= 0.05  # Decrease importance of open issues
    elif feedback == 'bad':
        metadata_weights['stars'] -= 0.05  # Decrease importance of stars
        metadata_weights['forks'] -= 0.05  # Decrease importance of forks
        metadata_weights['issues'] += 0.05  # Increase importance of open issues

    # Normalize weights to ensure they're in the valid range
    total = sum(metadata_weights.values())
    for key in metadata_weights:
        metadata_weights[key] = max(0, min(metadata_weights[key], 1))
    return metadata_weights

# Function to suggest a repository and collect feedback
def suggest_repository_with_feedback(user_query, repos, index, metadata_weights):
    # Step 1: Get the embedding similarity-based top-k results
    query_embedding = embedding_model.embed_query(user_query)
    query_embedding = np.array([query_embedding])
    embedding_scores, top_k_indices = index.search(query_embedding, k=10)  # Get top 10 based on embeddings

    # Step 2: Rank these repositories based on metadata
    ranked_repositories = rank_repositories(repos, top_k_indices[0], embedding_scores[0], metadata_weights)

    # Step 3: Return the top repository recommendation
    top_repo = ranked_repositories[0]

    repo_name = top_repo.get('name', 'Unknown Repository')
    repo_description = top_repo.get('description', 'No description available')
    stars_or_likes = top_repo.get('stargazers_count', top_repo.get('likes', 'N/A'))
    repo_language = top_repo.get('language',
                                 top_repo.get('library_name', top_repo.get('pipeline_tag', 'N/A')))

    response = (f"Based on your query, I recommend the repository '{repo_name}'. "
                f"It has {stars_or_likes} stars/likes and is primarily written in {repo_language}. "
                f"Description: {repo_description}.")

    print(response)

    # Collect feedback from the user
    feedback = input("Was this suggestion helpful? (yes/no): ").strip().lower()
    if feedback == "yes":
        metadata_weights = update_metadata_weights('good', metadata_weights)
    elif feedback == "no":
        metadata_weights = update_metadata_weights('bad', metadata_weights)

    return metadata_weights
