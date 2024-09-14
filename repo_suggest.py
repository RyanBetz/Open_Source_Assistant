from openai import OpenAI
import faiss
import json
import numpy as np
import base64  # Import base64 to handle Base64-encoded content
from langchain_openai import OpenAIEmbeddings
from rank_and_suggest import rank_repositories, get_recent_update_score, metadata_weights


# Load API key from the 'API Keys' file (assuming key-value format)
def load_api_key():
    with open('API Keys', 'r') as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY"):
                return line.split('=')[1].strip()


# Initialize OpenAI client and embeddings with the API key
api_key = load_api_key()
client = OpenAI(api_key=api_key)
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)


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


# Function to decode Base64 content if needed
def decode_base64_content(content, encoding):
    if encoding == "base64":
        return base64.b64decode(content).decode('utf-8', errors='ignore')
    return content


# Function to create embeddings for repository data
def create_embeddings(repositories):
    descriptions = []
    for repo in repositories:
        # Handle Base64 decoding for README content if necessary
        if repo.get('description'):
            descriptions.append(repo['description'])
        elif repo.get('readme'):
            readme_content = repo['readme']
            # Decode Base64 README content if encoded
            if isinstance(readme_content, dict) and readme_content.get('content') and readme_content.get('encoding'):
                readme_content = decode_base64_content(readme_content['content'], readme_content['encoding'])
            descriptions.append(readme_content)

    if descriptions:
        return embedding_model.embed_documents(descriptions)
    else:
        print("Error: No descriptions or README content available to create embeddings.")
        return []


# Function to create a FAISS index
def create_faiss_index(embeddings):
    if embeddings:
        dimension = len(embeddings[0])  # Size of the embeddings
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        index.add(np.array(embeddings))  # Ensure embeddings are a NumPy array
        return index
    else:
        print("Error: No embeddings provided to create the FAISS index.")
        return None


# Function to use OpenAI to generate a freeform summary of the repository
def generate_freeform_summary(metadata):
    prompt = (
        "Based on the following repository metadata, generate a brief and natural summary:\n\n"
        f"Repository Name: {metadata.get('name', 'Unknown Repository')}\n"
        f"Description: {metadata.get('description', 'No description available')}\n"
        f"Stars/Likes: {metadata.get('stargazers_count', metadata.get('likes', 'N/A'))}\n"
        f"Language: {metadata.get('language', metadata.get('library_name', metadata.get('pipeline_tag', 'N/A')))}\n"
        f"Forks: {metadata.get('forks_count', 'N/A')}\n"
        f"Open Issues: {metadata.get('open_issues_count', 'N/A')}\n"
        f"Contributors: {', '.join(metadata.get('contributors', ['various developers']))}\n"
        f"Created At: {metadata.get('created_at', 'N/A')}\n"
        f"URL: {metadata.get('html_url', 'N/A')}\n\n"
        "Now, give a brief summary of this repository."
    )

    # Use OpenAI's correct API for generating a freeform summary
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using the correct model here
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    summary = response.choices[0].message.content.strip()
    return summary


# Function to suggest a repository using embeddings, metadata-based ranking, and freeform summaries
def suggest_repository_with_embeddings(user_query, repos, index, metadata_weights):
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
                f"Description: {repo_description}.\n")

    # Generate a freeform summary using OpenAI
    summary = generate_freeform_summary(top_repo)

    # Combine the recommendation and freeform summary
    return response + "\nSummary:\n" + summary


# Example usage
if __name__ == "__main__":
    # Load GitHub and Hugging Face repository data
    github_repos = load_data('github_metadata.json')
    huggingface_repos = load_data('huggingface_metadata.json')

    # Combine both sources
    repos = github_repos + huggingface_repos

    if not repos:
        print("No repositories found.")
    else:
        # Create embeddings for the repositories
        embeddings = create_embeddings(repos)

        # Create a FAISS index
        faiss_index = create_faiss_index(embeddings)

        # Sample user query
        user_query = "machine learning library"

        # Suggest a repository based on the query
        recommendation = suggest_repository_with_embeddings(user_query, repos, faiss_index, metadata_weights)
        print(recommendation)
