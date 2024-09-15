import json
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses the input text by tokenizing, converting to lowercase,
    removing stopwords, and non-alphabetic tokens.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

def load_repos(file_path):
    """
    Loads repository data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of repositories.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

def perform_embedding_clustering(repositories, n_clusters=5):
    """
    Performs clustering on repository descriptions using sentence embeddings.

    Args:
        repositories (list): A list of repository dictionaries.
        n_clusters (int): The number of clusters to form.

    Returns:
        np.ndarray: An array of cluster labels.
    """
    # Preprocess descriptions and handle missing ones
    descriptions = []
    for repo in repositories:
        description = repo.get('description', '')
        if description:
            descriptions.append(preprocess_text(description))
        else:
            descriptions.append('')

    # Generate embeddings using SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model if preferred
    embeddings = model.encode(descriptions)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    return cluster_labels

def group_repos_by_topic(repositories, topics):
    """
    Groups repositories by the assigned topic labels.

    Args:
        repositories (list): A list of repository dictionaries.
        topics (np.ndarray): An array of topic labels.

    Returns:
        dict: A dictionary grouping repositories by topic.
    """
    grouped_repos = {f"Topic {i + 1}": [] for i in range(max(topics) + 1)}

    for i, repo in enumerate(repositories):
        topic_label = topics[i]
        grouped_repos[f"Topic {topic_label + 1}"].append({
            "name": repo.get('name', 'Unknown Repository'),
            "description": repo.get('description', 'No description available'),
            "stars": repo.get('stargazers_count', 0),
            "language": repo.get('language', 'N/A'),
            "url": repo.get('html_url', 'N/A')
        })

    # Sort repositories within each topic by stars in descending order
    for topic_name in grouped_repos:
        grouped_repos[topic_name] = sorted(
            grouped_repos[topic_name],
            key=lambda x: x.get('stars', 0),
            reverse=True
        )
        # Limit to top 5 repositories per topic
        grouped_repos[topic_name] = grouped_repos[topic_name][:5]

    return grouped_repos

def main():
    """
    Main function to execute the topic modeling process.
    """
    repos = load_repos('github_metadata.json') + load_repos('huggingface_metadata.json')

    if repos:
        # Perform embedding and clustering
        cluster_labels = perform_embedding_clustering(repos, n_clusters=5)

        # Group repositories by topic
        grouped_repos = group_repos_by_topic(repos, cluster_labels)

        # Save grouped repositories to a file
        with open('grouped_repositories_by_topic.json', 'w') as f:
            json.dump(grouped_repos, f, indent=4)
        print("Grouped repositories saved.")
    else:
        print("No repositories found.")

if __name__ == "__main__":
    main()
