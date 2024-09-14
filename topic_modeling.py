import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load stopwords
stop_words = set(stopwords.words('english'))

# Preprocess text: tokenize, lowercase, remove stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

# Function to generate topic names based on top keywords
def generate_topic_name(keywords):
    return f"Topic based on: {' '.join(keywords[:3]).capitalize()}"

# Load repository data from JSON
def load_repos(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

# Topic modeling with LDA
def perform_topic_modeling(repositories):
    descriptions = [preprocess_text(repo['description']) for repo in repositories if repo.get('description')]

    # Vectorize the descriptions
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dt_matrix = vectorizer.fit_transform(descriptions)

    # Fit LDA to the data
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dt_matrix)

    # Extract topics and top words
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topic_name = generate_topic_name(top_keywords)
        topics.append((topic_name, top_keywords))

    return topics

# Group repositories by topics and include full metadata
def group_repos_by_topic(repositories, topics):
    grouped_repos = {topic_name: [] for topic_name, _ in topics}

    # Randomly assign repositories to topics and include full metadata
    for repo in repositories:
        random_topic = random.choice(topics)[0]
        # Ensure repo is a dictionary with full metadata
        grouped_repos[random_topic].append({
            "name": repo.get('name', 'Unknown Repository'),
            "description": repo.get('description', 'No description available'),
            "stars": repo.get('stargazers_count', 'N/A'),
            "language": repo.get('language', 'N/A'),
            "url": repo.get('html_url', 'N/A')
        })

    # Limit to top 5 repositories per topic
    for topic_name in grouped_repos:
        grouped_repos[topic_name] = grouped_repos[topic_name][:5]

    return grouped_repos

# Save grouped repositories to a JSON file
def save_grouped_repos(grouped_repos, output_file='grouped_repositories_by_topic.json'):
    with open(output_file, 'w') as f:
        json.dump(grouped_repos, f, indent=4)
    print(f"Grouped repositories saved to {output_file}")

# Main function to run the topic modeling script
if __name__ == "__main__":
    repos = load_repos('github_metadata.json') + load_repos('huggingface_metadata.json')

    if repos:
        # Perform topic modeling
        topics = perform_topic_modeling(repos)

        # Group repositories by topic
        grouped_repos = group_repos_by_topic(repos, topics)

        # Save the result to a file
        save_grouped_repos(grouped_repos)

        print("Topics found in the repository descriptions:")
        for topic_name, keywords in topics:
            print(f"{topic_name}: {' '.join(keywords)}")
    else:
        print("No repositories found.")
