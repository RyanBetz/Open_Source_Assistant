from flask import Flask, render_template, request
from repo_suggest import load_data, create_embeddings, create_faiss_index, suggest_repository_with_embeddings
from rank_and_suggest import metadata_weights  # Import metadata_weights from rank_and_suggest script
import json

app = Flask(__name__)

# Load repository data and handle potential errors
github_repos = load_data('github_metadata.json')
huggingface_repos = load_data('huggingface_metadata.json')

# Initialize variables
faiss_index = None
all_repos = []

if github_repos or huggingface_repos:  # Proceed if we have any data
    all_repos = github_repos + huggingface_repos  # Combine the repositories

    # Create embeddings and FAISS index
    repo_embeddings = create_embeddings(all_repos)
    if repo_embeddings:
        faiss_index = create_faiss_index(repo_embeddings)
        print("FAISS index successfully created.")
    else:
        print("Error: No embeddings were created for the repositories.")
else:
    print("Error: No repository data loaded.")


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    if request.method == 'POST':
        user_input = request.form['query']  # Get user query from the form
        if user_input and faiss_index:
            # Get recommendation using embeddings and metadata-based ranking
            recommendation = suggest_repository_with_embeddings(user_input, all_repos, faiss_index, metadata_weights)
        else:
            recommendation = "No valid query or FAISS index is missing."

    return render_template('index.html', recommendation=recommendation)


# New route to display grouped repositories by topic
@app.route('/topics', methods=['GET'])
def topics():
    # Load the grouped repositories by topic
    try:
        with open('grouped_repositories_by_topic.json', 'r') as f:
            grouped_repos = json.load(f)
    except FileNotFoundError:
        return "Error: 'grouped_repositories_by_topic.json' not found."

    # Pass the correct variable 'grouped_repos' as 'topics' to the template
    return render_template('topics.html', topics=grouped_repos)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
