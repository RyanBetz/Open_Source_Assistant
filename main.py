import argparse
from web_app import app  # Flask app import
from search_filter import load_data  # Load data from search_filter.py
from repo_suggest import suggest_repository_with_embeddings

def run_web_app():
    """Run the web app on port 5001."""
    app.run(debug=True, port=5001)

def run_cli():
    """Run the CLI version of the tool."""
    # Load GitHub and Hugging Face data
    github_repos = load_data('github_metadata.json')
    huggingface_repos = load_data('huggingface_metadata.json')

    # Get user input for the query
    query = input("Ask your question: ").strip()

    # Get the recommendation using OpenAI
    recommendation = suggest_repository_with_embeddings(query, github_repos, huggingface_repos)

    # Print the recommendation
    print("Recommended repository/model:")
    print(recommendation)

def main():
    """Main entry point for the tool."""
    parser = argparse.ArgumentParser(description="Repository Suggestion Tool")
    parser.add_argument('--mode', type=str, choices=['cli', 'web'], default='cli',
                        help="Choose the mode: 'cli' for command-line interface or 'web' for web interface")
    args = parser.parse_args()

    if args.mode == 'cli':
        run_cli()  # Run the CLI tool
    elif args.mode == 'web':
        run_web_app()  # Run the web app

if __name__ == "__main__":
    main()
