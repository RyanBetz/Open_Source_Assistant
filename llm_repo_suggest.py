import json
import openai


# Load the JSON data for GitHub and Hugging Face repositories
def load_repositories():
    with open('github_metadata.json', 'r') as github_file:
        github_data = json.load(github_file)

    with open('huggingface_metadata.json', 'r') as huggingface_file:
        huggingface_data = json.load(huggingface_file)

    combined_data = github_data + huggingface_data

    # Print the first few repositories for debugging
    print("Loaded repositories:", combined_data[:5])

    return combined_data


# Filter the repositories based on the user's query
def filter_repositories(query):
    repositories = load_repositories()

    # Filter repositories where the query matches the name or description
    filtered_repos = []
    query = query.lower().strip()  # Normalize the query

    for repo in repositories:
        if 'name' in repo and 'description' in repo:
            repo_name = repo['name'].lower().strip()  # Normalize repo name
            repo_description = repo['description'].lower().strip()  # Normalize description

            if query in repo_name or query in repo_description:
                filtered_repos.append(repo)

    return filtered_repos


# Use OpenAI to suggest a repository based on the filtered data
def suggest_repository_with_llm(query):
    # Filter repositories using the local JSON data
    filtered_repos = filter_repositories(query)

    if not filtered_repos:
        return "Sorry, no repositories match your query."

    # Prepare repository information for OpenAI to analyze
    repo_list = "\n".join([f"Name: {repo['name']}, Description: {repo['description']}" for repo in filtered_repos[:5]])

    # Construct the prompt for OpenAI
    prompt = f"""
    I have the following repositories related to the query "{query}":
    {repo_list}

    Based on this list, choose the best repository and explain why it is a good match for someone looking for "{query}".
    """

    # Call OpenAI to generate the recommendation
    response = openai.Completion.create(
        engine="text-davinci-003",  # or another GPT model
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].text.strip()
