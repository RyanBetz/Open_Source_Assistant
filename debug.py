import requests

# Replace with your GitHub username and repository name
username = 'RyanBetz'
repo = 'Open_Source_Assistant'

# GitHub API URL to list contents of the repository
url = f'https://api.github.com/repos/{username}/{repo}/contents/'

response = requests.get(url)

if response.status_code == 200:
    files = [file['name'] for file in response.json()]
    if 'API Keys' in files:
        print("WARNING: 'API Keys' is in the repository!")
    else:
        print("'API Keys' is NOT in the repository. You're safe.")
else:
    print(f"Failed to fetch repository contents. Status code: {response.status_code}")
