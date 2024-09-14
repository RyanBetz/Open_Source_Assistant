import json


def check_json_format(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        # Check if data is a list
        if not isinstance(data, list):
            print(f"Error: The data in {file_path} is not a list.")
            return

        # Iterate through each repository
        for idx, repo in enumerate(data):
            # Check if each repository is a dictionary
            if not isinstance(repo, dict):
                print(f"Error: Entry {idx} in {file_path} is not a dictionary. Found: {type(repo)}")
                continue

            # Check for the 'name' and 'description' fields
            if 'name' not in repo:
                print(f"Error: Entry {idx} is missing the 'name' field.")
            if 'description' not in repo:
                print(f"Error: Entry {idx} is missing the 'description' field.")
            else:
                # Print name and description for confirmation
                print(f"Repo Name: {repo['name']}, Description: {repo['description']}")

        print(f"Check complete for {file_path}")

    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON in {file_path}. Error: {str(e)}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while checking {file_path}: {str(e)}")


if __name__ == "__main__":
    # Specify the paths to the JSON files you want to check
    github_json = 'github_metadata.json'
    huggingface_json = 'huggingface_metadata.json'

    # Run the format check on each JSON file
    print("Checking GitHub metadata JSON...")
    check_json_format(github_json)

    print("\nChecking Hugging Face metadata JSON...")
    check_json_format(huggingface_json)
