import argparse
from llm_repo_suggest import suggest_repository_with_llm

def run_cli(query=None):
    if query is None:
        # If no query is passed, prompt the user for one
        query = input("Enter your search query: ")

    recommendations = suggest_repositor(query)
    print("Recommended Repository:", recommendations)
