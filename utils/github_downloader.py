import os
import requests
from urllib.parse import urlparse
import time

def extract_repo_info(github_url):
    """Extract owner and repo name from GitHub URL."""
    if not github_url.startswith("https://github.com/"):
        return None, None
    
    parts = urlparse(github_url).path.strip('/').split('/')
    if len(parts) < 2:
        return None, None
    
    return parts[0], parts[1]

def check_branch_exists(owner, repo, branch):
    """Check if a specific branch exists in the repository."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}"
    try:
        response = requests.get(api_url)
        return response.status_code == 200
    except:
        return False

def get_default_branch(owner, repo):
    """Get the default branch of a GitHub repository."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            repo_info = response.json()
            return repo_info.get("default_branch")
    except Exception as e:
        print(f"Error getting default branch: {e}")
    
    return None

def download_readme_from_branch(github_url, branch, save_path):
    """Download README.md from a specific branch."""
    owner, repo = extract_repo_info(github_url)
    if not owner or not repo:
        return False
    
    readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
    try:
        response = requests.get(readme_url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded README from branch '{branch}': {save_path}")
            return True
    except Exception as e:
        print(f"Error downloading from branch '{branch}': {e}")
    
    return False

def download_github_readme(github_url, save_dir):
    """Downloads the README.md file from a GitHub repository checking multiple branches."""
    if not github_url.startswith("https://github.com/"):
        print(f"Invalid GitHub URL: {github_url}")
        return False
    
    # Ensure the target directory exists
    readme_dir = os.path.join(save_dir, "GitHubReadme")
    os.makedirs(readme_dir, exist_ok=True)
    save_path = os.path.join(readme_dir, "README.md")
    
    # Extract owner and repo name
    owner, repo = extract_repo_info(github_url)
    if not owner or not repo:
        print(f"Could not parse GitHub URL: {github_url}")
        return False
    
    # First try to get the default branch
    default_branch = get_default_branch(owner, repo)
    if default_branch and download_readme_from_branch(github_url, default_branch, save_path):
        return True
    
    # Common branch names to try
    branches_to_try = ["main", "master", "develop", "development"]
    
    # If default branch is in our list, move it to the front to try first
    if default_branch in branches_to_try:
        branches_to_try.remove(default_branch)
        branches_to_try.insert(0, default_branch)
    elif default_branch:
        branches_to_try.insert(0, default_branch)
    
    # Try each branch
    for branch in branches_to_try:
        if download_readme_from_branch(github_url, branch, save_path):
            return True
        # Wait a bit between requests to avoid rate limiting
        time.sleep(0.5)
    
    # If we couldn't find README.md in any branch, save the repo URL to a text file
    if not os.path.exists(save_path):
        with open(os.path.join(readme_dir, "repo_info.txt"), "w") as file:
            file.write(f"GitHub Repository: {github_url}\n")
            file.write(f"README.md could not be found in common branches.")
        print(f"Could not find README.md in repository. Saved repo info instead.")
    
    return False

def download_additional_files(github_url, save_dir, files_to_download=None):
    """Download additional files from the repository."""
    if files_to_download is None:
        files_to_download = ["LICENSE", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md"]
    
    owner, repo = extract_repo_info(github_url)
    if not owner or not repo:
        return
    
    # Get default branch
    default_branch = get_default_branch(owner, repo)
    if not default_branch:
        default_branch = "main"  # Fallback to main
    
    # Create directory for additional files
    additional_dir = os.path.join(save_dir, "GitHubReadme", "additional")
    os.makedirs(additional_dir, exist_ok=True)
    
    # Try to download each file
    for filename in files_to_download:
        file_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{filename}"
        save_path = os.path.join(additional_dir, filename)
        
        try:
            response = requests.get(file_url, timeout=10)
            if response.status_code == 200:
                with open(save_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded additional file: {filename}")
        except Exception:
            pass  # Skip if file doesn't exist

def download_requirement_files(github_url, save_dir):
    """Download requirement files from the repository."""
    owner, repo = extract_repo_info(github_url)
    if not owner or not repo:
        return
    
    # Get default branch
    default_branch = get_default_branch(owner, repo)
    if not default_branch:
        default_branch = "main"  # Fallback to main
    
    # Common names for requirement files
    requirement_files = [
        "requirements.txt", 
        "requirements-dev.txt",
        "requirements-test.txt",
        "requirements.pip",
        "Pipfile",
        "pyproject.toml",
        "setup.py",
        "environment.yml"
    ]
    
    # Create directory for requirement files
    req_dir = os.path.join(save_dir, "GitHubReadme", "requirements")
    os.makedirs(req_dir, exist_ok=True)
    
    found_files = False
    
    # Try to download each file
    for filename in requirement_files:
        file_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{filename}"
        save_path = os.path.join(req_dir, filename)
        
        try:
            response = requests.get(file_url, timeout=10)
            if response.status_code == 200:
                with open(save_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded requirement file: {filename}")
                found_files = True
        except Exception as e:
            pass  # Skip if file doesn't exist
    
    if not found_files:
        print("No requirement files found in the repository.")

# Example usage if run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        github_url = sys.argv[1]
        save_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
        download_github_readme(github_url, save_dir)
        download_additional_files(github_url, save_dir)
        download_requirement_files(github_url, save_dir)
    else:
        print("Usage: python github_downloader.py <GitHub_repo_URL> [output_directory]")
