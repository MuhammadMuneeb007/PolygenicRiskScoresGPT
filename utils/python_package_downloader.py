from git import Repo
import os
import shutil
import re

def download_pythonpackage_github(r_package: str, tool_dir: str, branch: str = None):
    """
    Downloads an R package from a GitHub repository.
    
    Args:
        r_package: GitHub repository URL (can be web URL or git URL)
        tool_dir: Directory where to clone the repository
        branch: Specific branch to clone (default: None, which means default branch)
    """
    
    # Convert web URL format to git clone URL
    # Handle URLs like https://github.com/username/repo/tree/branch
    if r_package.startswith("https://github.com"):
        # Extract username and repo
        pattern = r"https://github\.com/([^/]+)/([^/]+)"
        match = re.match(pattern, r_package)
        
        if match:
            username, repo = match.groups()
            # Clean up repo name in case there's a trailing /tree/branch
            repo = repo.split('/')[0]
            r_package = f"https://github.com/{username}/{repo}.git"
            
            # Check if a branch was specified in the URL
            branch_match = re.search(r"/tree/([^/]+)", r_package)
            if branch_match and not branch:
                branch = branch_match.group(1)
    
    # Validate the repository URL
    if not (r_package.startswith("http") or r_package.startswith("git@")):
        print("Invalid repository URL. Please provide a valid GitHub URL.")
        return
    
    # Check if directory exists and is not empty
    if os.path.exists(tool_dir) and os.listdir(tool_dir):
        print(f"Directory {tool_dir} already exists and is not empty.")
        user_input = input("Do you want to remove and re-clone? (y/n): ").lower()
        
        if user_input == 'y':
            print(f"Removing existing directory: {tool_dir}")
            shutil.rmtree(tool_dir)
        else:
            print("Operation cancelled by user.")
            return
    
    # Ensure tool_dir exists (in case we just deleted it)
    os.makedirs(tool_dir, exist_ok=True)
    
    try:
        # Clone the repository with specific branch if provided
        if branch:
            print(f"Cloning repository {r_package} (branch: {branch}) into {tool_dir}")
            Repo.clone_from(r_package, tool_dir, branch=branch)
        else:
            print(f"Cloning repository {r_package} (default branch) into {tool_dir}")
            Repo.clone_from(r_package, tool_dir)
            
        print(f"Repository cloned successfully into: {tool_dir}")
        
        # Check if there's an R package subdirectory
        rpackage_dirs = [d for d in os.listdir(tool_dir) if d.lower() == "rpackage" or d.lower() == "r"]
        if rpackage_dirs:
            print(f"Found potential R package directories: {', '.join(rpackage_dirs)}")
            print("You may want to use the specific R package subdirectory instead of the whole repository.")
            
    except Exception as e:
        print(f"Error cloning repository: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the repository URL is correct")
        print("2. Check if the repository is private (you need proper authentication)")
        print("3. Try using the HTTPS URL: https://github.com/username/repo.git")
        print("4. If specifying a branch, make sure the branch exists")