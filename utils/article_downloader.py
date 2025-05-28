import os
import re
import subprocess
import sys

def extract_doi_url(article_field):
    """Extracts DOI URL from the given Markdown-like format."""
    match = re.search(r"\[.*?\]\((https://doi\.org/.*?)\)", article_field)
    return match.group(1) if match else None

def extract_doi(doi_url):
    """Extracts the DOI string from a DOI URL."""
    if doi_url and doi_url.startswith("https://doi.org/"):
        return doi_url.replace("https://doi.org/", "")
    return None

def ensure_scidownl_installed():
    """Ensures the scidownl package is installed."""
    try:
        import scidownl
        print("scidownl is already installed.")
        return True
    except ImportError:
        print("Installing scidownl package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scidownl"])
            print("scidownl installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install scidownl. Please install it manually: pip install scidownl")
            return False

def download_article_from_doi(article_field, save_dir):
    """Downloads the article if the link is a DOI in Markdown format using scidownl."""
    doi_url = extract_doi_url(article_field)
    
    if not doi_url:
        print(f"Invalid DOI format: {article_field}")
        return
    
    # Extract the DOI string
    doi = extract_doi(doi_url)
    if not doi:
        print(f"Could not extract DOI from {doi_url}")
        return
    
    # Ensure output directory exists
    article_dir = os.path.join(save_dir, "Article")
    os.makedirs(article_dir, exist_ok=True)
    
    # Ensure scidownl is installed
    if not ensure_scidownl_installed():
        # Fallback to saving just the DOI info
        save_path = os.path.join(article_dir, "doi.txt")
        with open(save_path, "w") as file:
            file.write(f"DOI: {doi}\nURL: {doi_url}")
        print(f"Saved DOI information to {save_path}")
        return
    
    try:
        # Import after installation check
        from scidownl import scihub_download
        
        # Set output path for the PDF
        output_path = os.path.join(article_dir, "article.pdf")
        
        # Download the paper using scidownl
        print(f"Downloading article with DOI: {doi_url}")
        scihub_download(doi_url, paper_type="doi", out=output_path)
        
        print(f"Article PDF saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to download article for DOI {doi}: {e}")
        # Fallback to saving just the DOI info
        save_path = os.path.join(article_dir, "doi.txt")
        with open(save_path, "w") as file:
            file.write(f"DOI: {doi}\nURL: {doi_url}")
        print(f"Saved DOI information to {save_path}")
        return False

# Optional: Add a command-line interface for standalone use
if __name__ == "__main__":
    if len(sys.argv) > 1:
        article_field = sys.argv[1]
        save_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
        download_article_from_doi(article_field, save_dir)
    else:
        print("Usage: python article_downloader.py '[Article DOI URL]' [output_directory]")
