import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import concurrent.futures
import multiprocessing

from utils.article_downloader import download_article_from_doi
from utils.github_downloader import download_github_readme, download_additional_files
# Import the advanced website downloader
from utils.website_downloader import download_website
from utils.r_package_downloader import download_rpackage_github
from utils.python_package_downloader import download_pythonpackage_github


def read_dataset():
    """Reads the dataset from an Excel file."""
    excel_path = os.path.join(os.getcwd(), "PRSGPT-Tools-DatasetLinks.xlsx")
    df = pd.read_excel(excel_path)
    print("Columns in dataset:", df.columns.tolist())
    
    return df


def create_tool_directory(tool_name):
    """Creates the directory structure for a given tool."""
    base_dir = os.path.join(os.getcwd(), "Tools")
    tool_dir = os.path.join(base_dir, tool_name)
    os.makedirs(tool_dir, exist_ok=True)
    
    for sub in ["GitHubReadme", "Website", "Article"]:
        os.makedirs(os.path.join(tool_dir, sub), exist_ok=True)
    
    print(f"Created directory: {tool_dir}")
    return tool_dir

def download_file(url, save_path):
    """Downloads content from a URL and saves it to a file."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, "wb") as file:
            file.write(response.content)
        
        print(f"Downloaded: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

# Replace the old download_readme function
def download_readme(github_url, save_dir):
    """Downloads the README.md file from a GitHub repository."""
    download_github_readme(github_url, save_dir)
    # Also download additional useful files
    download_additional_files(github_url, save_dir)

 

def download_article(article_field, save_dir):
    """Wrapper function to download articles using the article_downloader module."""
    download_article_from_doi(article_field, save_dir)

def download_pdf_manual(pdf_url, save_dir):
    """Downloads a PDF manual from the given URL and saves it to the specified directory."""
    if not pdf_url or pdf_url.lower() in ["nan", "na"] or pdf_url.strip() == "":
        print("No PDF manual URL provided")
        return
    
    try:
        pdf_dir = os.path.join(save_dir, "PDFManual")
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Extract filename from URL or use default name
        filename = os.path.basename(pdf_url.split('?')[0]) or "manual.pdf"
        if not filename.endswith('.pdf'):
            filename += '.pdf'
            
        save_path = os.path.join(pdf_dir, filename)
        
        print(f"Downloading PDF manual from: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, "wb") as file:
            file.write(response.content)
            
        print(f"PDF manual downloaded to: {save_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF manual from {pdf_url}: {e}")
    except Exception as e:
        print(f"Error downloading PDF manual: {e}")

def process_single_tool(row, tool_idx, total_tools):
    """Process a single tool from the dataset. This function is designed to run in parallel."""
    tool_name = str(row["Tool"]).strip()
    
    print(f"Processing tool {tool_idx}/{total_tools}: {tool_name}")
    
    if tool_name.lower() == "nan":
        return
   
    # Step 1: Create directory for the tool
    tool_dir = create_tool_directory(tool_name)

    # Step 2: Download resources
    github_url = str(row["GitHubReadme"]).strip()
    website_url = str(row["Website"]).strip()
    article_url = str(row["Article"]).strip()
    r_package = str(row["Rpackage"]).strip()
    py_package = str(row["Python"]).strip()
    pdf_manual = str(row["PDFManual"]).strip()   

    #Download GitHub README and related files if available
    if github_url.lower() != "nan":
        download_readme(github_url, tool_dir)
        
    # Download R package, treating as GitHub repo if it ends with .git
    if r_package.lower() not in ["nan", "na"] and r_package.strip() != "":
        print(f"Downloading R package from: {r_package}")
        download_rpackage_github(r_package, tool_dir+os.sep+"Rpackage")
    
    # # Download article if available
    if article_url.lower() != "nan":
        download_article(article_url, tool_dir)
        
    # # Download PDF manual if available
    if pdf_manual.lower() not in ["nan", "na"] and pdf_manual.strip() != "":
        download_pdf_manual(pdf_manual, tool_dir)
    
    if py_package.lower() not in ["nan", "na"] and pdf_manual.strip() != "":
        download_pythonpackage_github(py_package, tool_dir+os.sep+"PythonPackage")
    
    
    
    # # Download website if available
    if website_url.lower() != "nan":
        download_website(website_url, tool_dir, max_pages=500)
        
    return f"Completed processing {tool_name}"

def process_tools():
    """Processes each tool from the dataset and downloads required resources using multiple threads."""
    df = read_dataset()

    # Process only a specific number of tools
    num_tools_to_process = 28  # Change this number as needed
    
    # Limit to the number of tools we want to process
    df_to_process = df.iloc[:num_tools_to_process]
    
    # Determine the number of workers (threads)
    # Use at most the number of CPU cores, but not more than the number of tools
    max_workers = min(multiprocessing.cpu_count(), len(df_to_process))
    print(f"Using {max_workers} worker threads")
    
    # Create a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each tool processing to the pool
        futures = [
            executor.submit(
                process_single_tool, 
                row, 
                i+1, 
                len(df_to_process)
            ) 
            for i, (_, row) in enumerate(df_to_process.iterrows())
        ]
        
        # Wait for all futures to complete and print results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"Tool processing generated an exception: {exc}")

if __name__ == "__main__":
    process_tools()





