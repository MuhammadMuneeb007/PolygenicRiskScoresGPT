import os
import glob
import requests
from bs4 import BeautifulSoup
import hashlib
import html2text
import time
import re
from pprint import pformat


def remove_duplicate_files(text_files):
    """
    Identify and remove duplicate files based on content hash
    
    Args:
        text_files: List of text file paths to check for duplicates
    
    Returns:
        int: Number of duplicate sets removed
    """
    # Dictionary to store file hashes
    file_hashes = {}
    duplicates_removed = 0
    
    for text_file_path in text_files:
        # Calculate hash of file content
        with open(text_file_path, 'rb') as f:
            file_content = f.read()
            file_hash = hashlib.md5(file_content).hexdigest()
         
        # Check if this hash already exists
        if file_hash in file_hashes:
            print(f"Duplicate text file found:")
            print(f"Original: {file_hashes[file_hash]}")
            print(f"Duplicate: {text_file_path}")
            
            # Get the base filename and directory
            base_name = os.path.basename(text_file_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            dir_path = os.path.dirname(text_file_path)
            
            # Find and remove all associated files (html, link txt, and content txt)
            files_to_remove = []
            
            # The text file itself
            files_to_remove.append(text_file_path)
            
            # The HTML file
            html_file = os.path.join(dir_path, f"{base_name_without_ext}.html")
            if os.path.exists(html_file):
                files_to_remove.append(html_file)
            
            shtml_file = os.path.join(dir_path, f"{base_name_without_ext}.shtml")
            if os.path.exists(shtml_file):
                files_to_remove.append(shtml_file)
                
            # The link file
            link_file = os.path.join(dir_path, f"{base_name_without_ext}_page_link.txt")
            if os.path.exists(link_file):
                files_to_remove.append(link_file)
            
            # Remove all associated files
            print("Removing associated files:")
            for file_path in files_to_remove:
                print(f"  - {file_path}")
                os.remove(file_path)
            
            print("\n")
            duplicates_removed += 1
        else:
            # Store hash with file path
            file_hashes[file_hash] = text_file_path
    
    print(f"Total duplicate sets removed: {duplicates_removed}")
    return duplicates_removed


def process_html_to_text(input_path, output_path):
    """
    Process HTML/SHTML files using BeautifulSoup to extract text content
    
    Args:
        input_path: Path to the input HTML/SHTML file
        output_path: Path to save the extracted text
    """
    try:
        # Read HTML file
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()
        
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text (removing script and style elements)
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text (remove excessive whitespace)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Save the content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully processed {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path} to text: {e}")
        return False


def convert_html_to_markdown(html_file, markdown_file):
    """
    Convert HTML file to Markdown format
    
    Args:
        html_file: Path to the input HTML file
        markdown_file: Path to save the markdown output
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Create html2text converter instance
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Clean HTML with BeautifulSoup first
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Convert to markdown using html2text
        markdown_content = h.handle(str(soup))
        
        # Write markdown content to file
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Converted: {os.path.basename(html_file)} -> {os.path.basename(markdown_file)}")
        return True
    except Exception as e:
        print(f"Error converting {html_file} to markdown: {e}")
        return False


def process_tool_directory(tool_dir):
    """
    Process all HTML files in a tool directory
    
    Args:
        tool_dir: Path to the tool directory
    """
    website_dir = os.path.join(tool_dir, "Website")
    if not os.path.exists(website_dir):
        print(f"Warning: Website directory not found in {tool_dir}")
        return
    
    # Step 1: Process HTML/SHTML to text files
    print("\n=== Step 1: Converting HTML/SHTML to text files ===")
    html_files = glob.glob(os.path.join(website_dir, "*.html"))
    shtml_files = glob.glob(os.path.join(website_dir, "*.shtml"))
    web_files = html_files + shtml_files
    
    print(f"Found {len(web_files)} HTML/SHTML files")
    for web_file in web_files:
        base_name = os.path.basename(web_file).split('.')[0]
        output_file = os.path.join(website_dir, f"{base_name}.txt")
        process_html_to_text(web_file, output_file)
    
    # Step 2: Remove duplicate files
    print("\n=== Step 2: Removing duplicate files ===")
    text_files = glob.glob(os.path.join(website_dir, "*.txt"))
    content_text_files = [f for f in text_files if not f.endswith("_page_link.txt")]
    print(f"Found {len(content_text_files)} content text files to check for duplicates")
    remove_duplicate_files(content_text_files)
    
    # Step 3: Convert HTML to Markdown
    print("\n=== Step 3: Converting HTML to Markdown ===")
    # Refresh the list of HTML files as some might have been removed as duplicates
    html_files = glob.glob(os.path.join(website_dir, "*.html"))
    print(f"Found {len(html_files)} HTML files to convert to markdown")
    
    for html_file in html_files:
        file_base = os.path.basename(html_file)
        file_name = os.path.splitext(file_base)[0]
        markdown_file = os.path.join(website_dir, f"{file_name}.md")
        convert_html_to_markdown(html_file, markdown_file)


def main():
    """Main processing function"""
    # Get all tool directories in "Tools"
    tools_path = "Tools"
    if not os.path.exists(tools_path):
        print(f"Error: {tools_path} directory not found!")
        return
    
    tool_dirs = [d for d in os.listdir(tools_path) if os.path.isdir(os.path.join(tools_path, d))]
    
    for tool_name in tool_dirs:
        print("\n" + "="*50)
        print(f"Processing tool: {tool_name}")
        print("="*50)
        tool_dir = os.path.join(tools_path, tool_name)
        process_tool_directory(tool_dir)
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
