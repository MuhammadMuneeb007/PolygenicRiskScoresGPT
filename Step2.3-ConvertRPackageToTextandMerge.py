import os
import glob
import requests
from bs4 import BeautifulSoup
import hashlib
import html2text
import time
import re
from pprint import pformat

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
     
    tool_name = os.path.basename(tool_dir)
    print("Processing tool:", tool_name)

    # Create the GitHubQuestions directory for this tool
    github_questions_dir = os.path.join(tool_dir, "Rpackage")
    os.makedirs(github_questions_dir, exist_ok=True)
    
    # Process R packages - recursively search in all subdirectories
    mdfiles = glob.glob(os.path.join(tool_dir,"Rpackage", "**", "*.md"), recursive=True)
    print(f"Found {len(mdfiles)} MD package files")
    
    rmdfiles = glob.glob(os.path.join(tool_dir,"Rpackage", "**", "*.Rmd"), recursive=True)
    print(f"Found {len(rmdfiles)} Rmd package files")
    
    rdfiles = glob.glob(os.path.join(tool_dir,"Rpackage", "**", "*.rd"), recursive=True)
    print(f"Found {len(rdfiles)} RD package files")
    
    # Get ALL files in the Rpackage directory for the summary (not just md, Rmd, rd files)
    all_files = []
    for root, dirs, files in os.walk(os.path.join(tool_dir, "Rpackage")):
        for file in files:
            all_files.append(os.path.relpath(os.path.join(root, file), tool_dir))
    
    # Create summary of all files
    file_summary = f"\n\n{'='*50}\nALL FILES IN THE PACKAGE ({len(all_files)}):\n{'='*50}\n"
    for i, file_path in enumerate(sorted(all_files), 1):
        file_summary += f"{i}. {file_path}\n"
    
    # Save all files list to a separate file
    files_txt_path = os.path.join(github_questions_dir, "Files.txt")
    with open(files_txt_path, 'w', encoding='utf-8') as file:
        file.write(f"ALL FILES IN THE PACKAGE ({len(all_files)}):\n")
        for i, file_path in enumerate(sorted(all_files), 1):
            file.write(f"{i}. {file_path}\n")
    print(f"All files list saved to {files_txt_path}")
    
    # Process MD files
    if mdfiles:
        md_content = ""
        for file_path in mdfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    md_content += f"\nFile: {relative_path}\n"
                    md_content += file.read()
                    md_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Add summary of all files to the merged content
        md_content += file_summary
        
        # Save merged MD content
        md_output_path = os.path.join(github_questions_dir, "MD.txt")
        with open(md_output_path, 'w', encoding='utf-8') as file:
            file.write(md_content)
        print(f"MD files merged and saved to {md_output_path}")
    
    # Process RMD files
    if rmdfiles:
        rmd_content = ""
        for file_path in rmdfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    rmd_content += f"\nFile: {relative_path}\n"
                    rmd_content += file.read()
                    rmd_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Add summary of all files to the merged content
        rmd_content += file_summary
        
        # Save merged RMD content
        rmd_output_path = os.path.join(github_questions_dir, "RMD.txt")
        with open(rmd_output_path, 'w', encoding='utf-8') as file:
            file.write(rmd_content)
        print(f"RMD files merged and saved to {rmd_output_path}")
    
    # Process RD files
    if rdfiles:
        rd_content = ""
        for file_path in rdfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    rd_content += f"\nFile: {relative_path}\n"
                    rd_content += file.read()
                    rd_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Add summary of all files to the merged content
        rd_content += file_summary
        
        # Save merged RD content
        rd_output_path = os.path.join(github_questions_dir, "RD.txt")
        with open(rd_output_path, 'w', encoding='utf-8') as file:
            file.write(rd_content)
        print(f"RD files merged and saved to {rd_output_path}")

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



