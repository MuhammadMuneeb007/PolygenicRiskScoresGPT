import os
import glob
import requests
from bs4 import BeautifulSoup
import hashlib
import html2text
import time
import re
from pprint import pformat
import json

def process_tool_directory(tool_dir):
    """
    Process all Python-related files in a tool directory
    
    Args:
        tool_dir: Path to the tool directory
    """
    website_dir = os.path.join(tool_dir, "Website")
    if not os.path.exists(website_dir):
        print(f"Warning: Website directory not found in {tool_dir}")
        return
     
    tool_name = os.path.basename(tool_dir)
    print("Processing tool:", tool_name)

    # Create the PythonPackage directory for this tool
    python_package_dir = os.path.join(tool_dir, "PythonPackage")
    os.makedirs(python_package_dir, exist_ok=True)
    
    # Process Python packages - recursively search in all subdirectories
    mdfiles = glob.glob(os.path.join(tool_dir,"PythonPackage", "**", "*.md"), recursive=True)
    print(f"Found {len(mdfiles)} MD package files")
    
    pyfiles = glob.glob(os.path.join(tool_dir,"PythonPackage", "**", "*.py"), recursive=True)
    print(f"Found {len(pyfiles)} PY package files")
    
    ipynbfiles = glob.glob(os.path.join(tool_dir,"PythonPackage", "**", "*.ipynb"), recursive=True)
    print(f"Found {len(ipynbfiles)} IPYNB package files")
    
    # Get ALL files in the PythonPackage directory for the summary
    all_files = []
    for root, dirs, files in os.walk(os.path.join(tool_dir, "PythonPackage")):
        for file in files:
            all_files.append(os.path.relpath(os.path.join(root, file), tool_dir))
    
    # Create summary of all files
    file_summary = f"\n\n{'='*50}\nALL FILES IN THE PACKAGE ({len(all_files)}):\n{'='*50}\n"
    for i, file_path in enumerate(sorted(all_files), 1):
        file_summary += f"{i}. {file_path}\n"
    
    # Save all files list to a separate file
    files_txt_path = os.path.join(python_package_dir, "Files.txt")
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
        
        # Save merged MD content
        md_output_path = os.path.join(python_package_dir, "MD.txt")
        with open(md_output_path, 'w', encoding='utf-8') as file:
            file.write(md_content)
        print(f"MD files merged and saved to {md_output_path}")
    
    # Process PY files
    if pyfiles:
        py_content = ""
        for file_path in pyfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    py_content += f"\nFile: {relative_path}\n"
                    py_content += file.read()
                    py_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Save merged PY content
        py_output_path = os.path.join(python_package_dir, "PY.txt")
        with open(py_output_path, 'w', encoding='utf-8') as file:
            file.write(py_content)
        print(f"PY files merged and saved to {py_output_path}")
    
    # Process IPYNB files
    if ipynbfiles:
        ipynb_content = ""
        for file_path in ipynbfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    ipynb_content += f"\nFile: {relative_path}\n"
                    
                    # Parse the notebook JSON
                    notebook = json.load(file)
                    
                    # Extract cells content
                    for cell_idx, cell in enumerate(notebook.get('cells', [])):
                        cell_type = cell.get('cell_type', '')
                        source = cell.get('source', [])
                        
                        # If source is a list, join it
                        if isinstance(source, list):
                            source = ''.join(source)
                            
                        ipynb_content += f"\n--- Cell {cell_idx+1} ({cell_type}) ---\n"
                        ipynb_content += source
                        ipynb_content += "\n"
                    
                    ipynb_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Save merged IPYNB content
        ipynb_output_path = os.path.join(python_package_dir, "IPYNB.txt")
        with open(ipynb_output_path, 'w', encoding='utf-8') as file:
            file.write(ipynb_content)
        print(f"IPYNB files merged and saved to {ipynb_output_path}")
    
    # Create a merged file containing all Python-related content
    all_python_content = ""
    
    # Add MD content
    if mdfiles:
        all_python_content += f"\n\n{'='*50}\nMARKDOWN FILES\n{'='*50}\n"
        for file_path in mdfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    all_python_content += f"\nFile: {relative_path}\n"
                    all_python_content += file.read()
                    all_python_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Add PY content
    if pyfiles:
        all_python_content += f"\n\n{'='*50}\nPYTHON FILES\n{'='*50}\n"
        for file_path in pyfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    all_python_content += f"\nFile: {relative_path}\n"
                    all_python_content += file.read()
                    all_python_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Add IPYNB content
    if ipynbfiles:
        all_python_content += f"\n\n{'='*50}\nJUPYTER NOTEBOOK FILES\n{'='*50}\n"
        for file_path in ipynbfiles:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    relative_path = os.path.relpath(file_path, tool_dir)
                    all_python_content += f"\nFile: {relative_path}\n"
                    
                    # Parse the notebook JSON
                    notebook = json.load(file)
                    
                    # Extract cells content
                    for cell_idx, cell in enumerate(notebook.get('cells', [])):
                        cell_type = cell.get('cell_type', '')
                        source = cell.get('source', [])
                        
                        # If source is a list, join it
                        if isinstance(source, list):
                            source = ''.join(source)
                            
                        all_python_content += f"\n--- Cell {cell_idx+1} ({cell_type}) ---\n"
                        all_python_content += source
                        all_python_content += "\n"
                    
                    all_python_content += "\n\n" + "-"*50 + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Add summary of all files to the merged content
    all_python_content += file_summary
    
    # Save the merged content
    merged_output_path = os.path.join(python_package_dir, "MergedPythonFiles.txt")
    with open(merged_output_path, 'w', encoding='utf-8') as file:
        file.write(all_python_content)
    print(f"All Python files merged and saved to {merged_output_path}")

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
