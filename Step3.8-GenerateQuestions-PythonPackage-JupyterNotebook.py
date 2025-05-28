import os
import json
import csv
import math
import re
import os.path
from pathlib import Path
from google import genai
import time
from tqdm import tqdm
import pandas as pd
import nbformat
from nbconvert import PythonExporter

# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyA-kxIjJAFAeEYz69mBRUdZMNU1xx5bbGY"
client = genai.Client(api_key=GOOGLE_API_KEY)
import os
import glob
import requests
from bs4 import BeautifulSoup

# Define tools_path if not already defined
tools_path = os.path.join(os.getcwd(), "Tools")
dirs = [d for d in os.listdir(tools_path) if os.path.isdir(os.path.join(tools_path, d))]
 
tool_dirs = dirs
  

 
def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def fix_json_format(text):
    """Try to fix common JSON formatting issues in the response text."""
    # Fix missing or extra commas between objects in array
    text = re.sub(r'}\s*{', '},{', text)
    # Fix trailing commas before closing brackets
    text = re.sub(r',\s*]', ']', text)
    # Fix missing commas between properties
    text = re.sub(r'"\s*"', '","', text)
    # Replace single quotes with double quotes (if used for JSON keys/values)
    text = re.sub(r"'(\w+)':", r'"\1":', text)
    text = re.sub(r':\s*\'([^\']+)\'', r':"\1"', text)
    return text

def extract_json_from_response(response_text):
    """Extract JSON from model response with improved parsing."""
    # Try different regex patterns to extract JSON array
    patterns = [
        r'(\[\s*\{.*\}\s*\])',  # Standard JSON array pattern
        r'```json\s*(\[\s*\{.*\}\s*\])\s*```',  # JSON in code block
        r'```\s*(\[\s*\{.*\}\s*\])\s*```',      # JSON in unspecified code block
    ]
    
    # First try direct JSON parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find and extract JSON using patterns
        for pattern in patterns:
            try:
                match = re.search(pattern, response_text.replace('\n', ' '), re.DOTALL)
                if match:
                    json_str = match.group(1)
                    return json.loads(json_str)
            except (json.JSONDecodeError, AttributeError):
                continue
                
        # If extraction failed, try fixing common JSON issues
        try:
            fixed_text = fix_json_format(response_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
            
        # If all else fails, try to extract all JSON objects and build an array
        try:
            pattern = r'\{[^{}]*"instruction":[^{}]*"output":[^{}]*\}'
            matches = re.findall(pattern, response_text.replace('\n', ' '), re.DOTALL)
            if matches:
                combined = "[" + ",".join(matches) + "]"
                fixed = fix_json_format(combined)
                return json.loads(fixed)
        except:
            pass
            
        # Could not parse JSON
        return None

def generate_questions(tool_name, text_chunk, link=None, chunk_id=None):
    """Generate questions for a given text chunk."""
    chunk_info = f"chunk {chunk_id}" if chunk_id is not None else "text"
    print(f"Generating questions for {chunk_info}...")
  
 
    prompt = f"""
    You are an expert in generating high-quality training data for language models. Your task is to create **exactly 100** question-answer pairs about **{tool_name}**, using the Information below from the R package .md and .RMD files for the particular tool. Each question-answer pair should be **content-focused** and **detailed** to provide a comprehensive understanding of the tool. The questions should be **varied** and **cover different aspects** of the tool to ensure a broad coverage of information.  

    ## **Requirements for Question-Answer Pairs:**

    ### **1. Content-Focused Questions & Answers**
    - Every **question and answer must explicitly mention** the tool name **"{tool_name}"**.
    - Ensure all questions are **fully supported** by the provided content and .MD and .RMD files.
    - Avoid referencing specific locations in the .MD and .RMD files (e.g., "as mentioned in Equation 2" or "refer to Section 4").
    - If an **answer is incomplete or missing**, retain the question but use a **single hyphen ("-")** as the answer.

    ### **2. Depth & Detail**
    - Provide **thorough explanations** for complete answers (aim for ~500 words where possible).
    - Expand on concepts using:
    - **Technical explanations**
    - **Mathematical equations (if applicable)**
    - **Step-by-step breakdowns of complex topics**, where relevant.

    ### **3. Installation and exectusion Approach**
    - Since the manual is part of a **R package Readme**, prioritize:
    - **execution**
    - **functions signatures**
    - **functions paramters**
    - **Example usage**
    - **Technical depth**
    - **instalation. rigor**
    - Relevant **equations, theories, or methodologies,installation, techniqual, execution questions**, where applicable.

    ### **4. Format & Clarity**
    - Format the output as a **valid JSON array** using the following structure:

    ```json
    [
        {{
            "instruction1": "Question about {tool_name}?",
            "output1": "Answer about {tool_name}.\\nSource: {link}"
        }},
        {{
            "instruction2": "Question about {tool_name}?",
            "output2": "Answer about {tool_name}.\\nSource: {link}"
        }},
        ...
    ]
    ```

    - Use **Markdown triple backticks** (` ``` `) for any **code snippets**.
    - If relevant, include **tables** or structured comparisons for clarity.

    ### **5. Strict Adherence to Provided Information**
    - **Do not fabricate** any features, functionalities, or details **not explicitly found** in the manual.
    - Stay strictly within the **scope of the provided text**.


    ### **7. Citations & References**
    - At the end of each complete answer, **cite the source** as follows:
    {f"SOURCE LINK: {link}" if link else ""}

    ---

    ## **TOOL NAME:**  
    **{tool_name}**

    ---

    ## **TEXT FROM MANUAL:**  
    {text_chunk}

    ---
    """
    

 
 
    with open("prompt.txt", "w") as f:
        f.write(prompt)
    
    # Load the prompt from the file
    
    #exit(0)
    # Add retries for API reliability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                #model="gemini-2.0-flash",
                #model="gemini-1.5-pro",
                #model = "gemini-2.0-flash-lite",
                model = "gemini-2.5-pro-exp-03-25",

                
                contents=prompt  
            )
            # Process the response to ensure it's valid JSON
            response_text = response.text
            
            # Print a small preview of the response for debugging
            print(f"Response preview (first 100 chars): {response_text[:100]}...")
            
            # Use improved JSON extraction and parsing
            questions_json = extract_json_from_response(response_text)
            
            if questions_json and isinstance(questions_json, list):
                print(f"Successfully parsed JSON with {len(questions_json)} questions")
                return questions_json
            else:
                print(f"Failed to parse JSON from model response on attempt {attempt+1}/{max_retries}.")
                if attempt == max_retries - 1:
                    print(f"Raw response: {response_text[:500]}...")
                    # Save the problematic response to a file for debugging
                    debug_dir = "debug_responses"
                    os.makedirs(debug_dir, exist_ok=True)
                    with open(f"{debug_dir}/failed_response_{int(time.time())}.txt", "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(f"Full response saved to debug file for inspection")
                    return {"error": "Failed to parse JSON", "raw_response": response_text[:500]}
                time.sleep(2)  # Wait before retry
                
        except Exception as e:
            print(f"API error on attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2)  # Wait before retry

def save_to_json(questions, json_filename):
    """Save questions to JSON format"""
    try:
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(questions, jsonfile, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def file_exists_with_content(filepath):
    """Check if a file exists and has content (not empty)."""
    if not os.path.exists(filepath):
        return False
    
    # Check if file has content
    return os.path.getsize(filepath) > 0

# Dictionary to store all questions for each tool
all_tool_questions = {}

# Read the Excel file with tool URLs once
excel_path = os.path.join(os.getcwd(), "PRSGPT-Tools-DatasetLinks.xlsx")
tools_df = pd.read_excel(excel_path)

 
 
def convert_notebook_to_python(notebook_path):
    """Convert a Jupyter notebook to Python code."""
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Convert to Python
        python_exporter = PythonExporter()
        python_code, _ = python_exporter.from_notebook_node(notebook)
        
        return python_code
    except Exception as e:
        print(f"Error converting notebook {notebook_path}: {e}")
        return None

def find_jupyter_notebooks(directory):
    """Find all Jupyter notebooks in a directory."""
    notebook_paths = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".ipynb") and not file.startswith("."):
                    notebook_paths.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error finding notebooks in {directory}: {e}")
    
    return notebook_paths

def process_notebooks_for_tool(tool_name, python_package_dir):
    """Process all notebooks for a tool and merge them into a single text."""
    print(f"Processing Jupyter notebooks for {tool_name}...")
    
    # Find all notebooks
    notebook_paths = find_jupyter_notebooks(python_package_dir)
    
    if not notebook_paths:
        print(f"No Jupyter notebooks found for {tool_name}")
        return None
    
    # Convert notebooks to Python code and merge
    merged_content = ""
    for nb_path in notebook_paths:
        print(f"Converting notebook: {os.path.basename(nb_path)}")
        python_code = convert_notebook_to_python(nb_path)
        if python_code:
            # Add filename as a header comment for organization
            filename = os.path.basename(nb_path)
            merged_content += f"\n\n# ==== From Notebook: {filename} ====\n\n"
            merged_content += python_code
    
    return merged_content

def chunk_text(text, max_chunk_size=8000):
    """Split text into chunks of approximately max_chunk_size characters."""
    # Split by double newlines to keep paragraphs together
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the max size, start a new chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def generate_questions_for_python_package(tool_name, text_chunk, link=None, chunk_id=None):
    """Generate questions for a Python package text chunk."""
    chunk_info = f"chunk {chunk_id}" if chunk_id is not None else "text"
    print(f"Generating questions for {chunk_info}...")
  
    prompt = f"""
    You are an expert in generating high-quality training data for language models. Your task is to create **exactly 100** question-answer pairs about **{tool_name}** Python package, using the Information below from Jupyter notebooks and Python files for the particular tool. Each question-answer pair should be **content-focused** and **detailed** to provide a comprehensive understanding of the tool's Python implementation. The questions should be **varied** and **cover different aspects** of the tool to ensure a broad coverage of information.  

    ## **Requirements for Question-Answer Pairs:**

    ### **1. Content-Focused Questions & Answers**
    - Every **question and answer must explicitly mention** the tool name **"{tool_name}"**.
    - Ensure all questions are **fully supported** by the provided content from Jupyter notebooks and Python files.
    - Focus on Python-specific implementation, API usage, and functionality.
    - If an **answer is incomplete or missing**, retain the question but use a **single hyphen ("-")** as the answer.

    ### **2. Depth & Detail**
    - Provide **thorough explanations** for complete answers (aim for ~500 words where possible).
    - Expand on concepts using:
    - **Technical explanations of Python implementation**
    - **Code examples and usage patterns**
    - **Step-by-step breakdowns of complex functions or workflows**, where relevant.

    ### **3. Python Package Approach**
    - Since the content is from Python packages and Jupyter notebooks, prioritize:
    - **Python function usage**
    - **Function signatures and parameters**
    - **Example code snippets**
    - **Python class structure and inheritance**
    - **Proper import statements**
    - **Installation and dependencies**
    - **Common Python patterns and practices**

    ### **4. Format & Clarity**
    - Format the output as a **valid JSON array** using the following structure:

    ```json
    [
        {{
            "instruction1": "Question about {tool_name} Python package?",
            "output1": "Answer about {tool_name} Python implementation.\\nSource: {link}"
        }},
        {{
            "instruction2": "Question about {tool_name} Python package?",
            "output2": "Answer about {tool_name} Python implementation.\\nSource: {link}"
        }},
        ...
    ]
    ```

    - Use **Markdown triple backticks** (` ``` `) for any **code snippets**.
    - If relevant, include **tables** or structured comparisons for clarity.

    ### **5. Strict Adherence to Provided Information**
    - **Do not fabricate** any features, functionalities, or details **not explicitly found** in the provided text.
    - Stay strictly within the **scope of the provided text**.

    ### **6. Citations & References**
    - At the end of each complete answer, **cite the source** as follows:
    {f"SOURCE LINK: {link}" if link else ""}

    ---

    ## **TOOL NAME:**  
    **{tool_name}**

    ---

    ## **TEXT FROM PYTHON PACKAGE JUPYTER NOTEBOOKS:**  
    {text_chunk}

    ---
    """
    
    # Add retries for API reliability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model = "gemini-2.5-flash-preview-05-20",
                contents=prompt  
            )
            # Process the response to ensure it's valid JSON
            response_text = response.text
            
            # Print a small preview of the response for debugging
            print(f"Response preview (first 100 chars): {response_text[:100]}...")
            
            # Use improved JSON extraction and parsing
            questions_json = extract_json_from_response(response_text)
            
            if questions_json and isinstance(questions_json, list):
                print(f"Successfully parsed JSON with {len(questions_json)} questions")
                return questions_json
            else:
                print(f"Failed to parse JSON from model response on attempt {attempt+1}/{max_retries}.")
                if attempt == max_retries - 1:
                    print(f"Raw response: {response_text[:500]}...")
                    # Save the problematic response to a file for debugging
                    debug_dir = "debug_responses"
                    os.makedirs(debug_dir, exist_ok=True)
                    with open(f"{debug_dir}/failed_response_{int(time.time())}.txt", "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(f"Full response saved to debug file for inspection")
                    return {"error": "Failed to parse JSON", "raw_response": response_text[:500]}
                time.sleep(2)  # Wait before retry
                
        except Exception as e:
            print(f"API error on attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2)  # Wait before retry

def process_jupyter_notebooks():
    """Process Jupyter notebooks for each tool and generate questions."""
    print("\n=== Processing Jupyter Notebooks for Question Generation ===\n")
    
    # Read the Excel file with tool URLs
    excel_path = os.path.join(os.getcwd(), "PRSGPT-Tools-DatasetLinks.xlsx")
    tools_df = pd.read_excel(excel_path)
    
    for tool_name in tool_dirs:
        print(f"\nProcessing Jupyter notebooks for tool: {tool_name}")
        
        # Define the Python package directory path
        python_package_dir = os.path.join(tools_path, tool_name, "PythonPackage")
        
        if not os.path.exists(python_package_dir):
            print(f"No PythonPackage directory found for {tool_name}")
            continue
        
        # Process notebooks and merge content
        merged_content = process_notebooks_for_tool(tool_name, python_package_dir)
        
        if not merged_content:
            print(f"No content extracted from Jupyter notebooks for {tool_name}")
            continue
        
        # Define output path for questions
        questions_output_path = os.path.join(python_package_dir, "Questions_Jupyter.json")
        
        # Check if the output file already exists
        if file_exists_with_content(questions_output_path):
            print(f"Questions file already exists for {tool_name}. Skipping...")
            continue
            
        # Get tool link from Excel if available
        tool_row = tools_df[tools_df['Tool'] == tool_name]
        python_link = ""
        if not tool_row.empty and 'Python' in tool_row.columns:
            python_link = str(tool_row["Python"].values[0]).strip() if not pd.isna(tool_row["Python"].values[0]) else ""
        
        # Split content into chunks if it's too large
        chunks = chunk_text(merged_content)
        print(f"Split content into {len(chunks)} chunks")
        
        all_questions = []
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_questions = generate_questions_for_python_package(
                tool_name, 
                chunk, 
                link=python_link,
                chunk_id=i+1
            )
            
            if chunk_questions and isinstance(chunk_questions, list):
                all_questions.extend(chunk_questions)
                print(f"Added {len(chunk_questions)} questions from chunk {i+1}")
            else:
                print(f"Failed to generate questions for chunk {i+1}")
        
        # Save all questions to Questions_Jupyter.json
        if all_questions:
            os.makedirs(os.path.dirname(questions_output_path), exist_ok=True)
            if save_to_json(all_questions, questions_output_path):
                print(f"Successfully saved {len(all_questions)} questions to {questions_output_path}")
            else:
                print(f"Failed to save questions for {tool_name}")
        else:
            print(f"No questions generated for {tool_name}")

# Add this at the end of the file to run the new functionality
# After processing R packages and GitHub readmes, process Jupyter notebooks
process_jupyter_notebooks()





