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
 
# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyByI4RR_7WlpjzW-CKS-3V4F3R_mZ1GTto"
client = genai.Client(api_key=GOOGLE_API_KEY)
import os
import glob
import requests
from bs4 import BeautifulSoup

tools_path = "Tools"

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



def find_python_files(directory):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def read_python_file(file_path):
    """Read a Python file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def find_all_files(directory):
    """Find all files in a directory and its subdirectories."""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def is_genetic_data_file(filename):
    """Check if a file is a genetic data file based on extension."""
    genetic_extensions = [
        '.bed', '.bim', '.fam',  # PLINK binary format
        '.ped', '.map',          # PLINK text format
        '.vcf', '.vcf.gz',       # Variant Call Format
        '.geno', '.gen',         # Oxford format
        '.haps', '.sample',      # SHAPEIT format
        '.bgen',                 # BGEN format
        '.dosage',               # Dosage files
        '.frq', '.freq',         # Frequency files
        '.assoc',                # Association results
        '.gwas',                 # GWAS results
        '.sumstats',             # Summary statistics
        '.ldsc',                 # LD score files
        '.prs'                   # PRS files
    ]
    
    ext = os.path.splitext(filename)[1].lower()
    # Handle double extensions like .vcf.gz
    if ext == '.gz' and len(os.path.splitext(os.path.splitext(filename)[0])[1]) > 0:
        ext = os.path.splitext(os.path.splitext(filename)[0])[1].lower() + ext
    
    return ext in genetic_extensions

def process_python_files_for_tool(tool_name, python_package_dir):
    """Process all Python files for a tool and merge their content."""
    print(f"Processing Python files for {tool_name}...")
    
    # Find all files
    all_files = find_all_files(python_package_dir)
    
    # Separate Python files and other files
    python_files = [f for f in all_files if f.endswith('.py')]
    genetic_data_files = [f for f in all_files if is_genetic_data_file(f)]
    
    if not python_files:
        print(f"No Python files found for {tool_name}")
        return None
    
    # Create a directory structure representation
    dir_structure = "Directory structure:\n"
    for root, dirs, files in os.walk(python_package_dir):
        level = root.replace(python_package_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        dir_structure += f"{indent}{os.path.basename(root)}/\n"
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            file_path = os.path.join(root, f)
            if f.endswith('.py') or f.endswith('.ipynb') or f.endswith('.json') or f.endswith('.csv') or f.endswith('.txt') or is_genetic_data_file(f):
                # Highlight genetic data files
                highlight = " [GENETIC DATA]" if is_genetic_data_file(f) else ""
                dir_structure += f"{sub_indent}{f}{highlight}\n"
    
    # List all genetic data files
    genetic_data_section = ""
    if genetic_data_files:
        genetic_data_section = "\n# Genetic Data Files Detected\n\n"
        for file_path in genetic_data_files:
            relative_path = os.path.relpath(file_path, python_package_dir)
            file_size = os.path.getsize(file_path)
            file_size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
            genetic_data_section += f"- {relative_path} ({file_size_str})\n"
    
    # Merge Python file contents
    merged_content = f"# Directory Structure for {tool_name} Python Package\n{dir_structure}\n{genetic_data_section}\n# Python Files Content\n\n"
    
    for py_file in python_files:
        relative_path = os.path.relpath(py_file, python_package_dir)
        file_content = read_python_file(py_file)
        
        # Add filename as a header for organization
        merged_content += f"\n\n# ==== From Python File: {relative_path} ====\n\n"
        merged_content += file_content
    
    return merged_content

def chunk_text(text, max_chunk_size=8000):
    """Split text into chunks of approximately max_chunk_size characters."""
    # Split by file headers to keep files together if possible
    file_sections = re.split(r'(# ==== From Python File: .*? ====\n\n)', text)
    
    chunks = []
    current_chunk = ""
    
    i = 0
    while i < len(file_sections):
        # If this is a header section
        if i < len(file_sections) - 1 and file_sections[i].startswith('# ==== From Python File:'):
            header = file_sections[i]
            content = file_sections[i+1] if i+1 < len(file_sections) else ""
            
            # If adding this section would exceed the max size and we already have content, start a new chunk
            if len(current_chunk) + len(header) + len(content) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = header + content
            else:
                current_chunk += header + content
            
            i += 2  # Skip the content section we just processed
        else:
            # If adding this section would exceed the max size, start a new chunk
            if len(current_chunk) + len(file_sections[i]) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = file_sections[i]
            else:
                current_chunk += file_sections[i]
            i += 1
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def analyze_for_sample_data(merged_content):
    """Analyze the content to check if sample/test data is present."""
    sample_data_indicators = [
        # Genetic data patterns
        r'\.bed',
        r'\.bim',
        r'\.fam',
        r'\.vcf',
        r'\.ped',
        r'\.map',
        r'\.geno',
        r'\.gen',
        r'\.haps',
        r'\.sample',
        r'\.bgen',
        r'\.dosage',
        r'\.prs',
        r'\.gwas',
        r'\.assoc',
        r'\.sumstats',
        r'plink',
        r'snp',
        r'genotype',
        r'allele',
        r'locus',
        r'variant',
        r'chromosome',
        r'genomic',
        r'haplotype',
        
        # General data loading patterns
        r'test_data',
        r'sample_data',
        r'example_data',
        r'\.csv',
        r'\.txt',
        r'\.json',
        r'\.h5',
        r'load_data',
        r'read_csv',
        r'pd\.read_',
        r'open\([\'"].*?[\'"]\, ?[\'"]r[\'"]\)',
        r'datasets?\.load_',
        r'glob\('
    ]
    
    data_findings = []
    for indicator in sample_data_indicators:
        matches = re.finditer(indicator, merged_content, re.IGNORECASE)
        for match in matches:
            line_start = merged_content.rfind('\n', 0, match.start()) + 1
            line_end = merged_content.find('\n', match.end())
            if line_end == -1:
                line_end = len(merged_content)
            
            line = merged_content[line_start:line_end].strip()
            if line and not line.startswith('#'):
                data_findings.append(line)
    
    return list(set(data_findings))  # Remove duplicates

def generate_questions_for_python_code(tool_name, text_chunk, data_findings=None, genetic_files=None, link=None, chunk_id=None):
    """Generate questions focusing on Python code, hyperparameters, and implementation details."""
    chunk_info = f"chunk {chunk_id}" if chunk_id is not None else "text"
    print(f"Generating questions for {chunk_info}...")
    
    # Prepare data findings section
    data_findings_text = ""
    if data_findings and len(data_findings) > 0:
        data_findings_text = "## **SAMPLE/TEST DATA DETECTED:**\n"
        for finding in data_findings:
            data_findings_text += f"- {finding}\n"
  
    prompt = f"""
    You are an expert in generating high-quality training data for language models. Your task is to create **exactly 100** question-answer pairs about **{tool_name}** Python implementation, focusing on code structure, hyperparameters, and implementation details from the provided Python files. Each question-answer pair should be **content-focused** and **detailed** to provide a comprehensive understanding of how to use and configure the tool in Python. The questions should be **varied** and **cover different aspects** of the code.

    ## **Requirements for Question-Answer Pairs:**

    ### **1. Content-Focused Questions & Answers**
    - Every **question and answer must explicitly mention** the tool name **"{tool_name}"**.
    - Questions should focus on **hyperparameters**, **configuration options**, **function parameters**, and **implementation details**.
    - If an **answer is incomplete or missing**, retain the question but use a **single hyphen ("-")** as the answer.

    ### **2. Hyperparameter-Focused Questions**
    - Prioritize questions about **hyperparameters** and their impact on performance.
    - Identify **all possible configuration settings** that can be adjusted.
    - Explain **default values** and **recommended ranges** for parameters.
    - Discuss the **effects of changing parameters** on the tool's behavior.

    ### **3. Code Implementation Focus**
    - Focus on:
      - **Function signatures** and their parameters
      - **Class structure** and inheritance patterns
      - **Default parameter values**
      - **File organization** and dependencies
      - **Example usage patterns**
      - **Error handling**
      - **Performance considerations**
    - Identify if **sample or test data** is present and how it's used.

    ### **4. Genetic Data Format Questions**
    - Since this tool works with genetic data, include questions about:
      - **Supported genetic file formats** (BED, BIM, FAM, VCF, etc.)
      - **How to load genetic data** into the tool
      - **Required data formats and transformations**
      - **Sample data provided with the package**
      - **Genetic data preprocessing steps**
      - **How to handle different types of genetic variants**

    ### **5. Format & Clarity**
    - Format the output as a **valid JSON array** using the following structure:

    ```json
    [
        {{ 
            "instruction1": "Question about {tool_name} Python implementation or hyperparameters?", 
            "output1": "Answer about {tool_name} Python implementation, with specific parameter details.\\nSource: {link}" 
        }},
        {{ 
            "instruction2": "Question about configuring {tool_name} in Python?", 
            "output2": "Answer about {tool_name} configuration options and code structure.\\nSource: {link}" 
        }},
        ...
    ]
    ```

    - Use **Markdown triple backticks** (` ``` `) for any **code snippets**.
    - If relevant, include **tables** or structured comparisons for clarity.

    ### **6. Strict Adherence to Provided Information**
    - **Do not fabricate** any features, functionalities, or details **not explicitly found** in the provided code.
    - Stay strictly within the **scope of the provided Python files**.

    ---

    ## **TOOL NAME:**  
    **{tool_name}**

    {data_findings_text}

    ---

    ## **PYTHON CODE CONTENT:**  
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

def process_python_code():
    """Process Python code files for each tool and generate questions."""
    print("\n=== Processing Python Code Files for Question Generation ===\n")
    
    # Read the Excel file with tool URLs
    excel_path = os.path.join(os.getcwd(), "PRSGPT-Tools-DatasetLinks.xlsx")
    tools_df = pd.read_excel(excel_path)
    
    for tool_name in tool_dirs:
        print(f"\nProcessing Python code for tool: {tool_name}")
        
        # Define the Python package directory path
        python_package_dir = os.path.join(tools_path, tool_name, "PythonPackage")
        
        if not os.path.exists(python_package_dir):
            print(f"No PythonPackage directory found for {tool_name}")
            continue
        
        # Define output path for questions
        questions_output_path = os.path.join(python_package_dir, "Questions_Code.json")
        
        # Check if the output file already exists
        if file_exists_with_content(questions_output_path):
            print(f"Questions_Code.json file already exists for {tool_name}. Skipping...")
            continue
        
        # Find genetic data files
        all_files = find_all_files(python_package_dir)
        genetic_files = [f for f in all_files if is_genetic_data_file(f)]
        if genetic_files:
            print(f"Found {len(genetic_files)} genetic data files in {tool_name}")
        
        # Process Python files and merge content
        merged_content = process_python_files_for_tool(tool_name, python_package_dir)
        
        if not merged_content:
            print(f"No content extracted from Python files for {tool_name}")
            continue
        
        # Analyze for sample/test data
        data_findings = analyze_for_sample_data(merged_content)
        if data_findings:
            print(f"Found {len(data_findings)} potential sample/test data references")
            
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
            chunk_questions = generate_questions_for_python_code(
                tool_name, 
                chunk, 
                data_findings=data_findings,
                genetic_files=genetic_files,
                link=python_link,
                chunk_id=i+1
            )
            
            if chunk_questions and isinstance(chunk_questions, list):
                all_questions.extend(chunk_questions)
                print(f"Added {len(chunk_questions)} questions from chunk {i+1}")
            else:
                print(f"Failed to generate questions for chunk {i+1}")
        
        # Save all questions to Questions_Code.json
        if all_questions:
            os.makedirs(os.path.dirname(questions_output_path), exist_ok=True)
            if save_to_json(all_questions, questions_output_path):
                print(f"Successfully saved {len(all_questions)} questions to {questions_output_path}")
            else:
                print(f"Failed to save questions for {tool_name}")
        else:
            print(f"No questions generated for {tool_name}")

# After processing R packages, process Python code files
process_python_code()





