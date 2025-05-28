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
GOOGLE_API_KEY = "AIzaSyB5tuMb-jyFnO2pqICY8UcbdMmwAzmNfDg"
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
    You are an expert in generating high-quality training data for language models. Your task is to create **exactly 100** question-answer pairs about **{tool_name}**, using only the information available in the manual of the tool.

    ## **Requirements for Question-Answer Pairs:**

    ### **1. Content-Focused Questions & Answers**
    - Every **question and answer must explicitly mention** the tool name **"{tool_name}"**.
    - Ensure all questions are **fully supported** by the provided manual excerpt.
    - Avoid referencing specific locations in the manual (e.g., "as mentioned in Equation 2" or "refer to Section 4").
    - If an **answer is incomplete or missing**, retain the question but use a **single hyphen ("-")** as the answer.

    ### **2. Depth & Detail**
    - Provide **thorough explanations** for complete answers (aim for ~500 to 600 words where possible).
    - Expand on concepts using:
    - **Technical explanations**
    - **Mathematical equations (if applicable)**
    - **Step-by-step breakdowns of complex topics**, where relevant.

    ### **3. Focus of the questions*
    - Since the manual is part of a tools containing instructions on usage of the tool, prioritize:
    1. **Tool functionalities**
    2. **Technical specifications**
    5. **Troubleshooting tips**
    6. **Coding examples and explanations**
    7. **Paramters usage and examples**
    8. **Output interpretation**
    9. **Data input requirements**
    10. **Data formats**
    11. **Output formats**
    
 
    - Relevant **equations, theories, or methodologies**, where applicable.

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
    #exit(0)
    # Add retries for API reliability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                #model="gemini-2.0-flash",
                #model="gemini-1.5-pro",
                #model = "gemini-2.0-flash-lite",
                #model = "gemini-2.5-pro-exp-03-25",
                #model = "gemini-2.0-flash-thinking-exp-01-21",
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

def save_to_json(questions, json_filename):
    """Save questions to JSON format"""
    try:
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
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

def merge_questions(question_files):
    """Merge multiple question JSON files into a single list of questions"""
    all_questions = []
    for qfile in question_files:
        if os.path.exists(qfile) and os.path.getsize(qfile) > 0:
            try:
                with open(qfile, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                    if isinstance(questions, list):
                        all_questions.extend(questions)
                    else:
                        print(f"Warning: {qfile} does not contain a list of questions")
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {qfile}")
    return all_questions

# Dictionary to store all questions for each tool
all_tool_questions = {}

# Read the Excel file with tool URLs once
excel_path = os.path.join(os.getcwd(), "PRSGPT-Tools-DatasetLinks.xlsx")
tools_df = pd.read_excel(excel_path)

 
# Process articles for each tool
def process_articles():
    print("\n=== Processing Articles for Question Generation ===\n")
    
    for tool_name in tool_dirs:
        print(f"\nProcessing articles for tool: {tool_name}")
        
        # Define the article directory path
        article_dir = os.path.join(tools_path, tool_name, "PDFManual")
        # Use recursive glob to find all .txt files in the directory and its subdirectories
        files = glob.glob(os.path.join(article_dir, "**", "*.txt"), recursive=True)
        print(f"Found {len(files)} article text files for {tool_name}")
        if not files:
            print(f"No article text found for {tool_name}, skipping...")
            continue
        
        

        # Read all .txt files
        article_text = ""
        for txt_file in files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    article_text += f.read() + "\n"
            except Exception as e:
                print(f"Error reading file {txt_file}: {str(e)}")
        
        # Get article link from Excel if available
        tool_row = tools_df[tools_df['Tool'] == tool_name]
        article_link = ""
        if not tool_row.empty and 'PDFManual' in tool_row.columns:
            article_link = str(tool_row["PDFManual"].values[0]).strip() if not pd.isna(tool_row["PDFManual"].values[0]) else ""
        
        # Split the article text into chunks with overlap
        words = article_text.split()
        chunk_size = 10000
        overlap = 100
        
        # Create output directories
        chunk_dir = os.path.join(tools_path, tool_name, "Chunks")
        questions_dir = os.path.join("Tools", tool_name, "PDFManual")
        os.makedirs(chunk_dir, exist_ok=True)
        os.makedirs(questions_dir, exist_ok=True)
        
        # Prepare to collect all question files for later merging
        question_files = []
        
        # Process each chunk with overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk_num = len(question_files) + 1
            
            # Check if output for this chunk already exists
            questions_output_path = os.path.join(questions_dir, f"Questions_Chunk{chunk_num}.json")
            if file_exists_with_content(questions_output_path):
                print(f"{questions_output_path} already exists, skipping chunk {chunk_num}...")
                question_files.append(questions_output_path)
                continue
            
            # Extract chunk with overlap
            end_idx = min(i + chunk_size, len(words))
            chunk = ' '.join(words[i:end_idx])
            
            # Save chunk to file
            chunk_filename = f"{tool_name}_chunk_{chunk_num}.txt"
            chunk_path = os.path.join(chunk_dir, chunk_filename)
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            print(f"Chunk {chunk_num}: {len(chunk.split())} words")
            
            # Generate questions for this chunk
            chunk_questions = generate_questions(tool_name, chunk, link=article_link, chunk_id=f"chunk_{chunk_num}")
            
            # Save questions to separate JSON file
            if chunk_questions and isinstance(chunk_questions, list):
                if save_to_json(chunk_questions, questions_output_path):
                    print(f"Successfully saved {len(chunk_questions)} questions to {questions_output_path}")
                    question_files.append(questions_output_path)
                else:
                    print(f"Failed to save questions for {tool_name} chunk {chunk_num}")
            else:
                print(f"Failed to generate questions for {tool_name} chunk {chunk_num} or no questions generated")
        
        # After processing all chunks, merge the questions
        if question_files:
            all_questions = merge_questions(question_files)
            merged_output_path = os.path.join(questions_dir, "Questions_Merged.json")
            
            if save_to_json(all_questions, merged_output_path):
                print(f"Successfully merged and saved {len(all_questions)} questions to {merged_output_path}")
            else:
                print(f"Failed to save merged questions for {tool_name}")
        else:
            print(f"No question files were generated for {tool_name}")
        
# After processing R packages and GitHub readmes, process articles
process_articles()






