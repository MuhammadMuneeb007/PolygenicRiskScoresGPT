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


#tool_dirs = [ "DBSLMM" ]

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
    You are an expert in creating high-quality training data for language models. Generate 50 comprehensive instruction-output pairs specifically about {tool_name} based on the manual excerpt below.

    TOOL NAME: {tool_name}
    
    TEXT FROM MANUAL:
    {text_chunk}
    
    {"WEBPAGE LINK: " + link if link else ""}

    REQUIREMENTS:
    1. Create diverse code-focused instruction types:
       - "How to" questions about specific procedures in {tool_name} with COMPLETE command-line examples
       - Parameter explanations with syntax and usage examples
       - Code examples for analyses 
       - Troubleshooting scenarios with command-line solutions
       - File format explanations with examples of how to create/use them

    2. Format guidelines:
       - Instructions should be phrased as direct questions about performing specific tasks
       - EVERY output MUST include complete, runnable command-line examples
       - Provide detailed parameter specifications with their types and default values
       - Show example command outputs where appropriate
       - Include complete workflows with all necessary commands for multi-step processes
       - Aim for comprehensive answers of approximately 500 words when the information allows
       - At the end of each output, include "Source: {link}" so users know where the information comes from

    3. Focus on {tool_name}-specific code examples:
       - Complete commands with all required parameters
       - Show exact syntax with proper flags and options
       - Include realistic file names and paths in examples
       - Demonstrate parameter combinations for different use cases
       - Include any preprocessing or post-processing steps

    Format your response as a valid JSON array with this structure:
    [
        {{
            "instruction": "How do I perform PCA analysis using {tool_name}?",
            "output": "To perform PCA analysis using {tool_name}, use the following command:\\n\\n```\\n{tool_name} [complete command with all parameters]\\n```\\n\\nThis command will [detailed explanation of what happens]...\\n\\nFor example, if your input files are named 'data.bed', 'data.bim', and 'data.fam', you would run:\\n\\n```\\n{tool_name} [specific example with these file names]\\n```\\n\\nThe output will include [explanation of output files and their contents]...\\n\\nSource: {link}"
        }},
        ...
    ]
    
    IMPORTANT INSTRUCTIONS:
    1. Generate EXACTLY 50 instruction-output pairs
    2. For each instruction, check if COMPLETE information is available in the manual excerpt
    3. If complete information for an instruction is NOT available, keep the instruction but set the output value to "-" (a single hyphen)
    4. For instructions with complete information, provide comprehensive answers (aim for ~500 words)
    5. Every complete output MUST include runnable command-line examples that could be directly copied and run by a user
    6. Only include commands and features explicitly stated or implied in the manual excerpt
    7. Don't invent features or capabilities not mentioned in the text
    8. Structure code examples with proper formatting using markdown code blocks with triple backticks
    9. ALWAYS include 'Source: {link}' at the end of each complete output
    10. Prioritize questions about HOW TO DO specific tasks with the tool, rather than conceptual questions
    """
 
    #print(prompt)
    #exit(0)

    # Add retries for API reliability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                #model="gemini-2.0-flash",
                #model="gemini-1.5-pro",
                model="gemini-2.5-flash-preview-05-20",

                
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

def split_text_into_chunks(text, chunk_size=8000):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

# Dictionary to store all questions for each tool
all_tool_questions = {}

# Read the Excel file with tool URLs once
excel_path = os.path.join(os.getcwd(), "PRSGPT-Tools-DatasetLinks.xlsx")
tools_df = pd.read_excel(excel_path)

for tool_name in tool_dirs:
    print("Processing tool:", tool_name)

    print(f"Processing R package for {tool_name}")
    tool_dir = os.path.join(tools_path, tool_name)

    # Create the GitHubQuestions directory for this tool
    github_questions_dir = os.path.join(tool_dir, "GitHubQuestions")
    os.makedirs(github_questions_dir, exist_ok=True)
    
    # Process R packages - recursively search in all subdirectories
    mdfiles = glob.glob(os.path.join(tool_dir,"Rpackage", "**", "*.md"), recursive=True)
    print(f"Found {len(mdfiles)} MD package files")
    rmdfiles = glob.glob(os.path.join(tool_dir,"Rpackage", "**", "*.Rmd"), recursive=True)
    print(f"Found {len(rmdfiles)} Rmd package files")
    rdfiles = glob.glob(os.path.join(tool_dir,"Rpackage", "**", "*.rd"), recursive=True)
    print(f"Found {len(rdfiles)} Rd package files")
    githubreadme = os.path.join(tool_dir, "GitHubReadme", "README.md")
    print(f"Found {githubreadme}")

    # Get GitHub, website, and R package URLs for this tool from the Excel file
    tool_row = tools_df[tools_df['Tool'] == tool_name]
    if not tool_row.empty:
        github_url = str(tool_row["GitHubReadme"].values[0]).strip()
        website_url = str(tool_row["Website"].values[0]).strip()
        r_package = str(tool_row["Rpackage"].values[0]).strip()
    else:
        print(f"Warning: No URL data found for {tool_name} in Excel file")
        github_url = ""
        website_url = ""
        r_package = ""

    # Collect all file contents to merge
    all_content = []
    
    # Process all files and collect their content
    allfiles = [mdfiles, rmdfiles, rdfiles]
    for files in allfiles:
        for f in files:
            try:
                # Read the file content
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    file_content = file.read()
                    if file_content.strip():  # Only add non-empty content
                        all_content.append(f"=== Content from {os.path.basename(f)} ===\n\n{file_content}\n\n")
                        print(f"Added content from {os.path.basename(f)}")
            except Exception as e:
                print(f"Error reading file {f}: {e}")
    
    # Add README content if it exists
    if os.path.exists(githubreadme):
        try:
            with open(githubreadme, 'r', encoding='utf-8', errors='ignore') as file:
                readme_content = file.read()
                if readme_content.strip():  # Only add non-empty content
                    all_content.append(f"=== Content from README.md ===\n\n{readme_content}\n\n")
                    print(f"Added content from README.md")
        except Exception as e:
            print(f"Error reading README file: {e}")
    
    # Check if we found any content
    if not all_content:
        print(f"No content found for {tool_name}, skipping...")
        continue
    
    # Merge all content into a single string
    merged_content = "\n".join(all_content)
    
    # Save the merged content to Github.txt
    github_txt_path = os.path.join(tool_dir, "Github.txt")
    with open(github_txt_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(merged_content)
    print(f"Saved merged content to {github_txt_path}")
    
    # Define questions output path
    questions_output_path = os.path.join(github_questions_dir, f"{tool_name}_all_questions.json")
    
    # Check if Questions.json already exists
    if os.path.exists(questions_output_path) and os.path.getsize(questions_output_path) > 0:
        print(f"Questions.json already exists for {tool_name}, skipping...")
        continue
    
    # Check if the merged content is large and needs to be split into chunks
    word_count = count_words(merged_content)
    print(f"Total word count for {tool_name}: {word_count}")
    
    all_questions = []
    
    if word_count > 8000:
        print(f"Content is large, splitting into chunks...")
        chunks = split_text_into_chunks(merged_content)
        print(f"Split into {len(chunks)} chunks")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_questions = generate_questions(tool_name, chunk, link=github_url or r_package, chunk_id=i+1)
            
            if chunk_questions and isinstance(chunk_questions, list):
                all_questions.extend(chunk_questions)
                print(f"Added {len(chunk_questions)} questions from chunk {i+1}")
                
                # Save interim results after each chunk to prevent data loss
                interim_path = os.path.join(github_questions_dir, f"{tool_name}_chunk{i+1}_questions.json")
                save_to_json(chunk_questions, interim_path)
                print(f"Saved {len(chunk_questions)} questions from chunk {i+1} to {interim_path}")
            else:
                print(f"Failed to generate questions for chunk {i+1}")
    else:
        # Process the entire text as a single chunk
        all_questions = generate_questions(tool_name, merged_content, link=github_url or r_package)
    
    # Save all questions to the output file
    if all_questions and isinstance(all_questions, list):
        if save_to_json(all_questions, questions_output_path):
            print(f"Successfully saved {len(all_questions)} questions to {questions_output_path}")
        else:
            print(f"Failed to save questions for {tool_name}")
    else:
        print(f"Failed to generate questions for {tool_name} or no questions generated")





