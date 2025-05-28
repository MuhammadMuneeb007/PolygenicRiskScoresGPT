import os
import json
import csv
import re
import time
from tqdm import tqdm
import glob
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google import genai
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Configure multiple Google Generative AI API keys
API_KEYS = [
    "AIzaSyA-kxIjJAFAeEYz69mBRUdZMNU1xx5bbGY",
     
]

# Create clients for each API key
clients = [genai.Client(api_key=key) for key in API_KEYS]

# Thread-safe lock for file operations
file_lock = threading.Lock()

tools_path = "Tools"
dirs = [d for d in os.listdir(tools_path) if os.path.isdir(os.path.join(tools_path, d))]
#dirs = ['Samtools']
 


def html_to_markdown(html_content):
    """Convert HTML content to clean markdown-like text."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove scripts, styles, and other unwanted elements
    for script in soup(["script", "style", "meta", "link"]):
        script.extract()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Remove extra newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def find_unique_pages(pages_data, similarity_threshold=0.9):
    """Use cosine similarity to find unique pages by comparing ALL pages with each other."""
    if len(pages_data) <= 1:
        return pages_data
    
    print(f"Comparing {len(pages_data)} pages with each other...")
    
    # Extract texts for vectorization
    texts = [page['content'] for page in pages_data]
    page_names = [page['name'] for page in pages_data]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity matrix for ALL pages
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    print("Similarity matrix calculated. Finding similar pairs...")
    
    # Find all similar pairs
    similar_pairs = []
    for i in range(len(pages_data)):
        for j in range(i + 1, len(pages_data)):
            similarity = similarity_matrix[i][j]
            if similarity > similarity_threshold:
                similar_pairs.append((i, j, similarity))
                print(f"SIMILAR: {page_names[i]} <-> {page_names[j]} (similarity: {similarity:.3f})")
    
    print(f"Found {len(similar_pairs)} similar pairs")
    
    # Create a set to track pages to remove
    pages_to_remove = set()
    
    # For each similar pair, remove the page with fewer words (keep the more detailed one)
    for i, j, similarity in similar_pairs:
        page_i_words = pages_data[i]['word_count']
        page_j_words = pages_data[j]['word_count']
        
        if page_i_words >= page_j_words:
            pages_to_remove.add(j)
            print(f"REMOVING: {page_names[j]} (keeping {page_names[i]} - more words: {page_i_words} vs {page_j_words})")
        else:
            pages_to_remove.add(i)
            print(f"REMOVING: {page_names[i]} (keeping {page_names[j]} - more words: {page_j_words} vs {page_i_words})")
    
    # Keep only unique pages
    unique_pages = [pages_data[i] for i in range(len(pages_data)) if i not in pages_to_remove]
    
    print(f"FINAL RESULT: Keeping {len(unique_pages)} unique pages, removed {len(pages_to_remove)} similar pages")
    
    for page in unique_pages:
        print(f"KEPT: {page['name']} ({page['word_count']} words)")
    
    return unique_pages

def merge_pages(unique_pages):
    """Merge unique pages into single content."""
    merged_content = ""
    
    for page in unique_pages:
        merged_content += f"\n\n## Page: {page['name']}\n\n"
        merged_content += page['content']
    
    return merged_content.strip()

def chunk_text(text, max_words=8000):
    """Divide text into chunks of max_words."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0
    
    for word in words:
        if current_count >= max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_count = 1
        else:
            current_chunk.append(word)
            current_count += 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

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
            # Fix missing or extra commas between objects in array
            fixed_text = re.sub(r'}\s*{', '},{', response_text)
            # Fix trailing commas before closing brackets
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
            
        # Could not parse JSON
        return None

def get_client_for_thread(thread_id):
    """Get a client based on thread ID to distribute API usage."""
    return clients[thread_id % len(clients)]

def generate_questions_threaded(tool_name, text_chunk, chunk_id, thread_id):
    """Generate questions for a text chunk with thread-specific client."""
    print(f"Thread {thread_id}: Generating questions for chunk {chunk_id}...")
    
    client = get_client_for_thread(thread_id)
    
    prompt = f"""
    You are an expert in creating high-quality training data for language models. Generate 50 detailed instruction-output pairs specifically about {tool_name} based on the manual excerpt below.

    TOOL NAME: {tool_name}
    
    TEXT FROM Website:
    {text_chunk}

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
            "output": "To perform PCA analysis using {tool_name}, use the following command:\\n\\n```\\n{tool_name} [complete command with all parameters]\\n```\\n\\nThis command will [detailed explanation of what happens]..."
        }},
        ...
    ]
    
    IMPORTANT: Generate EXACTLY 50 instruction-output pairs. If complete information is not available for an instruction, set the output value to "-".
    """
    
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt
            )
            
            response_text = response.text
            print(f"Thread {thread_id}: Response preview: {response_text[:100]}...")
            
            # Use improved JSON extraction
            questions_json = extract_json_from_response(response_text)
            
            if questions_json and isinstance(questions_json, list):
                print(f"Thread {thread_id}: Successfully generated {len(questions_json)} questions for chunk {chunk_id}")
                return questions_json, chunk_id, thread_id
            else:
                print(f"Thread {thread_id}: Failed to parse JSON on attempt {attempt+1} for chunk {chunk_id}")
                if attempt == max_retries - 1:
                    print(f"Thread {thread_id}: Raw response: {response_text[:500]}...")
                    return [], chunk_id, thread_id
                time.sleep(2)
                
        except Exception as e:
            print(f"Thread {thread_id}: API error on attempt {attempt+1} for chunk {chunk_id}: {e}")
            if attempt == max_retries - 1:
                return [], chunk_id, thread_id
            time.sleep(2)
    
    return [], chunk_id, thread_id

def save_chunk_questions_threadsafe(questions, chunk_id, tool_name, questions_dir):
    """Save chunk questions in a thread-safe manner."""
    if not questions:
        return
        
    chunk_file = os.path.join(questions_dir, f"chunk_{chunk_id}_questions.json")
    
    with file_lock:
        try:
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=2)
            print(f"Chunk {chunk_id} questions saved to {chunk_file}")
        except Exception as e:
            print(f"Error saving chunk {chunk_id}: {e}")

def process_chunks_multithreaded(tool_name, chunks, questions_dir, max_workers=12):
    """Process all chunks using multithreading."""
    print(f"\nProcessing {len(chunks)} chunks with {max_workers} threads...")
    
    all_questions = []
    chunks_to_process = []
    
    # Check which chunks need processing
    for i, chunk in enumerate(chunks):
        chunk_id = i + 1
        chunk_exists, existing_questions = check_chunk_exists(chunk_id, tool_name, questions_dir)
        
        if chunk_exists and existing_questions:
            print(f"Using existing questions for chunk {chunk_id}")
            # Add metadata to existing questions
            for q in existing_questions:
                q['chunk_id'] = chunk_id
                q['tool_name'] = tool_name
            all_questions.extend(existing_questions)
        else:
            chunks_to_process.append((chunk, chunk_id))
    
    if not chunks_to_process:
        print("All chunks already processed!")
        return all_questions
    
    print(f"Need to process {len(chunks_to_process)} new chunks")
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk processing tasks
        future_to_chunk = {}
        for i, (chunk, chunk_id) in enumerate(chunks_to_process):
            thread_id = i  # This will be used to select API client
            future = executor.submit(generate_questions_threaded, tool_name, chunk, chunk_id, thread_id)
            future_to_chunk[future] = (chunk_id, thread_id)
        
        # Collect results as they complete
        progress_bar = tqdm(total=len(chunks_to_process), desc="Processing chunks")
        
        for future in as_completed(future_to_chunk):
            chunk_id, thread_id = future_to_chunk[future]
            try:
                questions, returned_chunk_id, returned_thread_id = future.result()
                
                if questions:
                    # Add metadata to each question
                    for q in questions:
                        q['chunk_id'] = chunk_id
                        q['tool_name'] = tool_name
                    
                    # Save chunk questions thread-safely
                    save_chunk_questions_threadsafe(questions, chunk_id, tool_name, questions_dir)
                    
                    # Add to overall results
                    all_questions.extend(questions)
                    
                    print(f"Thread {returned_thread_id}: Completed chunk {returned_chunk_id} with {len(questions)} questions")
                else:
                    print(f"Thread {returned_thread_id}: No questions generated for chunk {returned_chunk_id}")
                    
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
            
            progress_bar.update(1)
        
        progress_bar.close()
    
    print(f"Multithreaded processing complete. Total questions: {len(all_questions)}")
    return all_questions

def save_questions_to_files(questions, tool_name, output_dir):
    """Save questions to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_filename = os.path.join(output_dir, f"{tool_name}_all_questions.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2)
    
    # Save as CSV - dynamically determine all fieldnames
    if questions:
        # Get all unique fieldnames from all questions
        all_fieldnames = set()
        for q in questions:
            all_fieldnames.update(q.keys())
        
        # Sort fieldnames for consistent output, with common fields first
        common_fields = ['instruction', 'output', 'chunk_id', 'tool_name']
        other_fields = sorted([f for f in all_fieldnames if f not in common_fields])
        fieldnames = [f for f in common_fields if f in all_fieldnames] + other_fields
        
        csv_filename = os.path.join(output_dir, f"{tool_name}_all_questions.csv")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in questions:
                writer.writerow(q)
        
        print(f"All questions saved to {json_filename} and {csv_filename}")
        return json_filename, csv_filename
    else:
        print(f"No questions to save for {tool_name}")
        return json_filename, None

def merge_all_questions(all_questions, tool_name, output_dir):
    """Merge all questions from all chunks and save master files."""
    print(f"\n=== Merging all questions for {tool_name} ===")
    
    # Remove duplicates based on instruction text
    seen_instructions = set()
    unique_questions = []
    
    for question in all_questions:
        instruction = question.get('instruction', '').strip().lower()
        if instruction and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            unique_questions.append(question)
        else:
            print(f"Removed duplicate: {question.get('instruction', '')[:50]}...")
    
    print(f"Removed {len(all_questions) - len(unique_questions)} duplicate questions")
    print(f"Final unique questions: {len(unique_questions)}")
    
    # Save merged questions
    master_json, master_csv = save_questions_to_files(unique_questions, f"{tool_name}_merged", output_dir)
    
    # Create summary statistics
    stats = {
        'tool_name': tool_name,
        'total_questions': len(unique_questions),
        'questions_by_chunk': {},
        'questions_with_output': len([q for q in unique_questions if q.get('output', '') != '-']),
        'questions_without_output': len([q for q in unique_questions if q.get('output', '') == '-'])
    }
    
    # Count questions by chunk
    for question in unique_questions:
        chunk_id = question.get('chunk_id', 'unknown')
        stats['questions_by_chunk'][chunk_id] = stats['questions_by_chunk'].get(chunk_id, 0) + 1
    
    # Save statistics
    stats_file = os.path.join(output_dir, f"{tool_name}_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {stats_file}")
    print(f"Questions with complete answers: {stats['questions_with_output']}")
    print(f"Questions with incomplete answers: {stats['questions_without_output']}")
    
    return unique_questions, master_json, master_csv

def check_chunk_exists(chunk_id, tool_name, questions_dir):
    """Check if questions for a specific chunk already exist."""
    chunk_file = os.path.join(questions_dir, f"chunk_{chunk_id}_questions.json")
    if os.path.exists(chunk_file):
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
                if questions and len(questions) > 0:
                    print(f"Chunk {chunk_id} questions already exist with {len(questions)} questions, skipping...")
                    return True, questions
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading existing chunk file: {e}")
            return False, None
    return False, None

# Main processing loop
all_tools_questions = {}

for tool_name in dirs:
    print(f"\n{'='*50}")
    print(f"Processing tool: {tool_name}")
    print(f"{'='*50}")
    
    tool_dir = os.path.join(tools_path, tool_name)
    
    # Check if output files already exist
    questions_dir = os.path.join(tool_dir, "WebsiteQuestions")
    json_output_path = os.path.join(questions_dir, f"{tool_name}_all_questions.json")
    
    #if os.path.exists(json_output_path):
    #    print(f"Output file {json_output_path} already exists, skipping {tool_name}...")
    #    continue
    
    # Step 1: Load data
    print("\nStep 1: Loading HTML/SHTML files...")
    html_files = glob.glob(os.path.join(tool_dir, "Website", "*.html"))
    shtml_files = glob.glob(os.path.join(tool_dir, "Website", "*.shtml"))
    web_files = html_files + shtml_files
    print(f"Found {len(web_files)} files")
    
    if not web_files:
        print(f"No web files found for {tool_name}, skipping...")
        continue
    
    # Step 2: Convert to markdown and collect pages
    print("\nStep 2: Converting to markdown...")
    pages_data = []
    
    for web_file in tqdm(web_files, desc="Converting files"):
        try:
            base_name = os.path.basename(web_file).split('.')[0]
            
            with open(web_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Convert to markdown
            markdown_content = html_to_markdown(html_content)
            
            # Skip very short content
            if len(markdown_content.split()) < 50:
                print(f"Skipping {base_name} - too short ({len(markdown_content.split())} words)")
                continue
            
            pages_data.append({
                'name': base_name,
                'content': markdown_content,
                'word_count': len(markdown_content.split())
            })
            
        except Exception as e:
            print(f"Error processing {web_file}: {e}")
    
    print(f"Loaded {len(pages_data)} pages")
    
    if not pages_data:
        print(f"No valid pages found for {tool_name}, skipping...")
        continue
    
    # Step 3: Find unique pages by comparing ALL pages with each other
    print("\nStep 3: Comparing ALL pages with each other to find unique ones...")
    unique_pages = find_unique_pages(pages_data, similarity_threshold=0.9)
    print(f"FINAL: Kept {len(unique_pages)} unique pages out of {len(pages_data)} total pages")
    
    # Step 4: Merge unique pages
    print("\nStep 4: Merging unique pages...")
    merged_content = merge_pages(unique_pages)
    print(f"Merged content has {len(merged_content.split())} words")
    
    # Save merged content
    merged_file = os.path.join(tool_dir, "Website", f"{tool_name}_merged_unique.txt")
    os.makedirs(os.path.dirname(merged_file), exist_ok=True)
    with open(merged_file, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    print(f"Merged content saved to {merged_file}")
    
    # Step 5: Divide into chunks
    print("\nStep 5: Dividing into chunks...")
    chunks = chunk_text(merged_content, max_words=8000)
    print(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.split())} words")
    
    # Step 6: Process all chunks with multithreading
    print("\nStep 6: Processing all chunks with multithreading...")
    
    # Create output directory
    os.makedirs(questions_dir, exist_ok=True)
    
    # Process chunks using multithreading
    all_questions = process_chunks_multithreaded(tool_name, chunks, questions_dir, max_workers=12)
    
    # Step 7: Merge all questions from all chunks
    if all_questions:
        print(f"\nStep 7: Merging all {len(all_questions)} questions...")
        unique_questions, master_json, master_csv = merge_all_questions(all_questions, tool_name, questions_dir)
        all_tools_questions[tool_name] = unique_questions
        
        print(f"\n{tool_name} Processing Summary:")
        print(f"- Total questions generated: {len(all_questions)}")
        print(f"- Unique questions after deduplication: {len(unique_questions)}")
        print(f"- Master files: {master_json}, {master_csv}")
    else:
        print(f"No questions generated for {tool_name}")