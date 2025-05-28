import os
import glob
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import requests
from bs4 import BeautifulSoup
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tqdm

# Force CPU usage and clear CUDA cache
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all CUDA devices
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def process_pdf_article(input_path, output_dir):
    """
    Process PDF articles using marker to extract content
    
    Args:
        input_path: Path to the input PDF file
        output_dir: Directory to save the extracted text
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename without extension for the output file
        base_name = os.path.basename(input_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # Create PDF converter with device explicitly set to CPU
        converter = PdfConverter(
            artifact_dict=create_model_dict(device="cpu"),
        )
        
        # Convert PDF file
        rendered = converter(input_path)
        
        # Extract text from rendered document
        text, _, _ = text_from_rendered(rendered)
        
        # Save extracted content to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully processed {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

# Process files one by one without multiprocessing
def process_files_sequential(file_args):
    successful = 0
    failed = 0
    for args in tqdm.tqdm(file_args):
        result = process_pdf_article(*args)
        if result:
            successful += 1
        else:
            failed += 1
    return successful, failed

# Main processing loop
def main():
    # Get all tool directories in "Tools"
    tools_path = "Tools"
    if not os.path.exists(tools_path):
        print(f"Error: {tools_path} directory not found!")
        return
    
    tool_dirs = [d for d in os.listdir(tools_path) if os.path.isdir(os.path.join(tools_path, d))]
    
    for tool_name in tool_dirs:
        print("Processing tool:", tool_name)
        tool_dir = os.path.join(tools_path, tool_name)
        
        # Process PDF articles
        pdf_files = glob.glob(os.path.join(tool_dir, "Article", "*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in Article directory")
        
        if pdf_files:
            # Prepare article processing arguments
            article_args = []
            for pdf_file in pdf_files:
                base_name = os.path.basename(pdf_file).split('.')[0]
                output_dir = os.path.join(tool_dir, "Article", base_name)
                article_args.append((pdf_file, output_dir))
            
            # Process articles sequentially
            print("Processing articles...")
            successful, failed = process_files_sequential(article_args)
            print(f"Article processing completed: {successful} successful, {failed} failed")
            
        # Process PDF manuals
        pdf_files = glob.glob(os.path.join(tool_dir, "PDFManual", "*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in PDFManual directory")
        
        if pdf_files:
            # Prepare manual processing arguments
            manual_args = []
            for pdf_file in pdf_files:
                base_name = os.path.basename(pdf_file).split('.')[0]
                output_dir = os.path.join(tool_dir, "PDFManual", base_name)
                manual_args.append((pdf_file, output_dir))
            
            # Process manuals sequentially
            print("Processing manuals...")
            successful, failed = process_files_sequential(manual_args)
            print(f"Manual processing completed: {successful} successful, {failed} failed")

if __name__ == "__main__":
    # This is the safer approach when dealing with libraries that might 
    # spawn their own processes internally
    main()