from unsloth import FastLanguageModel
import torch
import json
import os
import numpy as np
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import csv
import html
from tabulate import tabulate  # For creating nice tables in the HTML report
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import re
import ast
from typing import List, Dict, Any, Optional, Tuple, Union
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from collections import defaultdict
import google.generativeai as genai
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextStreamer, 
    RobertaTokenizer, RobertaModel, AutoModel
)

# Set up NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize availability flags
SPACY_AVAILABLE = False
BERT_AVAILABLE = False
WMD_AVAILABLE = False
NLI_AVAILABLE = False
BERTSCORE_AVAILABLE = False
CODEBERT_AVAILABLE = False

# Setup paths
os.environ['GENSIM_DATA_DIR'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/'
os.environ['TRANSFORMERS_CACHE'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/huggingface'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/sentence_transformers'

# ====== CONFIGURATION ======
train_samples = 23278  # Number of training samples to use
validation_samples = 5000  # Number of validation samples to use
test_samples = 100     # Number of test samples to use
max_seq_length = 2048  # Context length for the model

dtype = None
load_in_4bit = True
model_name = "unsloth/Llama-3.2-3B-Instruct"
#model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
#model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

num_epochs = 50
output_dir = "outputs_llama3.2"

#output_dir = "outputsllama3.370b"

import os
os.environ["TRITON_CACHE_DIR"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.triton_cache"

os.environ["HF_HOME"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.huggingface_cache"
os.environ["TORCH_HOME"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.torch_cache"

# ====== INITIALIZE EVALUATION TOOLS ======
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize
except ImportError:
 
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize

try:
    from rouge import Rouge
except ImportError:
    
    from rouge import Rouge

# ====== DATA LOADING ======
def load_jsonl(file_path, num_samples=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    if num_samples and num_samples < len(data):
        data = data[:num_samples]
    
    return data

# Load data from existing files
print(f"Loading data: Train.jsonl ({train_samples} samples)")
raw_train_data = load_jsonl("Train.jsonl", train_samples )  # Load more data initially to filter
print(f"Loading data: Test.jsonl ({test_samples} samples)")
raw_test_data = load_jsonl("Train.jsonl", test_samples )  # Load more data initially to filter

# Function to check if text contains code (common code indicators)
def contains_code(text):
    code_indicators = [
        "```", "def ", "class ", "import ", "from ", 
        "#include", "function(", "function ", "<script", 
        "for(", "for (", "while(", "while (", 
        "if(", "if (", "else{", "else {", 
        "return ", "var ", "let ", "const ", 
        "public ", "private ", "int ", "float ", 
        "void ", "package ", "namespace ", "<div", "<p>",
        "SELECT ", "INSERT ", "UPDATE ", "DELETE FROM",
        "$", "print(", "console.log", "System.out"
    ]
    
    for indicator in code_indicators:
        if indicator in text:
            return True
    return False

# Filter out data that doesn't contain code
print("Filtering questions without code...")
train_data = [item for item in raw_train_data if not contains_code(item["prompt"])][:train_samples]
test_data = [item for item in raw_test_data if not contains_code(item["prompt"])][:test_samples]

print(f"Selected {len(train_data)} training samples without code")
print(f"Selected {len(test_data)} test samples without code")


# If we couldn't find enough samples without code, warn the user
if len(train_data) < train_samples:
    print(f"Warning: Could only find {len(train_data)} training samples without code (requested {train_samples})")
if len(test_data) < test_samples:
    print(f"Warning: Could only find {len(test_data)} test samples without code (requested {test_samples})")





#print(f"Loading data: Test.jsonl ({test_samples} samples)")
#test_data = load_jsonl("Test.jsonl", test_samples)


# Create conversations in the correct format for the model
def create_conversations(data):
    conversations = []
    for item in data:
        conversation = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]}
        ]
        conversations.append({"conversations": conversation})
    return conversations

# ====== MODEL LOADING ======
print(f"Loading model: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# Set up chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# ====== DATASET PREPARATION ======
# Create datasets
train_dataset = Dataset.from_list(create_conversations(train_data))

# Prepare the datasets
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create directories for logs and metrics
os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Create a CSV file to store loss values
loss_file_path = os.path.join(output_dir, "training_loss.csv")
with open(loss_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'step', 'loss'])

# Create a list to store all loss values
all_losses = []

# Create a TensorBoard writer
tb_writer = SummaryWriter(os.path.join(output_dir, "logs"))

# Custom callback to capture loss values
class LossLoggingCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if logs and "loss" in logs:
            current_epoch = state.epoch
            current_step = state.global_step
            current_loss = logs["loss"]
            
            # Store the loss in CSV
            with open(loss_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([current_epoch, current_step, current_loss])
            
            # Store in memory for plotting later
            all_losses.append({
                'epoch': current_epoch,
                'step': current_step,
                'loss': current_loss
            })
            
            # Log to TensorBoard
            tb_writer.add_scalar('train/loss', current_loss, current_step)
            
            # Print loss every 10 steps
            if current_step % 10 == 0:
                print(f"Step {current_step}: Loss = {current_loss:.4f}")

# Function to configure publication-quality plot style
def setup_publication_plot_style():
    """Configure matplotlib for publication-quality plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    mpl.rcParams['grid.linewidth'] = 0.8
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', 
        '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9'
    ])

# Function to plot the loss curve
def plot_loss_curve():
    """Create publication-quality plots of the training loss"""
    if not all_losses:  # Check if the list is empty
        print("Warning: No loss data available to plot.")
        return
        
    setup_publication_plot_style()
    
    # Convert all_losses to DataFrame
    loss_df = pd.DataFrame(all_losses)
    
    # Check if required columns exist
    required_columns = ['step', 'loss']
    missing_columns = [col for col in required_columns if col not in loss_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in loss data: {missing_columns}")
        # If 'step' is missing but we have 'global_step', use that instead
        if 'step' in missing_columns and 'global_step' in loss_df.columns:
            loss_df['step'] = loss_df['global_step']
            missing_columns.remove('step')
        # If still missing required columns, we can't plot
        if missing_columns:
            print("Cannot generate loss plot due to missing data.")
            return
    
    # Save as CSV (in addition to the per-step CSV we're already creating)
    loss_df.to_csv(os.path.join(output_dir, "loss_summary.csv"), index=False)
    
    # Plot loss curve
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot with a professional style
    ax.plot(loss_df['step'], loss_df['loss'], 
            marker='o', markersize=4, linestyle='-', 
            color='#0173B2', alpha=0.7, markevery=5,
            markerfacecolor='white', markeredgewidth=1.5)
    
    # Add a trend line using moving average
    window_size = min(15, len(loss_df) // 5) if len(loss_df) > 20 else 1
    if window_size > 1:
        loss_df['trend'] = loss_df['loss'].rolling(window=window_size, center=True).mean()
        ax.plot(loss_df['step'], loss_df['trend'], 
                linestyle='-', linewidth=2, color='#D55E00', alpha=0.9,
                label=f'Trend (MA{window_size})')
    
    ax.set_title('Training Loss', fontweight='bold', pad=15)
    ax.set_xlabel('Training Steps', fontweight='bold', labelpad=10)
    ax.set_ylabel('Loss', fontweight='bold', labelpad=10)
    
    # Set axis limits with some padding
    ymin = max(0, loss_df['loss'].min() * 0.95)
    ymax = loss_df['loss'].max() * 1.05
    ax.set_ylim(ymin, ymax)
    
    # Use professional gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and final loss
    initial_loss = loss_df['loss'].iloc[0]
    final_loss = loss_df['loss'].iloc[-1]
    ax.annotate(f'Initial: {initial_loss:.4f}', 
                xy=(loss_df['step'].iloc[0], initial_loss),
                xytext=(10, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    ax.annotate(f'Final: {final_loss:.4f}', 
                xy=(loss_df['step'].iloc[-1], final_loss),
                xytext=(-10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2'))
    
    # Add legend if we have a trend line
    if window_size > 1:
        ax.legend(loc='upper right')
    
    # Add descriptive text
    plt.figtext(0.02, 0.02, f"Total steps: {len(loss_df)}", 
                ha="left", fontsize=8, fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "loss_curve.png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "plots", "loss_curve.pdf"), bbox_inches='tight')
    plt.close()
    
    # Plot loss curve per epoch
    if 'epoch' in loss_df.columns and len(loss_df['epoch'].unique()) > 1:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        
        # Get unique epochs
        unique_epochs = sorted(loss_df['epoch'].unique())
        
        # Use a color map for distinct epochs
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_epochs)))
        
        # Plot each epoch with a different color
        for i, epoch in enumerate(unique_epochs):
            epoch_df = loss_df[loss_df['epoch'] == epoch]
            ax.plot(epoch_df['step'], epoch_df['loss'], 
                    marker='o', markersize=4, linestyle='-', 
                    color=colors[i], alpha=0.8, markevery=5,
                    markerfacecolor='white', markeredgewidth=1.5,
                    label=f'Epoch {epoch:.1f}')
        
        ax.set_title('Training Loss by Epoch', fontweight='bold', pad=15)
        ax.set_xlabel('Training Steps', fontweight='bold', labelpad=10)
        ax.set_ylabel('Loss', fontweight='bold', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create a custom legend with better placement
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=min(5, len(unique_epochs)), frameon=True, 
                  fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "loss_curve_by_epoch.png"), 
                    bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, "plots", "loss_curve_by_epoch.pdf"), 
                    bbox_inches='tight')
        plt.close()
    
    # Plot average loss per epoch
    if 'epoch' in loss_df.columns and len(loss_df['epoch'].unique()) > 1:
        # Compute average loss per epoch
        avg_loss_by_epoch = loss_df.groupby('epoch')['loss'].mean().reset_index()
        std_loss_by_epoch = loss_df.groupby('epoch')['loss'].std().reset_index()
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot with error bars
        ax.errorbar(avg_loss_by_epoch['epoch'], avg_loss_by_epoch['loss'],
                    yerr=std_loss_by_epoch['loss'], 
                    marker='o', markersize=8, linestyle='-', linewidth=2,
                    color='#0173B2', ecolor='#D55E00', capsize=5, 
                    markerfacecolor='white', markeredgewidth=2)
        
        # Add a trend line
        z = np.polyfit(avg_loss_by_epoch['epoch'], avg_loss_by_epoch['loss'], 1)
        p = np.poly1d(z)
        ax.plot(avg_loss_by_epoch['epoch'], p(avg_loss_by_epoch['epoch']), 
                linestyle='--', linewidth=1.5, color='#029E73', alpha=0.8,
                label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        
        ax.set_title('Average Loss per Epoch', fontweight='bold', pad=15)
        ax.set_xlabel('Epoch', fontweight='bold', labelpad=10)
        ax.set_ylabel('Average Loss', fontweight='bold', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Force x-axis to use integer ticks for epochs
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotations for each point
        for i, (epoch, loss) in enumerate(zip(avg_loss_by_epoch['epoch'], avg_loss_by_epoch['loss'])):
            ax.annotate(f'{loss:.4f}', 
                        xy=(epoch, loss),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "avg_loss_by_epoch.png"), 
                    bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, "plots", "avg_loss_by_epoch.pdf"), 
                    bbox_inches='tight')
        plt.close()

# ====== EVALUATION FUNCTION ======
def evaluate_model(model, tokenizer, test_data, epoch):
    """Evaluate model on test data and compute metrics"""
    print(f"\n===== Evaluating model after Epoch {epoch} =====")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Setup metrics
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    
    predictions = []
    references = []
    prompts = []
    bleu_scores = []
    rouge_scores = []
    results_data = []
    
    # Generate predictions for test data
    for i, item in enumerate(test_data):
        prompt = item["prompt"]
        reference = item["completion"]
        prompts.append(prompt)
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=1024,
                use_cache=True,
                temperature=0.7,
                min_p=0.1
            )
        
        # Decode prediction
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full_output.split("assistant\n\n")[1]
        
       
        
        predictions.append(prediction)
        references.append(reference)
        
        # Calculate BLEU
        ref_tokens = word_tokenize(reference.lower())
        pred_tokens = word_tokenize(prediction.lower())
        
        # BLEU scores (1, 2, 3, 4-gram)
        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        
        bleu_scores.append({
            'bleu-1': bleu1,
            'bleu-2': bleu2,
            'bleu-3': bleu3,
            'bleu-4': bleu4
        })
        
        # Calculate ROUGE
        try:
            rouge_score = rouge.get_scores(prediction, reference)[0]
            rouge_scores.append(rouge_score)
        except Exception as e:
            print(f"Rouge error on sample {i}: {e}")
            rouge_scores.append({'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}})
        
        # Store individual result data
        results_data.append({
            'sample_id': i,
            'prompt': prompt,
            'actual': reference,
            'predicted': prediction,
            'bleu-1': bleu1,
            'bleu-2': bleu2,
            'bleu-3': bleu3,
            'bleu-4': bleu4,
            'rouge-1': rouge_scores[-1]['rouge-1']['f'],
            'rouge-2': rouge_scores[-1]['rouge-2']['f'],
            'rouge-l': rouge_scores[-1]['rouge-l']['f']
        })
            
        print(f"Sample {i+1}/{len(test_data)} processed")
    
    # Aggregate metrics
    avg_bleu1 = np.mean([score['bleu-1'] for score in bleu_scores])
    avg_bleu2 = np.mean([score['bleu-2'] for score in bleu_scores])
    avg_bleu3 = np.mean([score['bleu-3'] for score in bleu_scores])
    avg_bleu4 = np.mean([score['bleu-4'] for score in bleu_scores])
    
    avg_rouge1 = np.mean([score['rouge-1']['f'] for score in rouge_scores])
    avg_rouge2 = np.mean([score['rouge-2']['f'] for score in rouge_scores])
    avg_rougeL = np.mean([score['rouge-l']['f'] for score in rouge_scores])
    
    # Print metrics
    print(f"\nEvaluation Metrics - Epoch {epoch}:")
    print(f"BLEU-1: {avg_bleu1:.4f}")
    print(f"BLEU-2: {avg_bleu2:.4f}")
    print(f"BLEU-3: {avg_bleu3:.4f}")
    print(f"BLEU-4: {avg_bleu4:.4f}")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    
    # Save metrics to file
    metrics = {
        'epoch': epoch,
        'bleu-1': float(avg_bleu1),
        'bleu-2': float(avg_bleu2),
        'bleu-3': float(avg_bleu3),
        'bleu-4': float(avg_bleu4),
        'rouge-1': float(avg_rouge1),
        'rouge-2': float(avg_rouge2),
        'rouge-l': float(avg_rougeL)
    }
    
    with open(os.path.join(output_dir, f"metrics_epoch_{epoch}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(output_dir, f"results_epoch_{epoch}.csv"), index=False)
    
    # Save example generations in a more readable format
    with open(os.path.join(output_dir, f"generations_epoch_{epoch}.txt"), 'w') as f:
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            f.write(f"Example {i+1}:\n")
            f.write(f"Prompt: {prompts[i]}\n\n")
            f.write(f"REFERENCE:\n{ref}\n\n")
            f.write(f"PREDICTION:\n{pred}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Save raw responses (useful for further analysis)
    with open(os.path.join(output_dir, f"raw_responses_epoch_{epoch}.jsonl"), 'w') as f:
        for i, (prompt, ref, pred) in enumerate(zip(prompts, references, predictions)):
            result = {
                "id": i,
                "prompt": prompt,
                "reference": ref,
                "prediction": pred,
                "metrics": {
                    "bleu-1": bleu_scores[i]['bleu-1'],
                    "bleu-2": bleu_scores[i]['bleu-2'],
                    "bleu-3": bleu_scores[i]['bleu-3'],
                    "bleu-4": bleu_scores[i]['bleu-4'],
                    "rouge-1": rouge_scores[i]['rouge-1']['f'],
                    "rouge-2": rouge_scores[i]['rouge-2']['f'],
                    "rouge-l": rouge_scores[i]['rouge-l']['f']
                }
            }
            f.write(json.dumps(result) + "\n")
    
    # Create HTML report with side-by-side comparisons
    generate_html_report(results_data, epoch)
    
    # Create individual plots for this epoch
    plot_epoch_metrics(epoch, results_data)
    
    # Return to training mode
    model.train()
    
    return metrics

def generate_html_report(results_data, epoch):
    """Generate an HTML report showing side-by-side comparison of actual and predicted responses"""
    html_dir = os.path.join(output_dir, "html_reports")
    os.makedirs(html_dir, exist_ok=True)
    
    # Using regular string (not f-string) for the template part with backslashes
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results - Epoch {epoch_num}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .sample {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }}
        .prompt {{
            background-color: #e8f4f8;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
        }}
        .comparison {{
            display: flex;
            gap: 20px;
        }}
        .reference, .prediction {{
            flex: 1;
            padding: 15px;
            border-radius: 5px;
            overflow-wrap: break-word;
        }}
        .reference {{
            background-color: #e8f8e8;
            border-left: 4px solid #27ae60;
        }}
        .prediction {{
            background-color: #f8f4e8;
            border-left: 4px solid #f39c12;
        }}
        .metrics {{
            margin-top: 15px;
            font-size: 0.9em;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .metric {{
            background-color: #eee;
            padding: 5px 10px;
            border-radius: 3px;
        }}
        .highlight {{
            background-color: #ffffcc;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .summary {{
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Evaluation Results - Epoch {epoch_num}</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Average</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
"""
    
    # Use .format() instead of f-string for the template part
    html_content = html_template.format(epoch_num=epoch)
    
    # Calculate statistics
    metrics = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'rouge-1', 'rouge-2', 'rouge-l']
    results_df = pd.DataFrame(results_data)
    
    for metric in metrics:
        avg_val = results_df[metric].mean()
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        
        # Add metric row to the table
        html_content += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{avg_val:.4f}</td>
                    <td>{min_val:.4f}</td>
                    <td>{max_val:.4f}</td>
                </tr>"""
    
    # Using regular string (not f-string) for template parts
    html_content += """
            </table>
        </div>
        
        <h2>Sample-by-Sample Results</h2>
"""
    
    # Add each sample
    for result in results_data:
        sample_id = result['sample_id'] + 1
        prompt = html.escape(result['prompt']).replace('\n', '<br>')
        actual = html.escape(result['actual']).replace('\n', '<br>')
        predicted = html.escape(result['predicted']).replace('\n', '<br>')
        bleu1 = result['bleu-1']
        bleu2 = result['bleu-2']
        bleu3 = result['bleu-3']
        bleu4 = result['bleu-4']
        rouge1 = result['rouge-1']
        rouge2 = result['rouge-2']
        rougeL = result['rouge-l']
        
        # Build each sample div without using f-strings for large blocks
        sample_html = """
        <div class="sample">
            <h3>Sample {}</h3>
            <div class="prompt">
                <strong>Prompt:</strong><br>
                {}
            </div>
            
            <div class="comparison">
                <div class="reference">
                    <h4>Reference (Actual):</h4>
                    {}
                </div>
                
                <div class="prediction">
                    <h4>Prediction (Generated):</h4>
                    {}
                </div>
            </div>
            
            <div class="metrics">
                <div class="metric">BLEU-1: {:.4f}</div>
                <div class="metric">BLEU-2: {:.4f}</div>
                <div class="metric">BLEU-3: {:.4f}</div>
                <div class="metric">BLEU-4: {:.4f}</div>
                <div class="metric">ROUGE-1: {:.4f}</div>
                <div class="metric">ROUGE-2: {:.4f}</div>
                <div class="metric">ROUGE-L: {:.4f}</div>
            </div>
        </div>
        """.format(
            sample_id, prompt, actual, predicted, 
            bleu1, bleu2, bleu3, bleu4, 
            rouge1, rouge2, rougeL
        )
        
        html_content += sample_html
    
    # End tags
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(os.path.join(html_dir, f"results_epoch_{epoch}.html"), 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to '{html_dir}/results_epoch_{epoch}.html'")

def save_test_data_comparison(test_data, predictions, output_path):
    """Save test data and predictions in multiple formats for easy comparison"""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data
    formatted_data = []
    for i, (test_item, pred) in enumerate(zip(test_data, predictions)):
        formatted_data.append({
            'id': i,
            'prompt': test_item['prompt'],
            'actual': test_item['completion'],
            'generated': pred
        })
    
    # Save as CSV
    df = pd.DataFrame(formatted_data)
    df.to_csv(f"{output_path}.csv", index=False)
    
    # Save as JSON
    with open(f"{output_path}.json", 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    # Save as text
    with open(f"{output_path}.txt", 'w') as f:
        for item in formatted_data:
            f.write(f"ID: {item['id']}\n")
            f.write(f"PROMPT:\n{item['prompt']}\n\n")
            f.write(f"ACTUAL:\n{item['actual']}\n\n")
            f.write(f"GENERATED:\n{item['generated']}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Create a markdown table version
    with open(f"{output_path}.md", 'w') as f:
        f.write("# Test Data Comparison\n\n")
        
        for item in formatted_data:
            f.write(f"## Sample {item['id']}\n\n")
            f.write(f"### Prompt\n\n{item['prompt']}\n\n")
            f.write(f"### Actual\n\n{item['actual']}\n\n")
            f.write(f"### Generated\n\n{item['generated']}\n\n")
            f.write("---\n\n")
    
    return formatted_data

# Function to plot metrics for a specific epoch
def plot_epoch_metrics(epoch, results_data):
    """Create publication-quality plots for the metrics for a specific epoch"""
    setup_publication_plot_style()
    
    results_df = pd.DataFrame(results_data)
    metrics_dir = os.path.join(output_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot BLEU scores
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Define positions for bars
    x = np.arange(len(results_df))
    width = 0.2  # Width of the bars
    
    # Plot each BLEU score with a specific color and pattern
    ax.bar(x - width*1.5, results_df['bleu-1'], width, label='BLEU-1', 
           color='#0173B2', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x - width*0.5, results_df['bleu-2'], width, label='BLEU-2', 
           color='#DE8F05', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width*0.5, results_df['bleu-3'], width, label='BLEU-3', 
           color='#029E73', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width*1.5, results_df['bleu-4'], width, label='BLEU-4', 
           color='#D55E00', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'BLEU Scores per Sample - Epoch {epoch}', fontweight='bold', pad=15)
    ax.set_ylabel('Score', fontweight='bold', labelpad=10)
    ax.set_xlabel('Sample ID', fontweight='bold', labelpad=10)
    
    # Set x-tick positions and labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sample {i+1}' for i in range(len(results_df))])
    
    # Add a horizontal line for the average of BLEU-4 (often considered the most important)
    avg_bleu4 = results_df['bleu-4'].mean()
    ax.axhline(y=avg_bleu4, color='#D55E00', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(results_df)-1, avg_bleu4, f'Avg BLEU-4: {avg_bleu4:.4f}', 
            ha='right', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Format y-axis to start at 0 and use percentages
    ax.set_ylim(0, min(1, results_df[['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']].max().max() * 1.2))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.1)])
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Place the legend at the bottom 
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f'bleu_scores_epoch_{epoch}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(metrics_dir, f'bleu_scores_epoch_{epoch}.pdf'), bbox_inches='tight')
    plt.close()
    
    # Plot ROUGE scores with a similar style
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Define positions for bars
    x = np.arange(len(results_df))
    width = 0.25  # Width of the bars
    
    # Plot each ROUGE score with specific colors
    ax.bar(x - width, results_df['rouge-1'], width, label='ROUGE-1', 
           color='#0173B2', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x, results_df['rouge-2'], width, label='ROUGE-2', 
           color='#DE8F05', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width, results_df['rouge-l'], width, label='ROUGE-L', 
           color='#029E73', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'ROUGE Scores per Sample - Epoch {epoch}', fontweight='bold', pad=15)
    ax.set_ylabel('Score', fontweight='bold', labelpad=10)
    ax.set_xlabel('Sample ID', fontweight='bold', labelpad=10)
    
    # Set x-tick positions and labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sample {i+1}' for i in range(len(results_df))])
    
    # Add a horizontal line for the average of ROUGE-L
    avg_rougeL = results_df['rouge-l'].mean()
    ax.axhline(y=avg_rougeL, color='#029E73', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(results_df)-1, avg_rougeL, f'Avg ROUGE-L: {avg_rougeL:.4f}', 
            ha='right', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Format y-axis to start at 0
    ax.set_ylim(0, min(1, results_df[['rouge-1', 'rouge-2', 'rouge-l']].max().max() * 1.2))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.1)])
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Place the legend at the bottom 
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f'rouge_scores_epoch_{epoch}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(metrics_dir, f'rouge_scores_epoch_{epoch}.pdf'), bbox_inches='tight')
    plt.close()

# Function to plot all metrics across epochs
def plot_all_metrics(all_metrics):
    """Create publication-quality plots for all metrics across epochs"""
    setup_publication_plot_style()
    
    metrics_dir = os.path.join(output_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate average loss per epoch and add to metrics
    if all_losses:
        loss_df = pd.DataFrame(all_losses)
        avg_loss_by_epoch = loss_df.groupby('epoch')['loss'].mean().reset_index()
        
        # Join avg_loss_by_epoch with metrics_df on the 'epoch' column
        # Round epoch numbers for joining
        avg_loss_by_epoch['epoch_int'] = avg_loss_by_epoch['epoch'].apply(lambda x: int(round(x)))
        metrics_df['epoch_int'] = metrics_df['epoch']
        
        # Merge metrics with loss data
        metrics_df = pd.merge(metrics_df, 
                             avg_loss_by_epoch[['epoch_int', 'loss']],
                             on='epoch_int', 
                             how='left')
        
        # Drop the temporary column
        metrics_df.drop('epoch_int', axis=1, inplace=True)
        
        # Rename the loss column
        metrics_df.rename(columns={'loss': 'avg_loss'}, inplace=True)
    
    # Save as CSV
    metrics_df.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)
    
    # Plot BLEU scores over epochs (publication quality)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    
    # Plot each BLEU metric with distinct markers
    ax.plot(metrics_df['epoch'], metrics_df['bleu-1'], marker='o', markersize=8, 
            label='BLEU-1', color='#0173B2', linewidth=2, markerfacecolor='white', 
            markeredgewidth=2)
    ax.plot(metrics_df['epoch'], metrics_df['bleu-2'], marker='s', markersize=8, 
            label='BLEU-2', color='#DE8F05', linewidth=2, markerfacecolor='white', 
            markeredgewidth=2)
    ax.plot(metrics_df['epoch'], metrics_df['bleu-3'], marker='^', markersize=8, 
            label='BLEU-3', color='#029E73', linewidth=2, markerfacecolor='white', 
            markeredgewidth=2)
    ax.plot(metrics_df['epoch'], metrics_df['bleu-4'], marker='d', markersize=8, 
            label='BLEU-4', color='#D55E00', linewidth=2, markerfacecolor='white', 
            markeredgewidth=2)
    
    ax.set_title('BLEU Scores Across Epochs', fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontweight='bold', labelpad=10)
    ax.set_ylabel('Score', fontweight='bold', labelpad=10)
    
    # Set x-axis ticks for epochs
    ax.set_xticks(metrics_df['epoch'])
    
    # Format y-axis for scores
    ax.set_ylim(bottom=0)  # Start from 0
    
    # Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a prettier legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=True, fancybox=True, shadow=True)
    
    # Add value annotations
    for col in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']:
        for i, v in enumerate(metrics_df[col]):
            ax.annotate(f'{v:.3f}', 
                      xy=(metrics_df['epoch'].iloc[i], v), 
                      xytext=(0, 5),  # 5 points vertical offset
                      textcoords='offset points',
                      ha='center', va='bottom',
                      fontsize=8, alpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'bleu_scores_all_epochs.png'), bbox_inches='tight')
    plt.savefig(os.path.join(metrics_dir, 'bleu_scores_all_epochs.pdf'), bbox_inches='tight')
    plt.close()
    
    # Plot ROUGE scores over epochs
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    
    # Plot each ROUGE metric with distinct markers
    ax.plot(metrics_df['epoch'], metrics_df['rouge-1'], marker='o', markersize=8, 
            label='ROUGE-1', color='#0173B2', linewidth=2, markerfacecolor='white', 
            markeredgewidth=2)
    ax.plot(metrics_df['epoch'], metrics_df['rouge-2'], marker='s', markersize=8, 
            label='ROUGE-2', color='#DE8F05', linewidth=2, markerfacecolor='white', 
            markeredgewidth=2)
    ax.plot(metrics_df['epoch'], metrics_df['rouge-l'], marker='^', markersize=8, 
            label='ROUGE-L', color='#029E73', linewidth=2, markerfacecolor='white', 
            markeredgewidth=2)
    
    ax.set_title('ROUGE Scores Across Epochs', fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontweight='bold', labelpad=10)
    ax.set_ylabel('Score', fontweight='bold', labelpad=10)
    
    # Set x-axis ticks for epochs
    ax.set_xticks(metrics_df['epoch'])
    
    # Format y-axis for scores
    ax.set_ylim(bottom=0)  # Start from 0
    
    # Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a prettier legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=True, fancybox=True, shadow=True)
    
    # Add value annotations
    for col in ['rouge-1', 'rouge-2', 'rouge-l']:
        for i, v in enumerate(metrics_df[col]):
            ax.annotate(f'{v:.3f}', 
                      xy=(metrics_df['epoch'].iloc[i], v), 
                      xytext=(0, 5),  # 5 points vertical offset
                      textcoords='offset points',
                      ha='center', va='bottom',
                      fontsize=8, alpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'rouge_scores_all_epochs.png'), bbox_inches='tight')
    plt.savefig(os.path.join(metrics_dir, 'rouge_scores_all_epochs.pdf'), bbox_inches='tight')
    plt.close()
    
    # If we have loss data, create a combined metrics vs loss plot
    if 'avg_loss' in metrics_df.columns:
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Top plot: Metrics
        ax1.plot(metrics_df['epoch'], metrics_df['bleu-4'], marker='d', markersize=8, 
                label='BLEU-4', color='#D55E00', linewidth=2, markerfacecolor='white', 
                markeredgewidth=2)
        ax1.plot(metrics_df['epoch'], metrics_df['rouge-l'], marker='^', markersize=8, 
                label='ROUGE-L', color='#029E73', linewidth=2, markerfacecolor='white', 
                markeredgewidth=2)
        
        ax1.set_title('Metrics and Loss Progression', fontweight='bold', pad=15)
        ax1.set_ylabel('Metric Score', fontweight='bold', labelpad=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                frameon=True, fancybox=True, shadow=True)
        
        # Bottom plot: Loss
        ax2.plot(metrics_df['epoch'], metrics_df['avg_loss'], marker='s', markersize=8, 
                color='#CC78BC', linewidth=2, markerfacecolor='white', markeredgewidth=2,
                label='Avg Loss')
        ax2.fill_between(metrics_df['epoch'], metrics_df['avg_loss'] * 0.95, 
                        metrics_df['avg_loss'] * 1.05, color='#CC78BC', alpha=0.2)
        
        ax2.set_xlabel('Epoch', fontweight='bold', labelpad=10)
        ax2.set_ylabel('Loss', fontweight='bold', labelpad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Force integer x-ticks
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotations
        for i, (epoch, bleu4, rougeL, loss) in enumerate(zip(
            metrics_df['epoch'], 
            metrics_df['bleu-4'], 
            metrics_df['rouge-l'],
            metrics_df['avg_loss']
        )):
            # Top plot annotations
            ax1.annotate(f'{bleu4:.3f}', xy=(epoch, bleu4), xytext=(0, 7),
                       textcoords='offset points', ha='center', fontsize=8)
            ax1.annotate(f'{rougeL:.3f}', xy=(epoch, rougeL), xytext=(0, -15),
                       textcoords='offset points', ha='center', fontsize=8)
            
            # Bottom plot annotations
            ax2.annotate(f'{loss:.4f}', xy=(epoch, loss), xytext=(0, 7),
                       textcoords='offset points', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'metrics_and_loss_all_epochs.png'), 
                    bbox_inches='tight')
        plt.savefig(os.path.join(metrics_dir, 'metrics_and_loss_all_epochs.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        # Create a single comprehensive figure with all metrics and loss
        if len(metrics_df) >= 2:  # Only create if we have enough data
            fig = plt.figure(figsize=(15, 10))
            
            # Create a 2x2 subplot layout
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Top-left: BLEU scores
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(metrics_df['epoch'], metrics_df['bleu-1'], marker='o', label='BLEU-1')
            ax1.plot(metrics_df['epoch'], metrics_df['bleu-2'], marker='s', label='BLEU-2')
            ax1.plot(metrics_df['epoch'], metrics_df['bleu-3'], marker='^', label='BLEU-3')
            ax1.plot(metrics_df['epoch'], metrics_df['bleu-4'], marker='d', label='BLEU-4')
            ax1.set_title('BLEU Scores', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Score')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='best')
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Top-right: ROUGE scores
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(metrics_df['epoch'], metrics_df['rouge-1'], marker='o', label='ROUGE-1')
            ax2.plot(metrics_df['epoch'], metrics_df['rouge-2'], marker='s', label='ROUGE-2')
            ax2.plot(metrics_df['epoch'], metrics_df['rouge-l'], marker='^', label='ROUGE-L')
            ax2.set_title('ROUGE Scores', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='best')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Bottom-left: Loss
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(metrics_df['epoch'], metrics_df['avg_loss'], marker='o', 
                    color='#CC78BC', linewidth=2)
            ax3.set_title('Training Loss', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Average Loss')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Bottom-right: Combined metrics (BLEU-4, ROUGE-L, Loss)
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Plot metrics on primary y-axis
            color1, color2 = '#D55E00', '#029E73'
            l1 = ax4.plot(metrics_df['epoch'], metrics_df['bleu-4'], marker='d', 
                        label='BLEU-4', color=color1, linewidth=2)
            l2 = ax4.plot(metrics_df['epoch'], metrics_df['rouge-l'], marker='^', 
                        label='ROUGE-L', color=color2, linewidth=2)
            
            # Create secondary y-axis for loss
            ax4_2 = ax4.twinx()
            color3 = '#CC78BC'
            l3 = ax4_2.plot(metrics_df['epoch'], metrics_df['avg_loss'], marker='s', 
                          label='Avg Loss', color=color3, linewidth=2, linestyle=':')
            
            # Add labels and title
            ax4.set_title('Key Performance Indicators', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Metric Score')
            ax4_2.set_ylabel('Loss')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Add legend
            lns = l1 + l2 + l3
            labs = [l.get_label() for l in lns]
            ax4.legend(lns, labs, loc='center', bbox_to_anchor=(0.5, -0.25), ncol=3)
            
            # Add a title for the whole figure
            plt.suptitle('Comprehensive Training Metrics Summary', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
            plt.savefig(os.path.join(metrics_dir, 'comprehensive_metrics_summary.png'), 
                        bbox_inches='tight', dpi=300)
            plt.savefig(os.path.join(metrics_dir, 'comprehensive_metrics_summary.pdf'), 
                        bbox_inches='tight')
            plt.close()

# ====== TRAINING SETUP ======
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,   
        num_train_epochs=1,  # We'll do one epoch at a time for evaluation
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=output_dir,
        report_to="none",
    ),
    callbacks=[LossLoggingCallback()]
)

# Only train on assistant responses, not user prompts
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Empty CUDA cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared CUDA cache")

# ====== TRAINING AND EVALUATION ======
# Show memory stats before training
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

all_metrics = []

# Train for specified number of epochs with evaluation after each
for epoch in range(1, num_epochs + 1):
    print(f"\n===== Starting Epoch {epoch}/{num_epochs} =====")
    
    # Train for one epoch
    trainer_stats = trainer.train()
    
    # Save checkpoint after each epoch
    epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    model.save_pretrained(epoch_output_dir)
    tokenizer.save_pretrained(epoch_output_dir)
    print(f"Saved model checkpoint for epoch {epoch}")
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Evaluate on test data
    metrics = evaluate_model(model, tokenizer, test_data, epoch)
    all_metrics.append(metrics)
    
    # Re-initialize trainer for next epoch (to avoid state issues)
    if epoch < num_epochs:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=1,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_steps=5,
                num_train_epochs=1,
                learning_rate=2e-4 * (0.9 ** (epoch)),  # Learning rate decay
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                evaluation_strategy="no",
                save_strategy="no",
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=output_dir,
                report_to="none",
            ),
            callbacks=[LossLoggingCallback()]
        )
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

# Show final memory and time stats
if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ====== SAVE FINAL MODEL ======
print("Saving final model to 'finetuned_model'")
model.save_pretrained("finetuned_model")
tokenizer.save_pretrained("finetuned_model")

# Create final test predictions using the best model
print("Generating final predictions with the trained model...")

# Enable inference mode
FastLanguageModel.for_inference(model)

final_predictions = []
for i, item in enumerate(test_data):
    prompt = item["prompt"]
    
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0.7,
            min_p=0.1
        )
    
    # Decode prediction
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    prediction = full_output.split("assistant")[1]
     
    
    final_predictions.append(prediction)
    print(f"Final prediction {i+1}/{len(test_data)} generated")

# Save the final test data comparison
save_test_data_comparison(
    test_data, 
    final_predictions, 
    os.path.join(output_dir, "final_test_comparison")
)

# Save all metrics in one file
with open(os.path.join(output_dir, "all_metrics.json"), 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Generate plots for all metrics across epochs
plot_all_metrics(all_metrics)

# Generate loss curve plots
plot_loss_curve()

print("\nTraining and evaluation complete!")
print("Final model saved to 'finetuned_model'")
print(f"Metrics and plots saved to '{output_dir}'")
print(f"Test data comparisons saved to '{output_dir}/final_test_comparison.[csv|json|txt|md]'")