import os
import pandas as pd

tool_dir = "Tools"

results = []
print(f"{'Directory':<30} {'PDF_Text_Exists':<20} {'PDF_Questions_Json':<22} {'Article_Text_Exists':<22} {'Article_Questions_Json':<25} {'GitHub_README':<18} {'GitHub_Questions_Json':<25} {'Website_Has_Content':<22} {'Website_Questions_CSV':<25} {'R_Package_Questions_Json':<25} {'R_Package_Has_Files':<22} {'R_Package_Has_Folder':<25} {'Jupyter_Notebook_Exists':<25} {'Python_Jupyter_Questions':<25}")

for directory in os.listdir(tool_dir):
    dir_path = os.path.join(tool_dir, directory)
    if not os.path.isdir(dir_path):
        continue

    # PDFManual checks
    pdfmanual_dir = os.path.join(dir_path, "PDFManual")
    txt_exists = 0
    if os.path.isdir(pdfmanual_dir):
        for pdf_name in os.listdir(pdfmanual_dir):
            pdf_path = os.path.join(pdfmanual_dir, pdf_name)
            if os.path.isdir(pdf_path):
                for fname in os.listdir(pdf_path):
                    if fname.endswith(".txt"):
                        txt_exists = 1
                        break
            if txt_exists:
                break
    questions_chunk1_json = os.path.join(pdfmanual_dir, "Questions_Merged.json")
    questions_chunk1_exists = 1 if os.path.exists(questions_chunk1_json) else 0

    # GitHubReadme checks
    gh_dir = os.path.join(dir_path, "GitHubReadme")
    readme_md = os.path.join(gh_dir, "README.md")
    gh_questions_json = os.path.join(gh_dir, "Questions.json")
    readme_exists = 1 if os.path.exists(readme_md) else 0
    gh_questions_exists = 1 if os.path.exists(gh_questions_json) else 0

    # Website folder check
    website_dir = os.path.join(dir_path, "Website")
    website_not_empty = 0
    if os.path.isdir(website_dir):
        if os.listdir(website_dir):
            website_not_empty = 1

    # WebsiteQuestions CSV check
    website_questions_csv = os.path.join(dir_path, "WebsiteQuestions", f"{directory}_merged_all_questions.csv")
    website_questions_exists = 1 if os.path.exists(website_questions_csv) else 0



    # Rpackage checks
    rpackage_dir = os.path.join(dir_path, "Rpackage")
    rpackage_questions_exists = 0
    rpackage_gt3_files = 0
    rpackage_has_folder = 0

    if os.path.isdir(rpackage_dir):
        file_count = 0
        folder_found = False
        for root, dirs, files in os.walk(rpackage_dir):
            file_count += len(files)
            if not folder_found and dirs:
                folder_found = True

        if file_count > 10:
            questions_json_path = os.path.join(rpackage_dir, "Questions.json")
            if os.path.exists(questions_json_path):
                rpackage_questions_exists = 1

        if file_count > 1:
            rpackage_gt3_files = 1

        if folder_found:
            rpackage_has_folder = 1
    
    # Article checks
    article_dir = os.path.join(dir_path, "Article")
    article_txt_exists = 0
    article_questions_exists = 0
    
    if os.path.isdir(article_dir):
        # Check for article.txt
        article_txt_path = os.path.join(article_dir, "article", "article.txt")
        if os.path.exists(article_txt_path):
            article_txt_exists = 1
        
        # Check for Questions.json
        article_questions_path = os.path.join(article_dir, "Questions.json")
        if os.path.exists(article_questions_path):
            article_questions_exists = 1
    
    # PythonPackage checks
    python_package_dir = os.path.join(dir_path, "PythonPackage")
    jupyter_notebook_exists = 0
    jupyter_questions_exists = 0
    python_questions_code_exists = 0
    python_package_not_empty = 0
    python_file_exists = 0
    
    if os.path.isdir(python_package_dir):
        # Check if directory is not empty
        if os.listdir(python_package_dir):
            python_package_not_empty = 1
            
            # Check for any Jupyter notebook files and Python files
            for root, dirs, files in os.walk(python_package_dir):
                for file in files:
                    if file.endswith(".ipynb"):
                        jupyter_notebook_exists = 1
                    if file.endswith(".py"):
                        python_file_exists = 1
                    if jupyter_notebook_exists and python_file_exists:
                        break
                if jupyter_notebook_exists and python_file_exists:
                    break
            
            # Check for Questions_Jupyter.json
            questions_jupyter_json = os.path.join(python_package_dir, "Questions_Jupyter.json")
            if os.path.exists(questions_jupyter_json):
                jupyter_questions_exists = 1
            
            # Check for Questions_Code.json
            questions_code_json = os.path.join(python_package_dir, "Questions_Code.json")
            if os.path.exists(questions_code_json):
                python_questions_code_exists = 1

    results.append({
        "Directory": directory,
         "PDF_Text_Exists": txt_exists,
         "PDF_Questions_Exists": questions_chunk1_exists,
         "Article_Text_Exists": article_txt_exists,
         "Article_Questions_Exists": article_questions_exists,
         "GitHub_README": readme_exists,
         "GitHub_Questions_Exists": gh_questions_exists,
         "Website_Has_Content": website_not_empty,
         "Website_Questions_Exists": website_questions_exists,
        "R_Package_Has_Files": rpackage_gt3_files,
        "R_Package_Questions_Exists": rpackage_questions_exists,
        "Jupyter_Notebook_Exists": jupyter_notebook_exists,
        "Python_Jupyter_Questions": jupyter_questions_exists,
        "Python_File_Exists": python_file_exists,
        "Python_Code_Questions": python_questions_code_exists
    })

df = pd.DataFrame(results)
print(df)
df.to_csv("FileExistanceCheck.csv", index=False)

# Create comparative visualizations for publication
def create_comparison_plots(df):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams
    
    # Set publication-quality parameters following top CS journals
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman']
    rcParams['font.size'] = 10
    rcParams['axes.linewidth'] = 0.8
    rcParams['grid.linewidth'] = 0.5
    rcParams['lines.linewidth'] = 1.5
    rcParams['patch.linewidth'] = 0.5
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.minor.width'] = 0.6
    rcParams['ytick.minor.width'] = 0.6
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 11
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.titlesize'] = 12
    
    # Define pairs to compare
    comparison_pairs = [
        ("PDF_Text_Exists", "PDF_Questions_Exists", "PDF\nManual"),
        ("Article_Text_Exists", "Article_Questions_Exists", "Research\nArticle"),
        ("GitHub_README", "GitHub_Questions_Exists", "GitHub\nREADME"),
        ("Website_Has_Content", "Website_Questions_Exists", "Project\nWebsite"),
        ("R_Package_Has_Files", "R_Package_Questions_Exists", "R\nPackage"),
        ("Python_File_Exists", "Python_Code_Questions", "Python\nCode"),
        ("Jupyter_Notebook_Exists", "Python_Jupyter_Questions", "Jupyter\nNotebook")
    ]
    
    # Calculate summary statistics
    summary_stats = {}
    for source, questions, title in comparison_pairs:
        source_count = df[source].sum()
        questions_count = df[questions].sum()
        success_rate = questions_count / source_count if source_count > 0 else 0
        summary_stats[title.replace('\n', ' ')] = {
            "Source Count": source_count,
            "Questions Count": questions_count,
            "Success Rate": success_rate
        }
    
    # Create the main comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left panel: Bar chart comparison
    barWidth = 0.35
    r1 = np.arange(len(comparison_pairs))
    r2 = [x + barWidth for x in r1]
    
    source_counts = [df[pair[0]].sum() for pair in comparison_pairs]
    question_counts = [df[pair[1]].sum() for pair in comparison_pairs]
    
    # Use professional color scheme (colorbrewer-inspired)
    color1 = '#2166ac'  # Blue
    color2 = '#d6604d'  # Orange-red
    
    bars1 = ax1.bar(r1, source_counts, width=barWidth, color=color1, alpha=0.8, 
                    edgecolor='black', linewidth=0.5, label='Sources Available')
    bars2 = ax1.bar(r2, question_counts, width=barWidth, color=color2, alpha=0.8,
                    edgecolor='black', linewidth=0.5, label='Questions Generated')
    
    ax1.set_xlabel('Data Source Type', fontweight='normal')
    ax1.set_ylabel('Count', fontweight='normal')
    ax1.set_title('(a) Source Availability vs. Question Generation', fontweight='bold', loc='left')
    
    ax1.set_xticks([r + barWidth/2 for r in range(len(comparison_pairs))])
    ax1.set_xticklabels([pair[2] for pair in comparison_pairs], fontsize=8)
    
    # Add value labels on bars (only if > 0)
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    ax1.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right panel: Success rate visualization
    categories = [pair[2].replace('\n', ' ') for pair in comparison_pairs]
    success_rates = []
    source_counts_for_rate = []
    
    for source, questions, title in comparison_pairs:
        source_count = df[source].sum()
        questions_count = df[questions].sum()
        success_rate = (questions_count / source_count * 100) if source_count > 0 else 0
        success_rates.append(success_rate)
        source_counts_for_rate.append(source_count)
    
    # Create scatter plot for success rates
    scatter = ax2.scatter(range(len(categories)), success_rates, 
                         s=[count*20 + 50 for count in source_counts_for_rate],
                         c=success_rates, cmap='RdYlBu_r', alpha=0.7,
                         edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Data Source Type', fontweight='normal')
    ax2.set_ylabel('Success Rate (%)', fontweight='normal')
    ax2.set_title('(b) Question Generation Success Rate', fontweight='bold', loc='left')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels([pair[2] for pair in comparison_pairs], fontsize=8)
    ax2.set_ylim(-5, 105)
    
    # Add horizontal reference lines
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(y=100, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
    
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8, aspect=20)
    cbar.set_label('Success Rate (%)', rotation=270, labelpad=15, fontsize=9)
    
    plt.tight_layout()
    plt.savefig("question_generation_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("question_generation_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table following academic standards
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data with proper formatting
    table_data = []
    for title, stats in summary_stats.items():
        success_rate = stats["Success Rate"] * 100
        table_data.append([
            title, 
            f"{stats['Source Count']:,}",
            f"{stats['Questions Count']:,}", 
            f"{success_rate:.1f}\\%"
        ])
    
    # Calculate totals
    total_sources = sum(stats["Source Count"] for stats in summary_stats.values())
    total_questions = sum(stats["Questions Count"] for stats in summary_stats.values())
    overall_rate = (total_questions / total_sources) * 100 if total_sources > 0 else 0
    
    # Add horizontal line before total
    table_data.append(["\\rule{0pt}{1pt}", "", "", ""])  # Empty row for spacing
    table_data.append(["\\textbf{Total}", f"\\textbf{{{total_sources:,}}}", 
                      f"\\textbf{{{total_questions:,}}}", f"\\textbf{{{overall_rate:.1f}\\%}}"])
    
    # Create professional table
    table = ax.table(
        cellText=table_data,
        colLabels=["Data Source", "Available", "Generated", "Success Rate"],
        loc='center',
        cellLoc='center',
        colWidths=[0.35, 0.2, 0.2, 0.2]
    )
    
    # Style the table professionally
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Apply consistent styling
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(fontfamily='serif')
        if row == 0:  # Header
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
        elif row == len(table_data):  # Total row
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')
            cell.set_edgecolor('black')
            cell.set_linewidth(0.8)
        else:
            cell.set_facecolor('white')
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)
    
    plt.title('Table 1: Summary of Question Generation Performance by Data Source', 
              fontweight='bold', fontsize=11, pad=20, loc='left')
    
    plt.tight_layout()
    plt.savefig("question_generation_table.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("question_generation_table.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_stats

# Generate the plots and get summary statistics
summary_stats = create_comparison_plots(df)

# Print summary report
print("\n" + "="*80)
print("SUMMARY OF QUESTION GENERATION SUCCESS")
print("="*80)

for source, stats in summary_stats.items():
    success_rate = stats["Success Rate"] * 100
    print(f"{source + ':':25} {stats['Source Count']:3} sources available, {stats['Questions Count']:3} with questions generated ({success_rate:.1f}%)")

total_sources = sum(stats["Source Count"] for stats in summary_stats.values())
total_questions = sum(stats["Questions Count"] for stats in summary_stats.values())
overall_rate = (total_questions / total_sources) * 100 if total_sources > 0 else 0

print("-"*80)
print(f"{'Overall:':25} {total_sources:3} total sources, {total_questions:3} with questions generated ({overall_rate:.1f}%)")
print("="*80)
