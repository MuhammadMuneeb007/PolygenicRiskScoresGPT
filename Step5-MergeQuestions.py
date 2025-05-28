import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_label_rotation(angle, offset):
    """Helper function to determine label rotation and alignment"""
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    """Add labels to the circular barplot"""
    padding = 4
    for angle, value, label in zip(angles, values, labels):
        rotation, alignment = get_label_rotation(angle, offset)
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor",
            fontsize=8
        )

def create_publication_plot(df, save_path=None):
    """Create a publication-quality circular barplot"""
    
    # Prepare data for visualization
    plot_df = df.melt(id_vars=['Tool'], 
                     value_vars=['Article_Questions', 'Github_Readme_Questions', 
                               'Github_Questions', 'Python_Questions', 
                               'PDF_Questions', 'R_Questions',
                                'Website_All_Questions'],
                     var_name='Source', value_name='Count')
    
    # Filter out zero counts for cleaner visualization
    plot_df = plot_df[plot_df['Count'] > 0]
    
    # Sort by tool and source for better grouping
    plot_df = plot_df.sort_values(['Tool', 'Source'])
    
    VALUES = plot_df["Count"].values
    LABELS = [f"{row['Tool']}\n({row['Source'].replace('_Questions', '')})" 
              for _, row in plot_df.iterrows()]
    GROUPS = plot_df["Tool"].values
    
    # Create groups for visualization
    unique_tools = plot_df['Tool'].unique()
    group_mapping = {tool: i for i, tool in enumerate(unique_tools)}
    GROUP_NUMS = [group_mapping[tool] for tool in GROUPS]
    
    PAD = 2
    ANGLES_N = len(VALUES) + PAD * len(unique_tools)
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)
    
    # Calculate group sizes
    GROUPS_SIZE = [len(plot_df[plot_df['Tool'] == tool]) for tool in unique_tools]
    
    # Calculate indices for positioning
    offset = 0
    IDXS = []
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD
    
    OFFSET = np.pi / 2
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw={"projection": "polar"})
    
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-max(VALUES) * 0.3, max(VALUES) * 1.2)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create colors for different sources
    source_colors = {
        'Article_Questions': '#1f77b4',
        'Github_Readme_Questions': '#ff7f0e', 
        'Github_Questions': '#2ca02c',
        'Python_Questions': '#d62728',
        'PDF_Questions': '#9467bd',
        'R_Questions': '#8c564b',
        'Website_All_Questions': '#17becf'
    }
    
    COLORS = [source_colors[source] for source in plot_df['Source']]
    
    # Add bars
    bars = ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, 
        edgecolor="white", linewidth=1.5, alpha=0.8
    )
    
    # Add labels
    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)
    
    # Add group annotations and reference lines
    offset = 0
    for tool, size in zip(unique_tools, GROUPS_SIZE):
        if size > 0:  # Only add if group has data
            # Add line below bars
            x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
            ax.plot(x1, [-max(VALUES) * 0.1] * 50, color="#333333", linewidth=2)
            
            # Add tool name
            ax.text(
                np.mean(x1), -max(VALUES) * 0.2, tool, 
                color="#333333", fontsize=12, fontweight="bold", 
                ha="center", va="center"
            )
            
            # Add reference lines
            for ref_val in [max(VALUES) * 0.25, max(VALUES) * 0.5, max(VALUES) * 0.75]:
                x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
                ax.plot(x2, [ref_val] * 50, color="#bebebe", lw=0.8, alpha=0.5)
        
        offset += size + PAD
    
    # Add title
    plt.title("Question Counts by Tool and Source Type", 
              fontsize=20, fontweight="bold", pad=20)
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=source.replace('_Questions', '').replace('_', ' '))
                      for source, color in source_colors.items()]
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.3, 0.5), 
              fontsize=10, frameon=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig, ax

def create_total_counts_plot(df, save_path=None):
    """Create a publication-quality circular barplot for total question counts by source using dynamic data"""
    
    # Extract dynamic data from DataFrame
    data = {
        'Article Questions': df['Article_Questions'].sum(),
        'GitHub Readme Questions': df['Github_Readme_Questions'].sum(),
        'GitHub Questions': df['Github_Questions'].sum(),
        'Python Questions': df['Python_Questions'].sum(),
        'PDF Questions': df['PDF_Questions'].sum(),
        'R Questions': df['R_Questions'].sum(),
        'Website All Questions': df['Website_All_Questions'].sum()
    }
    
    LABELS = list(data.keys())
    VALUES = list(data.values())
    
    # Set up angles and positioning
    ANGLES = np.linspace(0, 2 * np.pi, len(VALUES), endpoint=False)
    WIDTH = 2 * np.pi / len(VALUES)
    OFFSET = np.pi / 2
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"projection": "polar"})
    
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-max(VALUES) * 0.1, max(VALUES) * 1.2)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Define colors for different question types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf']
    
    # Add bars
    bars = ax.bar(ANGLES, VALUES, width=WIDTH, color=colors, 
                  edgecolor="white", linewidth=2, alpha=0.8)
    
    # Add labels with values
    for angle, value, label in zip(ANGLES, VALUES, LABELS):
        rotation = np.rad2deg(angle + OFFSET)
        if angle <= np.pi:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"
        
        # Add label with count
        ax.text(angle, value + max(VALUES) * 0.05, f"{label}\n{value:,}", 
                ha=alignment, va="center", rotation=rotation, rotation_mode="anchor",
                fontsize=11, fontweight="bold")
    
    # Add reference circles
    max_val = max(VALUES)
    reference_values = []
    if max_val > 30000:
        reference_values = [5000, 10000, 15000, 20000, 25000, 30000]
    elif max_val > 15000:
        reference_values = [2500, 5000, 7500, 10000, 12500, 15000]
    elif max_val > 5000:
        reference_values = [1000, 2000, 3000, 4000, 5000]
    else:
        reference_values = [500, 1000, 1500, 2000, 2500]
    
    for ref_val in reference_values:
        if ref_val <= max_val:
            circle = plt.Circle((0, 0), ref_val, fill=False, color='#bebebe', 
                              linestyle='--', alpha=0.5, linewidth=1)
            ax.add_patch(circle)
            # Add reference value text
            ax.text(0, ref_val, f"{ref_val:,}", ha="center", va="center", 
                   fontsize=9, color='#666666', alpha=0.8)
    
    # Calculate total dynamically
    total_questions = sum(VALUES)
    
    # Add title
    plt.title(f"Question Counts by Source Type\nTotal: {total_questions:,} Questions", 
              fontsize=18, fontweight="bold", pad=30)
    
    # Add center text with total
    ax.text(0, 0, f"Total\n{total_questions:,}\nQuestions", ha="center", va="center", 
           fontsize=16, fontweight="bold", color='#333333',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig, ax

def create_simple_distribution_plot(df, save_path=None):
    """Create a simple bar plot to show distribution of questions by source using dynamic data"""
    
    # Extract dynamic data from DataFrame
    sources = ['Article', 'GitHub Readme', 'GitHub', 'Python', 'PDF', 'R', 'Website All']
    counts = [
        df['Article_Questions'].sum(),
        df['Github_Readme_Questions'].sum(),
        df['Github_Questions'].sum(),
        df['Python_Questions'].sum(),
        df['PDF_Questions'].sum(),
        df['R_Questions'].sum(),
        df['Website_All_Questions'].sum()
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(sources)))
    bars = ax1.bar(sources, counts, color=colors, edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Question Distribution by Source Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Questions', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax2.pie(counts, labels=sources, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Question Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()
    return fig


def count_questions_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return len(data)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0

def count_questions_in_directory(directory_path):
    """
    Recursively finds all .json files in the directory, merges their contents (assumed to be lists), and returns the total count.
    """
    total_questions = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_questions += len(data)
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
    return total_questions
      

def count_questions_in_articles():
    tools_directory = "Tools"
    
    # Get all tool names from Tools directory only
    all_tools = set()
    
    if os.path.exists(tools_directory):
        all_tools = set(d for d in os.listdir(tools_directory) 
                       if os.path.isdir(os.path.join(tools_directory, d)))
    
    results = []
    
    for tool in all_tools:
        # Count Article questions
        article_file = os.path.join(tools_directory, tool, "Article", "Questions.json")
        article_count = count_questions_in_file(article_file) if os.path.isfile(article_file) else 0

        # Count GitHubReadme questions
        github_readme_file = os.path.join(tools_directory, tool, "GitHubReadme", "Questions.json")
        github_readme_count = count_questions_in_file(github_readme_file) if os.path.isfile(github_readme_file) else 0

        # Count GitHubQuestions questions
        github_questions_file = os.path.join(tools_directory, tool, "GitHubQuestions")
        github_questions_count = count_questions_in_directory(github_questions_file)  

        # Count Python package questions
        python_file = os.path.join(tools_directory, tool, "PythonPackage", "Questions_Code.json")
        python_count = count_questions_in_file(python_file) if os.path.isfile(python_file) else 0

        # Count PDF manual questions
        pdf_file = os.path.join(tools_directory, tool, "PDFManual", "Questions_Merged.json")
        pdf_count = count_questions_in_file(pdf_file) if os.path.isfile(pdf_file) else 0

        # Count R package questions
        r_file = os.path.join(tools_directory, tool, "Rpackage", "Questions.json")
        r_count = count_questions_in_file(r_file) if os.path.isfile(r_file) else 0
        
        # Count WebsiteQuestions - all questions file
        website_all_file = os.path.join(tools_directory, tool,"WebsiteQuestions")
        website_all_count = count_questions_in_directory(website_all_file)
        
        results.append({
            'Tool': tool,
            'Article_Questions': article_count,
            'Github_Readme_Questions': github_readme_count,
            'Github_Questions': github_questions_count,
            'Python_Questions': python_count,
            'PDF_Questions': pdf_count,
            'R_Questions': r_count,
            'Website_All_Questions': website_all_count,
            'Total_Questions': article_count + github_readme_count + github_questions_count + python_count + pdf_count + r_count + website_all_count
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Tool')
    
    # Display results
    print(df)
    print(f"\nTotal Article Questions: {df['Article_Questions'].sum()}")
    print(f"Total GitHub Readme Questions: {df['Github_Readme_Questions'].sum()}")
    print(f"Total GitHub Questions: {df['Github_Questions'].sum()}")
    print(f"Total Python Questions: {df['Python_Questions'].sum()}")
    print(f"Total PDF Questions: {df['PDF_Questions'].sum()}")
    print(f"Total R Questions: {df['R_Questions'].sum()}")
    print(f"Total Website All Questions: {df['Website_All_Questions'].sum()}")
    print(f"Grand Total Questions: {df['Total_Questions'].sum()}")
    
    # Create total counts plot with DataFrame
    print("\nGenerating publication-quality visualization...")
    create_total_counts_plot(df, save_path="question_counts_by_source.png")
    
    # Create simple distribution plot with DataFrame
    print("\nGenerating distribution visualization...")
    create_simple_distribution_plot(df, save_path="question_distribution.png")
    
    return df
 

def convert_question_format(question, tool_name):
    """
    Convert question format: instruction* -> Question, output* -> output
    Add ToolName as source
    """
    converted = {}
    
    # Convert instruction* fields to Question (instruction, instruction1, instruction2, etc.)
    for key, value in question.items():
        if key.lower().startswith('instruction'):
            converted['Question'] = value
            break  # Take the first instruction field found
    
    # Convert output* fields to output (output, output1, output2, etc.)
    for key, value in question.items():
        if key.lower().startswith('output'):
            converted['output'] = value
            break  # Take the first output field found
    
    # Add tool name as source
    converted['ToolName'] = tool_name
    
    return converted

def filter_questions_by_output_length(questions, min_words=30):
    """
    Filter questions where output length is greater than specified number of words
    """
    filtered = []
    for q in questions:
        if 'output' in q and q['output']:
            word_count = len(q['output'].split())
            if word_count > min_words:
                filtered.append(q)
    return filtered

def remove_duplicate_questions(questions, similarity_threshold=0.95):
    """
    Remove duplicate questions using cosine similarity.
    Keep the question with longer output when duplicates are found.
    """
    if not questions:
        return questions
    
    # Extract question texts for similarity comparison
    question_texts = []
    for q in questions:
        if 'Question' in q and q['Question']:
            question_texts.append(q['Question'])
        else:
            question_texts.append("")
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(question_texts)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find duplicates and keep the one with longer output
    to_remove = set()
    n_questions = len(questions)
    
    for i in range(n_questions):
        if i in to_remove:
            continue
            
        for j in range(i + 1, n_questions):
            if j in to_remove:
                continue
                
            if similarity_matrix[i][j] > similarity_threshold:
                # Get output lengths
                output_i = questions[i].get('output', '')
                output_j = questions[j].get('output', '')
                
                len_i = len(output_i.split()) if output_i else 0
                len_j = len(output_j.split()) if output_j else 0
                
                # Remove the one with shorter output
                if len_i >= len_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    
    # Return questions that are not marked for removal
    deduplicated = [q for idx, q in enumerate(questions) if idx not in to_remove]
    
    print(f"    Removed {len(to_remove)} duplicate questions out of {n_questions}")
    return deduplicated

def merge_all_questions():
    """
    Merge all deduplicated and filtered questions from all sources into one file
    """
    print("\nMerging all questions into a single file...")
    
    # List of source files to merge (deduplicated and filtered versions)
    source_files = [
        'Article_deduplicated_filtered.json',
        'GitHubReadme_deduplicated_filtered.json',
        'GitHubQuestions_deduplicated_filtered.json',
        'Python_deduplicated_filtered.json',
        'PDF_deduplicated_filtered.json',
        'R_deduplicated_filtered.json',
        'WebsiteAll_deduplicated_filtered.json'
    ]
    
    all_questions = []
    
    for filename in source_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Add source type to each question
                        source_type = filename.replace('_deduplicated_filtered.json', '')
                        for q in data:
                            q['SourceType'] = source_type
                        all_questions.extend(data)
                        print(f"Added {len(data)} questions from {filename}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"File not found: {filename}")
    
    # Save merged questions
    output_filename = 'Allquestions.json'
    with open(output_filename, 'w') as f:
        json.dump(all_questions, f, indent=2)
    
    print(f"Saved {len(all_questions)} total questions to {output_filename}")
    return all_questions

def analyze_question_distribution():
    """
    Analyze and visualize the distribution of questions using clustering
    """
    # Import required libraries for clustering analysis
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import seaborn as sns
    from tqdm import tqdm
    from collections import Counter
    import time
    from wordcloud import WordCloud
    import matplotlib.patheffects as PathEffects
    from sklearn.decomposition import PCA
    
    # Set publication-quality plot settings
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    def load_json_data(file_path):
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []
    
    def create_embeddings(texts, max_features=1500):
        """Create TF-IDF embeddings for a list of texts"""
        print(f"Creating embeddings for {len(texts)} texts...")
        
        # Use TF-IDF vectorizer with max_features to keep dimensionality manageable
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85
        )
        
        # Fit and transform the texts to get embeddings
        try:
            embeddings = vectorizer.fit_transform(texts)
            print(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings, vectorizer
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None, None
    
    def extract_outputs(data):
        """Extract output texts from JSON data"""
        all_outputs = []
        output_sources = []
        valid_indices = []
        
        print("Extracting outputs from data...")
        
        for i, item in enumerate(data):
            output = item.get('output', '')
            
            if output:
                all_outputs.append(output)
                source_type = item.get('SourceType', 'Unknown')
                output_sources.append(source_type)
                valid_indices.append(i)
        
        stats = {
            'total_items': len(data),
            'items_with_output': len(all_outputs),
            'source_distribution': Counter(output_sources)
        }
        
        print(f"Found {stats['items_with_output']} items with outputs")
        print("Source distribution:", dict(stats['source_distribution']))
        
        return all_outputs, output_sources, valid_indices, stats
    
    def find_optimal_clusters(embeddings, max_clusters=15):
        """Find optimal number of clusters using silhouette score"""
        print("Determining optimal number of clusters...")
        
        # Convert sparse to dense if needed
        if hasattr(embeddings, "toarray"):
            dense_embeddings = embeddings.toarray()
        else:
            dense_embeddings = embeddings
        
        # Sample data if there's too much
        max_samples = 10000
        if dense_embeddings.shape[0] > max_samples:
            indices = np.random.choice(dense_embeddings.shape[0], max_samples, replace=False)
            sample_data = dense_embeddings[indices]
        else:
            sample_data = dense_embeddings
        
        # Try different numbers of clusters
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, sample_data.shape[0]))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(sample_data)
            silhouette_avg = silhouette_score(sample_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"  Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")
        
        # Find the best number of clusters
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(cluster_range), silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method For Optimal k')
        plt.grid(True, alpha=0.3)
        
        # Create the ClusterAnalysis directory if it doesn't exist
        os.makedirs('ClusterAnalysis', exist_ok=True)
        
        plt.savefig('ClusterAnalysis/optimal_clusters.png')
        print("Saved optimal clusters plot to ClusterAnalysis/optimal_clusters.png")
        
        return optimal_clusters
    
    def cluster_embeddings(embeddings, n_clusters):
        """Cluster the embeddings using KMeans"""
        print(f"Clustering embeddings into {n_clusters} clusters...")
        
        # Convert sparse matrix to dense if needed
        if hasattr(embeddings, "toarray"):
            dense_embeddings = embeddings.toarray()
        else:
            dense_embeddings = embeddings
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(dense_embeddings)
        
        # Calculate cluster statistics
        cluster_counts = Counter(clusters)
        print("Cluster distribution:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"  Cluster {cluster_id}: {count} questions ({count/len(clusters)*100:.1f}%)")
        
        return clusters, kmeans
    
    def reduce_dimensions(embeddings, n_components=2):
        """Reduce dimensions for visualization using t-SNE"""
        print(f"Reducing dimensions with t-SNE...")
        start_time = time.time()
        
        # Convert sparse matrix to dense if needed
        if hasattr(embeddings, "toarray"):
            dense_embeddings = embeddings.toarray()
        else:
            dense_embeddings = embeddings
        
        # Use PCA first to reduce dimensionality for faster t-SNE processing
        print("Applying PCA to reduce initial dimensionality before t-SNE...")
        pca = PCA(n_components=min(50, dense_embeddings.shape[1]))
        pca_result = pca.fit_transform(dense_embeddings)
        print(f"PCA reduced dimensions from {dense_embeddings.shape[1]} to {pca_result.shape[1]}")
        
        # Apply t-SNE on all data points after PCA reduction
        tsne = TSNE(
            n_components=n_components, 
            random_state=42, 
            perplexity=min(40, len(pca_result) // 100 + 10),
            learning_rate='auto', 
            init='random',
            n_iter=1000,
            verbose=1
        )
        print(f"Applying t-SNE on all {len(pca_result)} points...")
        reduced = tsne.fit_transform(pca_result)
        indices = np.arange(dense_embeddings.shape[0])
        
        print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
        return reduced, indices
    
    def get_cluster_labels(texts, clusters, n_clusters, vectorizer=None):
        """Get the most representative terms for each cluster"""
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
        else:
            X = vectorizer.transform(texts)
            feature_names = vectorizer.get_feature_names_out()
        
        # For each cluster, find the top terms
        cluster_labels = {}
        
        for i in range(n_clusters):
            # Get indices of texts in this cluster
            indices = [j for j, c in enumerate(clusters) if c == i]
            
            if not indices:
                cluster_labels[i] = f"Cluster {i}"
                continue
                
            # Get the average tfidf values for this cluster
            cluster_values = X[indices].toarray().mean(axis=0)
            
            # Get the top terms for this cluster
            top_indices = cluster_values.argsort()[-5:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            # Create a short readable label from the top terms
            if len(top_terms) > 0:
                label = " / ".join(top_terms[:2])
            else:
                label = f"Cluster {i}"
                
            cluster_labels[i] = label
        
        return cluster_labels
    
    def create_publication_plot(reduced_embeddings, clusters, output_sources, cluster_labels, output_file="ClusterAnalysis/answer_clusters_publication.png"):
        """Create a high-quality visualization of the clusters with labels"""
        plt.figure(figsize=(16, 12))
        
        unique_clusters = sorted(list(set(clusters)))
        n_clusters = len(unique_clusters)
        
        if n_clusters <= 10:
            palette = sns.color_palette("tab10", n_clusters)
        elif n_clusters <= 20:
            palette = sns.color_palette("tab20", n_clusters)
        else:
            palette1 = sns.color_palette("tab20", 20)
            palette2 = sns.color_palette("husl", n_clusters - 20)
            palette = palette1 + palette2[:n_clusters-20]
        
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'cluster': [str(c) for c in clusters],
            'source': output_sources
        })
        
        cluster_sizes = Counter(clusters)
        
        ax = plt.subplot(111)
        
        # Plot background
        plt.scatter(df['x'], df['y'], alpha=0.1, s=30, c='gray', edgecolor=None)
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = df[df['cluster'] == str(cluster_id)]
            plt.scatter(
                cluster_points['x'], 
                cluster_points['y'],
                alpha=0.7,
                s=50,
                c=[palette[i]],
                label=f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} items",
                edgecolor='w',
                linewidth=0.2
            )
        
        # Add cluster annotations
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = df[df['cluster'] == str(cluster_id)]
            centroid_x = cluster_points['x'].mean()
            centroid_y = cluster_points['y'].mean()
            
            std_x = cluster_points['x'].std() * 0.3
            std_y = cluster_points['y'].std() * 0.3
            
            count = cluster_sizes[cluster_id]
            size_text = f"Cluster {cluster_id}\n{count} items"
            terms_text = f"{cluster_labels[cluster_id]}"
            
            txt1 = plt.text(
                centroid_x, centroid_y + std_y, 
                size_text,
                fontsize=12,
                weight='bold',
                ha='center',
                va='bottom',
                color='black',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=palette[i], boxstyle='round,pad=0.5', linewidth=2)
            )
            
            txt2 = plt.text(
                centroid_x, centroid_y - std_y, 
                terms_text,
                fontsize=11,
                ha='center',
                va='top',
                style='italic'
            )
            txt2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        
        plt.title('Clustering of Question Answers in PRSGPT Dataset', fontsize=22, pad=20, weight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=16, labelpad=10)
        plt.ylabel('t-SNE Dimension 2', fontsize=16, labelpad=10)
        
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.xticks([])
        plt.yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
        
        dataset_info = f"Dataset: {len(df)} answers | {n_clusters} clusters"
        plt.figtext(0.02, 0.02, dataset_info, ha="left", fontsize=12, weight='bold')
        
        handles, labels = ax.get_legend_handles_labels()
        
        if n_clusters > 10:
            ncols = min(4, (n_clusters + 4) // 5)
            legend = plt.legend(
                handles, labels,
                title="Clusters",
                loc='best',
                frameon=True,
                framealpha=0.95,
                fontsize=8,
                title_fontsize=10,
                ncol=ncols,
                markerscale=0.8
            )
        else:
            legend = plt.legend(
                handles, labels,
                title="Clusters",
                loc='lower right',
                frameon=True,
                framealpha=0.95,
                fontsize=10,
                title_fontsize=12,
                ncol=2
            )
        
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_edgecolor('lightgray')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        print(f"Publication-quality plot saved as '{output_file}'")
        
        plt.savefig(output_file.replace('.png', '_white_bg.png'), dpi=400, 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"White background version saved as '{output_file.replace('.png', '_white_bg.png')}'")
        
        return df
    
    # Main clustering analysis execution
    os.makedirs('ClusterAnalysis', exist_ok=True)
    
    MAX_DATAPOINTS = -1
    FORCE_CLUSTERS = None
    
    # Load the data
    file_path = "Allquestions.json"
    data = load_json_data(file_path)
    
    if not data:
        print(f"No data found or could not parse {file_path}")
        return
    
    print(f"Loaded {len(data)} items from {file_path}")
    
    # Extract outputs from the data
    all_outputs, output_sources, valid_indices, stats = extract_outputs(data)
    
    # Apply data limit if specified
    if MAX_DATAPOINTS > 0 and len(all_outputs) > MAX_DATAPOINTS:
        print(f"Limiting analysis to {MAX_DATAPOINTS} data points as configured")
        indices = np.random.choice(len(all_outputs), MAX_DATAPOINTS, replace=False)
        all_outputs = [all_outputs[i] for i in indices]
        output_sources = [output_sources[i] for i in indices]
        valid_indices = [valid_indices[i] for i in indices]
    
    print(f"Processing {len(all_outputs)} outputs...")
    
    # Create embeddings for the texts
    embeddings, vectorizer = create_embeddings(all_outputs)
    if embeddings is None:
        print("Failed to create embeddings. Exiting.")
        return
    
    # Determine number of clusters
    if FORCE_CLUSTERS is not None:
        n_clusters = FORCE_CLUSTERS
        print(f"Using forced cluster count: {n_clusters}")
    else:
        n_clusters = find_optimal_clusters(embeddings)
    
    # Cluster the embeddings
    clusters, kmeans = cluster_embeddings(embeddings, n_clusters=n_clusters)
    
    # Reduce dimensions for visualization
    reduced_embeddings, sample_indices = reduce_dimensions(embeddings)
    
    # Map clusters to the sampled indices
    sampled_clusters = [clusters[i] for i in sample_indices]
    sampled_sources = [output_sources[i] for i in sample_indices]
    
    # Get descriptive labels for each cluster
    sampled_texts = [all_outputs[i] for i in sample_indices]
    cluster_labels = get_cluster_labels(sampled_texts, sampled_clusters, n_clusters, vectorizer)
    
    # Create publication quality plot
    df = create_publication_plot(reduced_embeddings, sampled_clusters, sampled_sources, cluster_labels)
    
    print("\nClustering analysis complete! Results saved to the ClusterAnalysis directory.")

def merge_questions_by_source():
    """
    Merge all questions by source type across all tools and save as separate JSON files.
    """
    tools_directory = "Tools"
    
    # Get all tool names from Tools directory
    all_tools = set()
    if os.path.exists(tools_directory):
        all_tools = set(d for d in os.listdir(tools_directory) 
                       if os.path.isdir(os.path.join(tools_directory, d)))
    
    # Initialize collections for each source type
    merged_data = {
        'Article': [],
        'GitHubReadme': [],
        'GitHubQuestions': [],
        'Python': [],
        'PDF': [],
        'R': [],
        'WebsiteAll': []
    }
    
    for tool in all_tools:
        print(f"Processing tool: {tool}")
        
        # Article questions
        article_file = os.path.join(tools_directory, tool, "Article", "Questions.json")
        if os.path.isfile(article_file):
            try:
                with open(article_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        converted_questions = [convert_question_format(q, tool) for q in data]
                        merged_data['Article'].extend(converted_questions)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # GitHub Readme questions
        github_readme_file = os.path.join(tools_directory, tool, "GitHubReadme", "Questions.json")
        if os.path.isfile(github_readme_file):
            try:
                with open(github_readme_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        converted_questions = [convert_question_format(q, tool) for q in data]
                        merged_data['GitHubReadme'].extend(converted_questions)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # GitHub Questions (from directory)
        github_questions_dir = os.path.join(tools_directory, tool, "GitHubQuestions")
        if os.path.exists(github_questions_dir):
            for root, _, files in os.walk(github_questions_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    converted_questions = [convert_question_format(q, tool) for q in data]
                                    merged_data['GitHubQuestions'].extend(converted_questions)
                        except (json.JSONDecodeError, FileNotFoundError):
                            continue
        
        # Python package questions
        python_file = os.path.join(tools_directory, tool, "PythonPackage", "Questions_Code.json")
        if os.path.isfile(python_file):
            try:
                with open(python_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        converted_questions = [convert_question_format(q, tool) for q in data]
                        merged_data['Python'].extend(converted_questions)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # PDF manual questions
        pdf_file = os.path.join(tools_directory, tool, "PDFManual", "Questions_Merged.json")
        if os.path.isfile(pdf_file):
            try:
                with open(pdf_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        converted_questions = [convert_question_format(q, tool) for q in data]
                        merged_data['PDF'].extend(converted_questions)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # R package questions
        r_file = os.path.join(tools_directory, tool, "Rpackage", "Questions.json")
        if os.path.isfile(r_file):
            try:
                with open(r_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        converted_questions = [convert_question_format(q, tool) for q in data]
                        merged_data['R'].extend(converted_questions)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Website all questions (from directory)
        website_all_dir = os.path.join(tools_directory, tool, "WebsiteQuestions")
        if os.path.exists(website_all_dir):
            for root, _, files in os.walk(website_all_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    converted_questions = [convert_question_format(q, tool) for q in data]
                                    merged_data['WebsiteAll'].extend(converted_questions)
                        except (json.JSONDecodeError, FileNotFoundError):
                            continue
    
    # Deduplicate questions for each source
    print("\nDeduplicating questions using cosine similarity...")
    deduplicated_data = {}
    for source, data in merged_data.items():
        if data:
            print(f"  Processing {source}...")
            deduplicated_data[source] = remove_duplicate_questions(data, similarity_threshold=0.8)
        else:
            deduplicated_data[source] = data
    
    # Create DataFrame for deduplicated counts and generate plots
    print("\n" + "="*60)
    print("POST-DEDUPLICATION VISUALIZATION")
    print("="*60)
    
    # Create deduplicated counts DataFrame for plotting
    deduplicated_counts_data = []
    for source_key, source_display in [
        ('Article', 'Article_Questions'),
        ('GitHubReadme', 'Github_Readme_Questions'), 
        ('GitHubQuestions', 'Github_Questions'),
        ('Python', 'Python_Questions'),
        ('PDF', 'PDF_Questions'),
        ('R', 'R_Questions'),
        ('WebsiteAll', 'Website_All_Questions')
    ]:
        deduplicated_counts_data.append({
            'Tool': 'All_Tools',  # Single row for all tools combined
            source_display: len(deduplicated_data[source_key])
        })
    
    # Combine all counts into a single row
    combined_row = {'Tool': 'All_Tools'}
    for item in deduplicated_counts_data:
        combined_row.update({k: v for k, v in item.items() if k != 'Tool'})
    
    # Calculate total
    total_deduplicated_questions = sum([v for k, v in combined_row.items() if k != 'Tool'])
    combined_row['Total_Questions'] = total_deduplicated_questions
    
    deduplicated_df = pd.DataFrame([combined_row])
    
    print("Deduplicated question counts:")
    print(deduplicated_df)
    
    # Generate post-deduplication visualizations
    print("\nGenerating post-deduplication visualizations...")
    create_total_counts_plot(deduplicated_df, save_path="deduplicated_question_counts_by_source.png")
    create_simple_distribution_plot(deduplicated_df, save_path="deduplicated_question_distribution.png")
    
    # Save merged data to separate JSON files
    output_files = {
        'Article.json': merged_data['Article'],
        'GitHubReadme.json': merged_data['GitHubReadme'],
        'GitHubQuestions.json': merged_data['GitHubQuestions'],
        'Python.json': merged_data['Python'],
        'PDF.json': merged_data['PDF'],
        'R.json': merged_data['R'],
        'WebsiteAll.json': merged_data['WebsiteAll']
    }
    
    # Save deduplicated data
    deduplicated_output_files = {
        'Article_deduplicated.json': deduplicated_data['Article'],
        'GitHubReadme_deduplicated.json': deduplicated_data['GitHubReadme'],
        'GitHubQuestions_deduplicated.json': deduplicated_data['GitHubQuestions'],
        'Python_deduplicated.json': deduplicated_data['Python'],
        'PDF_deduplicated.json': deduplicated_data['PDF'],
        'R_deduplicated.json': deduplicated_data['R'],
        'WebsiteAll_deduplicated.json': deduplicated_data['WebsiteAll']
    }
    
    # Filter questions by output length and save filtered versions
    filtered_output_files = {}
    for filename, data in output_files.items():
        if data:  # Only process if there's data
            # Save original
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} questions to {filename}")
            
            # Filter and save filtered version
            filtered_data = filter_questions_by_output_length(data, min_words=30)
            filtered_filename = filename.replace('.json', '_filtered.json')
            filtered_output_files[filtered_filename] = filtered_data
            
            with open(filtered_filename, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Saved {len(filtered_data)} filtered questions to {filtered_filename}")
        else:
            print(f"No questions found for {filename}")
    
    # Save deduplicated files
    for filename, data in deduplicated_output_files.items():
        if data:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} deduplicated questions to {filename}")
            
            # Also save filtered deduplicated version
            filtered_data = filter_questions_by_output_length(data, min_words=30)
            filtered_filename = filename.replace('.json', '_filtered.json')
            with open(filtered_filename, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Saved {len(filtered_data)} deduplicated filtered questions to {filtered_filename}")
    
    print(f"\nMerging complete! Total questions by source:")
    for source, data in merged_data.items():
        print(f"{source}: {len(data)} questions")
    
    # Print total number of questions across all sources
    total_questions = sum(len(data) for data in merged_data.values())
    print(f"\nGrand Total Questions across all sources: {total_questions}")
    
    # Print deduplicated totals
    print(f"\nDeduplicated questions by source:")
    for source, data in deduplicated_data.items():
        print(f"{source}: {len(data)} questions")
    
    total_deduplicated = sum(len(data) for data in deduplicated_data.values())
    print(f"\nGrand Total Deduplicated Questions: {total_deduplicated}")
    print(f"Total Questions Removed: {total_questions - total_deduplicated}")
    print(f"Overall Reduction: {((total_questions - total_deduplicated) / total_questions * 100):.1f}%")
    
    # Print deduplicated + filtered totals
    print(f"\nDeduplicated + Filtered questions (output > 30 words) by source:")
    deduplicated_filtered_totals = {}
    for source, data in deduplicated_data.items():
        filtered_count = len(filter_questions_by_output_length(data, min_words=30))
        deduplicated_filtered_totals[source] = filtered_count
        print(f"{source}: {filtered_count} questions")
    
    total_deduplicated_filtered = sum(deduplicated_filtered_totals.values())
    print(f"\nGrand Total Deduplicated + Filtered Questions: {total_deduplicated_filtered}")
    
    # Create DataFrame for final filtered counts and generate plots
    print("\n" + "="*60)
    print("POST-FILTERING VISUALIZATION")
    print("="*60)
    
    # Create filtered counts DataFrame for plotting
    filtered_counts_data = []
    for source_key, source_display in [
        ('Article', 'Article_Questions'),
        ('GitHubReadme', 'Github_Readme_Questions'), 
        ('GitHubQuestions', 'Github_Questions'),
        ('Python', 'Python_Questions'),
        ('PDF', 'PDF_Questions'),
        ('R', 'R_Questions'),
        ('WebsiteAll', 'Website_All_Questions')
    ]:
        filtered_count = len(filter_questions_by_output_length(deduplicated_data[source_key], min_words=30))
        filtered_counts_data.append({
            'Tool': 'All_Tools',  # Single row for all tools combined
            source_display: filtered_count
        })
    
    # Combine all filtered counts into a single row
    filtered_combined_row = {'Tool': 'All_Tools'}
    for item in filtered_counts_data:
        filtered_combined_row.update({k: v for k, v in item.items() if k != 'Tool'})
    
    # Calculate total
    total_filtered_questions = sum([v for k, v in filtered_combined_row.items() if k != 'Tool'])
    filtered_combined_row['Total_Questions'] = total_filtered_questions
    
    filtered_df = pd.DataFrame([filtered_combined_row])
    
    print("Final filtered question counts (deduplicated + output > 30 words):")
    print(filtered_df)
    
    # Generate post-filtering visualizations
    print("\nGenerating final filtered visualizations...")
    create_total_counts_plot(filtered_df, save_path="final_filtered_question_counts_by_source.png")
    create_simple_distribution_plot(filtered_df, save_path="final_filtered_question_distribution.png")
    
    # Merge all questions into one file
    merge_all_questions()
    
    # Analyze question distribution
    print("\nStarting clustering analysis of question answers...")
    analyze_question_distribution()

# Run the function
if __name__ == "__main__":
    count_questions_in_articles()
    print("\n" + "="*50)
    print("Merging questions by source type...")
    merge_questions_by_source()


