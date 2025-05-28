import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import pandas as pd
from collections import Counter
import os
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(filename):
    """Load and parse the JSON data file."""
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            # Try loading as JSONL
            file.seek(0)
            data = [json.loads(line) for line in file if line.strip()]
    
    return data

def create_prompt_completion_pairs(data):
    """Convert Question and output to prompt/completion pairs with group information."""
    processed_data = []
    
    for i, item in enumerate(data):
        # Create a unique group ID for each question
        group_id = item.get('ToolName', f'group_{i}')
        
        # Process Question/output pair
        if 'Question' in item and 'output' in item:
            processed_data.append({
                'prompt': item['Question'],
                'completion': item['output'],
                'group_id': group_id,
                'pair_type': 'question_output',
                'tool_name': item.get('ToolName', 'Unknown'),
                'source_type': item.get('SourceType', 'Unknown')
            })
    
    return processed_data

def create_semantic_embeddings(data):
    """Create semantic embeddings using a pre-trained language model."""
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract prompts
    prompts = [item['prompt'] for item in data]
    
    # Generate embeddings
    print("Generating semantic embeddings...")
    embeddings = model.encode(prompts, show_progress_bar=True)
    
    return embeddings

def semantic_clustering(embeddings, min_cluster_size=None):
    """Perform semantic clustering using UMAP dimensionality reduction and HDBSCAN with optimal cluster parameters."""
    # First reduce dimensionality with UMAP
    print("Performing dimensionality reduction with UMAP...")
    umap_embeddings = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        metric='cosine',
        random_state=42
    ).fit_transform(embeddings)
    
    # Find optimal min_cluster_size if not provided
    if min_cluster_size is None:
        print("Finding optimal number of clusters...")
        best_score = -1
        best_min_cluster_size = 0
        best_labels = None
        
        # Try a range of min_cluster_size values
        min_sizes_to_try = range(5, min(100, len(embeddings) // 20), 5)
        for min_size in min_sizes_to_try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            clusterer.fit(umap_embeddings)
            
            # Get the number of clusters and non-noise points
            unique_clusters = set(clusterer.labels_)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            non_noise_mask = clusterer.labels_ != -1
            
            # Only evaluate clustering if we have more than one cluster and some non-noise points
            if n_clusters > 1 and np.sum(non_noise_mask) > min_size:
                try:
                    # Use silhouette score instead of DBCV
                    non_noise_data = umap_embeddings[non_noise_mask]
                    non_noise_labels = clusterer.labels_[non_noise_mask]
                    score = silhouette_score(non_noise_data, non_noise_labels)
                    
                    print(f"Min cluster size {min_size} produced {n_clusters} clusters with silhouette score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_min_cluster_size = min_size
                        best_labels = clusterer.labels_
                except Exception as e:
                    print(f"Error calculating score for min_size={min_size}: {e}")
        
        if best_min_cluster_size > 0:
            print(f"Optimal min_cluster_size: {best_min_cluster_size} with silhouette score: {best_score:.4f}")
            labels = best_labels
        else:
            # Fallback to default if optimization fails
            print("Optimization failed, using default min_cluster_size=10")
            min_cluster_size = 10
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit(umap_embeddings)
            labels = clusterer.labels_
    else:
        # Use provided min_cluster_size
        print(f"Using provided min_cluster_size={min_cluster_size}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        ).fit(umap_embeddings)
        labels = clusterer.labels_
    
    # Get cluster labels (including outliers labeled as -1)
    outlier_count = sum(labels == -1)
    print(f"Found {outlier_count} outliers ({outlier_count/len(labels)*100:.1f}% of data)")
    
    # Count non-outlier clusters
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Created {num_clusters} semantic clusters (excluding outliers)")
    
    return labels

def split_data_by_clusters_and_groups(data, clusters, train_size=0.8, val_size=10000, test_size=100, remove_outliers=False):
    """
    Split data ensuring proportional representation from each cluster and keeping question pairs together.
    
    Parameters:
    -----------
    data : list
        The data to split
    clusters : numpy.ndarray
        The cluster assignments including outliers (-1)
    train_size : float
        The proportion of data to allocate to the training set
    val_size : int
        The size of the validation set
    test_size : int
        The size of the test set
    remove_outliers : bool
        Whether to remove outliers from the datasets
    """
    # If remove_outliers is True, filter out outliers
    if remove_outliers:
        print("Removing outliers from all datasets...")
        # Create a mask for non-outlier data
        non_outlier_mask = clusters != -1
        # Filter data and clusters
        filtered_indices = np.where(non_outlier_mask)[0]
        filtered_data = [data[i] for i in filtered_indices]
        filtered_clusters = clusters[non_outlier_mask]
        
        # Continue with filtered data
        data_to_process = filtered_data
        clusters_to_process = filtered_clusters
        
        # Count removed outliers
        removed_count = len(data) - len(filtered_data)
        print(f"Removed {removed_count} outliers ({removed_count/len(data)*100:.1f}% of data)")
    else:
        print("Keeping outliers in all datasets...")
        # Use all data
        data_to_process = data
        clusters_to_process = clusters
    
    # Group data by group_id
    groups = {}
    for i, item in enumerate(data_to_process):
        group_id = item['group_id']
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(i)
    
    # Calculate average cluster for each group
    group_clusters = {}
    for group_id, indices in groups.items():
        group_clusters[group_id] = int(np.mean([clusters_to_process[idx] for idx in indices]))
    
    # Create a dictionary of cluster to groups
    cluster_groups = {}
    for group_id, cluster in group_clusters.items():
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(group_id)
    
    total_groups = len(groups)
    total_samples = len(data_to_process)
    
    print(f"Total groups: {total_groups}, Total samples: {total_samples}")
    print(f"Average samples per group: {total_samples/total_groups:.2f}")
    
    # Work with individual samples to get exactly the requested sizes
    # Shuffle all data indices for random selection
    all_indices = list(range(len(data_to_process)))
    random.shuffle(all_indices)
    
    # Create lists to track which cluster each sample belongs to
    sample_clusters = [clusters_to_process[i] for i in all_indices]
    
    # Get unique clusters and their proportions
    unique_clusters = list(set(clusters_to_process))
    cluster_proportions = {}
    for cluster in unique_clusters:
        cluster_proportions[cluster] = sum(1 for c in clusters_to_process if c == cluster) / len(clusters_to_process)
    
    print(f"Cluster proportions: {cluster_proportions}")
    
    # Allocate samples to splits maintaining cluster proportions
    test_indices = []
    val_indices = []
    
    # For test set - select exactly test_size samples
    samples_per_cluster_test = {}
    for cluster in unique_clusters:
        samples_per_cluster_test[cluster] = max(1, int(test_size * cluster_proportions[cluster]))
    
    # Adjust if we have too many samples allocated
    total_allocated_test = sum(samples_per_cluster_test.values())
    if total_allocated_test > test_size:
        # Reduce proportionally
        scale_factor = test_size / total_allocated_test
        for cluster in unique_clusters:
            samples_per_cluster_test[cluster] = max(1, int(samples_per_cluster_test[cluster] * scale_factor))
    
    # Select test samples
    cluster_counts_test = {cluster: 0 for cluster in unique_clusters}
    for i, sample_idx in enumerate(all_indices):
        if len(test_indices) >= test_size:
            break
        sample_cluster = sample_clusters[i]
        if cluster_counts_test[sample_cluster] < samples_per_cluster_test[sample_cluster]:
            test_indices.append(sample_idx)
            cluster_counts_test[sample_cluster] += 1
    
    # Fill remaining test slots if needed
    remaining_indices = [idx for idx in all_indices if idx not in test_indices]
    while len(test_indices) < test_size and remaining_indices:
        test_indices.append(remaining_indices.pop(0))
    
    # For validation set - select exactly val_size samples from remaining
    remaining_indices = [idx for idx in all_indices if idx not in test_indices]
    random.shuffle(remaining_indices)
    
    samples_per_cluster_val = {}
    remaining_cluster_props = {}
    for cluster in unique_clusters:
        remaining_in_cluster = sum(1 for idx in remaining_indices if clusters_to_process[idx] == cluster)
        remaining_cluster_props[cluster] = remaining_in_cluster / len(remaining_indices) if remaining_indices else 0
        samples_per_cluster_val[cluster] = max(1, int(val_size * remaining_cluster_props[cluster]))
    
    # Adjust validation allocation
    total_allocated_val = sum(samples_per_cluster_val.values())
    if total_allocated_val > val_size and total_allocated_val > 0:
        scale_factor = val_size / total_allocated_val
        for cluster in unique_clusters:
            samples_per_cluster_val[cluster] = max(1, int(samples_per_cluster_val[cluster] * scale_factor))
    
    # Select validation samples
    cluster_counts_val = {cluster: 0 for cluster in unique_clusters}
    for sample_idx in remaining_indices:
        if len(val_indices) >= val_size:
            break
        sample_cluster = clusters_to_process[sample_idx]
        if cluster_counts_val[sample_cluster] < samples_per_cluster_val[sample_cluster]:
            val_indices.append(sample_idx)
            cluster_counts_val[sample_cluster] += 1
    
    # Fill remaining validation slots if needed
    remaining_after_val = [idx for idx in remaining_indices if idx not in val_indices]
    while len(val_indices) < val_size and remaining_after_val:
        val_indices.append(remaining_after_val.pop(0))
    
    # All remaining samples go to training
    train_indices = [idx for idx in all_indices if idx not in test_indices and idx not in val_indices]
    
    print(f"Actual allocation - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create the datasets
    train_data = [data_to_process[i] for i in train_indices]
    val_data = [data_to_process[i] for i in val_indices]
    test_data = [data_to_process[i] for i in test_indices]
    
    # Create mapping from filtered indices to original indices if outliers were removed
    if remove_outliers:
        train_original_indices = [filtered_indices[i] for i in train_indices]
        val_original_indices = [filtered_indices[i] for i in val_indices]
        test_original_indices = [filtered_indices[i] for i in test_indices]
        
        return train_data, val_data, test_data, train_original_indices, val_original_indices, test_original_indices
    else:
        return train_data, val_data, test_data, train_indices, val_indices, test_indices

def analyze_semantic_distribution(train_data, val_data, test_data, embeddings, clusters, train_indices, val_indices, test_indices, remove_outliers=False):
    """
    Create an enhanced publication-quality visualization of semantic distribution across datasets.
    """
 
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 600
    })
    
    # Get indices for each set
    all_data = train_data + val_data + test_data
    
    # Create 2D projection for visualization with optimized parameters
    print("Creating UMAP projection for visualization...")
    umap_2d = umap.UMAP(
        n_neighbors=20,
        n_components=2,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    ).fit_transform(embeddings)
    
    # Enhanced color scheme suitable for publication (colorblind-friendly with improved contrast)
    train_color = '#0173B2'  # Blue
    val_color = '#029E73'    # Green
    test_color = '#D55E00'   # Orange-red
    outlier_color = '#CC78BC'  # Purple for outliers
    
    # Create a figure with higher resolution
    fig = plt.figure(figsize=(12, 8))
    
    # Create gridspec for plot and legend with better proportions
    gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
    
    # Main plot area with improved styling
    ax = fig.add_subplot(gs[0])
    
    # Add enhanced background and styling
    ax.set_facecolor('#F8F8F8')
    ax.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC')
    
    # Identify clusters for visualization
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)  # Exclude outlier cluster
    
    # Map clusters to colors for better visualization (excluding -1 for outliers)
    cmap = plt.cm.viridis
    cluster_colors = {c: cmap(i / max(1, num_clusters)) for i, c in enumerate(unique_clusters) if c != -1}
    
    # Identify outliers and non-outliers in each dataset
    # When remove_outliers=True, these lists should be empty since outliers were removed
    train_outlier_indices = [i for i in train_indices if clusters[i] == -1]
    val_outlier_indices = [i for i in val_indices if clusters[i] == -1]
    test_outlier_indices = [i for i in test_indices if clusters[i] == -1]
    
    train_non_outlier_indices = [i for i in train_indices if clusters[i] != -1]
    val_non_outlier_indices = [i for i in val_indices if clusters[i] != -1]
    test_non_outlier_indices = [i for i in test_indices if clusters[i] != -1]
    
    # Add enhanced density contours with better transparency and granularity
    def add_density_contours(points, color, alpha=0.2, levels=7):
        if len(points) < 10:
            return
            
        x, y = points[:, 0], points[:, 1]
        
        # Calculate the point density with improved bandwidth
        xy = np.vstack([x, y])
        try:
            # Use Scott's rule for bandwidth selection for better contour definition
            kernel = gaussian_kde(xy, bw_method='scott')
            
            # Create a meshgrid with more points for smoother contours
            margin = 0.1
            x_min, x_max = min(x) - margin, max(x) + margin
            y_min, y_max = min(y) - margin, max(y) + margin
            xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Get the density values
            z = np.reshape(kernel(positions).T, xx.shape)
            
            # Create custom colormap for better visualization
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [color, color], N=100)
            
            # Plot contours with improved styling
            contour = ax.contour(xx, yy, z, levels=levels, colors=color, 
                               alpha=alpha*1.5, linewidths=0.8)
            contourf = ax.contourf(xx, yy, z, levels=levels, cmap=custom_cmap, 
                                 alpha=alpha)
        except Exception as e:
            print(f"Could not create contours: {e}")
    
    # Visualize cluster distribution with improved styling
    print("Plotting improved data visualization...")
    
    # Create scatter plot handles for legend
    scatter_handles = []
    
    # Add cluster labels as a semi-transparent background
    if num_clusters > 1:  # Only if we have more than one non-outlier cluster
        # Plot points colored by cluster for the background context
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip outliers for this background layer
                continue
                
            # Get indices for this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                ax.scatter(
                    umap_2d[cluster_indices, 0], umap_2d[cluster_indices, 1],
                    c=[cluster_colors[cluster_id]], s=15, alpha=0.1,
                    edgecolors='none', zorder=1
                )
    
    # Add density contours for each dataset (non-outliers only)
    if len(train_non_outlier_indices) > 10:
        add_density_contours(umap_2d[train_non_outlier_indices], train_color, alpha=0.15, levels=8)
    if len(val_non_outlier_indices) > 10:
        add_density_contours(umap_2d[val_non_outlier_indices], val_color, alpha=0.15, levels=8)
    if len(test_non_outlier_indices) > 10:
        add_density_contours(umap_2d[test_non_outlier_indices], test_color, alpha=0.15, levels=8)
    
    # Plot the non-outlier scatter points with improved styling
    if train_non_outlier_indices:
        train_scatter = ax.scatter(
            umap_2d[train_non_outlier_indices, 0], umap_2d[train_non_outlier_indices, 1],
            c=train_color, s=40, alpha=0.7, edgecolors='white', linewidths=0.3, zorder=10
        )
        scatter_handles.append((train_scatter, f'Training (n={len(train_indices)})'))
    
    if val_non_outlier_indices:
        val_scatter = ax.scatter(
            umap_2d[val_non_outlier_indices, 0], umap_2d[val_non_outlier_indices, 1],
            c=val_color, s=50, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=11
        )
        scatter_handles.append((val_scatter, f'Validation (n={len(val_indices)})'))
    
    if test_non_outlier_indices:
        test_scatter = ax.scatter(
            umap_2d[test_non_outlier_indices, 0], umap_2d[test_non_outlier_indices, 1],
            c=test_color, s=60, alpha=0.9, edgecolors='white', linewidths=0.3, zorder=12
        )
        scatter_handles.append((test_scatter, f'Test (n={len(test_indices)})'))
    
    # Plot outliers with star markers and enhanced styling ONLY if outliers are kept
    # This is the fixed part - only show outliers in the plot if remove_outliers=False
    if not remove_outliers:
        if train_outlier_indices:
            train_outlier_scatter = ax.scatter(
                umap_2d[train_outlier_indices, 0], umap_2d[train_outlier_indices, 1],
                c=outlier_color, s=70, alpha=0.9, marker='*', edgecolors='white', linewidths=0.5, zorder=20
            )
            scatter_handles.append((train_outlier_scatter, f'Training Outliers (n={len(train_outlier_indices)})'))
        
        if val_outlier_indices:
            val_outlier_scatter = ax.scatter(
                umap_2d[val_outlier_indices, 0], umap_2d[val_outlier_indices, 1],
                c=outlier_color, s=80, alpha=0.9, marker='*', edgecolors='white', linewidths=0.5, zorder=21
            )
            scatter_handles.append((val_outlier_scatter, f'Validation Outliers (n={len(val_outlier_indices)})'))
        
        if test_outlier_indices:
            test_outlier_scatter = ax.scatter(
                umap_2d[test_outlier_indices, 0], umap_2d[test_outlier_indices, 1],
                c=outlier_color, s=90, alpha=0.9, marker='*', edgecolors='white', linewidths=0.5, zorder=22
            )
            scatter_handles.append((test_outlier_scatter, f'Test Outliers (n={len(test_outlier_indices)})'))
    
    # Set axis limits with better padding
    x_min, x_max = np.min(umap_2d[:, 0])-1.5, np.max(umap_2d[:, 0])+1.5
    y_min, y_max = np.min(umap_2d[:, 1])-1.5, np.max(umap_2d[:, 1])+1.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add statistics about total clusters
    num_clusters_text = f"Total clusters: {num_clusters}"
    ax.text(0.02, 0.98, num_clusters_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Calculate diversity metrics
    def calculate_diversity(indices):
        if len(indices) < 2:
            return 0
        
        subset_embeddings = embeddings[indices]
        similarities = cosine_similarity(subset_embeddings)
        np.fill_diagonal(similarities, 0)
        return 1 - similarities.sum() / (len(indices) * (len(indices) - 1))
    
    train_diversity = calculate_diversity(train_indices)
    val_diversity = calculate_diversity(val_indices)
    test_diversity = calculate_diversity(test_indices)
    
    # Calculate outlier percentage in each set
    train_outlier_pct = len(train_outlier_indices) / len(train_indices) * 100 if train_indices else 0
    val_outlier_pct = len(val_outlier_indices) / len(val_indices) * 100 if val_indices else 0
    test_outlier_pct = len(test_outlier_indices) / len(test_indices) * 100 if test_indices else 0
    
    # Formatting and labels with improved typography
    ax.set_xlabel('UMAP Dimension 1', fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontweight='bold')
    outlier_status = "with Outliers Removed" if remove_outliers else "with Outliers Included"
    ax.set_title(f'Semantic Distribution of Question Datasets ({outlier_status})', pad=15, fontweight='bold')
    
    # Remove tick labels to reduce clutter but improve tick visibility
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    
    # Fine-tune aesthetics for spines with enhanced styling
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.2)
    
    # Create legend axis (outside the main plot) with improved styling
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')
    
    # Create custom legend with better spacing and organization
    handles = [h[0] for h in scatter_handles]
    labels = [h[1] for h in scatter_handles]
    
    legend = legend_ax.legend(
        handles, 
        labels,
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        fancybox=False,
        title='Dataset Distribution',
        borderpad=1.0,
        labelspacing=1.2
    )
    legend.get_title().set_fontweight('bold')
    
    # Add information panel with improved formatting showing the diversity metrics and outlier percentages
    outlier_status_text = "Outlier Percentages (Removed from Dataset):" if remove_outliers else "Outlier Percentages (Included in Dataset):"
    
    info_text = (
        f"Semantic Diversity Metrics:\n"
        f"Training set: {train_diversity:.4f}\n"
        f"Validation set: {val_diversity:.4f}\n"
        f"Test set: {test_diversity:.4f}\n\n"
        f"{outlier_status_text}\n"
        f"Training set: {train_outlier_pct:.1f}%\n"
        f"Validation set: {val_outlier_pct:.1f}%\n"
        f"Test set: {test_outlier_pct:.1f}%"
    )
    
    # Add the info box with improved styling
    info_box = legend_ax.text(
        0.05, 0.5, info_text, 
        transform=legend_ax.transAxes,
        fontsize=10, 
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F8F8', edgecolor='#CCCCCC', alpha=0.95)
    )
    
    # Add explanation about outliers
    if remove_outliers:
        outlier_note = (
            f"Note: Outliers were removed from all datasets.\n"
            f"They represented semantically unique questions\n"
            f"that didn't fit well into any cluster."
        )
    else:
        outlier_note = (
            f"Note: Outliers are included in all datasets.\n"
            f"They represent semantically unique questions\n"
            f"that don't fit well into any cluster."
        )
    
    legend_ax.text(
        0.05, 0.10, outlier_note, 
        transform=legend_ax.transAxes,
        fontsize=9, 
        style='italic',
        verticalalignment='bottom'
    )
    
    plt.tight_layout()
    
    # Save in multiple formats for publication with increased quality
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    outlier_filename = "without_outliers" if remove_outliers else "with_outliers"
    print(f"Saving enhanced visualization in multiple formats...")
    plt.savefig(f'{output_dir}/semantic_distribution_{outlier_filename}.png', dpi=600, bbox_inches='tight')
    
    outlier_status_message = "with outliers removed" if remove_outliers else "with outliers included"
    print(f"Enhanced publication-quality visualization saved {outlier_status_message}")
    
    # Print diversity results
    print("\nSemantic Diversity Results:")
    print(f"Train set: {train_diversity:.4f}")
    print(f"Validation set: {val_diversity:.4f}")
    print(f"Test set: {test_diversity:.4f}")
    
    # Print outlier results
    outlier_status_print = "removed from datasets" if remove_outliers else "included in datasets"
    print(f"\nOutlier Percentages ({outlier_status_print}):")
    print(f"Train set: {train_outlier_pct:.1f}% ({len(train_outlier_indices)} out of {len(train_indices)})")
    print(f"Validation set: {val_outlier_pct:.1f}% ({len(val_outlier_indices)} out of {len(val_indices)})")
    print(f"Test set: {test_outlier_pct:.1f}% ({len(test_outlier_indices)} out of {len(test_indices)})")
    print(f"Total clusters: {num_clusters}")
    
    return {
        'train_diversity': train_diversity,
        'val_diversity': val_diversity,
        'test_diversity': test_diversity,
        'train_outlier_pct': train_outlier_pct,
        'val_outlier_pct': val_outlier_pct,
        'test_outlier_pct': test_outlier_pct,
        'num_clusters': num_clusters,
        'figure': fig  # Return the figure for further adjustments if needed
    }

def verify_question_pairs(train_data, val_data, test_data, remove_outliers=False):
    """Verify that question pairs are kept together and create visualization."""
    # Group data by group_id in each dataset
    train_groups = {}
    val_groups = {}
    test_groups = {}
    
    for item in train_data:
        if item['group_id'] not in train_groups:
            train_groups[item['group_id']] = []
        train_groups[item['group_id']].append(item['pair_type'])
    
    for item in val_data:
        if item['group_id'] not in val_groups:
            val_groups[item['group_id']] = []
        val_groups[item['group_id']].append(item['pair_type'])
    
    for item in test_data:
        if item['group_id'] not in test_groups:
            test_groups[item['group_id']] = []
        test_groups[item['group_id']].append(item['pair_type'])
    
    # Check for any overlap
    all_groups = set(list(train_groups.keys()) + list(val_groups.keys()) + list(test_groups.keys()))
    split_issues = 0
    
    for group in all_groups:
        datasets_with_group = 0
        if group in train_groups:
            datasets_with_group += 1
        if group in val_groups:
            datasets_with_group += 1
        if group in test_groups:
            datasets_with_group += 1
        
        if datasets_with_group > 1:
            split_issues += 1
    
    if split_issues == 0:
        print("✓ All question pairs are kept together in the same dataset!")
    else:
        print(f"✗ Found {split_issues} groups split across multiple datasets.")
    
    # Count question-output pairs in each dataset
    train_qa_pairs = len(train_groups)
    val_qa_pairs = len(val_groups)
    test_qa_pairs = len(test_groups)
    
    print(f"\nQuestion-Answer pairs:")
    print(f"Train set: {train_qa_pairs} pairs")
    print(f"Validation set: {val_qa_pairs} pairs")
    print(f"Test set: {test_qa_pairs} pairs")

def save_to_jsonl(data, filename):
    """Save data to a JSONL file."""
    # Create output directory if it doesn't exist
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Create a copy of the data without the internal fields used for grouping
    clean_data = []
    for item in data:
        clean_item = item.copy()
        if 'group_id' in clean_item:
            del clean_item['group_id']
        if 'pair_type' in clean_item:
            del clean_item['pair_type']
        clean_data.append(clean_item)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in clean_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(clean_data)} items to {output_path}")

def main(remove_outliers=False):
    input_file = "Allquestions.json"
    
    # Create output directory
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and parse the data
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    
    # For debugging with smaller dataset, uncomment the next line
    #data = data[:1000]
    print(f"Loaded {len(data)} items")
    
    # Convert to prompt/completion pairs with group information
    processed_data = create_prompt_completion_pairs(data)
    print(f"Created {len(processed_data)} prompt/completion pairs")
    
    # Create semantic embeddings
    embeddings = create_semantic_embeddings(processed_data)
    
    # Perform semantic clustering with automatic cluster determination
    print("Performing semantic clustering with automatic cluster determination...")
    clusters = semantic_clustering(embeddings)
    
    # Calculate more reasonable sizes for train/val/test
    total_samples = len(processed_data)
    
    # Use percentage-based allocation that makes more sense
    test_size = min(100, 200)      # 2% for test, minimum 100
    val_size = max(500, 5000)       # 10% for validation, minimum 500
    
    # Ensure we don't exceed total samples
    if test_size + val_size >= total_samples:
        test_size = min(100, 200)    # Fallback to smaller sizes
        val_size = max(100, 5000)
    
    outlier_status = "removing outliers" if remove_outliers else "keeping outliers"
    print(f"Splitting data into train, validation, and test sets ({outlier_status})...")
    print(f"Target sizes: train (~{total_samples - val_size - test_size:.0f} samples), "
          f"validation ({val_size} samples), and test ({test_size} samples)")
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Split the data using the function with outlier removal option
    train_data, val_data, test_data, train_indices, val_indices, test_indices = split_data_by_clusters_and_groups(
        processed_data, clusters, 
        train_size=0.8, 
        val_size=val_size, 
        test_size=test_size,
        remove_outliers=remove_outliers
    )
    
    print(f"Final split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
    
    # Print tool distribution across splits
    print("\nTool distribution across splits:")
    for dataset_name, dataset in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        if len(dataset) > 0:
            tool_counts = Counter(item['tool_name'] for item in dataset)
            print(f"{dataset_name}: {dict(tool_counts)}")
        else:
            print(f"{dataset_name}: No data")
    
    # Analyze semantic distribution with improved visualization
    print("Analyzing semantic distribution across splits...")
    
    # Use full original embeddings for visualization (handle differently based on outlier removal)
    distribution_results = analyze_semantic_distribution(
        train_data, val_data, test_data, 
        embeddings, clusters, 
        train_indices, val_indices, test_indices,
        remove_outliers=remove_outliers
    )
    
    # Create filenames with outlier status
    outlier_suffix = "_without_outliers" if remove_outliers else "_with_outliers"
    
    # Save to JSONL files
    save_to_jsonl(train_data, f'Train{outlier_suffix}.jsonl')
    save_to_jsonl(val_data, f'Validation{outlier_suffix}.jsonl')
    save_to_jsonl(test_data, f'Test{outlier_suffix}.jsonl')
    
    # Print outlier statistics
    outlier_status_print = "removed from" if remove_outliers else "included in"
    
    # Identify outliers in each dataset
    train_outlier_indices = [i for i in train_indices if clusters[i] == -1]
    val_outlier_indices = [i for i in val_indices if clusters[i] == -1]
    test_outlier_indices = [i for i in test_indices if clusters[i] == -1]
    
    print(f"\nData processing complete!")
    print(f"Train set: {len(train_data)} questions ({len(train_data)/len(processed_data)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} questions ({len(val_data)/len(processed_data)*100:.1f}%)")
    print(f"Test set: {len(test_data)} questions ({len(test_data)/len(processed_data)*100:.1f}%)")
    
    print(f"\nOutlier statistics ({outlier_status_print} datasets):")
    print(f"Train set: {len(train_outlier_indices)} outliers ({len(train_outlier_indices)/len(train_indices)*100 if train_indices else 0:.1f}%)")
    print(f"Validation set: {len(val_outlier_indices)} outliers ({len(val_outlier_indices)/len(val_indices)*100 if val_indices else 0:.1f}%)")
    print(f"Test set: {len(test_outlier_indices)} outliers ({len(test_outlier_indices)/len(test_indices)*100 if test_indices else 0:.1f}%)")
    print(f"Total clusters found: {distribution_results['num_clusters']}")

if __name__ == "__main__":
    # Set remove_outliers to False to keep outliers (default behavior)
    # Set to True to remove outliers from all datasets
    remove_outliers = False
    main(remove_outliers=remove_outliers)
