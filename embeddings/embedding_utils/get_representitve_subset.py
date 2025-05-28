import numpy as np
from scipy.spatial.distance import cdist
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.insert(0, os.getenv("HOME"))
from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_df, load_npy_to_nparray, load_paths_from_npy
from NOVA.src.datasets.dataset_config import DatasetConfig
from NOVA.src.figures.plot_config import PlotConfig
from NOVA.src.common.utils import load_config_file
from NOVA.src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels
import logging


def filter_by_labels(labels_df: pd.DataFrame,embeddings_df: pd.DataFrame,paths_df: pd.DataFrame, filters: dict
):
    """
    Filter labels, embeddings, and paths based on multiple column-value conditions.

    Parameters:
        labels_df (pd.DataFrame): DataFrame containing labels and metadata.
        embeddings_df (pd.DataFrame): DataFrame with embeddings, aligned by index.
        paths_df (pd.DataFrame): DataFrame with file paths or additional info, aligned by index.
        filters (dict): Dictionary of {column_name: value} to filter on.

    Returns:
        filtered_labels, filtered_embeddings, filtered_paths: Filtered DataFrames.
    """
    
    # Apply all filters
    mask = pd.Series(True, index=labels_df.index)
    for col, val in filters.items():
        mask &= (labels_df[col] == val)

    filtered_labels = labels_df[mask]
    filtered_embeddings = embeddings_df.loc[filtered_labels.index]
    filtered_paths = paths_df.loc[filtered_labels.index]

    return filtered_labels, filtered_embeddings, filtered_paths


def compute_distances(a1:np.array, a2:np.array, metric='euclidean'):
    """"Compute all pairwise distances between 2 vectors.
    parameters:
        a1: first array 
        a2: second array
        mteric: metric to calculate the dist accordinly (defualt: cosine)
    
    return:
        dim1: dimension of a1
        dim2: dimension of a2
        flattened_distances: distance matrix 
    """
    distances = cdist(a1, a2, metric=metric)  
    return distances

def get_pairs(flattened_distances, n_pairs, dim2):
    # get min/max/middle dist pairs
    sorted_indices = np.argsort(flattened_distances) # sort (ascendingly)
    pairs = [(idx // dim2, idx % dim2) for idx in sorted_indices] # extract original (matrix) indices 
    min_pairs = pairs[:n_pairs]
    max_pairs = pairs[-n_pairs:]
    middle_start = (len(pairs) // 2) - (n_pairs // 2)
    middle_pairs = pairs[middle_start:middle_start + n_pairs]
    
    return min_pairs, max_pairs, middle_pairs

def visualize_pairs(distances, flattened_distances, min_pairs, max_pairs, middle_pairs, metric, output_dir=None):
    """
        create a distribution plot of the distanced and marker the pairs on top of it.
    """
    # Distances of selected pairs
    selected_min_distances = [distances[i, j] for i, j in min_pairs]
    selected_max_distances = [distances[i, j] for i, j in max_pairs]
    selected_middle_distances = [distances[i, j] for i, j in middle_pairs]


    plt.figure(figsize=(10, 6))
    plt.hist(flattened_distances, bins=50, alpha=0.7, color='blue', label='All distances')

    # Plot vertical lines
    for d in selected_min_distances:
        plt.axvline(d, color='green', linestyle='--', linewidth=1)

    for d in selected_max_distances:
        plt.axvline(d, color='red', linestyle='--', linewidth=1)

    for d in selected_middle_distances:
        plt.axvline(d, color='yellow', linestyle='--', linewidth=1)

    plt.xlabel(f"{metric} Distance")
    plt.ylabel("Count")
    plt.title("Distribution of All Pairwise Distances")
    plt.legend()

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{metric}_distance_distribution.png"))
        plt.close()
    else:
        plt.show()


def extract_subset(marker_labels:pd.DataFrame, marker_embeddings:pd.DataFrame, marker_paths:pd.DataFrame, metric:str, num_pairs:int, set_type:str, data_config:DatasetConfig, output_dir:str, mutual_attr:str = "cell_lines", compare_by_attr:str = "condition", compare_by_attr_idx:list = [0,1]):
                """
                extract subset of samples from marker_embeddings by -
                    1) computing all pair-wise distances
                    2) sorting the distances and taking the num_pairs with minimal/middle/maxinal distance
                    3) save csv file with the distances
                    4) extract data (emb, labels, paths) of the curresponding pairs and save them in npy files
                """

                mutual_param = getattr(data_config, mutual_attr.upper())[0]
                compare_param_c1 = getattr(data_config,compare_by_attr.upper())[compare_by_attr_idx[0]]
                compare_param_c2 = getattr(data_config,compare_by_attr.upper())[compare_by_attr_idx[1]]

                filtered_labels_c1, filtered_embeddings_c1, filtered_paths_c1  = filter_by_labels(marker_labels, marker_embeddings, marker_paths, {mutual_attr.lower():mutual_param, compare_by_attr.lower():compare_param_c1})
                filtered_labels_c2, filtered_embeddings_c2, filtered_paths_c2 = filter_by_labels(marker_labels, marker_embeddings, marker_paths, {mutual_attr.lower():mutual_param, compare_by_attr.lower():compare_param_c2})
                
                #convert to nparraay
                filtered_embeddings_c1 = np.array(filtered_embeddings_c1)
                filtered_embeddings_c2 = np.array(filtered_embeddings_c2)
                
                num_samples_c1, num_samples_c2 = len(filtered_embeddings_c1), len(filtered_embeddings_c2)
                
                # Compute all pairwise distances between condition 1 and 2
                distances = compute_distances(filtered_embeddings_c1, filtered_embeddings_c2, metric)
                flattened_distances = distances.flatten()
                
                # extract min/max/middle pairs
                min_pairs, max_pairs, middle_pairs = get_pairs(flattened_distances, num_pairs, dim2 = num_samples_c2)

                # Combine all pairs with their labels
                labeled_pairs = [
                    ("min", pair) for pair in min_pairs
                ] + [
                    ("middle", pair) for pair in middle_pairs
                ] + [
                    ("max", pair) for pair in max_pairs
                ]

                # Get indices for condition 1 and 2 samples
                c1_indices = np.array(list(set([i for (_, (i, j)) in labeled_pairs])))
                c2_indices = np.array(list(set([j for (_, (i, j)) in labeled_pairs])))
                logging.info(f"Selected {len(c1_indices)} {compare_param_c1} samples and {len(c2_indices)} {compare_param_c2} samples.")

                # Save distances
                distances_df = pd.DataFrame()
                distances_df["pair_type"] = [label for (label, (i, j)) in labeled_pairs]
                distances_df[f"{metric}_distance"] = [distances[i, j] for (label, (i, j)) in labeled_pairs]
                distances_df[f"path_{compare_param_c1}"] = [filtered_paths_c1.iloc[i]["Path"] for (label, (i, j)) in labeled_pairs]
                distances_df[f"path_{compare_param_c2}"] = [filtered_paths_c2.iloc[j]["Path"] for (label, (i, j)) in labeled_pairs]
                distances_df.to_csv(os.path.join(output_dir, f"{metric}_distances.csv"), index=False)

                # extract embeding,labels and paths values
                c1_embeddings = filtered_embeddings_c1[c1_indices]
                c1_labels = filtered_labels_c1.iloc[c1_indices]
                c1_paths = filtered_paths_c1.iloc[c1_indices]

                c2_embeddings = filtered_embeddings_c2[c2_indices]
                c2_labels = filtered_labels_c2.iloc[c2_indices]
                c2_paths = filtered_paths_c2.iloc[c2_indices]

                set_embeddings = np.concatenate([c1_embeddings, c2_embeddings], axis=0)
                set_labels = pd.concat([c1_labels, c2_labels], axis=0).reset_index(drop=True)
                set_paths= pd.concat([c1_paths, c2_paths], axis=0).reset_index(drop=True)

                # Save npy files
                np.save(os.path.join(output_dir, f"{set_type}.npy"), set_embeddings)
                np.save(os.path.join(output_dir, f"{set_type}_labels.npy"), set_labels["full_label"].values)
                np.save(os.path.join(output_dir, f"{set_type}_paths.npy"), np.array(set_paths["Path"].values, dtype=str))

                visualize_pairs(distances, flattened_distances, min_pairs, max_pairs, middle_pairs, metric, output_dir = output_dir)    


def main(emb_dir:str,  config_path_data:str, config_path_subset:str):

    dataset_name = os.path.basename(config_path_data)
    input_folder_path = os.path.join(emb_dir, dataset_name, "embeddings")
    output_folder_path = os.path.join(emb_dir, dataset_name, "pairs")

    # init subset config with a dataset config
    data_config:SubsetConfig = load_config_file(config_path_subset, "data", args = load_config_file(config_path_data, "data"))
    data_config.OUTPUTS_FOLDER = output_folder_path

    metric:str = data_config.METRIC
    num_pairs:int = data_config. NUM_PAIRS 
    mutual_attr:str = data_config.MUTUAL_ATTR
    compare_by_attr:str = data_config.COMPARE_BY_ATTR
    compare_by_attr_idx:list = data_config.COMPARE_BY_ATTR_IDX

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        # get batche names by all subdirs that starts with "batch"
        temp_input_folder_path = os.path.join(input_folder_path, data_config.EXPERIMENT_TYPE)
        batches_names = [name for name in os.listdir(temp_input_folder_path)
              if os.path.isdir(os.path.join(temp_input_folder_path, name)) and name.lower().startswith("batch")]
        if not batches_names:
            logging.info(f"Error: No batches dirs found. exiting")
            sys.exit()

        for batch in batches_names:
            temp_input_folder_path = os.path.join(input_folder_path, data_config.EXPERIMENT_TYPE, batch)
            # load data
            batch_labels = load_npy_to_df(temp_input_folder_path,f"{set_type}_labels.npy", columns=['full_label'])
            batch_paths = load_npy_to_df(temp_input_folder_path, f"{set_type}_paths.npy", columns=['Path'])
            batch_embeddings = load_npy_to_df(temp_input_folder_path, f"{set_type}.npy")

            batch_labels[['markers', 'cell_lines', 'conditions', 'batch', 'replicate']] = batch_labels['full_label'].str.split('_', expand=True)
            grouped = batch_labels.groupby(['markers', 'cell_lines', 'conditions', 'batch'])
            logging.info(f"\nlabels groups for {set_type}:")
            logging.info(grouped.size())

            marker_names = batch_labels['markers'].unique()
            for marker in marker_names:
                # filter by marker
                marker_labels, marker_embeddings, marker_paths = filter_by_labels(batch_labels, batch_embeddings, batch_paths, {"markers": marker})
                if marker_labels.empty:
                    logging.info(f"Error: No samples found for marker {marker}. continues.")
                    continue

                temp_output_dir = os.path.join(output_folder_path, metric, data_config.EXPERIMENT_TYPE, batch, marker)
                os.makedirs(temp_output_dir, exist_ok=True)

                # run extraction of subset logic
                extract_subset(marker_labels, marker_embeddings, marker_paths, metric, num_pairs, set_type, data_config, temp_output_dir, mutual_attr= mutual_attr, compare_by_attr =compare_by_attr, compare_by_attr_idx = compare_by_attr_idx)
                
                            
if __name__ == "__main__":
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply emb_dir and data config.")
        emb_dir = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_subset = sys.argv[3]

        main(emb_dir, config_path_data, config_path_subset)
        
    except Exception as e:
        print(e)
    print("Done.")
