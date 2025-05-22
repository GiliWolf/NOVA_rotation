import numpy as np
from scipy.spatial.distance import cdist
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.insert(0, os.getenv("HOME"))
from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_df, load_npy_to_nparray
from NOVA.src.datasets.dataset_config import DatasetConfig
from NOVA.src.figures.plot_config import PlotConfig
from NOVA.src.common.utils import load_config_file
from NOVA.src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels



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


def filter_and_save(labels_df:pd.DataFrame, embeddings_df:pd.DataFrame,paths_df:pd.DataFrame, marker, batch,  cell_line:str, condition:str, output_dir:str=None):
    # extract labels and indices
    filtered_labels= labels_df[
    (labels_df['cell_line'] == cell_line) &
    (labels_df['condition'] == condition)
    ]

    if filtered_labels.empty():
        print(f"No samples found for: {marker} & {batch} & {cell_line} & {condition}")
        return None

    # extract data 
    filtered_embeddings = embeddings_df.iloc[filtered_labels.index]
    filtered_paths = paths_df.iloc[filtered_labels.index]

    # save
    if output_dir:
        filtered_labels["full_label"].to_csv(os.path.join(output_dir, f"{cell_line}_{condition}_labels.csv"), index=False)
        np.save(os.path.join(output_dir, f"{cell_line}_{condition}_embedding.npy"),  filtered_embeddings.to_numpy())
        filtered_paths.to_csv(os.path.join(output_dir, f"{cell_line}_{condition}_paths.csv"), index=False)

    return filtered_labels, filtered_embeddings, filtered_paths

def main(output_folder_path:str, config_path_data:str):

    data_config:DatasetConfig = load_config_file(config_path_data, "data")
    data_config.OUTPUTS_FOLDER = output_folder_path


    home_dir = os.getenv("HOME")
    emb_out_dir = "NOVA_rotation/embeddings/embedding_output"
    run_name = "RotationDatasetConfig"
    embd_dir  = os.path.join(home_dir, emb_out_dir, run_name, "embeddings/neurons/batch9")


    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        # load data
        labels_df = load_npy_to_df(embd_dir,f"{set_type}_labels.npy", columns=['full_label'])
        paths_df = load_npy_to_df(embd_dir, f"{set_type}_paths.npy", columns=['Path'])
        embeddings_df = load_npy_to_df(embd_dir, f"{set_type}.npy")

        # split to groups
        labels_df[['marker', 'cell_line', 'condition', 'batch', 'replicate']] = labels_df['full_label'].str.split('_', expand=True)
        grouped = labels_df.groupby(['marker', 'cell_line', 'condition', 'batch'])
        print(f"\nlabels groups for {set_type}:")
        print(grouped.size())

        batches_names = labels_df['batch'].unique()
        for batch in batches_names:
            batch_labels, batch_embeddings, batch_paths = filter_by_labels(labels_df, embeddings_df, paths_df, {"batch":batch})

            if batch_labels.empty:
                print(f"Error: No samples found.")
                continue

            marker_names = batch_labels['marker'].unique()
            for marker in marker_names:
                marker_labels, marker_embeddings, marker_paths = filter_by_labels(batch_labels, batch_embeddings, batch_paths, {"marker": marker})
                if marker_labels.empty:
                    print(f"Error: No samples found.")
                    continue
                temp_output_dir = os.path.join(output_folder_path, data_config.EXPERIMENT_TYPE, batch, marker)
                os.makedirs(temp_output_dir, exist_ok=True)
                filtered_labels_c1, filtered_embeddings_c1, filtered_paths_c1  = filter_by_labels(marker_labels, marker_embeddings, marker_paths, {'cell_line':data_config.CELL_LINES[0], 'condition':data_config.CONDITIONS[0]})
                filtered_labels_c2, filtered_embeddings_c2, filtered_paths_c2 = filter_by_labels(marker_labels, marker_embeddings, marker_paths, {'cell_line':data_config.CELL_LINES[0], 'condition':data_config.CONDITIONS[1]})

            
if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply outputs folder path, data config.")
        outputs_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]

        main(outputs_folder_path, config_path_data)
        
    except Exception as e:
        print(e)
    print("Done.")
