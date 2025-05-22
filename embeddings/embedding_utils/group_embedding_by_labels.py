import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.insert(0, os.getenv("HOME"))
from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_df, load_npy_to_nparray

# init paths and parameteres
home_dir = os.getenv("HOME")
emb_out_dir = "NOVA_rotation/embeddings/embedding_output"
run_name = "RotationDatasetConfig_New_paths"
embd_dir  = os.path.join(home_dir, emb_out_dir, run_name, "embeddings/neurons/batch9")
output_dir = os.path.join(home_dir, emb_out_dir, run_name, "grouped_embedding")
data_set_types = ['testset'] # OR ['trainset','valset','testset']
os.makedirs(output_dir, exist_ok=True)



def filter_and_save(labels_df:pd.DataFrame, embeddings_df:pd.DataFrame,paths_df:pd.DataFrame, condition:str, treatment:str, output_dir:str=None):
    # extract labels and indices
    filtered_labels= labels_df[
    (labels_df['condition'] == condition) &
    (labels_df['treatment'] == treatment)
    ]

    # extract data 
    print(f"\n{condition}_{treatment}_labels:")
    print(filtered_labels.shape)

    print(f"\n{condition}_{treatment}_embeddings:")
    filtered_embeddings = embeddings_df.iloc[filtered_labels.index]
    print(filtered_embeddings.shape)

    print(f"\n{condition}_{treatment}_paths:")
    filtered_paths = paths_df.iloc[filtered_labels.index]
    print(filtered_paths.shape)

    # save
    if output_dir:
        filtered_labels["full_label"].to_csv(os.path.join(output_dir, f"{condition}_{treatment}_labels.csv"), index=False)
        np.save(os.path.join(output_dir, f"{condition}_{treatment}_embedding.npy"),  filtered_embeddings.to_numpy())
        filtered_paths.to_csv(os.path.join(output_dir, f"{condition}_{treatment}_paths.csv"), index=False)

    return filtered_labels, filtered_embeddings, filtered_paths



if __name__ == "__main__":
    for i, set_type in enumerate(data_set_types):
            # load data
            labels_df = load_npy_to_df(embd_dir,f"{set_type}_labels.npy", columns=['full_label'])
            paths_df = load_npy_to_df(embd_dir, f"{set_type}_paths.npy", columns=['Path'])
            embeddings_df = load_npy_to_df(embd_dir, f"{set_type}.npy")

            
            # split to groups
            labels_df[['protein', 'condition', 'treatment', 'batch', 'replicate']] = labels_df['full_label'].str.split('_', expand=True)
            grouped = labels_df.groupby(['protein', 'condition', 'treatment'])
            print("\nlabels groups:")
            print(grouped.size())

            temp_output_dir = os.path.join(output_dir, set_type)
            os.makedirs(temp_output_dir, exist_ok = True)
            filter_and_save(labels_df, embeddings_df,paths_df, condition = "WT", treatment="Untreated", output_dir = temp_output_dir)
            filter_and_save(labels_df, embeddings_df,paths_df, condition = "WT", treatment="stress", output_dir = temp_output_dir)





