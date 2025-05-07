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

# init paths
home_dir = os.getenv("HOME")
emb_out_dir = "NOVA_rotation/embeddings/embedding_output"
run_name = "RotationDatasetConfig_New_paths"
emb_dir = os.path.join(home_dir, emb_out_dir, run_name)
groups_dir  = os.path.join(emb_dir, "grouped_embedding")
# output control 
output_dir = os.path.join(emb_dir, "pairs")
os.makedirs(output_dir, exist_ok=True)

# parameters
data_set_types = ['testset'] # OR ['trainset','valset','testset']
NUM_PAIRS = 25 # numer of pairs for each catagory: {min_dist, max_dist, moderate_dist}

def load_files(groups_dir, condition, treatment):
    emb = load_npy_to_nparray(groups_dir,f"{condition}_{treatment}_embedding.npy")
    labels = pd.read_csv(os.path.join(groups_dir, f"{condition}_{treatment}_labels.csv"))
    paths = pd.read_csv(os.path.join(groups_dir, f"{condition}_{treatment}_paths.csv"))
    return emb, labels, paths


def compute_distances(a1:np.array, a2:np.array, metric='cosine'):
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
    dim_1, dim_2 = distances.shape
    return dim_1, dim_2, distances

def get_pairs(flattened_distances, N_PAIRS, dim2):
    # get min/max/middle dist pairs
    sorted_indices = np.argsort(flattened_distances) # sort (ascendingly)
    pairs = [(idx // dim2, idx % dim2) for idx in sorted_indices] # extract original (matrix) indices 
    min_pairs = pairs[:N_PAIRS]
    max_pairs = pairs[-N_PAIRS:]
    middle_start = (len(pairs) // 2) - (N_PAIRS // 2)
    middle_pairs = pairs[middle_start:middle_start + N_PAIRS]
    
    return min_pairs, max_pairs, middle_pairs


def visualize_pairs(distances, flattened_distances, min_pairs, max_pairs, middle_pairs, output_dir=None):
    # Distances of selected pairs
    selected_min_distances = [distances[i, j] for i, j in min_pairs]
    selected_max_distances = [distances[i, j] for i, j in max_pairs]
    selected_middle_distances = [distances[i, j] for i, j in middle_pairs]

    print("selected_max_distances: ", selected_max_distances)
    print("selected_middle_distances: ", selected_middle_distances)
    print("selected_min_distances: ", selected_min_distances)


    plt.figure(figsize=(10, 6))
    plt.hist(flattened_distances, bins=50, alpha=0.7, color='blue', label='All distances')

    # Plot vertical lines
    for d in selected_min_distances:
        plt.axvline(d, color='green', linestyle='--', linewidth=1)

    for d in selected_max_distances:
        plt.axvline(d, color='red', linestyle='--', linewidth=1)

    for d in selected_middle_distances:
        plt.axvline(d, color='yellow', linestyle='--', linewidth=1)

    plt.xlabel("Cosine Distance")
    plt.ylabel("Count")
    plt.title("Distribution of All Pairwise Distances")
    plt.legend()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "cosine_dist.png"))
        plt.close()
    else:
        plt.show()



def main(NUM_PAIRS, data_set_types):

    for i, set_type in enumerate(data_set_types):
        temp_output_dir = os.path.join(output_dir, set_type)
        os.makedirs(temp_output_dir, exist_ok=True)

        #load data
        input_dir  = os.path.join(groups_dir, set_type)
        wt_untreated, wt_untreated_labels, wt_untreated_paths = load_files(input_dir, "WT", "Untreated")
        wt_stress, wt_stress_labels, wt_stress_paths = load_files(input_dir, "WT", "stress")

        # Compute all pairwise distances between untreated and stress
        num_untreated, num_stress, distances = compute_distances(wt_untreated, wt_stress,metric='cosine')
        flattened_distances = distances.flatten()

        # get min/max/middle dist pairs
        min_pairs, max_pairs, middle_pairs = get_pairs(flattened_distances, NUM_PAIRS, dim2 = num_stress)

        # get indices for treated/stress samples 
        all_pairs = min_pairs + middle_pairs + max_pairs
        untreated_indices = list(set([i for (i, j) in all_pairs]))
        stress_indices = list(set([j for (i, j) in all_pairs]))
        print(f"Selected {len(untreated_indices)} untreated samples and {len(stress_indices)} stress samples.")

        # save distances
        distances_df = pd.DataFrame(all_pairs, columns=["i_untreated", "j_stress"])
        distances_df["cosine_distance"] = [distances[i, j] for (i, j) in all_pairs]
        distances_df["path_untreated"] = [wt_untreated_paths.iloc[i]["Path"] for (i, j) in all_pairs] 
        distances_df["path_stress"] = [wt_stress_paths.iloc[j]["Path"] for (i, j) in all_pairs] 
        distances_df.to_csv(os.path.join(temp_output_dir, f"{set_type}_distances.csv"), index=False)

        # extract embeding,labels and paths values
        filtered_untreated_embeddings = wt_untreated[untreated_indices]
        filtered_stress_embeddings = wt_stress[stress_indices]
        filtered_untreated_labels = wt_untreated_labels.iloc[untreated_indices]
        filtered_stress_labels = wt_stress_labels.iloc[stress_indices]
        filtered_untreated_paths = wt_untreated_paths.iloc[untreated_indices]
        filtered_stress_paths = wt_stress_paths.iloc[stress_indices]

        # save all together as nyp 
        # Concatenate embeddings and labels correspondly 
        set_type_embeddings = np.concatenate([filtered_untreated_embeddings, filtered_stress_embeddings], axis=0)
        set_type_labels = pd.concat([filtered_untreated_labels, filtered_stress_labels], axis=0).reset_index(drop=True)
        set_type_paths= pd.concat([filtered_untreated_paths, filtered_stress_paths], axis=0).reset_index(drop=True)

        # Save npy files
        np.save(os.path.join(temp_output_dir, f"{set_type}.npy"), set_type_embeddings)
        np.save(os.path.join(temp_output_dir, f"{set_type}_labels.npy"), set_type_labels["full_label"].values)
        np.save(os.path.join(temp_output_dir, f"{set_type}_paths.npy"), np.array(set_type_paths["Path"].values, dtype=str))

        visualize_pairs(distances, flattened_distances, min_pairs, max_pairs, middle_pairs, output_dir = temp_output_dir)

if __name__ == "__main__":
    main(NUM_PAIRS, data_set_types)
        