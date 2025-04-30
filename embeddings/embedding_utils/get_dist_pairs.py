import numpy as np
from scipy.spatial.distance import cdist
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

N_PAIRS = 25

# Load embeddings  labels and paths
embd_dir  = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_and_paths/RotationDatasetConfig"
wt_untreated = np.load(os.path.join(embd_dir, "grouped_embeddings", "wt_untreated_embedding.npy"))
wt_untreated_labels = pd.read_csv(os.path.join(embd_dir, "grouped_embeddings", "wt_untreated_labels.csv"))
wt_untreated_paths = pd.read_csv(os.path.join(embd_dir, "grouped_embeddings", "wt_untreated_paths.csv"))
wt_stress = np.load(os.path.join(embd_dir, "grouped_embeddings", "wt_stress_embedding.npy"))
wt_stress_labels = pd.read_csv(os.path.join(embd_dir, "grouped_embeddings", "wt_stress_labels.csv"))
wt_stress_paths = pd.read_csv(os.path.join(embd_dir, "grouped_embeddings", "wt_stress_paths.csv"))

output_dir = os.path.join(embd_dir, "pairs")
os.makedirs(output_dir, exist_ok=True)

# Compute all pairwise distances between untreated and stress
distances = cdist(wt_untreated, wt_stress, metric='cosine')  # shape: (n_untreated [1475], n_stress [1343])
num_untreated, num_stress = distances.shape
flattened_distances = distances.flatten() # flatten to shape: (n_untreated X n_stress)

# get min/max/middle dist pairs
sorted_indices = np.argsort(flattened_distances) # sort (ascendingly)
pairs = [(idx // num_stress, idx % num_stress) for idx in sorted_indices] # extract original (matrix) indices 
min_pairs = pairs[:N_PAIRS]
max_pairs = pairs[-N_PAIRS:]
middle_start = (len(pairs) // 2) - (N_PAIRS // 2)
middle_pairs = pairs[middle_start:middle_start + N_PAIRS]

# get indices for treated/stress samples 
all_pairs = min_pairs + middle_pairs + max_pairs
untreated_indices = list(set([i for (i, j) in all_pairs]))
stress_indices = list(set([j for (i, j) in all_pairs]))
print(f"Selected {len(untreated_indices)} untreated samples and {len(stress_indices)} stress samples.")


# save distances
distances_df = pd.DataFrame(all_pairs, columns=["i_untreated", "j_stress"])
distances_df["cosine_distance"] = [distances[i, j] for (i, j) in all_pairs]
distances_df["path_untreated"] = [wt_untreated_paths.iloc[i]["full_path"] for (i, j) in all_pairs] 
distances_df["path_stress"] = [wt_stress_paths.iloc[j]["full_path"] for (i, j) in all_pairs] 
distances_df.to_csv(os.path.join(output_dir, "testsets_distances.csv"), index=False)

# extract embeding,labels and paths values
filtered_untreated_embeddings = wt_untreated[untreated_indices]
filtered_stress_embeddings = wt_stress[stress_indices]
filtered_untreated_labels = wt_untreated_labels.iloc[untreated_indices]
filtered_stress_labels = wt_stress_labels.iloc[stress_indices]
filtered_untreated_paths = wt_untreated_labels.iloc[untreated_indices]
filtered_stress_paths = wt_stress_labels.iloc[stress_indices]

# save all together as nyp 
# Concatenate embeddings and labels correspondly 
testsets_embeddings = np.concatenate([filtered_untreated_embeddings, filtered_stress_embeddings], axis=0)
testsets_labels = pd.concat([filtered_untreated_labels, filtered_stress_labels], axis=0).reset_index(drop=True)

# Save npy files
np.save(os.path.join(output_dir, "testsets.npy"), testsets_embeddings)
np.save(os.path.join(output_dir, "testsets_labels.npy"), testsets_labels["full_label"].values)


# visulaize:
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

plt.xlabel("Cosine Distance")
plt.ylabel("Count")
plt.title("Distribution of All Pairwise Distances")
plt.legend()
plt.savefig(os.path.join(output_dir, "cosine_dist.png"))
plt.close()


print("selected_max_distances: ", selected_max_distances)
print("selected_middle_distances: ", selected_middle_distances)
print("selected_min_distances: ", selected_min_distances)
