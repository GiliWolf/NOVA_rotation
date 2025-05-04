import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

embd_dir  = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/RotationDatasetConfig/embeddings/neurons/batch9"
output_dir = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/RotationDatasetConfig/grouped_embeddings"
os.makedirs(output_dir, exist_ok=True)
label_path = os.path.join(embd_dir, "testset_labels.npy")
labels = np.load(label_path)

paths_path = os.path.join(embd_dir, "testset_paths.npy")
paths = np.load(paths_path)
paths_df = pd.DataFrame(paths, columns=['Path'])

embed_path = os.path.join(embd_dir, "testset.npy")
embeddings = np.load(embed_path)
embeddings_df = pd.DataFrame(embeddings)


# Convert to pandas DataFrame
labels_df = pd.DataFrame(labels, columns=['full_label'])

#split to groups
labels_df[['protein', 'condition', 'treatment', 'batch', 'replicate']] = labels_df['full_label'].str.split('_', expand=True)
grouped = labels_df.groupby(['protein', 'condition', 'treatment'])
print("\nlabels groups:")
print(grouped.size())


# WT UNTREATED 

wt_untreated_labels= labels_df[
    (labels_df['condition'] == 'WT') &
    (labels_df['treatment'] == 'Untreated')
]
print("\nwt_untreated_labels:")
print(wt_untreated_labels.shape)

print("\nwt_untreated_embeddings:")
wt_untreated_embeddings = embeddings_df.iloc[wt_untreated_labels.index]
print(wt_untreated_embeddings.shape)

print("\nwt_untreated_paths:")
wt_untreated_paths = paths_df.iloc[wt_untreated_labels.index]
print(wt_untreated_paths.shape)

#save
wt_untreated_labels["full_label"].to_csv(os.path.join(output_dir, "wt_untreated_labels.csv"), index=False)
np.save(os.path.join(output_dir, "wt_untreated_embedding.npy"),  wt_untreated_embeddings.to_numpy())
wt_untreated_paths.to_csv(os.path.join(output_dir, "wt_untreated_paths.csv"), index=False)


# WT stress 
wt_stress_labels= labels_df[
    (labels_df['condition'] == 'WT') &
    (labels_df['treatment'] == 'stress')
]
print("\nwt_stress_labels:")
print(wt_stress_labels.shape)

print("\nwt_stress_embeddings:")
wt_stress_embeddings = embeddings_df.iloc[wt_stress_labels.index]
print(wt_stress_embeddings.shape)

print("\nwt_stress_paths:")
wt_stress_paths = paths_df.iloc[wt_stress_labels.index]
print(wt_stress_paths.shape)

#save
wt_stress_labels["full_label"].to_csv(os.path.join(output_dir, "wt_stress_labels.csv"), index=False)
np.save(os.path.join(output_dir, "wt_stress_embedding.npy"),  wt_stress_embeddings.to_numpy())
wt_stress_paths.to_csv(os.path.join(output_dir, "wt_stress_paths.csv"), index=False)

