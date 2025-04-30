import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

embd_dir  = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/RotationDatasetConfig"
label_path = os.path.join(embd_dir, "testset_labels.npy")
labels = np.load(label_path)

embed_path = os.path.join(embd_dir, "testset.npy")
embeddings = np.load(embed_path)
embeddings_df = pd.DataFrame(embeddings)
print("embeddings_df: ")
print(embeddings_df.head())

# Convert to pandas DataFrame
labels_df = pd.DataFrame(labels, columns=['full_label'])

# Split the 'full_label' into parts (assuming '_' is the separator)
# example label - G3BP1_FUSHomozygous_Untreated_batch9_rep1
labels_df[['protein', 'condition', 'treatment', 'batch', 'replicate']] = labels_df['full_label'].str.split('_', expand=True)

#  print group sizes
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
print(wt_untreated_labels.head())

print("\nwt_untreated_embeddings:")
wt_untreated_embeddings = embeddings_df.iloc[wt_untreated_labels.index]
print(wt_untreated_embeddings.shape)
print(wt_untreated_embeddings.head())

wt_untreated_labels.reset_index().to_csv(os.path.join(embd_dir, "wt_untreated_labels.csv"), index=False)
np.save(os.path.join(embd_dir, "wt_untreated_embedding.npy"),  wt_untreated_embeddings.to_numpy())


# WT stress 
wt_stress_labels= labels_df[
    (labels_df['condition'] == 'WT') &
    (labels_df['treatment'] == 'stress')
]
print("\nwt_stress_labels:")
print(wt_stress_labels.shape)
print(wt_stress_labels.head())

print("\nwt_stress_embeddings:")
wt_stress_embeddings = embeddings_df.iloc[wt_stress_labels.index]
print(wt_stress_embeddings.shape)
print(wt_stress_embeddings.head())

wt_stress_labels.reset_index().to_csv(os.path.join(embd_dir, "wt_stress_labels.csv"), index=False)
np.save(os.path.join(embd_dir, "wt_stress_embedding.npy"),  wt_stress_embeddings.to_numpy())

