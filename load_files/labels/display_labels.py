import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os


embd_dir  = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/try_with_new_config/embeddings/neurons/batch9"
labels_path = os.path.join(embd_dir, "testset_labels.npy")
labels = np.load(labels_path)
labels_df = pd.DataFrame(labels, columns=['full_label'])
labels_df[['protein', 'condition', 'treatment', 'batch', 'replicate']] = labels_df['full_label'].str.split('_', expand=True)
grouped = labels_df.groupby(['protein', 'condition', 'treatment'])
print("\nlabels groups:")
print(grouped.size())
