import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os


embd_dir  = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_and_paths/RotationDatasetConfig/embeddings/neurons/batch9"
paths_path = os.path.join(embd_dir, "testset_paths.npy")
paths = np.load(paths_path)
paths_df = pd.DataFrame(paths, columns=['full_path'])
paths_df["file_name"] = paths_df['full_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
print("\n paths_df:", paths_df.head())
paths_df[['replicate', 'meaningless', 'antibody', 'site', 'panel', 'cell_line', 'suffix']] = paths_df['file_name'].str.split('_', expand=True)
grouped = paths_df.groupby(['replicate', 'site'])
print("\n paths_df:", paths_df)
print("\npaths groups:")
print(grouped.size())
