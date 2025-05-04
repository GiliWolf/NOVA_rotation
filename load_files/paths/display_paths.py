import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os


embd_dir  = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_and_paths/RotationDatasetConfig/embeddings/neurons/batch9"
embd_dir = "/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model/embeddings/neurons_iu/batch9"
paths_path = os.path.join(embd_dir, "testset_paths.npy")
paths = np.load(paths_path)
paths_df = pd.DataFrame(paths, columns=['index', 'full_path'])
# paths_df = pd.DataFrame(paths, columns=['full_path'])
# paths_df["file_name"] = paths_df['full_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
paths_df.head().to_csv("paths.csv")
sys.exit()
print("\n paths_df:", paths_df.iloc[0])
paths_df[['replicate', 'meaningless', 'antibody', 'site', 'panel', 'cell_line', 'suffix']] = paths_df['file_name'].str.split('_', expand=True)
grouped = paths_df.groupby(['replicate', 'site'])
print("\n paths_df:", paths_df)
print("\npaths groups:")
print(grouped.size())
