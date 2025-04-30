import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os


embd_dir  = "/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model/embeddings/neurons/batch9"
embeddings_path = os.path.join(embd_dir, "testset.npy")
embeddings = np.load(embeddings_path)
embeddings_df = pd.DataFrame(embeddings)