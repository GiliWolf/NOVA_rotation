
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np

img_dir  = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/batch9/WT/stress/G3BP1/"
img_sample = "rep2_R11_w3confCy5_s200_panelA_WT_processed"
img_path = os.path.join(img_dir, f"{img_sample}.npy")
img = np.load(img_path)
print(img.shape) # (3, 100, 100, 2) (?, H, W, C)