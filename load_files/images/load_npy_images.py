import numpy as np
import matplotlib.pyplot as plt

img_path = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/batch9/WT/Untreated/G3BP1/rep2_R11_w3confCy5_s391_panelA_WT_processed.npy"
img = np.load(img_path)

tile_ch = img[4, :, :, 1]
quant = 0.95
threshold = np.quantile(tile_ch,quant)
mask = tile_ch >= threshold

masked_tile = np.zeros_like(tile_ch)
masked_tile[mask] = tile_ch[mask]

# Create the figure
plt.imshow(masked_tile, cmap='gray')
plt.axis('off')

# Save the figure (choose your desired path and format)
plt.savefig(f"masked_tile_{quant}.png", bbox_inches='tight', pad_inches=0, dpi=300)

# Optional: close the figure to free memory
plt.close()