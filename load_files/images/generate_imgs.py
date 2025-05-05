import sys
import os
working_dir = os.getcwd()
sys.path.append(working_dir)
print(f"working_dir: {working_dir}")
from NOVA_rotation.load_files.load_data_from_npy import load_paths_from_npy, load_labels_from_npy , load_npy_to_df, load_npy_to_nparray, load_tile, Parse_Path_Item
from NOVA_rotation.attention_maps.attention_maps_utils.generate_attention_maps import __process_attn_map

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

attn_maps_dir = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/attention_maps/attention_maps_output/RotationDatasetConfig_Pairs/raw/attn_maps/neurons/batch9"
save_dir = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/attention_maps/attention_maps_output/RotationDatasetConfig_Pairs/attn_layers"

def init_globals(attn_maps):
    global num_samples, num_layers, num_heads, num_patches, img_shape, patch_dim
    num_samples, num_layers, num_heads, num_patches, _ = attn_maps.shape
    img_shape = (100, 100)
    patch_dim = int(np.sqrt(num_patches))

def create_fig(path, sample_attn, save_dir = None):
    img_path, tile, site = Parse_Path_Item(path)
    marker, nucleus, overlay = load_tile(img_path, tile)

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])  # Top (overlay), Bottom (attention maps)

    # display input overlay at the top
    ax_overlay = plt.subplot(gs[0])
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Overlay Image")
    ax_overlay.axis("off")

    # Grid for 12 attention maps (3 rows x 4 columns)
    gs_attn = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[1], wspace=0.2, hspace=0.3)

    for layer_idx in range(num_layers):
        # Get attention for this layer and average over heads
        attn = sample_attn[layer_idx]  # (num_heads, num_patches+1, num_patches+1)
        attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape)

        ax = plt.subplot(gs_attn[layer_idx])
        ax.imshow(heatmap_colored)
        ax.set_title(f"Layer {layer_idx+1}")
        ax.axis("off")
    
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "all_attn_layers.png"), dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    """
    1 - CHANGE SAVING PATH
    2 - ADD INFORMATIVE TITLE
    3 - ADD SELECTION OF IMG BASED ON ITS PATH
    """
    attn_maps = load_npy_to_nparray(attn_maps_dir, "testset_attn.npy") 
    labels = load_labels_from_npy(attn_maps_dir)
    paths = load_paths_from_npy(attn_maps_dir)

    init_globals(attn_maps)

    os.makedirs(save_dir, exist_ok=True)

    for index, path in enumerate(paths.itertuples()):
        sample_attn = attn_maps[index]
        create_fig(path, sample_attn, save_dir)
        break

if __name__ == "__main__":
    main()

    
    
   