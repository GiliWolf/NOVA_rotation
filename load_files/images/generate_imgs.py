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
import matplotlib

attn_maps_dir = "./NOVA_rotation/attention_maps/attention_maps_output/RotationDatasetConfig_Pairs/raw/attn_maps/neurons/batch9"
save_dir = "./NOVA_rotation/attention_maps/attention_maps_output/RotationDatasetConfig_Pairs/layers"

def init_globals(attn_maps):
    global num_samples, num_layers, num_heads, num_patches, img_shape, patch_dim
    num_samples, num_layers, num_heads, num_patches, _ = attn_maps.shape
    img_shape = (100, 100)
    patch_dim = int(np.sqrt(num_patches))


def create_fig(path, sample_attn, label, save_dir=None):
    img_path, tile, site = Parse_Path_Item(path)
    file_name = path.File_Name
    marker, nucleus, overlay = load_tile(img_path, tile)

    fig = plt.figure(figsize=(13, 11), facecolor="#d3ebe3")
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.4)

    # Main title
    fig.suptitle(f"Attention Layers\n{file_name}\n{label}\n\n", fontsize=18, fontweight='bold', y=0.98)

    # Overlay section (light gray background)
    ax_overlay = plt.subplot(gs[0])
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Overlay Image", fontsize=14, pad=10)
    ax_overlay.axis("off")

    # Attention maps section (soft pastel background)
    gs_attn = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[1], wspace=0.3, hspace=0.4)

    for layer_idx in range(num_layers):
        attn = sample_attn[layer_idx]
        attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape)

        ax = plt.subplot(gs_attn[layer_idx])  
        ax.imshow(heatmap_colored)
        ax.set_title(f"Layer {layer_idx+1}", fontsize=10)
        ax.axis("off")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{file_name}.png"), dpi=300, bbox_inches="tight",  facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def extract_path(paths, path_to_plot):
    match = paths[paths["Path"] == path_to_plot]

    if not match.empty:
        idx = match.index[0]
        item = match.iloc[0]
        return idx, item
    else:
        print("No match found.")
        return None


def main():
    """
    1 - CHANGE SAVING PATH - V
    2 - ADD INFORMATIVE TITLE - V
    3 - ADD SELECTION OF IMG BASED ON ITS PATH
    """
    img_input_dir = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk"
    path_name_to_plot = "batch9/WT/Untreated/G3BP1/rep1_R11_w3confCy5_s208_panelA_WT_processed.npy/4"
    path_to_plot = os.path.join(img_input_dir, path_name_to_plot)

    attn_maps = load_npy_to_nparray(attn_maps_dir, "testset_attn.npy") 
    labels = load_labels_from_npy(attn_maps_dir)
    paths = load_paths_from_npy(attn_maps_dir)

    init_globals(attn_maps)

    os.makedirs(save_dir, exist_ok=True)

    index, path = extract_path(paths, path_to_plot)

    sample_attn = attn_maps[index]
    label = labels.iloc[index].full_label
    create_fig(path, sample_attn, label, save_dir)


if __name__ == "__main__":
    main()

    
    
   