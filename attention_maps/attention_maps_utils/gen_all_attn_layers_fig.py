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
from scipy.stats import pearsonr



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

    # List to store correlations for this sample
    corr_nucleus_list = []
    corr_marker_list = []

    for layer_idx in range(num_layers):
        # get attn map
        attn = sample_attn[layer_idx]
        attn = attn.mean(axis=0) #avg across heads
        attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape)

        # Flatten and compute correlations
        flat_attn = attn_map.flatten()
        corr_nucleus = pearsonr(flat_attn, nucleus.flatten())[0]
        corr_marker = pearsonr(flat_attn, marker.flatten())[0]
        corr_nucleus_list.append(corr_nucleus)
        corr_marker_list.append(corr_marker)

        #plot layer attn mape   
        ax = plt.subplot(gs_attn[layer_idx])  
        ax.imshow(heatmap_colored)
        ax.set_title(f"Layer {layer_idx+1}", fontsize=10)
        ax.axis("off")

        # Add correlation values below the attention map
        ax.text(0.5, -0.1, f"Corr(Nucleus): {corr_nucleus:.2f}\nCorr(Marker): {corr_marker:.2f}", 
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='black')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{file_name}.png"), dpi=300, bbox_inches="tight",  facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return corr_nucleus_list, corr_marker_list


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
    input_dir = "./NOVA_rotation/attention_maps/attention_maps_output"
    run_name = "RotationDatasetConfig_Pairs"
    attn_maps_dir = os.path.join(input_dir, run_name, "raw/attn_maps/neurons/batch9")
    save_dir =  os.path.join(input_dir, run_name,"layers_corr")

    img_input_dir = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk"
    path_name_to_plot = "batch9/WT/Untreated/G3BP1/rep1_R11_w3confCy5_s208_panelA_WT_processed.npy/4"
    path_to_plot = os.path.join(img_input_dir, path_name_to_plot)

    attn_maps = load_npy_to_nparray(attn_maps_dir, "testset_attn.npy") 
    labels = load_labels_from_npy(attn_maps_dir, "testset")
    paths = load_paths_from_npy(attn_maps_dir, "testset")

    init_globals(attn_maps)

    os.makedirs(save_dir, exist_ok=True)

    index, path = extract_path(paths, path_to_plot)

    sample_attn = attn_maps[index]
    label = labels.iloc[index].full_label
    create_fig(path, sample_attn, label, save_dir)


if __name__ == "__main__":
    main()

    
    
   