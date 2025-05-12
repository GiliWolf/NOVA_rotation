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
from scipy.stats import pearsonr, entropy



def init_globals(attn_maps):
    global num_samples, num_layers, num_heads, num_patches, img_shape, patch_dim
    num_samples, num_layers, num_heads, num_patches, _ = attn_maps.shape
    img_shape = (100, 100)
    patch_dim = int(np.sqrt(num_patches))


def compute_parameters(img_path, tile, sample_attn):
    marker, nucleus, overlay = load_tile(img_path, tile)

    # List to store results
    corr_nucleus_list = []
    corr_marker_list = []
    entropy_list = []

    for layer_idx in range(num_layers):
        # get attn map
        attn = sample_attn[layer_idx]
        attn = attn.mean(axis=0) # avg across heads
        attn_map, _ = __process_attn_map(attn, patch_dim, img_shape)

        # Flatten
        flat_attn = attn_map.flatten()

        # compute correlations - 
        corr_nucleus = pearsonr(flat_attn, nucleus.flatten())[0]
        corr_marker = pearsonr(flat_attn, marker.flatten())[0]
        corr_nucleus_list.append(corr_nucleus)
        corr_marker_list.append(corr_marker)

        # compute entropy -
        attn_probs = flat_attn + 1e-8  # avoid log(0)
        attn_probs /= attn_probs.sum()  # normalize to sum to 1
        layer_ent = entropy(attn_probs, base=2)  # base-2 entropy (bits)
        entropy_list.append(layer_ent)

    return corr_nucleus_list, corr_marker_list, entropy_list

def plot_fig(img_path, tile, site, file_name, sample_attn, label, corr_nucleus_list, corr_marker_list, entropy_list, save_dir=None):
    marker, nucleus, overlay = load_tile(img_path, tile)

    fig = plt.figure(figsize=(13, 11), facecolor="#d3ebe3")
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.2)

    # Main title
    fig.suptitle(f"{file_name}\n{label}\n\n", fontsize=18, fontweight='bold', y=0.98)

    # Overlay section 
    ax_overlay = plt.subplot(gs[0])
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Input Image", fontsize=14, fontweight='bold', pad=10)
    ax_overlay.axis("off")

    # Attention maps section 
    gs_attn = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[1], wspace=0.3, hspace=0.8)
    fig.text(0.5, 0.68, "Attention Maps", ha='center', va='center', fontsize=16, fontweight='bold')


    for layer_idx, (corr_nucleus, corr_marker, layer_ent) in enumerate(zip(corr_nucleus_list, corr_marker_list, entropy_list)):
        # get attn map
        attn = sample_attn[layer_idx]
        attn = attn.mean(axis=0) #avg across heads
        _, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape)


        # plot layer attn maps
        ax = plt.subplot(gs_attn[layer_idx])  
        ax.imshow(heatmap_colored)
        ax.set_title(f"Layer {layer_idx}", fontsize=14, fontweight='bold')
        ax.axis("off")

        # Add correlation values below the attention map
        ax.text(0.5, -0.25, f"Corr(Nucleus): {corr_nucleus:.2f}\nCorr(Marker): {corr_marker:.2f}\nEntropy: {layer_ent:.2f}", 
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='black')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{file_name}.png"), dpi=300, bbox_inches="tight",  facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def get_percentiles(data, prc_list = [50, 25, 75], axis=0):
    perc_tuple = ()
    for prc in prc_list:
        res = np.percentile(data, prc, axis=axis)
        perc_tuple += (res,)
    return perc_tuple


def plot_correlation(corr_nucleus_all, corr_marker_all, entropy_all, num_layers, save_dir=None):
    """
    Plot correlation for each layer across multiple samples.
    """
    # Convert to numpy arrays for easier manipulation
    corr_nucleus_all = np.array(corr_nucleus_all)
    corr_marker_all = np.array(corr_marker_all)
    entropy_all = np.array(entropy_all)

    # Calculate median and percentiles (25th and 75th)
    median_nucleus, p25_nucleus, p75_nucleus = get_percentiles(corr_nucleus_all)
    median_marker, p25_marker, p75_marker = get_percentiles(corr_marker_all)
    median_entropy, p25_entropy, p75_entropy = get_percentiles(entropy_all)

    layers_range = range(num_layers)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Correlation plots
    ax1.plot(layers_range, median_nucleus, label="Nucleus (Median)", color='green', marker='o')
    ax1.fill_between(layers_range, p25_nucleus, p75_nucleus, color='green', alpha=0.3)

    ax1.plot(layers_range, median_marker, label="Marker (Median)", color='red', marker='o')
    ax1.fill_between(layers_range, p25_marker, p75_marker, color='red', alpha=0.3)

    ax1.set_xlabel("Layer Number", fontsize=12)
    ax1.set_ylabel("Correlation", fontsize=12)
    ax1.set_xticks(layers_range)
    ax1.legend(loc="upper left")

    # Entropy on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(layers_range, median_entropy, label="Entropy (Median)", color='gray', linestyle='--', marker='x', alpha=0.4)
    ax2.fill_between(layers_range, p25_entropy, p75_entropy, color='gray', alpha=0.1)
    ax2.set_ylabel("Entropy", fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    fig.suptitle("Correlation and Entropy Across Layers", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"correlation_entropy_plot.png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
    else:
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


def run_one_sample(paths, path_to_plot, attn_maps,labels, save_dir=None):
    # get specific sample by path name 
    index, path = extract_path(paths, path_to_plot)
    sample_attn = attn_maps[index]
    label = labels.iloc[index].full_label
    img_path, tile, site = Parse_Path_Item(path)
    file_name = path.File_Name
    corr_nucleus_list, corr_marker_list, entropy_list = compute_parameters(img_path, tile, sample_attn)
    plot_fig(img_path, tile, site, file_name, sample_attn, label, corr_nucleus_list, corr_marker_list, entropy_list, save_dir=save_dir)



def run_all_samples(paths, attn_maps,labels, save_dir=None):
    corr_nucleus_all = []
    corr_marker_all = []
    entropy_all = []

    for index, (sample_attn, label) in enumerate(zip(attn_maps, labels["full_label"])):
        path = paths.iloc[index]
        img_path, tile, site = Parse_Path_Item(path)
        file_name = path.File_Name
        corr_nucleus_list, corr_marker_list, entropy_list = compute_parameters(img_path, tile, sample_attn)
        corr_nucleus_all.append(corr_nucleus_list)
        corr_marker_all.append(corr_marker_list)
        entropy_all.append(entropy_list)

    # plot correlations across samples
    plot_correlation(corr_nucleus_all, corr_marker_all, entropy_all, num_layers, save_dir)

def main(run_all=False):
    input_dir = "./NOVA_rotation/attention_maps/attention_maps_output"
    run_name = "RotationDatasetConfig_Pairs"
    attn_maps_dir = os.path.join(input_dir, run_name, "raw/attn_maps/neurons/batch9")
    save_dir =  os.path.join(input_dir, run_name,"layers_corr")

    img_input_dir = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk"
    path_name_to_plot = "batch9/WT/stress/G3BP1/rep2_R11_w3confCy5_s180_panelA_WT_processed.npy/2"
    path_to_plot = os.path.join(img_input_dir, path_name_to_plot)

    attn_maps = load_npy_to_nparray(attn_maps_dir, "testset_attn.npy") 
    labels = load_labels_from_npy(attn_maps_dir, "testset")
    paths = load_paths_from_npy(attn_maps_dir, "testset")

    init_globals(attn_maps)

    os.makedirs(save_dir, exist_ok=True)

    if run_all:
        run_all_samples(paths, attn_maps,labels, save_dir=save_dir)
    else:
        run_one_sample(paths, path_to_plot, attn_maps,labels, save_dir=save_dir)



if __name__ == "__main__":
    main(run_all=True)
    print("Done.")

    
    
   