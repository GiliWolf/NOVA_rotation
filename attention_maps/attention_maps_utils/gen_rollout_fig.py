import sys
import os
working_dir = os.getcwd()
sys.path.append(working_dir)
print(f"working_dir: {working_dir}")
from NOVA_rotation.load_files.load_data_from_npy import load_paths_from_npy, load_labels_from_npy , load_npy_to_df, load_npy_to_nparray, load_tile, Parse_Path_Item
from NOVA_rotation.attention_maps.attention_maps_utils.generate_attention_maps import __process_attn_map, __attn_map_rollout

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from scipy.stats import pearsonr, entropy
import cv2
from matplotlib.colors import LinearSegmentedColormap


def init_globals(attn_maps):
    global num_samples, num_layers, num_heads, num_patches, img_shape, patch_dim
    num_samples,num_layers, num_heads, num_patches, _ = attn_maps.shape
    img_shape = (100, 100)
    patch_dim = int(np.sqrt(num_patches))


def compute_parameters(img_path, tile, sample_attn, min_attn_threshold = None, heads_reduce_fn=np.mean):
    marker, nucleus, overlay = load_tile(img_path, tile)

    #attn = heads_reduce_fn(attn, axis=0)
    attn = __attn_map_rollout(sample_attn, attn_layer_dim=0,heads_reduce_fn = heads_reduce_fn)
    attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape, min_attn_threshold= min_attn_threshold)

    # Flatten
    flat_attn = attn_map.flatten()

    # compute correlations - 
    corr_nucleus = pearsonr(flat_attn, nucleus.flatten())[0]
    corr_marker = pearsonr(flat_attn, marker.flatten())[0]

    # compute entropy -
    attn_probs = flat_attn + 1e-8  # avoid log(0)
    attn_probs /= attn_probs.sum()  # normalize to sum to 1
    layer_ent = entropy(attn_probs, base=2)  # base-2 entropy (bits)
    normalized_ent = layer_ent / np.log2(len(attn_probs))  # normalize to [0, 1]

    return attn_map, heatmap_colored, corr_nucleus, corr_marker, normalized_ent

def plot_fig(img_path, tile, site, file_name, label, attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy, save_dir=None):

        _, _, input_img = load_tile(img_path, tile)
        attn_map = np.clip(attn_map, 0, 1)   
        alpha = 0.4

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{file_name}\n{label}\n\n", fontsize=16, fontweight='bold')

        ax[1].text(0.5, -0.25, f"Corr(Nucleus): {corr_nucleus:.2f}\nCorr(Marker): {corr_marker:.2f}\nEntropy: {entropy:.2f}", 
                transform=ax[1].transAxes, ha='center', va='center', fontsize=12, color='black')

        ax[0].set_title(f'Input - Marker (blue), Nucleus (green)', fontsize=12)
        ax[0].imshow(input_img)
        ax[0].set_axis_off()

        ax[1].set_title(f'Attention Heatmap', fontsize=12)
        ax[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        ax[1].set_axis_off()


        custom_cmap = LinearSegmentedColormap.from_list(
            'black_yellow_red', 
            ['black', 'yellow', 'red']
        )

        ax[2].set_title('Attention Overlay', fontsize=12)
        ax[2].imshow(input_img)  # Show the original image
        ax[2].imshow(attn_map, cmap=custom_cmap, alpha=alpha)  # Overlay attention map transparently
        ax[2].set_axis_off()



        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.subplots_adjust(top=0.80)
            plt.savefig(os.path.join(save_dir, f"{file_name}_rollout.png"), dpi=300, bbox_inches="tight",  facecolor=fig.get_facecolor())
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


def plot_correlation(corr_nucleus_all, corr_marker_all, entropy_all, save_dir=None):
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

    layers_range = range(1)

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

def plot_correlation_boxplot(corr_nucleus_all, corr_marker_all, entropy_all, save_dir=None):


    # Convert to numpy arrays
    corr_nucleus_all = np.array(corr_nucleus_all)
    corr_marker_all = np.array(corr_marker_all)
    entropy_all = np.array(entropy_all)

    # Boxplot positions
    positions_corr = [1, 2]
    positions_entropy = [1.5]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Boxplots for correlation (left y-axis)
    bp_nuc = ax1.boxplot(corr_nucleus_all, positions=[positions_corr[0]], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor='green', color='black'),
                         medianprops=dict(color='black'), showfliers=False)

    bp_marker = ax1.boxplot(corr_marker_all, positions=[positions_corr[1]], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor='red', color='black'),
                            medianprops=dict(color='black'), showfliers=False)

    ax1.set_ylabel("Correlation", fontsize=12)
    ax1.set_ylim(-1, 1)

    # Boxplot for entropy (right y-axis)
    ax2 = ax1.twinx()
    bp_entropy = ax2.boxplot(entropy_all, positions=positions_entropy, widths=0.3, patch_artist=True,
                             boxprops=dict(facecolor='gray', color='black', alpha=0.5),
                             medianprops=dict(color='black'), showfliers=False)

    ax2.set_ylabel("Entropy", fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 1)

    # Title and spacing
    fig.suptitle("Rollout: Correlation and Entropy", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"rollout_correlation_entropy_boxplot.png"),
                    dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
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


def run_one_sample(paths, path_to_plot, attn_maps,labels, min_attn_threshold = None, heads_reduce_fn = None, save_dir=None):
    # get specific sample by path name 
    index, path = extract_path(paths, path_to_plot)
    sample_attn = attn_maps[index]
    label = labels.iloc[index].full_label
    img_path, tile, site = Parse_Path_Item(path)
    file_name = path.File_Name
    attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy = compute_parameters(img_path, tile, sample_attn, min_attn_threshold, heads_reduce_fn)
    plot_fig(img_path, tile, site, file_name, label, attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy, save_dir)



def run_all_samples(paths, attn_maps,labels, min_attn_threshold = None, heads_reduce_fn = None, save_dir=None):
    corr_nucleus_all = []
    corr_marker_all = []
    entropy_all = []

    for index, (sample_attn, label) in enumerate(zip(attn_maps, labels["full_label"])):
        path = paths.iloc[index]
        img_path, tile, site = Parse_Path_Item(path)
        file_name = path.File_Name
        attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy = compute_parameters(img_path, tile, sample_attn, min_attn_threshold, heads_reduce_fn)
        corr_nucleus_all.append(corr_nucleus)
        corr_marker_all.append(corr_marker)
        entropy_all.append(entropy)

    # plot correlations across samples
    plot_correlation_boxplot(corr_nucleus_all, corr_marker_all, entropy_all,  save_dir)

def main(run_all=False, min_attn_threshold=None, heads_reduce_fn = np.mean):

    input_dir = "./NOVA_rotation/attention_maps/attention_maps_output"
    run_name = "RotationDatasetConfig_Euc_Pairs_rollout"
    attn_maps_dir = os.path.join(input_dir, run_name, "raw/attn_maps/neurons/batch9")
    save_dir =  os.path.join(input_dir, run_name,"layers_corr", "np_mean_threshold0.5")

    img_input_dir = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk"
    path_name_to_plot = "batch9/WT/Untreated/G3BP1/rep1_R11_w3confCy5_s225_panelA_WT_processed.npy/1"
    path_to_plot = os.path.join(img_input_dir, path_name_to_plot)

    attn_maps = load_npy_to_nparray(attn_maps_dir, "testset_attn.npy") 
    labels = load_labels_from_npy(attn_maps_dir, "testset")
    paths = load_paths_from_npy(attn_maps_dir, "testset")

    init_globals(attn_maps)

    os.makedirs(save_dir, exist_ok=True)

    if run_all:
        run_all_samples(paths, attn_maps,labels, min_attn_threshold, heads_reduce_fn, save_dir=save_dir)
    else:
        run_one_sample(paths, path_to_plot, attn_maps,labels, min_attn_threshold, heads_reduce_fn, save_dir=save_dir)



if __name__ == "__main__":
    main(run_all=False, min_attn_threshold = 0.5, heads_reduce_fn = np.mean)
    print("Done.")

    
    
   