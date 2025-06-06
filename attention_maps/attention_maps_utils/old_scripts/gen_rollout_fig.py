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
from scipy.stats import entropy
import cv2
from matplotlib.colors import LinearSegmentedColormap
import itertools


def init_globals(attn_maps):
    global num_samples, num_layers, num_heads, num_patches, img_shape, patch_dim
    num_samples,num_layers, num_heads, num_patches, _ = attn_maps.shape
    img_shape = (100, 100)
    patch_dim = int(np.sqrt(num_patches))

def corr_pearsonr(m1, m2):
    from scipy.stats import pearsonr
    v1 = m1.flatten()
    v2 = m2.flatten()
    return pearsonr(v1, v2)[0]

def corr_mutual_info(m1, m2, bins=32):
    from sklearn.metrics import mutual_info_score
    # Flatten and discretize
    v1 = np.digitize(m1.flatten(), bins=np.histogram_bin_edges(m1, bins))
    v2 = np.digitize(m2.flatten(), bins=np.histogram_bin_edges(m2, bins))
    return mutual_info_score(v1, v2)

def corr_ssim(m1, m2):
    """
    assumes m1 and m2 are normalized 
    """
    from skimage.metrics import structural_similarity as ssim
    score, ssim_map = ssim(m1, m2, full=True)

    return score

def corr_attn_overlap(m1, m2, m2_binary_perc = 0.9):
    # Use top X% of m2 (img) as binary mask
    threshold = np.quantile(m2, m2_binary_perc)
    m2_mask = m2 >= threshold
    if m2_mask.sum() == 0:
        return 0.0
    score = (m1[m2_mask].sum()) / m2_mask.sum() #normalize by the mask size
    return score

def normalize(v1):
    denom = v1.max() - v1.min()
    if denom == 0:
        return np.zeros_like(v1)
    else:
        return (v1 - v1.min()) / denom

def compute_correlation(attn, img_ch, corrleation_method:str):
    # make sure both are normalized
    attn = normalize(attn) 
    img_ch = normalize(img_ch)
    return globals()[f"corr_{corrleation_method}"](attn, img_ch)


def compute_parameters(img_path, tile, sample_attn, min_attn_threshold = None, heads_reduce_fn=np.mean, start_layer_index = 0, corr_method = "pearsonr"):
    marker, nucleus, overlay = load_tile(img_path, tile)

    #attn = heads_reduce_fn(attn, axis=0)
    attn = __attn_map_rollout(sample_attn, attn_layer_dim=0,heads_reduce_fn = heads_reduce_fn, start_layer_index=start_layer_index)
    attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape, min_attn_threshold= min_attn_threshold)

    # compute correlations - 
    corr_nucleus = compute_correlation(attn_map, nucleus, corr_method)
    corr_marker = compute_correlation(attn_map, marker, corr_method)

    # compute entropy -
    flat_attn = attn_map.flatten()
    attn_probs = flat_attn + 1e-8  # avoid log(0)
    attn_probs /= attn_probs.sum()  # normalize to sum to 1
    layer_ent = entropy(attn_probs, base=2)  # base-2 entropy (bits)
    normalized_ent = layer_ent / np.log2(len(attn_probs))  # normalize to [0, 1]

    return attn_map, heatmap_colored, corr_nucleus, corr_marker, normalized_ent

def plot_fig(img_path, tile, site, file_name, label, attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy, corr_method, save_dir=None):

        _, _, input_img = load_tile(img_path, tile)
        attn_map = np.clip(attn_map, 0, 1)   
        alpha = 0.35

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{file_name}\n{label}\n\n", fontsize=16, fontweight='bold')

        ax[1].text(0.5, -0.25, f"{corr_method} Corr(Nucleus): {corr_nucleus:.2f}\n{corr_method} Corr(Marker): {corr_marker:.2f}\nEntropy: {entropy:.2f}", 
                transform=ax[1].transAxes, ha='center', va='center', fontsize=12, color='black')

        ax[0].set_title(f'Input - Marker (blue), Nucleus (green)', fontsize=12)
        ax[0].imshow(input_img)
        ax[0].set_axis_off()

        ax[1].set_title(f'Attention Heatmap', fontsize=12)
        ax[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        ax[1].set_axis_off()


        custom_cmap = LinearSegmentedColormap.from_list(
            'attn_overlay_colors', 
            ['black', 'white', 'yellow', 'orange', 'red']
        )

        ax[2].set_title('Attention Overlay', fontsize=12)
        ax[2].imshow(input_img)  # Show the original image
        ax[2].imshow(attn_map, cmap=custom_cmap, alpha=alpha)  # Overlay attention map transparently
        ax[2].set_axis_off()



        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.subplots_adjust(top=0.80)
            plt.savefig(os.path.join(save_dir, f"{file_name}_rollout_{corr_method}.png"), dpi=300, bbox_inches="tight",  facecolor=fig.get_facecolor())
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



def plot_correlation_boxplot(corr_nucleus_all, corr_marker_all, entropy_all, corr_method, save_dir=None):


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

     # dashed line on y=0 for better visulization
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)

    ax1.set_ylabel(f"{corr_method} Correlation", fontsize=12)
    ax1.set_ylim(-1, 1)
    ax1.set_xticks([1, 1.5, 2])
    ax1.set_xticklabels(['Nucleus', 'Entropy', 'Marker'])

    # Boxplot for entropy (right y-axis)
    ax2 = ax1.twinx()
    bp_entropy = ax2.boxplot(entropy_all, positions=positions_entropy, widths=0.3, patch_artist=True,
                             boxprops=dict(facecolor='gray', color='black', alpha=0.5),
                             medianprops=dict(color='black'), showfliers=False)

    ax2.set_ylabel("Entropy", fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 1)
    ax2.set_xticks([1, 1.5, 2])
    ax2.set_xticklabels(['Nucleus', 'Entropy', 'Marker'])

    # Title and spacing
    fig.suptitle("Rollout: Correlation and Entropy", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"rollout_{corr_method}_correlation_entropy_boxplot.png"),
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


def run_one_sample(paths, path_to_plot, attn_maps,labels, min_attn_threshold = None, heads_reduce_fn = None, start_layer_index = None, corr_method = None, save_dir=None):
    # get specific sample by path name 
    index, path = extract_path(paths, path_to_plot)
    sample_attn = attn_maps[index]
    label = labels.iloc[index].full_label
    img_path, tile, site = Parse_Path_Item(path)
    file_name = path.File_Name
    attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy = compute_parameters(img_path, tile, sample_attn, min_attn_threshold, heads_reduce_fn, start_layer_index, corr_method)
    plot_fig(img_path, tile, site, file_name, label, attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy, corr_method, save_dir)



def run_all_samples(paths, attn_maps,labels, min_attn_threshold = None, heads_reduce_fn = None, start_layer_index = None, corr_method = None, save_dir=None):
    corr_nucleus_all = []
    corr_marker_all = []
    entropy_all = []

    for index, (sample_attn, label) in enumerate(zip(attn_maps, labels["full_label"])):
        path = paths.iloc[index]
        img_path, tile, site = Parse_Path_Item(path)
        file_name = path.File_Name
        attn_map, heatmap_colored, corr_nucleus, corr_marker, entropy = compute_parameters(img_path, tile, sample_attn, min_attn_threshold, heads_reduce_fn, start_layer_index, corr_method)
        corr_nucleus_all.append(corr_nucleus)
        corr_marker_all.append(corr_marker)
        entropy_all.append(entropy)

    # plot correlations across samples
    plot_correlation_boxplot(corr_nucleus_all, corr_marker_all, entropy_all, corr_method, save_dir)

def main(run_all=False, min_attn_threshold=None, heads_reduce_fn = None, start_layer_index = None, corr_method = None):

    input_dir = "./NOVA_rotation/attention_maps/attention_maps_output"
    run_name = "RotationDatasetConfig_Euc_Pairs_rollout"
    attn_maps_dir = os.path.join(input_dir, run_name, "raw/attn_maps/neurons/batch9")
    save_dir =  os.path.join(input_dir, run_name,"layers_corr", f"threshold{min_attn_threshold:.1f}_{heads_reduce_fn.__name__}_start{start_layer_index}")


    img_input_dir = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk"
    samples_paths = ["batch9/WT/stress/G3BP1/rep1_R11_w3confCy5_s19_panelA_WT_processed.npy/0", 
                    "batch9/WT/stress/G3BP1/rep1_R11_w3confCy5_s38_panelA_WT_processed.npy/1",
                    "batch9/WT/Untreated/G3BP1/rep1_R11_w3confCy5_s204_panelA_WT_processed.npy/2",
                    "batch9/WT/Untreated/G3BP1/rep1_R11_w3confCy5_s281_panelA_WT_processed.npy/8",
                    "batch9/WT/stress/G3BP1/rep1_R11_w3confCy5_s26_panelA_WT_processed.npy/4",
                    "batch9/WT/stress/G3BP1/rep1_R11_w3confCy5_s60_panelA_WT_processed.npy/1",
                    "batch9/WT/Untreated/G3BP1/rep1_R11_w3confCy5_s208_panelA_WT_processed.npy/4",
                    "batch9/WT/Untreated/G3BP1/rep1_R11_w3confCy5_s225_panelA_WT_processed.npy/1"]

    attn_maps = load_npy_to_nparray(attn_maps_dir, "testset_attn.npy") 
    labels = load_labels_from_npy(attn_maps_dir, "testset")
    paths = load_paths_from_npy(attn_maps_dir, "testset")

    init_globals(attn_maps)

    os.makedirs(save_dir, exist_ok=True)

    if run_all:
        run_all_samples(paths, attn_maps,labels, min_attn_threshold, heads_reduce_fn, start_layer_index, corr_method,save_dir=save_dir)
    else:
        for path_name_to_plot in samples_paths:
            path_to_plot = os.path.join(img_input_dir, path_name_to_plot)
            run_one_sample(paths, path_to_plot, attn_maps,labels, min_attn_threshold, heads_reduce_fn, start_layer_index,corr_method,save_dir=save_dir)




if __name__ == "__main__":

    # Define parameter grid
    run_all_options = [False, True]
    min_attn_threshold_options = [0.2] #np.arange(0, 0.5, 0.1) # 0 to 0.4
    heads_reduce_fn_options = [np.mean]
    start_layer_index_options = [0]
    correlation_methods = ["pearsonr", "mutual_info", "ssim", "attn_overlap"]


    # Iterate over all combinations
    for run_all, min_attn_threshold, heads_reduce_fn, start_layer_index, corr_method in itertools.product(
        run_all_options, min_attn_threshold_options, heads_reduce_fn_options, start_layer_index_options, correlation_methods
    ):
        print(f"Running: run_all={run_all}, min_attn_threshold={min_attn_threshold}, "
            f"heads_reduce_fn={heads_reduce_fn.__name__}, start_layer_index={start_layer_index}")
        
        # Call your main function with the current parameter set
        main(run_all, min_attn_threshold, heads_reduce_fn, start_layer_index, corr_method)

    print("Done.")

    
    
   