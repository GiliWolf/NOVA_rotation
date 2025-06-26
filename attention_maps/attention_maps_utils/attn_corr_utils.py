import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

def corr_pearsonr(m1, m2):
    """
    calcaultes person r correlation score betweeb the 2 flattened matrices 
    """
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
    assumes m1 and m2 are normalized.
    Calculates structural similarity index measure between the 2 matrices.
    """
    from skimage.metrics import structural_similarity as ssim
    score, ssim_map = ssim(m1, m2, full=True)

    return score

def corr_attn_overlap(m1, m2, m2_binary_perc = 0.7):
    """
        for attention maps:
            sums the values of attention (m1) only in the masked area of the input (m2).
                --> "segment" to get only the most important pixels of the input images and calculate
                    the average attention value in those areas.  
    """
    # Use top X% of m2 (img) as binary mask
    threshold = np.quantile(m2, m2_binary_perc)
    m2_mask = m2 >= threshold
    if m2_mask.sum() == 0:
        return 0.0
    score = (m1[m2_mask].sum()) / m2_mask.sum() #normalize by the mask size
    return score

def corr_soft_overlap(attn_map, img_ch):

    # Element-wise product (overlap)
    overlap = np.sum(attn_map * img_ch)

    # Normalizations
    total_attn = np.sum(attn_map)
    total_marker = np.sum(img_ch)

    # Avoid division by zero
    precision_like = overlap / total_attn if total_attn > 0 else 0
    recall_like = overlap / total_marker if total_marker > 0 else 0

    # Harmonic mean (F1-like)
    if precision_like + recall_like > 0:
        f1_like = 2 * precision_like * recall_like / (precision_like + recall_like)
    else:
        f1_like = 0
    
    return f1_like

def normalize(v1):
    denom = v1.max() - v1.min()
    if denom == 0:
        return np.zeros_like(v1)
    else:
        return (v1 - v1.min()) / denom

def compute_correlation(attn, img_ch, corrleation_method:str):
    assert attn.shape == img_ch.shape

    # make sure both are normalized
    attn = normalize(attn) 
    img_ch = normalize(img_ch)

    return globals()[f"corr_{corrleation_method}"](attn, img_ch)


def get_percentiles(data, prc_list = [25,50,75], axis=0):
    perc_tuple = ()
    for prc in prc_list:
        res = np.percentile(data, prc, axis=axis)
        perc_tuple += (res,)
    return perc_tuple


def compute_corr_data(attn_map, channels, corr_method = "pearsonr"):
    """
        input:
            attn_map: attention maps values, already in the img shape (H,W), rescale to [0,1]
            channels: image input channels
            corr_method
        
        returns:
            corrs: list of correlation for each channel
            normalized_ent: normalized [0,1] entropy of the attn map
            corr_method: as the argument
    """

    corrs = []
    for channel in channels:
        # compute correlation - 
        ch_corr = compute_correlation(attn_map, channel, corr_method)
        corrs.append(ch_corr)


    # compute entropy -
    flat_attn = attn_map.flatten()
    attn_probs = flat_attn + 1e-8  # avoid log(0)
    attn_probs /= attn_probs.sum()  # normalize to make suresum to 1
    layer_ent = entropy(attn_probs, base=2)  # base-2 entropy (bits)
    normalized_ent = layer_ent / np.log2(len(attn_probs))  # normalize to [0, 1]

    return corrs, normalized_ent

def parse_corr_data_rollout(corr_data):
    """
    corr_data: [ [float, float, ...], entropy ]
    Returns: np.array([ch_1, ch_2, ..., ch_M, entropy])
    """
    corr_values = corr_data[0]          # list of floats, length = num_channels
    entropy_value = corr_data[1]        # single float
    corr_data_reshaped = np.array(corr_values + [entropy_value])
    return corr_data_reshaped

def parse_corr_data_all_layers(corr_data):
    """
    corr_data: [ [ [float list], entropy ], ... ] with length = num_layers
    Returns: np.array of shape (num_channels + 1, num_layers)
    """
    corr_data_reshaped = [parse_corr_data_rollout(layer_data) for layer_data in corr_data]
    corr_data_reshaped = np.array(corr_data_reshaped).T  # Transpose to get (num_channels + 1, num_layers)
    return corr_data_reshaped


def parse_corr_data_list(corr_data):

        """
            Converts corr_data from list of corr data items returned from *compute_corr_data* into a an np.array
            
            args:
                corr_data: list of length N, such that each item is a ([list of correlation for each channel], entropy value)

            returns:
                corr_data_reshaped: np.array of shape (N, num_channels+1)
        """
        corr_data_reshaped = []

        for ch_index in range(len(corr_data[0])): # iterate on the number of channels
            corr_list = [item[0][ch_index] for item in corr_data]
            corr_data_reshaped.append(corr_list)
            
        ent_list = [item[1] for item in corr_data]
        corr_data_reshaped.append(ent_list)
        corr_data_reshaped = np.array(corr_data_reshaped) 
        # transpose to have (num_samples x [num_channels + 1 (entropy))])
        corr_data_reshaped = corr_data_reshaped.T

        return corr_data_reshaped



def plot_correlation_rollout(corr_data, corr_method, config_plot, channel_names=None,  sup_title = "Rollout_Correlation_Entropy", output_folder_path=None):
    """
    Plots correlation and entropy boxplots from corr_data.

    Parameters:
        corr_data: np.ndarray of shape (N, num_channels + 1), 
                   where last column is entropy and others are per-channel correlations.
        corr_method: string, name of the correlation method to display on y-axis label.
        channel_names: optional list of names for each channel (length = num_channels).
        output_folder_path: if provided, saves the plot to the directory.
    """
    num_channels = corr_data.shape[1] - 1
    correlations = [corr_data[:, i] for i in range(num_channels)]
    entropy = corr_data[:, -1]

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(num_channels)]
    assert len(channel_names) == num_channels, "Mismatch between channel names and number of channels"

    # Define x axis positions
    corr_positions = np.arange(1, num_channels + 1)
    entropy_position = num_channels + 1

    fig, ax1 = plt.subplots(figsize=(1.5 * (num_channels + 1), 6))

    # Correlation boxplots (left y-axis)
    for i, corr_values in enumerate(correlations):
        ax1.boxplot(corr_values,
                    positions=[corr_positions[i]],
                    widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor='C'+str(i), color='black'),
                    medianprops=dict(color='black'),
                    showfliers=False)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel(f"{corr_method} correlation", fontsize=12)
    ax1.set_ylim(-1, 1)
    ax1.set_xticks(list(corr_positions) + [entropy_position])
    ax1.set_xticklabels(channel_names + ['Entropy'])

    # Entropy boxplot (right y-axis)
    ax2 = ax1.twinx()
    ax2.boxplot(entropy,
                positions=[entropy_position],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor='gray', color='black', alpha=0.5),
                medianprops=dict(color='black'),
                showfliers=False)

    ax2.set_ylabel("Entropy", fontsize=config_plot.PLOT_TITLE_FONTSIZE, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(-1, 1)
    ax2.set_xticks(list(corr_positions) + [entropy_position])
    ax2.set_xticklabels(channel_names + ['Entropy'])

    fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if config_plot.SAVE_PLOT and (output_folder_path is not None):
        fig_name  = sup_title.split('\n', 1)[0] #either till the end of the line or the full str
        plt.savefig(os.path.join(output_folder_path, f"{fig_name}.png"),
                    dpi=config_plot.PLOT_SAVEFIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


def plot_correlation_rollout_by_markers(corr_by_markers, corr_method, config_plot, channel_names=None, sup_title="Rollout_Correlation_Entropy", output_folder_path=None):
    """
    Plots correlation and entropy boxplots for each marker group, clustering each marker together.

    Parameters:
        corr_by_markers: dict of {marker_name: np.ndarray} with shape (N, num_channels + 1)
        corr_method: string, correlation method name
        config_plot: config object with plotting settings
        channel_names: optional list of names for each channel (length = num_channels)
        sup_title: overall figure title
        output_folder_path: directory to save the plot
    """

    marker_names = list(corr_by_markers.keys())
    num_markers = len(marker_names)
    sample_corr = next(iter(corr_by_markers.values()))
    num_channels = (sample_corr.shape[1] - 1)  # Last column = entropy

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(num_channels)]
    assert len(channel_names) == num_channels, "Mismatch between channel names and number of channels"

    # Plot setup
    fig, ax1 = plt.subplots(figsize=(1.5 * (num_channels + 1) * num_markers, 6))

    box_width = 0.6
    intra_gap = 1.0    # spacing between channels within the same marker group
    inter_gap = 2.5    # spacing between different marker groups

    xtick_positions = []
    xtick_labels = []
    current_pos = 1

    for m_idx, marker in enumerate(marker_names):
        marker_data = corr_by_markers[marker]
        correlations = [marker_data[:, i] for i in range(num_channels)]
        entropy = marker_data[:, -1]

        # Plot correlations
        for i, corr_values in enumerate(correlations):
            pos = current_pos
            ax1.boxplot(corr_values,
                        positions=[pos],
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(facecolor=f'C{i}', color='black'),
                        medianprops=dict(color='black'),
                        showfliers=False)
            xtick_positions.append(pos)
            xtick_labels.append(f"{marker}\n{channel_names[i]}")
            current_pos += intra_gap

        # Plot entropy
        pos = current_pos
        ax1.boxplot(entropy,
                    positions=[pos],
                    widths=box_width * 0.8,
                    patch_artist=True,
                    boxprops=dict(facecolor='gray', color='black', alpha=0.5),
                    medianprops=dict(color='black'),
                    showfliers=False)
        xtick_positions.append(pos)
        xtick_labels.append(f"{marker}\nEntropy")

        # Move to next cluster position
        current_pos += inter_gap

    # Formatting
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel(f"{corr_method} correlation", fontsize=12)
    ax1.set_ylim(-1, 1)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE)

    plt.tight_layout()

    if config_plot.SAVE_PLOT and (output_folder_path is not None):
        fig_name = sup_title.split('\n', 1)[0]
        plt.savefig(os.path.join(output_folder_path, f"{fig_name}.png"),
                    dpi=config_plot.PLOT_SAVEFIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


def plot_correlation_all_layers(corr_data, corr_method, config_plot, channel_names=None,  sup_title = "All_Layers_Correlation_Entropy", output_folder_path=None):
    """
    Plot correlation for each layer across multiple samples with support for multiple channels.

    Parameters:
        corr_data: np.ndarray of shape (num_samples, num_channels + 1, num_layers)
                   last channel is entropy.
        num_layers: number of layers.
        corr_method: string for title labeling.
        save_dir: directory to save plot.
        channel_names: list of names for correlation channels (not including entropy).
    """
    num_channels = (corr_data.shape[1] - 1)  # last is entropy
    num_layers = corr_data.shape[2]

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(num_channels)]

    assert len(channel_names) == num_channels, "Mismatch between channel names and number of correlation channels"

    layers_range = np.arange(num_layers)

    # Percentiles for each correlation channel
    medians_corr = []
    p25s_corr = []
    p75s_corr = []

    for ch in range(num_channels):
        p25, median, p75 = get_percentiles(corr_data[:, ch, :], prc_list=[25,50,75])
        medians_corr.append(median)
        p25s_corr.append(p25)
        p75s_corr.append(p75)

    # Percentiles for entropy
    p25_entropy, median_entropy,p75_entropy = get_percentiles(corr_data[:, -1, :], prc_list=[25,50,75])

    # Plot
    fig, ax1 = plt.subplots(figsize=(1.5 * num_layers, 6))
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Plot correlations
    for ch in range(num_channels):
        ax1.plot(layers_range, medians_corr[ch], label=f"{channel_names[ch]} (Median)", 
                 marker='o', color=f"C{ch}")
        ax1.fill_between(layers_range, p25s_corr[ch], p75s_corr[ch], alpha=0.3, color=f"C{ch}")

    ax1.set_xlabel("Layer Number", fontsize=config_plot.PLOT_TITLE_FONTSIZE)
    ax1.set_ylabel(f"{corr_method} Correlation", fontsize=config_plot.PLOT_TITLE_FONTSIZE)
    ax1.set_xticks(layers_range)
    ax1.legend(loc="upper left")

    # Plot entropy on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(layers_range, median_entropy, label="Entropy (Median)", color='gray', linestyle='--', marker='x', alpha=0.5)
    ax2.fill_between(layers_range, p25_entropy, p75_entropy, color='gray', alpha=0.1)
    ax2.set_ylabel("Entropy", fontsize=config_plot.PLOT_TITLE_FONTSIZE, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if config_plot.SAVE_PLOT and (output_folder_path is not None):
        fig_name  = sup_title.split('\n', 1)[0] #either till the end of the line or the full str
        plt.savefig(os.path.join(output_folder_path, f"{fig_name}.png"),
                    dpi=config_plot.PLOT_SAVEFIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


def plot_correlation_all_layers_by_markers(corr_by_markers, corr_method, config_plot, channel_names=None, sup_title="All_Layers_Correlation_Entropy", output_folder_path=None):
    
    print("ERROR: plot_correlation_all_layers_by_markers is NOT IMPLEMENTED")
    pass

