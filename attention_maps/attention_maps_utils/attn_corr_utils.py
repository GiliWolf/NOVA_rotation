import numpy as np
from scipy.stats import entropy

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


def get_percentiles(data, prc_list = [50, 25, 75], axis=0):
    perc_tuple = ()
    for prc in prc_list:
        res = np.percentile(data, prc, axis=axis)
        perc_tuple += (res,)
    return perc_tuple


def compute_parameters(attn_map, channels, corr_method = "pearsonr"):
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
    attn_probs /= attn_probs.sum()  # normalize to sum to 1
    layer_ent = entropy(attn_probs, base=2)  # base-2 entropy (bits)
    normalized_ent = layer_ent / np.log2(len(attn_probs))  # normalize to [0, 1]

    return corrs, normalized_ent, corr_method