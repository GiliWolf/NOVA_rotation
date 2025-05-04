import os
import sys
import matplotlib.pyplot as plt
import cv2

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")
working_dir = os.getcwd()
sys.path.append(working_dir)
print(f"working_dir: {working_dir}")

import logging

from src.models.architectures.NOVA_model import NOVAModel
from src.embeddings.embeddings_utils import generate_embeddings, save_embeddings
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.common.utils import get_if_exists
from src.datasets.data_loader import get_dataloader
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels,\
    edit_labels_by_config, get_batches_from_input_folders, get_reps_from_labels, get_conditions_from_labels, get_cell_lines_from_labels
from torch.utils.data import DataLoader
from collections import OrderedDict

from NOVA_rotation.load_files.load_data_from_npy import parse_paths, load_tile, load_paths_from_npy


"""
based on - ./NOVA/runnables/generate_embeddings.py

changes:
    (1) Reasign outputs_folder_path in generate_attn_maps_with_model
    (2) __generate_attn_maps_with_dataloader - calls model.gen_attn_maps(data_loader) instead of model.infer()
    (3) in NOVAModel - added gen_attn_maps function, which is the same as infer, but calls get_all_selfattention instead of farward (model(X))

"""

def generate_attn_maps_with_model(outputs_folder_path:str, config_path_data:str, config_path_plot:str, batch_size:int=700)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    attn_maps, labels, paths = generate_attn_maps(model, config_data, batch_size=batch_size)
    
    # OUTPUT 
    outputs_folder_path = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/attention_maps/attention_maps_output/RotationDatasetConfig_Pairs"
    saveroot = outputs_folder_path

    # CHECK WHAT TO KEEP 
    # colored_by = get_if_exists(config_plot, 'MAP_LABELS_FUNCTION',None)
    # if colored_by is not None:
    #     saveroot += f'_colored_by_{colored_by}'
    # to_color = get_if_exists(config_plot, 'TO_COLOR',None)
    # if to_color is not None:
    #     saveroot += f'_coloring_{to_color[0].split("_")[0]}'

    # os.makedirs(saveroot, exist_ok=True)
    # logging.info(f'saveroot: {saveroot}')

    # extract samples to plot and save 
    pairs_dir = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/RotationDatasetConfig/pairs"

    
    ### NEEDS TO ADD - CAN BE IN BATCHES, FOR NOW TAKES THE FIRST ONE
    attn_maps = attn_maps[0]
    labels = labels[0]
    paths = paths[0]

    # filter by path names 
    samples_indices = __extract_samples_to_plot(keep_samples_dir=pairs_dir, paths = paths)
    attn_maps = attn_maps[samples_indices]
    labels = labels[samples_indices]
    paths = paths[samples_indices]
    plot_attn_maps(attn_maps, labels, paths, config_data, os.path.join(outputs_folder_path, "figures"))

    # save the raw attn_maps
    save_attn_maps(attn_maps, labels, config_data, os.path.join(outputs_folder_path, "raw"))

def generate_attn_maps(model:NOVAModel, config_data:DatasetConfig, 
                        batch_size:int=700, num_workers:int=6)->Tuple[List[np.ndarray[torch.Tensor]],
                                                                      List[np.ndarray[str]]]:
    logging.info(f"[generate_attn_maps] Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"[generate_attn_maps] Num GPUs Available: {torch.cuda.device_count()}")

    all_attn_maps, all_labels, all_paths = [], [], []

    train_paths:np.ndarray[str] = model.trainset_paths
    val_paths:np.ndarray[str] = model.valset_paths
    
    full_dataset = DatasetNOVA(config_data)
    full_paths = full_dataset.get_X_paths()
    full_labels = full_dataset.get_y()
    logging.info(f'[generate_embbedings]: total files in dataset: {full_paths.shape[0]}')
    for set_paths, set_type in zip([train_paths, val_paths, None],
                                   ['trainset','valset','testset']):
        
        if set_type=='testset':
            paths_to_remove = np.concatenate([train_paths, val_paths])
            current_paths = full_dataset.get_X_paths()
            current_labels = full_dataset.get_y()
            indices_to_keep = np.where(~np.isin(current_paths, paths_to_remove))[0]      
            new_set_paths = current_paths[indices_to_keep]
            new_set_labels = current_labels[indices_to_keep]
        
        else:
            indices_to_keep = np.where(np.isin(full_paths, set_paths))[0]
            if indices_to_keep.shape[0]==0:
                continue
            
            new_set_paths = full_paths[indices_to_keep]
            new_set_labels = full_labels[indices_to_keep]


        logging.info(f'[generate_embbedings]: for set {set_type}, there are {new_set_paths.shape} paths and {new_set_labels.shape} labels')
        new_set_dataset = deepcopy(full_dataset)
        new_set_dataset.set_Xy(new_set_paths, new_set_labels)
        
        attn_maps, labels, paths = __generate_attn_maps_with_dataloader(new_set_dataset, model, batch_size, num_workers)
        
        all_attn_maps.append(attn_maps)
        all_labels.append(labels)
        all_paths.append(paths)

    return all_attn_maps, all_labels, all_paths

def save_attn_maps(embeddings:List[np.ndarray[torch.Tensor]], 
                    labels:List[np.ndarray[str]], data_config:DatasetConfig, output_folder_path)->None:

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_attn_maps] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        cur_embeddings, cur_labels = embeddings[i], labels[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}
        for batch, batch_indexes in __dict_temp.items():
            # create folder if needed
            batch_save_path = os.path.join(output_folder_path, 'attn_maps', data_config.EXPERIMENT_TYPE, batch)
            os.makedirs(batch_save_path, exist_ok=True)
            
            if not data_config.SPLIT_DATA:
                # If we want to save a full batch (without splittint to train/val/test), the name still will be testset.npy.
                # This is why we want to make sure that in this case, we never saved already the train/val/test sets, because this would mean this batch was used as training batch...
                if os.path.exists(os.path.join(batch_save_path,f'trainset_labels.npy')) or os.path.exists(os.path.join(batch_save_path,f'valset_labels.npy')):
                    logging.warning(f"[save_attn_maps] SPLIT_DATA={data_config.SPLIT_DATA} BUT there exists trainset or valset in folder {batch_save_path}!! make sure you don't overwrite the testset!!")
            logging.info(f"[save_attn_maps] Saving {len(batch_indexes)} in {batch_save_path}")
            
            np.save(os.path.join(batch_save_path,f'{set_type}_attn_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}_attn.npy'), cur_embeddings[batch_indexes])

            logging.info(f'[save_attn_maps] Finished {set_type} set, saved in {batch_save_path}')



def __generate_attn_maps_with_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_attn_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    attn_maps, labels, paths = model.gen_attn_maps(data_loader) # (num_samples, num_layers, num_heads, num_patches, num_patches)
    logging.info(f'[generate_attn_maps_with_dataloader] total attn_maps: {attn_maps.shape}')
    
    return attn_maps, labels, paths


def __extract_samples_to_plot(keep_samples_dir:str, paths: np.ndarray[str]):
    keep_paths_df = load_paths_from_npy(keep_samples_dir)
    paths_df = parse_paths(paths)

    samples_indices = paths_df[paths_df["Path"].isin(keep_paths_df["Path"])].index.tolist()

    return samples_indices


def plot_attn_maps(attn_maps: np.ndarray[float], labels: np.ndarray[str], paths: np.ndarray[str], data_config: DatasetConfig, output_folder_path: str):
    """
    Plot attention maps for a specific sample.

    Parameters
    ----------
    attn_maps :         np.ndarray of shape (num_samples, num_layers, num_heads, num_patches, num_patches)
                        The attention maps for all samples. Each map shows how patches attend to each other across layers and heads.
    labels :            np.ndarray of shape (num_samples,)
                        Class labels for each sample (used for labeling plots).
    sample_index :      int
                        The index of the sample whose attention maps will be visualized.
    data_config:        DatasetConfig 
    output_folder_path : str
                        Path to the folder where the attention heatmaps and overlay plots will be saved.

    Algo
    ---------
    1. Extract the attention maps, label, and image corresponding to `sample_index`.
    2. For each layer:
        a. Aggregate the attention across all heads (e.g., by averaging or summing).
        b. Reshape the resulting attention map's patches (tokens) into pixels
        c. Save the heatmap of attention weights.
        d. Overlay the attention map on top of the original input image and save the result.

"""

    os.makedirs(output_folder_path, exist_ok=True)
    img_shape = data_config.IMAGE_SIZE # suppose to be square (100, 100)
    img_path_df = parse_paths(paths)

    # Extract attention and label for the sample
    logging.info(f"[plot_attn_maps] starting plotting {len(paths)} samples.")
    for index, (sample_attn, label, img_path) in enumerate(zip(attn_maps, labels, paths)):
        # load img details
        img_path = str(img_path_df.Path.iloc[index]).split('.npy')[0]+'.npy'
        tile = int(img_path_df.Tile.iloc[index])
        Site = img_path_df.Site.iloc[index]

        # plot
        output_folder_path = os.path.join(output_folder_path, os.path.basename(img_path).split('.npy')[0])
        os.makedirs(output_folder_path, exist_ok=True)
        __plot_attn(sample_attn, img_path, tile, Site, label, img_shape, output_folder_path)

def __plot_attn(sample_attn: np.ndarray[float], img_path:str, tile:int, Site:str,  label:str, img_shape:tuple, output_folder_path:str):
    num_layers, num_heads, num_patches, _ = sample_attn.shape
    patch_dim = int(np.sqrt(num_patches))

    marker, nucleus, input_overlay = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape == img_shape

    logging.info(f"[plot_attn_maps] extracting sample image path: {os.path.basename(img_path)}")
    logging.info(f"[plot_attn_maps] extracting sample label: {label}")
    logging.info(f"[plot_attn_maps] dimensions: {num_layers} layers, {num_heads} heads, {num_patches} patches, {img_shape} img_shape")
    

    for layer_idx in range(num_layers):
        # Get attention for this layer and average over heads
        attn = sample_attn[layer_idx]  # (num_heads, num_patches+1, num_patches+1)
        attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape) #(img_shape

        # Green = nucleus
        # Blue = marker
        # red = attn map
        attn_red = np.zeros_like(input_overlay)
        attn_inverted = 1.0 - attn_map  # higher attention => deeper red
        attn_red[..., 0] = np.clip(attn_inverted, 0, 1) 

        # overlay attention with input fig using alpha for transpercy 
        alpha = 0.45   
        attn_overlay = cv2.addWeighted(input_overlay, 1.0, attn_red, alpha, 0)
        attn_overlay_uint8 = (attn_overlay * 255).clip(0, 255).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].set_title(f'Input - Marker (blue), Nucleus (green)', fontsize=11)
        ax[0].imshow(input_overlay)
        ax[0].set_axis_off()

        ax[1].set_title(f'Attention Heatmap', fontsize=11)
        ax[1].imshow(heatmap_colored, cmap='hot')
        ax[1].set_axis_off()

        ax[2].set_title(f'Attention Overlay', fontsize=11)
        ax[2].imshow(attn_overlay_uint8)
        ax[2].set_axis_off()

        fig.suptitle(f"Site {Site} | Tile {tile} | Layer {layer_idx}\n{label}", fontsize=12)

        save_path = os.path.join(output_folder_path, f'site{Site}_tile{tile}_layer{layer_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")


def __process_attn_map(attn, patch_dim, img_shape):# (num_heads, num_patches+1, num_patches+1)
        avg_attn = attn.mean(axis=0)  # (num_patches+1, num_patches+1)

        # Take attention from CLS token to all other patches (assumes CLS is first)
        cls_attn = avg_attn[0, 1:] # (num_patches)
        cls_attn_map = cls_attn.reshape(patch_dim, patch_dim) # reshape to square 

        # Normalize attention for heatmap
        attn_heatmap = (cls_attn_map - cls_attn_map.min()) / (cls_attn_map.max() - cls_attn_map.min() + 1e-6)

        # Resize to match image size
        heatmap_resized = cv2.resize(attn_heatmap, img_shape)

        # # Normalize for color mapping
        attn_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)

        return heatmap_resized, heatmap_colored

"""
After constructung working plot_attn_map, use the follwoing function, based on 
    NOVA_rotation/NOVA/src/figures/umap_plotting.py/plot_umap
in order to use it in a "NOVA" like way 
"""
# def plot_attn_maps_fromumap(attn_maps: np.ndarray[float], labels: np.ndarray[str], config_data: DatasetConfig,
#               config_plot: PlotConfig, saveroot: str, figsize: Tuple[int,int] = (6,5), cmap='tab20') -> None:

#     if saveroot:
#         os.makedirs(saveroot, exist_ok=True)
#         save_config(config_data, saveroot)
#         save_config(config_plot, saveroot)
        
#     markers = get_unique_parts_from_labels(labels, get_markers_from_labels)
#     logging.info(f"[plot_attn_maps] Detected markers: {markers}")

#     multiple_markers = False
#     if multiple_markers:
#         for marker in markers:
#             logging.info(f"[plot_attn_maps]: Marker: {marker}")
#             indices = np.where(np.char.startswith(labels.astype(str), f"{marker}_"))[0]
#             logging.info(f"[plot_attn_maps]: {len(indices)} indexes have been selected")

#             if marker == 'DAPI':
#                 np.random.seed(config_plot.SEED)
#                 indices = np.random.choice(indices, size=int(len(indices) * 0.25), replace=False)
#                 logging.info(f"[plot_attn_maps]: {len(indices)} indexes have been selected after DAPI downsample")

#             if len(indices) == 0:
#                 logging.info(f"[plot_attn_maps] No data for marker {marker}, skipping.")
#                 continue

#             marker_attn_maps = attn_maps[indices]
#             marker_labels = labels[indices].reshape(-1,)

#             savepath = os.path.join(saveroot, f'{marker}') if saveroot else None
#             label_data = map_labels(marker_labels, config_plot, config_data)
            
#             __plot_attn_maps(marker_attn_maps, label_data, config_data, config_plot, savepath=savepath, title=marker,
#                                    ari_score=ari_score, figsize=figsize, cmap=cmap)
#         return

#     else:
#         # Mode: All markers together
#         #CHECK
#         savepath = os.path.join(saveroot, '??') if saveroot else None

    
#     label_data = map_labels(labels, config_plot, config_data)
    
#     __plot_attn_maps(attn_maps, label_data, config_data, config_plot, savepath,
#                            ari_score=ari_score, figsize=figsize, cmap=cmap)

# def __plot_attn_maps(attn_maps: np.ndarray[float], 
#                          label_data: np.ndarray[str], 
#                          config_data: DatasetConfig,
#                          config_plot: PlotConfig,
#                          savepath: str = None,
#                          title: str = None, 
#                          dpi: int = 500, 
#                          figsize: Tuple[int,int] = (6,5),
#                          cmap:str = 'tab20'
#                          ) -> None:
#     """Plots UMAP embeddings with given labels and configurations.

#     Args:
#         umap_embeddings (np.ndarray[float]): The 2D UMAP embeddings to be plotted.
#         label_data (np.ndarray[str]): Array of labels corresponding to the embeddings.
#         config_data (DatasetConfig): Configuration data containing metric settings.
#         config_plot (PlotConfig): Configuration plot containing visualization settings.
#         savepath (str, optional): Path to save the plot. If None, the plot is shown interactively. Defaults to None.
#         title (str, optional): Title for the plot. Defaults to 'UMAP projection of Embeddings'.
#         dpi (int, optional): Dots per inch for the saved plot. Defaults to 300.
#         figsize (Tuple[int, int], optional): Size of the figure. Defaults to (6, 5).
#         cmap (str, optional): Colormap to be used. Defaults to 'tab20'.
#         ari_score (float, optional): ari score to show on the umap. Defaults to None.

#     Raises:
#         ValueError: If the size of `umap_embeddings` and `label_data` are incompatible.

#     Returns:
#         None
#     """
#     if umap_embeddings.shape[0] != label_data.shape[0]:
#         raise ValueError("The number of embeddings and labels must match.")

#     name_color_dict =  config_plot.COLOR_MAPPINGS
#     name_key = config_plot.MAPPINGS_ALIAS_KEY
#     color_key = config_plot.MAPPINGS_COLOR_KEY
#     marker_size = config_plot.SIZE
#     to_color = get_if_exists(config_plot, 'TO_COLOR', None)
#     show_metric = config_data.SHOW_ARI
#     mix_groups = get_if_exists(config_plot, 'MIX_GROUPS', False)
#     logging.info(f'mix_groups: {mix_groups}')
#     unique_groups = np.unique(label_data)

#     ordered_marker_names = get_if_exists(config_plot, 'ORDERED_MARKER_NAMES', None)
#     if ordered_marker_names:
#         # Get the indices of each element in 'unique_groups' according to 'ordered_marker_names'
#         indices = [ordered_marker_names.index(item) for item in unique_groups]
#         # Sort the unique_groups based on the indices
#         unique_groups = unique_groups[np.argsort(indices)]

#     fig = plt.figure(figsize=figsize, dpi=300)
#     gs = GridSpec(2,1,height_ratios=[20,1])

#     ax = fig.add_subplot(gs[0])
#     indices = []
#     colors = []
#     for i, group in enumerate(unique_groups):
#         alpha = config_plot.ALPHA
#         logging.info(f'[_plot_attn_maps]: adding {group}')
#         group_indices = np.where(label_data==group)[0]
#         if group == 'DAPI':
#             np.random.seed(config_plot.SEED)
#             group_indices = np.random.choice(group_indices, size=int(len(group_indices) * 0.1), replace=False)
#         # Get hex color and convert to RGBA
#         if to_color is not None and group not in to_color:
#             base_color = '#bab5b5'
#             alpha = 0.2
#         else:
#             base_color = name_color_dict[group][color_key] if name_color_dict else plt.get_cmap(cmap)(i)

#         rgba_color = mcolors.to_rgba(base_color, alpha=alpha)  # Convert hex to RGBA and apply alpha
        
#         # Create a color array for each point
#         color_array = np.array([rgba_color] * group_indices.shape[0])

#         label = name_color_dict[group][name_key] if name_color_dict else group
#         if not mix_groups:
#             ax.scatter(
#                 umap_embeddings[group_indices, 0],
#                 umap_embeddings[group_indices, 1],
#                 s=marker_size,
#                 alpha=alpha,
#                 c=color_array,
#                 marker = 'o',
#                 label=label,
#                 linewidths=0,
#             )
#             logging.info(f'[_plot_attn_maps]: adding label {label}')
#         else:
#             colors.append(color_array)
#             indices.append(group_indices)
    
#     if mix_groups:
#         colors = np.concatenate(colors)
#         indices = np.concatenate(indices)
#         shuffled_indices = np.random.permutation(len(indices))
#         shuffled_colors = colors[shuffled_indices]
#         shuffled_indices = indices[shuffled_indices]
#         ax.scatter(
#             umap_embeddings[shuffled_indices, 0],
#             umap_embeddings[shuffled_indices, 1],
#             s=marker_size,
#             alpha=alpha,
#             c=shuffled_colors,
#             marker = 'o',
#             linewidths=0,
#         )                    
#     __format_UMAP_axes(ax, title)
#     if not mix_groups:
#         __format_UMAP_legend(ax, marker_size)
        
#     if show_metric:
#         gs_bottom = fig.add_subplot(gs[1])
#         ax = __get_metrics_figure(ari_score, ax=gs_bottom)
    
#     fig.tight_layout()
#     if savepath:
#         save_plot(fig, savepath, dpi, save_eps=True)
#     else:
#         plt.show()
        
#     return fig, ax

# def __format_UMAP_axes(ax:Axes, title:str)->None:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_xlabel('UMAP1')
#     ax.set_ylabel('UMAP2')
#     ax.set_title(title)
    
#     ax.set_xticklabels([]) 
#     ax.set_yticklabels([]) 
#     ax.set_xticks([]) 
#     ax.set_yticks([]) 
#     return

# def __format_UMAP_legend(ax:Axes, marker_size: int) -> None:
#     """Formats the legend in the plot."""
#     handles, labels = ax.get_legend_handles_labels()
#     leg = ax.legend(handles, labels, prop={'size': 6},
#                     bbox_to_anchor=(1, 1), loc='upper left',
#                     ncol=1 + len(labels) // 26, frameon=False)
#     for handle in leg.legendHandles:
#         handle.set_alpha(1)
#         handle.set_sizes([max(6, marker_size)])


if __name__ == "__main__":
    print("Starting generate attention maps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply outputs folder path, data config and plot config.")
        outputs_folder_path = sys.argv[1]
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        
        config_path_data = sys.argv[2]
        config_path_plot = sys.argv[3]
        if len(sys.argv)==5:
            try:
                batch_size = int(sys.argv[3])
            except ValueError:
                raise ValueError("Invalid batch size, must be integer")
        else:
            batch_size = 700
        generate_attn_maps_with_model(outputs_folder_path, config_path_data, config_path_plot, batch_size)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
