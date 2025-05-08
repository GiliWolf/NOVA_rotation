import os
import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))


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

from NOVA_rotation.load_files.load_data_from_npy import parse_paths, load_tile, load_paths_from_npy, Parse_Path_Item


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
    input_dir = "/home/projects/hornsteinlab/giliwo/NOVA_rotation"
    run_name = "RotationDatasetConfig_Rollout"
    outputs_folder_path = os.path.join(input_dir, "attention_maps/attention_maps_output", run_name)

    # filter by path names 
    samples_indices = __extract_indices_to_plot(keep_samples_dir=config_plot.SAMPLES_PATH, paths = paths, data_config = config_data)
    attn_maps = __extract_samples_to_plot(attn_maps, samples_indices, data_config = config_data)
    labels = __extract_samples_to_plot(labels, samples_indices, data_config = config_data)
    paths = __extract_samples_to_plot(paths, samples_indices, data_config = config_data)
    plot_attn_maps(attn_maps, labels, paths, config_data, config_plot, output_folder_path=os.path.join(outputs_folder_path, "figures"))

    # save the raw attn_map (AFTER FILTERING)
    # save att map according to ATTN_METHOD
    save_attn_maps(attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "raw"))


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


def save_attn_maps(attn_maps:List[np.ndarray[torch.Tensor]], 
                    labels:List[np.ndarray[str]], paths:List[np.ndarray[str]],
                    data_config:DatasetConfig, output_folder_path)->None:

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_attn_maps] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels, cur_paths = attn_maps[i], labels[i], paths[i]
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
            
            np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}_attn.npy'), cur_attn_maps[batch_indexes])
            np.save(os.path.join(batch_save_path,f'{set_type}_paths.npy'), cur_paths[batch_indexes])

            logging.info(f'[save_attn_maps] Finished {set_type} set, saved in {batch_save_path}')



def __generate_attn_maps_with_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_attn_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    attn_maps, labels, paths = model.gen_attn_maps(data_loader) # (num_samples, num_layers, num_heads, num_patches, num_patches)
    logging.info(f'[generate_attn_maps_with_dataloader] total attn_maps: {attn_maps.shape}')
    
    return attn_maps, labels, paths


def __extract_indices_to_plot(keep_samples_dir:str, paths: np.ndarray[str], data_config: DatasetConfig):

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_samples_indices = []
    for i, set_type in enumerate(data_set_types):
        cur_paths = paths[i]
        temp_keep_samples_dir = os.path.join(keep_samples_dir, set_type)
        keep_paths_df = load_paths_from_npy(temp_keep_samples_dir, set_type)
        paths_df = parse_paths(cur_paths)
        samples_indices = paths_df[paths_df["Path"].isin(keep_paths_df["Path"])].index.tolist()
        all_samples_indices.append(samples_indices)
    return all_samples_indices

def __extract_samples_to_plot(sampels: np.ndarray[str], indices:list, data_config: DatasetConfig):

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_filtered_sampels = []
    for i, set_type in enumerate(data_set_types):
        curr_samples, curr_indices = sampels[i], indices[i]
        filtered_samples = curr_samples[curr_indices]
        all_filtered_sampels.append(filtered_samples)
        
    
    return all_filtered_sampels


def plot_attn_maps(attn_maps: np.ndarray[float], labels: np.ndarray[str], paths: np.ndarray[str], data_config: DatasetConfig, config_plot, output_folder_path: str):
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
    config_plot:        PlotConfig
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

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    

    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels, cur_paths = attn_maps[i], labels[i], paths[i]

        img_path_df = parse_paths(cur_paths)

        logging.info(f'[plot_attn_maps]: for set {set_type}, starting plotting {len(cur_paths)} samples.')
        
        for index, (sample_attn, label, img_path) in enumerate(zip(cur_attn_maps, cur_labels, cur_paths)):
            # load img details
            path_item = img_path_df.iloc[index]
            img_path, tile, site = Parse_Path_Item(path_item)

            # plot
            temp_output_folder_path = os.path.join(output_folder_path, set_type, os.path.basename(img_path).split('.npy')[0])
            os.makedirs(temp_output_folder_path, exist_ok=True)
            __plot_attn(sample_attn, img_path, (site, tile, label), img_shape, config_plot, temp_output_folder_path)




def __plot_attn(sample_attn: np.ndarray[float], img_path:str, sample_info:tuple, img_shape:tuple, config_plot, output_folder_path:str):
    num_layers, num_heads, num_patches, _ = sample_attn.shape
    patch_dim = int(np.sqrt(num_patches))

    _, tile, label = sample_info

    marker, nucleus, input_overlay = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape == img_shape

    logging.info(f"[plot_attn_maps] extracting sample image path: {os.path.basename(img_path)}")
    logging.info(f"[plot_attn_maps] extracting sample label: {label}")
    logging.info(f"[plot_attn_maps] dimensions: {num_layers} layers, {num_heads} heads, {num_patches} patches, {img_shape} img_shape")

    attn_method = config_plot.ATTN_METHOD
    globals()[f"_plot_attn_map_{attn_method}"](sample_attn, sample_info, patch_dim, input_overlay, img_shape, config_plot, output_folder_path)

def _plot_attn_map_all_layers(attn, sample_info, patch_dim, input_img, img_shape, config_plot, output_folder_path):
    site, tile, label = sample_info
    attn = __attn_map_all_layers(attn)
    for layer_idx in range(num_layers):
        layer_attn = attn[layer_idx]
        layer_attn_map, heatmap_colored = __process_attn_map(layer_attn, patch_dim, img_shape, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP)
        __create_attn_map_img(layer_attn_map, input_img, heatmap_colored, config_plot, sup_title =f"Site{site}_Tile{tile}_Layer{layer_idx}\n{label}",  output_folder_path=output_folder_path)


def _plot_attn_map_rollout(attn, sample_info, patch_dim, input_img, img_shape, config_plot, output_folder_path):
    site, tile, label = sample_info
    attn = __attn_map_rollout(attn)
    attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP)
    __create_attn_map_img(attn_map, input_img, heatmap_colored, config_plot, sup_title= f"Site{site}_Tile{tile}_Rollout\n{label}", output_folder_path= output_folder_path)



def __create_attn_map_img(attn_map, input_img, heatmap_colored, config_plot, sup_title = "Attention Maps", output_folder_path = None):
        """
            Create attention map img with:
                (1) input image 
                (2) attention heatmap
                (3) attention overlay on the input img
            ** save/plot according to config_plot

            parameters:
                attn_map: attention maps values, already in the img shape (H,W)
                input_img: input img with marker and nucleus overlay (3,H,W)
                            ** assuming  Green = nucleus, Blue = marker, Red = zeroed out
                heatmap_colored: attention map colored by heatmap_color (3,H,W)
                config_plot: config with the plotting parameters 
                output_folder_path: for saving the output img.

            return:
                fig: matplot fig created. 
        """
        # Green = nucleus
        # Blue = marker
        # red = attn map
        attn_red = np.zeros_like(input_img)
        attn_inverted = 1.0 - attn_map  # higher attention => deeper red
        attn_red[..., 0] = np.clip(attn_inverted, 0, 1)     

        # overlay attention with input fig using alpha for transpercy 
        alpha = config_plot.ALPHA  
        attn_overlay = cv2.addWeighted(input_img, 1.0, attn_red, alpha, 0)
        attn_overlay_uint8 = (attn_overlay * 255).clip(0, 255).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=config_plot.FIG_SIZE)
        ax[0].set_title(f'Input - Marker (blue), Nucleus (green)', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[0].imshow(input_img)
        ax[0].set_axis_off()

        ax[1].set_title(f'Attention Heatmap', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[1].imshow(heatmap_colored, cmap='hot')
        ax[1].set_axis_off()

        ax[2].set_title(f'Attention Overlay', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[2].imshow(attn_overlay_uint8)
        ax[2].set_axis_off()

        fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE)

        if config_plot.SAVE_PLOT and (output_folder_path is not None):
            fig_name  = sup_title.split('\n', 1)[0] #either till the end of the line or the full str
            save_path = os.path.join(output_folder_path, f"{fig_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=config_plot.PLOT_SAVEFIG_DPI)
            plt.close()
        if config_plot.SHOW_PLOT:
            plt.show()

        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")

        return fig

def __process_attn_map(attn, patch_dim, img_shape, heatmap_color = cv2.COLORMAP_JET):
        """
            process attn map:
                (1) extract cls token attn to other pathces
                (2) normalize
                (3) resize to fit image shape
        
            parameteres:
                attn: attention map of shape (num_patches, num_patches)
                patch_dim: square root of num_patches to resize the attention vector into a square.
                img_shape: original img_shape (H, W)
                heatmap_color: int for cv2 coloring of the attention heatmap

            return:
                attn_resized: attention map processed into original img_shape (H,W)
                heatmap_colored: attention map colored by heatmap_color (3,H,W)

        """

        # Take attention from CLS token to all other patches (assumes CLS is first)
        cls_attn = attn[0, 1:] # (num_patches)
        cls_attn_map = cls_attn.reshape(patch_dim, patch_dim) # reshape to square 

        # Normalize attention for heatmap
        attn_heatmap = (cls_attn_map - cls_attn_map.min()) / (cls_attn_map.max() - cls_attn_map.min() + 1e-6)

        # Resize to match image size
        heatmap_resized = Image.fromarray((attn_heatmap * 255).astype(np.uint8)).resize(img_shape, resample=Image.BICUBIC)
        heatmap_resized = np.array(heatmap_resized).astype(np.uint8)
        attn_resized = np.array(heatmap_resized).astype(np.float32) / 255.0  # Rescale to [0,1] float

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, heatmap_color)

        #DUBUG
        # max_coord = np.unravel_index(np.argmax(attn_heatmap), attn_heatmap.shape)
        # min_coord = np.unravel_index(np.argmin(attn_heatmap), attn_heatmap.shape)
        # max_color = heatmap_colored[max_coord[0], max_coord[1]]  # OpenCV uses (y, x)
        # min_color = heatmap_colored[min_coord[0], min_coord[1]]

        # print(f"Max attention color at {max_coord}: {max_color}")
        # print(f"Min attention color at {min_coord}: {min_color}")

        return attn_resized, heatmap_colored

def __attn_map_all_layers(attn):
    """ average attention across heads, for each layer.

    parameteres:
        attn: attention values of shape: (num_layers, num_heads, num_patches, num_patches)

    return:
        avg_attn: attention map per layer: (num_layers, num_patches, num_patches)
    """
    avg_attn = attn.mean(axis=1)
    return avg_attn

def __attn_map_rollout(attn):
    """  aggregates attention maps across multiple layers, using the rollout method:

    parameteres:
        attn: attention values of shape: (num_layers, num_heads, num_patches, num_patches)

    returns:
        rollout: attention map for all layers and heads: (num_patches, num_patches)
    """

    # Initialize rollout with identity matrix
    rollout = np.eye(attn[0].shape[-1]) #(num_patches, num_patches)

    # Multiply attention maps layer by layer
    for layer_attn in attn:
        # CHECK layer_attn shape
        attention_heads_fused = layer_attn.mean(axis=0) # Average attention across heads = A
        attention_heads_fused += np.eye(attention_heads_fused.shape[-1]) # A + I
        attention_heads_fused /= attention_heads_fused.sum(axis=-1, keepdims=True) # Normalizing A
        rollout = rollout @ attention_heads_fused  # Matrix multiplication
    
    return rollout




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
