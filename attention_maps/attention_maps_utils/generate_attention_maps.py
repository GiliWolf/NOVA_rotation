import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))


import logging
from src.models.architectures.NOVA_model import NOVAModel
from src.embeddings.embeddings_utils import load_embeddings
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
from matplotlib import gridspec
from NOVA_rotation.load_files.load_data_from_npy import parse_paths, load_tile, load_paths_from_npy, Parse_Path_Item
from NOVA_rotation.attention_maps.attention_maps_utils.attn_corr_utils import *
from NOVA_rotation.Configs.subset_config import SubsetConfig



"""
based on - ./NOVA/runnables/generate_embeddings.py

changes:
    (1) Reasign outputs_folder_path in generate_attn_maps_with_model
    (2) __generate_attn_maps_with_dataloader - calls model.gen_attn_maps(data_loader) instead of model.infer()
    (3) in NOVAModel - added gen_attn_maps function, which is the same as infer, but calls get_all_selfattention instead of farward (model(X))

"""

REDUCE_HEAD_FUNC_MAP = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
}
def generate_attn_maps_with_model(outputs_folder_path:str, config_path_data:str, config_path_plot:str, batch_size:int=700)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    attn_maps, labels, paths = generate_attn_maps(model, config_data, batch_size=batch_size)
    # save the raw attn_map (BEFORE FILTERING)
    # OUTPUT 
    home_dir = "/home/projects/hornsteinlab/giliwo/NOVA_rotation"
    outputs_folder_path = os.path.join(home_dir, "attention_maps/attention_maps_output")
    save_attn_maps(attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "raw"))

    batches = get_batches_from_input_folders(config_data.INPUT_FOLDERS)
    markers = config_data.MARKERS

    all_proccesed_attn_maps = []
    for batch in batches:
        for marker in markers:
            continue
            # filter the subset
            if config_plot.FILTER_BY_PAIRS:
                samples_path = os.path.join(home_dir, "embeddings/embedding_output/pairs/euclidean", config_data.EXPERIMENT_TYPE, batch, marker)
                samples_indices = __extract_indices_to_plot(keep_samples_dir=samples_path, paths = paths, data_config = config_data)
                attn_maps = __extract_samples_to_plot(attn_maps, samples_indices, data_config = config_data)
                labels = __extract_samples_to_plot(labels, samples_indices, data_config = config_data)
                paths = __extract_samples_to_plot(paths, samples_indices, data_config = config_data)

            # process and plot attn_maps (AFTER FILTERING)
            proccesed_attn_maps, corr_data = plot_attn_maps(attn_maps, labels, paths, config_data, config_plot, output_folder_path=os.path.join(outputs_folder_path, "figures", config_plot.ATTN_METHOD))
            all_proccesed_attn_maps.append(proccesed_attn_maps)

            # save the correlation data between the attn maps and input images
            save_corr_data(corr_data, labels, config_data, output_folder_path=os.path.join(outputs_folder_path, "correlations", config_plot.ATTN_METHOD, config_plot.CORR_METHOD))

            # save summary plots of the correlations
            if config_plot.PLOT_CORR_SUMMARY:
                plot_corr_data(corr_data, labels, config_data, config_plot, output_folder_path=os.path.join(outputs_folder_path, "figures", config_plot.ATTN_METHOD))

    # save all the processed attn_map (AFTER FILTERING - without marker seperation) ??????????????????????
    #save_attn_maps(all_proccesed_attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "processed", config_plot.ATTN_METHOD))

def generate_attn_maps_with_model_old(outputs_folder_path:str, config_path_data:str, config_path_plot:str, batch_size:int=700)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    attn_maps, labels, paths = generate_attn_maps(model, config_data, batch_size=batch_size)
    
    # OUTPUT 
    home_dir = "/home/projects/hornsteinlab/giliwo/NOVA_rotation"
    outputs_folder_path = os.path.join(home_dir, "attention_maps/attention_maps_output")

    if config_plot.SAMPLES_PATH is not None:
        # filter by path names 
        samples_indices = __extract_indices_to_plot(keep_samples_dir=config_plot.SAMPLES_PATH, paths = paths, data_config = config_data)
        attn_maps = __extract_samples_to_plot(attn_maps, samples_indices, data_config = config_data)
        labels = __extract_samples_to_plot(labels, samples_indices, data_config = config_data)
        paths = __extract_samples_to_plot(paths, samples_indices, data_config = config_data)

    # save the raw attn_map (AFTER FILTERING)
    save_attn_maps(attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "raw"))
 
    # process and plot attn_maps (AFTER FILTERING)
    proccesed_attn_maps, corr_data = plot_attn_maps(attn_maps, labels, paths, config_data, config_plot, output_folder_path=os.path.join(outputs_folder_path, "figures", config_plot.ATTN_METHOD))
    
    # save the processed attn_map (AFTER FILTERING)
    save_attn_maps(proccesed_attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "processed", config_plot.ATTN_METHOD))

    # save the correlation data between the attn maps and input images
    save_corr_data(corr_data, labels, config_data, output_folder_path=os.path.join(outputs_folder_path, "correlations", config_plot.ATTN_METHOD, config_plot.CORR_METHOD))

    # save summary plots of the correlations
    if config_plot.PLOT_CORR_SUMMARY:
        plot_corr_data(corr_data, labels, config_data, config_plot, output_folder_path=os.path.join(outputs_folder_path, "figures", config_plot.ATTN_METHOD))


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
    logging.info(f'[generate_attn_maps]: total files in dataset: {full_paths.shape[0]}')
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


        logging.info(f'[generate_attn_maps]: for set {set_type}, there are {new_set_paths.shape} paths and {new_set_labels.shape} labels')
        new_set_dataset = deepcopy(full_dataset)
        new_set_dataset.set_Xy(new_set_paths, new_set_labels)
        
        attn_maps, labels, paths = __generate_attn_maps_with_dataloader(new_set_dataset, model, batch_size, num_workers)
        
        all_attn_maps.append(attn_maps)
        all_labels.append(labels)
        all_paths.append(paths)

    return all_attn_maps, all_labels, all_paths


def save_attn_maps(attn_maps:List[np.ndarray[torch.Tensor]], 
                    labels:List[np.ndarray[str]], 
                    paths:List[np.ndarray[str]],
                    data_config:DatasetConfig, 
                    output_folder_path:str)->None:
    """
        ** if attn_method is gover, process the attn_maps accordinly before saving 
    """

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
            batch_save_path = os.path.join(output_folder_path, data_config.EXPERIMENT_TYPE, batch)
            os.makedirs(batch_save_path, exist_ok=True)
            
            if not data_config.SPLIT_DATA:
                # If we want to save a full batch (without splittint to train/val/test), the name still will be testset.npy.
                # This is why we want to make sure that in this case, we never saved already the train/val/test sets, because this would mean this batch was used as training batch...
                if os.path.exists(os.path.join(batch_save_path,f'trainset_labels.npy')) or os.path.exists(os.path.join(batch_save_path,f'valset_labels.npy')):
                    logging.warning(f"[save_attn_maps] SPLIT_DATA={data_config.SPLIT_DATA} BUT there exists trainset or valset in folder {batch_save_path}!! make sure you don't overwrite the testset!!")
            logging.info(f"[save_attn_maps] Saving {len(batch_indexes)} in {batch_save_path}")


            np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}.npy'), cur_attn_maps[batch_indexes])
            np.save(os.path.join(batch_save_path,f'{set_type}_paths.npy'), cur_paths[batch_indexes])

            logging.info(f'[save_attn_maps] Finished {set_type} set, saved in {batch_save_path}')


def save_corr_data(corr_data:List[np.ndarray[torch.Tensor]], 
                    labels:List[np.ndarray[str]],
                    data_config:DatasetConfig, output_folder_path:str)->None:
    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_corr_data] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        cur_corr_data, cur_labels = corr_data[i], labels[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}
        for batch, batch_indexes in __dict_temp.items():
            # create folder if needed
            batch_save_path = os.path.join(output_folder_path, data_config.EXPERIMENT_TYPE, batch)
            
            if not data_config.SPLIT_DATA:
                # If we want to save a full batch (without splittint to train/val/test), the name still will be testset.npy.
                # This is why we want to make sure that in this case, we never saved already the train/val/test sets, because this would mean this batch was used as training batch...
                if os.path.exists(os.path.join(batch_save_path,f'trainset_labels.npy')) or os.path.exists(os.path.join(batch_save_path,f'valset_labels.npy')):
                    logging.warning(f"[save_corr_data] SPLIT_DATA={data_config.SPLIT_DATA} BUT there exists trainset or valset in folder {batch_save_path}!! make sure you don't overwrite the testset!!")
            logging.info(f"[save_corr_data] Saving {len(batch_indexes)} in {batch_save_path}")

            for marker in data_config.MARKERS:
                marker_save_path = os.path.join(batch_save_path, marker)
                os.makedirs(marker_save_path, exist_ok=True)
                # Extract lists
                for ch_index in range(cur_corr_data.shape[1] - 1): # num channels 
                    corr_list = cur_corr_data[:, ch_index]
                    np.save(os.path.join(marker_save_path,f'{set_type}_corrs_ch{ch_index}.npy'), np.array(corr_list)[batch_indexes])
                
                ent_list = cur_corr_data[:, -1]
                np.save(os.path.join(marker_save_path,f'{set_type}_ent.npy'), np.array(ent_list)[batch_indexes])
  
                logging.info(f'[save_corr_data] saved in {marker_save_path}')

def plot_corr_data(corr_data:List[np.ndarray[torch.Tensor]], labels:List[np.ndarray[torch.Tensor]], data_config, config_plot, output_folder_path):
    """
        Plotting summary correlation plots using corr_data.

        Args:
            corr_data: all samples correlation data.
            labels: corresponding labels
            data_config: confif with parameteres of the data. 
            config_plot: onfif with parameteres of the plotting.
            output_folder_path: path to save the plots.

    """
    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[plot_corr_data] unique_batches: {unique_batches}')

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    attn_method = config_plot.ATTN_METHOD
    corr_method = config_plot.CORR_METHOD

    for i, set_type in enumerate(data_set_types):
        cur_corr_data, cur_labels = corr_data[i], labels[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}
        logging.info(f'[plot_corr_data]: for set {set_type}, starting plotting {len(cur_labels)} samples.')

        for batch, batch_indexes in __dict_temp.items():
            batch_save_path = os.path.join(output_folder_path, data_config.EXPERIMENT_TYPE, batch)
            logging.info(f"[plot_attn_maps] Saving {len(batch_indexes)} in {batch_save_path}")

            #extract current batch samples
            batch_corr_data = cur_corr_data[batch_indexes]
            batch_labels = cur_labels[batch_indexes]
        
            batch_markers = np.array(get_markers_from_labels(batch_labels))

            # extract current marker samples and plot
            corr_by_markers = {}
            marker_names = np.unique(batch_markers)
            for marker in marker_names:
                cur_output_path = os.path.join(batch_save_path, marker, set_type)
                os.makedirs(cur_output_path, exist_ok=True)

                indices_to_keep = batch_markers == marker
                marker_cor = batch_corr_data[indices_to_keep]
                logging.info(f"[plot_attn_maps] Extracting {len(marker_cor)} samples of marker {marker}.")
                corr_by_markers[marker] = marker_cor
                if config_plot.PLOT_CORR_SEPERATE_MARKERS:
                    globals()[f"plot_correlation_{attn_method}"](marker_cor, corr_method, config_plot, channel_names=['Nucleus', 'Marker'],  sup_title = f"{marker}_{corr_method}_correlation", output_folder_path=cur_output_path)
            
            if config_plot.PLOT_CORR_ALL_MARKERS:
                globals()[f"plot_correlation_{attn_method}_by_markers"](corr_by_markers, corr_method, config_plot, channel_names=['Nucleus', 'Marker'],  sup_title = f"{corr_method}_correlation", output_folder_path=cur_output_path)
            


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
        keep_paths_df = load_paths_from_npy(keep_samples_dir, set_type)
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

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_attn_maps] unique_batches: {unique_batches}')

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_attn_maps = []
    all_corr_data = []
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels, cur_paths = attn_maps[i], labels[i], paths[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}

        img_path_df = parse_paths(cur_paths)
        logging.info(f'[plot_attn_maps]: for set {set_type}, starting plotting {len(cur_paths)} samples.')
        
        set_attn_maps = []
        set_corr_data = []
        for batch, batch_indexes in __dict_temp.items():
            batch_save_path = os.path.join(output_folder_path, data_config.EXPERIMENT_TYPE, batch)
            logging.info(f"[plot_attn_maps] Saving {len(batch_indexes)} in {batch_save_path}")

            #extract current batch samples
            batch_attn_maps = cur_attn_maps[batch_indexes]
            batch_labels = cur_labels[batch_indexes]
            batch_paths = cur_paths[batch_indexes]

            for index, (sample_attn, label, img_path) in enumerate(zip(batch_attn_maps, batch_labels, batch_paths)):
                # load img details
                path_item = img_path_df.iloc[index]
                img_path, tile, site = Parse_Path_Item(path_item)

                # plot
                marker = str(get_markers_from_labels(label))
                temp_output_folder_path = os.path.join(batch_save_path, marker, set_type, os.path.basename(img_path).split('.npy')[0])
                os.makedirs(temp_output_folder_path, exist_ok=True)
                attn_map, corr_data = __plot_attn(sample_attn, (img_path, site, tile, label), img_shape, config_plot, temp_output_folder_path)
                set_attn_maps.append(attn_map)
                set_corr_data.append(corr_data)
        
        # end of set type
        set_attn_maps = np.stack(set_attn_maps)
        set_corr_data = np.stack(set_corr_data)
        all_attn_maps.append(set_attn_maps)
        all_corr_data.append(set_corr_data)

    # end of samples
    return all_attn_maps, all_corr_data



def __plot_attn(sample_attn: np.ndarray[float], sample_info:tuple, img_shape:tuple, config_plot, output_folder_path:str):
    num_layers, num_heads, num_patches, _ = sample_attn.shape
    patch_dim = int(np.sqrt(num_patches))

    logging.info(f"[plot_attn_maps] dimensions: {num_layers} layers, {num_heads} heads, {num_patches} patches, {img_shape} img_shape")

    attn_method = config_plot.ATTN_METHOD
    attn_maps, corr_data = globals()[f"_plot_attn_map_{attn_method}"](sample_attn, sample_info, patch_dim, img_shape, config_plot, output_folder_path)
    corr_data = globals()[f"parse_corr_data_{attn_method}"](corr_data)
    return attn_maps, corr_data

def _plot_attn_map_all_layers(attn, sample_info, patch_dim, img_shape, config_plot, output_folder_path):
    # Sample Info
    img_path, site, tile, label = sample_info
    marker, nucleus, input_img = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape == img_shape
    logging.info(f"[plot_attn_maps] Sample Info: img_path:{img_path}, site:{site}, tile:{tile}, label:{label}")

    # Attn workflow
    attn = __attn_map_all_layers(attn, attn_layer_dim=0, heads_reduce_fn=REDUCE_HEAD_FUNC_MAP[config_plot.REDUCE_HEAD_FUNC])
    num_layers, _, _= attn.shape #(num_layers, num_patches, num_patches)
    attn_maps_all_layers = []
    corr_data_all_layers = []
    heatmap_colored_all_layers = []
    for layer_idx in range(num_layers):
        layer_attn = attn[layer_idx]
        layer_attn_map, heatmap_colored = __process_attn_map(layer_attn, patch_dim, img_shape, min_attn_threshold=config_plot.MIN_ATTN_THRESHOLD, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP)
        attn_maps_all_layers.append(layer_attn_map.flatten())
        heatmap_colored_all_layers.append(heatmap_colored)
        corr_data = compute_corr_data(layer_attn_map, [nucleus, marker], corr_method = config_plot.CORR_METHOD)
        corr_data_all_layers.append(corr_data)
        if config_plot.SAVE_SEPERATE_LAYERS:
            __create_attn_map_img(layer_attn_map, input_img, heatmap_colored,config_plot, corr_data, sup_title =f"Tile{tile}_Layer{layer_idx}\n{label}",  output_folder_path=output_folder_path)
    
    
    # plot all layers in one figure
    __create_all_layers_attn_map_img(heatmap_colored_all_layers, input_img, config_plot, corr_data_all_layers, sup_title = f"Tile{tile}_All_Layers\n{label}", output_folder_path = output_folder_path)
    attn_maps_all_layers = np.stack(attn_maps_all_layers)
    return attn_maps_all_layers, corr_data_all_layers

def _plot_attn_map_rollout(attn, sample_info, patch_dim, img_shape, config_plot, output_folder_path):
    # Sample Info
    img_path, site, tile, label = sample_info
    marker, nucleus, input_img = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape == img_shape
    logging.info(f"[plot_attn_maps] Sample Info: img_path:{img_path}, site:{site}, tile:{tile}, label:{label}")

    # Attn workflow
    attn = __attn_map_rollout(attn, attn_layer_dim=0, heads_reduce_fn=REDUCE_HEAD_FUNC_MAP[config_plot.REDUCE_HEAD_FUNC])
    attn_map, heatmap_colored = __process_attn_map(attn, patch_dim, img_shape, min_attn_threshold=config_plot.MIN_ATTN_THRESHOLD, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP)
    corr_data = compute_corr_data(attn_map, [nucleus, marker], corr_method = config_plot.CORR_METHOD)
    __create_attn_map_img(attn_map, input_img, heatmap_colored, config_plot, corr_data, corr_method = config_plot.CORR_METHOD, sup_title= f"Tile{tile}\nRollout\n{label}", output_folder_path= output_folder_path)
    return attn_map.flatten(), corr_data



def __create_attn_map_img(attn_map, input_img, heatmap_colored, config_plot, corr_data = None, corr_method = None, sup_title = "Attention Maps", output_folder_path = None):
        """
            Create attention map img with:
                (1) input image 
                (2) attention heatmap
                (3) attention overlay on the input img
            ** save/plot according to config_plot

            parameters:
                attn_map: attention maps values, already in the img shape (H,W), rescale to [0,1]
                input_img: input img with marker and nucleus overlay (3,H,W)
                            ** assuming  Green = nucleus, Blue = marker, Red = zeroed out
                heatmap_colored: attention map colored by heatmap_color (3,H,W)
                config_plot: config with the plotting parameters 
                corr_data: [optional] tuple of corrletion of the attention with the image channels, entropy and corr_method
                sup_title: [optional] main title for the figure
                output_folder_path: [optional] for saving the output fig.

            return:
                fig: matplot fig created. 
        """

        

        alpha = config_plot.ALPHA

        fig, ax = plt.subplots(1, 3, figsize=config_plot.FIG_SIZE)

        if corr_data:
            corrs, entropy = corr_data
            corr_nucleus, corr_marker = corrs[0], corrs[1]
            ax[1].text(0.5, -0.25, f"{corr_method} Correlation (Nucleus): {corr_nucleus:.2f}\n{corr_method} Correlation (Marker): {corr_marker:.2f}\nEntropy: {entropy:.2f}",
                    transform=ax[1].transAxes, ha='center', va='center', fontsize=config_plot.PLOT_TITLE_FONTSIZE, color='black')

        
        ax[0].set_title(f'Input - Marker (blue), Nucleus (green)', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[0].imshow(input_img)
        ax[0].set_axis_off()

        ax[1].set_title(f'Attention Heatmap', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        ax[1].set_axis_off()


        custom_cmap = LinearSegmentedColormap.from_list(
            'attn_overlay_colors',
            ['black', 'white', 'yellow', 'orange', 'red']
        )

        ax[2].set_title('Attention Overlay', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[2].imshow(input_img)  # Show the original image
        ax[2].imshow(attn_map, cmap=custom_cmap, alpha=alpha)  # Overlay attention map transparently
        ax[2].set_axis_off()

        fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE, y=1.1)
        
        if config_plot.SAVE_PLOT and (output_folder_path is not None):
            fig_name  = sup_title.split('\n', 1)[0] #either till the end of the line or the full str
            save_path = os.path.join(output_folder_path, f"{fig_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=config_plot.PLOT_SAVEFIG_DPI)
            plt.close()
        if config_plot.SHOW_PLOT:
            plt.show()

        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")
        return fig


def __create_all_layers_attn_map_img(attn_maps, input_img, config_plot, corr_data_list = None, corr_method = None, sup_title = "Attention Maps", output_folder_path = None):
 

        fig = plt.figure(figsize=config_plot.ALL_LAYERS_FIG_SIZE, facecolor="#d3ebe3")
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.2)

        # Main title
        fig.suptitle(f"{sup_title}\n\n", fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE, fontweight='bold', y=0.98)

        # Overlay section 
        ax_overlay = plt.subplot(gs[0])
        ax_overlay.imshow(input_img)
        ax_overlay.set_title("Input Image", fontsize=config_plot.PLOT_TITLE_FONTSIZE, fontweight='bold', pad=10)
        ax_overlay.axis("off")

        # Attention maps section 
        gs_attn = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[1], wspace=0.3, hspace=0.8)
        fig.text(0.5, 0.68, "Attention Maps", ha='center', va='center', fontsize=config_plot.PLOT_TITLE_FONTSIZE, fontweight='bold')


        for layer_idx, (attn_map, corr_data) in enumerate(zip(attn_maps, corr_data_list)):

            # plot layer attn maps
            ax = plt.subplot(gs_attn[layer_idx])  
            ax.imshow(cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Layer {layer_idx}", fontsize=config_plot.PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.axis("off")

            # Add correlation values below the attention map
            if corr_data:
                corrs, layer_ent = corr_data
                corr_nucleus, corr_marker = corrs[0], corrs[1]
                ax.text(0.5, -0.25, f"{corr_method} Correlation (Nucleus): {corr_nucleus:.2f}\n{corr_method} Correlation (Marker): {corr_marker:.2f}\nEntropy: {layer_ent:.2f}", 
                    transform=ax.transAxes, ha='center', va='center', fontsize=config_plot.PLOT_LAYER_FONTSIZE, color='black')
        
        plt.tight_layout()
        if config_plot.SAVE_PLOT and (output_folder_path is not None):
                fig_name  = sup_title.split('\n', 1)[0] #either till the end of the line or the full str
                save_path = os.path.join(output_folder_path, f"{fig_name}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=config_plot.PLOT_SAVEFIG_DPI)
                plt.close()
        if config_plot.SHOW_PLOT:
                plt.show()

        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")
        return fig
    

def __process_attn_map(attn, patch_dim, img_shape, min_attn_threshold = None, heatmap_color = cv2.COLORMAP_JET):
        """
            process attn map:
                (1) extract cls token attn to other pathces
                (2) normalize
                (3) resize to fit image shape
        
            parameteres:
                attn: attention map of shape (num_patches, num_patches) 
                patch_dim: square root of num_patches to resize the attention vector into a square.
                img_shape: original img_shape (H, W)
                min_attn_threshold [optional]: threshold for the minimum value if attention 
                heatmap_color [optional]: int for cv2 coloring of the attention heatmap

            return:
                attn_resized: float32 attention map heatmap processed into original img_shape (H,W) scaled to [0,1]
                heatmap_colored: uint8 attention map colored by heatmap_color (3,H,W)

        """

        # Take attention from CLS token to all other patches (assumes CLS is first)
        cls_attn = attn[0, 1:] # (num_patches)
        cls_attn_map = cls_attn.reshape(patch_dim, patch_dim) # reshape to square 

        # Normalize attention for heatmap
        attn_heatmap = (cls_attn_map - cls_attn_map.min()) / (cls_attn_map.max() - cls_attn_map.min() + 1e-6)

        # Apply minimum attention threshold (if provided)
        if min_attn_threshold is not None:
            attn_heatmap[attn_heatmap < min_attn_threshold] = 0.0

        # Resize to match image size
        heatmap_resized = Image.fromarray((attn_heatmap * 255).astype(np.uint8)).resize(img_shape, resample=Image.BICUBIC)
        heatmap_resized = np.array(heatmap_resized).astype(np.uint8)
        attn_resized = np.array(heatmap_resized).astype(np.float32) / 255.0  # Rescale to [0,1] float

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, heatmap_color)

        return attn_resized, heatmap_colored

def __attn_map_all_layers(attn, attn_layer_dim=0, heads_reduce_fn:callable=np.mean):
    """ reduce attention across heads according to heads_reduce_fn, for each layer.

    parameteres:
        attn: attention values of shape: ([<num_samples>], num_layers, num_heads, num_patches, num_patches)
        attn_layer_dim: the dimension of the attention layer to iterate the rollout through
                        ** for one sample should be 0 (as it the first dim)
                        ** for multiple samples should be 1 (as num_samples is the 0 dimension)
        heads_reduce_fn: numpy function to reduce the heads layer with (for example: np.mean/np.max/np.min...)

    return:
        reduced_attn: attention map per layer: (num_layers, num_patches, num_patches)
    """
    reduced_attn = heads_reduce_fn(attn, axis=(attn_layer_dim + 1))
    return reduced_attn

def __attn_map_rollout(attn, attn_layer_dim:int=0, heads_reduce_fn:callable=np.mean, start_layer_index:int=0):
    """  aggregates attention maps across multiple layers, using the rollout method:

    parameteres:
        attn: attention values of shape: ([<num_samples>], num_layers, num_heads, num_patches, num_patches)
        attn_layer_dim: the dimension of the attention layer to iterate the rollout through
                        ** for one sample should be 0 (as it the first dim)
                        ** for multiple samples should be 1 (as num_samples is the 0 dimension)
        heads_reduce_fn: numpy function to reduce the heads layer with (for example: np.mean/np.max/np.min...)
        start_layer_index: the index of the layer to start the rollput from.

    returns:
        rollout: attention map for all layers and heads: (num_patches, num_patches)
    """

    # Initialize rollout with identity matrix
    rollout = np.eye(attn.shape[-1]) #(num_patches, num_patches)

    attn = heads_reduce_fn(attn, axis=(attn_layer_dim + 1)) # Average attention across heads (A)

    # Multiply attention maps layer by layer
    for layer_idx in range(start_layer_index,attn.shape[attn_layer_dim]):
        # extract the layer data
        if attn_layer_dim == 0:
            layer_attn = attn[layer_idx]        # layers are in the first dimension
        elif attn_layer_dim == 1:
            layer_attn = attn[:, layer_idx]     # layers are in the second dimension (after batch)
        else:  
            idx = [slice(None)] * attn.ndim
            idx[attn_layer_dim] = layer_idx
            layer_attn = attn[tuple(idx)]

        # rollout mechanism 
        layer_attn += np.eye(layer_attn.shape[-1]) # A + I
        layer_attn /= layer_attn.sum(axis=-1, keepdims=True) # Normalizing A
        rollout = rollout @ layer_attn  # Matrix multiplication
    
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
            batch_size = 200
        generate_attn_maps_with_model(outputs_folder_path, config_path_data, config_path_plot, batch_size)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
