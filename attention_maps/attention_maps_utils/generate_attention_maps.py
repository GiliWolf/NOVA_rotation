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
    
    # load model
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)
    model_name = os.path.basename(outputs_folder_path)

    # outout path
    home_dir = os.path.join(os.getenv("HOME"),"NOVA_rotation")
    outputs_folder_path = os.path.join(home_dir, "attention_maps/attention_maps_output", model_name)

    if os.path.exists(os.path.join(outputs_folder_path, "raw", config_data.EXPERIMENT_TYPE)):
        attn_maps, labels, paths = load_embeddings(outputs_folder_path, config_data, emb_folder_name = "raw")
        attn_maps, labels, paths = [attn_maps], [labels], [paths] #TODO: fix, needed for settypes
    else:
        attn_maps, labels, paths = generate_attn_maps(model, config_data, batch_size=batch_size)#TODO: add option to load attention maps
        # save the raw attn_map (BEFORE FILTERING)
        save_attn_maps(attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "raw"))

    if os.path.exists(os.path.join(outputs_folder_path, "processed", config_data.EXPERIMENT_TYPE)):
        # load attn maps instead of generating them 
        processed_attn_maps, labels, paths = load_embeddings(outputs_folder_path, config_data, emb_folder_name = "processed")
        processed_attn_maps, labels, paths = [processed_attn_maps], [labels], [paths] #TODO: fix, needed for settypes
    else:
        # process the raw attn_map and save (BEFORE FILTERING) 
        processed_attn_maps = process_attn_maps(attn_maps, labels, config_data, config_plot)
        save_attn_maps(processed_attn_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "processed"))

    
    batches = get_batches_from_input_folders(config_data.INPUT_FOLDERS)
    markers = config_data.MARKERS
    # filter the subset
    if config_plot.FILTER_BY_PAIRS: #TODO: decide how to filter 
            samples_base_dir = os.path.join(home_dir, "embeddings/embedding_output", model_name, "pairs", config_data.METRIC, config_data.EXPERIMENT_TYPE, str(batches[0]), os.path.basename(config_path_data))
            sample_path_list = []
            for marker in markers:
                sample_path_list.append(os.path.join(samples_base_dir, marker))
            samples_indices = __extract_indices_to_plot(keep_samples_dirs=sample_path_list, paths = paths, data_config = config_data)
            processed_attn_maps = __extract_samples_to_plot(processed_attn_maps, samples_indices, data_config = config_data)
            labels = __extract_samples_to_plot(labels, samples_indices, data_config = config_data)
            paths = __extract_samples_to_plot(paths, samples_indices, data_config = config_data)

    # plot attn_maps (AFTER FILTERING)
    corr_data = plot_attn_maps(processed_attn_maps, labels, paths, config_data, config_plot, output_folder_path=os.path.join(outputs_folder_path, "figures", config_plot.ATTN_METHOD))

    # save summary plots of the correlations
    if config_plot.PLOT_CORR_SUMMARY:
        plot_corr_data(corr_data, labels, config_data, config_plot, output_folder_path=os.path.join(outputs_folder_path, "correlations", config_plot.ATTN_METHOD, config_plot.CORR_METHOD))

def get_all_subdirs(base_path: os.PathLike) -> list[str]:
    return [
        os.path.join(base_path, name)
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]
    
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

            # extract current markers 
            batch_markers = np.array(get_markers_from_labels(batch_labels))
            marker_names = np.unique(batch_markers)

            # iterate each marker and plot/ save 
            corr_by_markers = {}
            for marker in marker_names:
                marker_save_path = os.path.join(batch_save_path, marker)
                os.makedirs(marker_save_path, exist_ok=True)
                indices_to_keep = (batch_markers == marker)
                marker_cor = batch_corr_data[indices_to_keep]
                logging.info(f"[plot_attn_maps] Extracting {len(marker_cor)} samples of marker {marker}.")
                corr_by_markers[marker] = marker_cor

                # save corr data by seperate markers
                if config_plot.SAVE_CORR_SEPERATE_MARKERS:
                    for ch_index in range(cur_corr_data.shape[1] - 1): # num channels 
                        corr_list = marker_cor[:, ch_index]
                        np.save(os.path.join(marker_save_path,f'{set_type}_corrs_ch{ch_index}.npy'), np.array(corr_list))
                    ent_list = marker_cor[:, -1]
                    np.save(os.path.join(marker_save_path,f'{set_type}_ent.npy'), np.array(ent_list))
                    logging.info(f'[save_corr_data] saved in {marker_save_path}')
                # create correlation plots by seperate markers
                if config_plot.PLOT_CORR_SEPERATE_MARKERS:
                    globals()[f"plot_correlation_{attn_method}"](marker_cor, corr_method, config_plot, channel_names=['Nucleus', 'Marker'],  
                                                                sup_title = f"{marker}_{corr_method}_correlation", output_folder_path=marker_save_path)
            
            # save corr data for batch 
            if config_plot.SAVE_CORR_ALL_MARKERS:
                for ch_index in range(cur_corr_data.shape[1] - 1): # num channels 
                    corr_list = batch_corr_data[:, ch_index]
                    np.save(os.path.join(batch_save_path,f'{set_type}_corrs_ch{ch_index}.npy'), np.array(corr_list))
                ent_list = batch_corr_data[:, -1]
                np.save(os.path.join(batch_save_path,f'{set_type}_ent.npy'), np.array(ent_list))
                logging.info(f'[save_corr_data] saved in {batch_save_path}')

            # plot corr for all markers 
            if config_plot.PLOT_CORR_ALL_MARKERS:
                globals()[f"plot_correlation_{attn_method}_by_markers"](corr_by_markers, corr_method, config_plot, channel_names=['Nucleus', 'Marker'],  
                                                                        sup_title = f"{corr_method}_correlation", output_folder_path=batch_save_path)
            


def __generate_attn_maps_with_dataloader(dataset:DatasetNOVA, model:NOVAModel, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    logging.info(f"[generate_attn_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    attn_maps, labels, paths = model.gen_attn_maps(data_loader) # (num_samples, num_layers, num_heads, num_patches, num_patches)
    logging.info(f'[generate_attn_maps_with_dataloader] total attn_maps: {attn_maps.shape}')
    
    return attn_maps, labels, paths



def __extract_indices_to_plot(keep_samples_dirs: list[str], paths: np.ndarray, data_config: DatasetConfig):
    """
    Extract indices to plot from a list of keep_samples_dirs.
    For each dataset split (train/val/test or test), collects indices from all directories and concatenates them.

    Parameters:
        keep_samples_dirs: list of directories containing .npy files of sample paths
        paths: np.ndarray of path arrays, one per dataset split
        data_config: dataset configuration object

    Returns:
        all_samples_indices: list of lists, where each sublist contains indices for one dataset split
    """
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset', 'valset', 'testset']
    else:
        data_set_types = ['testset']

    all_samples_indices = []

    for i, set_type in enumerate(data_set_types):
        cur_paths = paths[i]
        paths_df = parse_paths(cur_paths)

        # Accumulate all keep_paths from all dirs
        combined_keep_paths = set()
        for dir_path in keep_samples_dirs:
            keep_paths_df = load_paths_from_npy(dir_path, set_type)
            combined_keep_paths.update(keep_paths_df["Path"].tolist())

        # Get indices of matching paths
        samples_indices = paths_df[paths_df["Path"].isin(combined_keep_paths)].index.tolist()
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


def plot_attn_maps(processed_attn_maps: np.ndarray[float], labels: np.ndarray[str], paths: np.ndarray[str], data_config: DatasetConfig, config_plot, output_folder_path: str):
    """


    """

    os.makedirs(output_folder_path, exist_ok=True)
    img_shape = data_config.IMAGE_SIZE # suppose to be square (100, 100)

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_attn_maps] unique_batches: {unique_batches}')

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_corr_data = []
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels, cur_paths = processed_attn_maps[i], labels[i], paths[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}

        img_path_df = parse_paths(cur_paths)
        logging.info(f'[plot_attn_maps]: for set {set_type}, starting plotting {len(cur_paths)} samples.')
        
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
                corr_data = __plot_attn(sample_attn, (img_path, site, tile, label), img_shape, config_plot, temp_output_folder_path)
                set_corr_data.append(corr_data)
        
        # end of set type
        set_corr_data = np.stack(set_corr_data)
        all_corr_data.append(set_corr_data)

    # end of samples
    return all_corr_data

## 

def process_attn_maps(attn_maps: np.ndarray[float], labels: np.ndarray[str], data_config: DatasetConfig, config_plot):
    """
    Plot attention maps for a specific sample.

    Parameters
    ----------
    attn_maps :         np.ndarray of shape (num_samples, num_layers, num_heads, num_patches, num_patches)
                        The attention maps for all samples. Each map shows how patches attend to each other across layers and heads.
    labels :            np.ndarray of shape (num_samples,)
                        Class labels for each sample (used for labeling plots).
    data_config:        DatasetConfig 
    config_plot:        PlotConfig

    algo:
        (1) process attention using attn_method and head_reduction_method
        (2) normalize and apply thershold if specified 
    return:
        processed attention maps 

    """

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[process_attn_maps] unique_batches: {unique_batches}')

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_attn_maps = []
    all_corr_data = []
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels = attn_maps[i], labels[i]
        batch_of_label = get_batches_from_labels(cur_labels, data_config)
        __dict_temp = {batch: np.where(batch_of_label==batch)[0] for batch in unique_batches}

        logging.info(f'[process_attn_maps]: for set {set_type}, starting proceesing {len(cur_labels)} samples.')
        
        set_attn_maps = []
        set_corr_data = []
        for batch, batch_indexes in __dict_temp.items():
            #extract current batch samples
            batch_attn_maps = cur_attn_maps[batch_indexes]

            for index, (sample_attn) in enumerate(batch_attn_maps):
                processed_attn_map = __process_attn(sample_attn, config_plot)
                set_attn_maps.append(processed_attn_map)
        
        # end of set type
        set_attn_maps = np.stack(set_attn_maps)
        all_attn_maps.append(set_attn_maps)

    # end of samples
    return all_attn_maps

def __process_attn(sample_attn: np.ndarray[float], config_plot):
    processed_attn_map = globals()[f"_process_attn_map_{config_plot.ATTN_METHOD}"](sample_attn, config_plot)
    return processed_attn_map

def _process_attn_map_rollout(attn, config_plot):
    # Attn workflow
    attn = __attn_map_rollout(attn, attn_layer_dim=0, heads_reduce_fn=REDUCE_HEAD_FUNC_MAP[config_plot.REDUCE_HEAD_FUNC])
    processed_attn_map = __process_attn_map(attn, min_attn_threshold=config_plot.MIN_ATTN_THRESHOLD)
    return processed_attn_map

def _process_attn_map_all_layers(attn, config_plot):
    # Attn workflow
    attn = __attn_map_all_layers(attn, attn_layer_dim=0, heads_reduce_fn=REDUCE_HEAD_FUNC_MAP[config_plot.REDUCE_HEAD_FUNC])
    num_layers, _, _= attn.shape #(num_layers, num_patches, num_patches)
    attn_maps_all_layers = []

    for layer_idx in range(num_layers):
        layer_attn = attn[layer_idx]
        processed_layer_attn_map = __process_attn_map(layer_attn, min_attn_threshold=config_plot.MIN_ATTN_THRESHOLD)
        attn_maps_all_layers.append(processed_layer_attn_map)
       
    attn_maps_all_layers = np.stack(attn_maps_all_layers)
    return attn_maps_all_layers

####

def __plot_attn(proccessed_sample_attn: np.ndarray[float], sample_info:tuple, img_shape:tuple, config_plot, output_folder_path:str):
    num_patches = proccessed_sample_attn.shape[-1]
    patch_dim = int(np.sqrt(num_patches))

    logging.info(f"[plot_attn_maps] {num_patches} patches, {img_shape} img_shape")

    attn_method = config_plot.ATTN_METHOD
    corr_data = globals()[f"_plot_attn_map_{attn_method}"](proccessed_sample_attn, sample_info, patch_dim, img_shape, config_plot, output_folder_path)
    corr_data = globals()[f"parse_corr_data_{attn_method}"](corr_data)
    return corr_data

def _plot_attn_map_all_layers(processed_attn_map, sample_info, patch_dim, img_shape, config_plot, output_folder_path):
    # Sample Info
    img_path, site, tile, label = sample_info
    marker, nucleus, input_img = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape == img_shape
    logging.info(f"[plot_attn_maps] Sample Info: img_path:{img_path}, site:{site}, tile:{tile}, label:{label}")

    # Attn workflow
    num_layers, _, _= attn.shape #(num_layers, num_patches, num_patches)
    corr_data_all_layers = []
    heatmap_colored_all_layers = []
    for layer_idx in range(num_layers):
        layer_attn = processed_attn_map[layer_idx]
        heatmap_colored = __color_heatmap_attn_map(layer_attn, patch_dim, img_shape, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP, resample_method=config_plot.RESAMPLE_METHOD)
        corr_data = compute_corr_data(layer_attn, [nucleus, marker], corr_method = config_plot.CORR_METHOD)
        corr_data_all_layers.append(corr_data)
        heatmap_colored_all_layers.append(heatmap_colored)
        if config_plot.SAVE_SEPERATE_LAYERS:
            layer_attn =__resize_attn_map(layer_attn)
            __create_attn_map_img(layer_attn, input_img, heatmap_colored,config_plot, corr_data, sup_title =f"Tile{tile}_Layer{layer_idx}\n{label}",  output_folder_path=output_folder_path)
    
    # plot all layers in one figure
    __create_all_layers_attn_map_img(heatmap_colored_all_layers, input_img, config_plot, corr_data_all_layers, sup_title = f"Tile{tile}_All_Layers\n{label}", output_folder_path = output_folder_path)
    return corr_data_all_layers

def _plot_attn_map_rollout(processed_attn_map, sample_info, patch_dim, img_shape, config_plot, output_folder_path):
    # Sample Info
    img_path, site, tile, label = sample_info
    marker, nucleus, input_img = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape == img_shape
    logging.info(f"[plot_attn_maps] Sample Info: img_path:{img_path}, site:{site}, tile:{tile}, label:{label}")

    # Attn workflow
    heatmap_colored = __color_heatmap_attn_map(processed_attn_map, patch_dim, img_shape, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP, resample_method=config_plot.RESAMPLE_METHOD)
    processed_attn_map =__resize_attn_map(processed_attn_map, patch_dim, img_shape, resample_method=config_plot.RESAMPLE_METHOD)
    corr_data = compute_corr_data(processed_attn_map, [nucleus, marker], corr_method = config_plot.CORR_METHOD)
    __create_attn_map_img(processed_attn_map, input_img, heatmap_colored, config_plot, corr_data, corr_method = config_plot.CORR_METHOD, sup_title= f"Tile{tile}\nRollout\n{label}", output_folder_path= output_folder_path)
    return corr_data

##

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
    



def __process_attn_map(attn, min_attn_threshold=None):
        """
        Process the attention from the attention matrix:
            (1) Extract CLS token attention
            (2) Normalize
            (3) Apply threshold
            (4) Scale to [0,1]

        Parameters:
            attn: attention matrix of shape (num_patches, num_patches)
            min_attn_threshold [optional]: minimum value threshold

        Returns:
            processed_attn: float32 attention vector of shape (num_patches - 1,), scaled to [0,1]
        """
        cls_attn = attn[0, 1:]  # shape: (num_patches - 1,)

        # Normalize
        attn_min = cls_attn.min()
        attn_max = cls_attn.max()
        processed_attn = (cls_attn - attn_min) / (attn_max - attn_min + 1e-6)

        # Apply optional threshold
        if min_attn_threshold is not None:
            processed_attn[processed_attn < min_attn_threshold] = 0.0

        return processed_attn.astype(np.float32)

def __resize_attn_map(processed_attn, patch_dim, img_shape, resample_method=Image.BICUBIC):
    """
    Resize flattened attention map to original image shape:
        (1) Reshape to (patch_dim, patch_dim)
        (2) Resample to image shape (interpolate)

    Parameters:
        processed_attn: float32 vector of shape (patch_dim * patch_dim,), scaled to [0,1]
        patch_dim: int, dimension of square patch grid
        img_shape: (H, W) tuple of the original image
        resample_method: PIL.Image resampling method (default: Image.BICUBIC)

    Returns:
        attn_resized: float32 attention map of shape (H, W), scaled to [0,1]
    """
    attn_square = processed_attn.reshape(patch_dim, patch_dim)
    attn_image = Image.fromarray((attn_square * 255).astype(np.uint8))
    attn_resized = attn_image.resize(img_shape[::-1], resample=resample_method)  # PIL uses (W, H)
    attn_resized = np.array(attn_resized).astype(np.float32) / 255.0
    return attn_resized


def __color_heatmap_attn_map(processed_attn, patch_dim, img_shape, heatmap_color=cv2.COLORMAP_JET, resample_method=Image.BICUBIC):
    """
    Create attention heatmap:
        (1) Reshape to (patch_dim, patch_dim)
        (2) Resize to image shape
        (3) Apply colormap

    Parameters:
        processed_attn: float32 vector of shape (patch_dim * patch_dim,), scaled to [0,1]
        patch_dim: dimension of square patch grid
        img_shape: (H, W) of the original image
        heatmap_color: OpenCV colormap type (default: cv2.COLORMAP_JET)

    Returns:
        heatmap_colored: uint8 colored attention heatmap of shape (H, W, 3)
    """
    attn_square = processed_attn.reshape(patch_dim, patch_dim)
    heatmap_uint8 = (attn_square * 255).astype(np.uint8)
    heatmap_resized = Image.fromarray(heatmap_uint8).resize(img_shape, resample=resample_method)
    heatmap_resized = np.array(heatmap_resized).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, heatmap_color)
    return heatmap_colored


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
