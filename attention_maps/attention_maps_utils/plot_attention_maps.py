import os
import sys
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
from NOVA_rotation.Configs.subset_config import SubsetConfig
from NOVA_rotation.Configs.attn_config import AttnConfig
from NOVA_rotation.Configs.plot_attn_map_config import PlotAttnMapConfig
from NOVA_rotation.attention_maps.attention_maps_utils.generate_attn_utils import plot_attn_maps, plot_corr_data, save_corr_data



def load_and_plot_attn_maps(outputs_folder_path:str, config_path_data:str, config_path_attn:str, config_path_plot:str):
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_attn:AttnConfig = load_config_file(config_path_attn, "data")
    config_plot:PlotAttnMapConfig = load_config_file(config_path_plot, 'plot')
    config_data.OUTPUTS_FOLDER = outputs_folder_path

    model_name = os.path.basename(outputs_folder_path)

    # outout path
    home_dir = os.path.join(os.getenv("HOME"),"NOVA_rotation")
    outputs_folder_path = os.path.join(home_dir, "attention_maps/attention_maps_output", model_name)

    processed_attn_maps, labels, paths = load_embeddings(outputs_folder_path, config_data, emb_folder_name = "processed")
    processed_attn_maps, labels, paths = [processed_attn_maps], [labels], [paths] #TODO: fix, needed for settypes
    
    # filter the subset - (!) TODO: decide how to filter 
    batches = get_batches_from_input_folders(config_data.INPUT_FOLDERS)
    markers = config_data.MARKERS
    if config_attn.FILTER_BY_PAIRS:
            samples_base_dir = os.path.join(home_dir, "embeddings/embedding_output", model_name, "pairs", config_data.METRIC, config_data.EXPERIMENT_TYPE, str(batches[0]), os.path.basename(config_path_data))
            sample_path_list = []
            for marker in markers:
                sample_path_list.append(os.path.join(samples_base_dir, marker))
            samples_indices = __extract_indices_to_plot(keep_samples_dirs=sample_path_list, paths = paths, data_config = config_data)
            processed_attn_maps = __extract_samples_to_plot(processed_attn_maps, samples_indices, data_config = config_data)
            labels = __extract_samples_to_plot(labels, samples_indices, data_config = config_data)
            paths = __extract_samples_to_plot(paths, samples_indices, data_config = config_data)

    # plot attn_maps (AFTER FILTERING)
    corr_data = plot_attn_maps(processed_attn_maps, labels, paths, config_data, config_attn, config_plot, output_folder_path=os.path.join(outputs_folder_path, "figures", config_attn.ATTN_METHOD))

    # save summary plots of the correlations
    if config_plot.PLOT_CORR_SUMMARY:
        plot_corr_data(corr_data, labels, config_data, config_attn, config_plot, output_folder_path=os.path.join(outputs_folder_path, "correlations", config_attn.ATTN_METHOD, config_attn.CORR_METHOD))



def get_all_subdirs(base_path: os.PathLike) -> list[str]:
    return [
        os.path.join(base_path, name)
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]



if __name__ == "__main__":
    print("Starting generate attention maps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply outputs folder path, data config , attn config, plot config.")
        outputs_folder_path = sys.argv[1]
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        
        config_path_data = sys.argv[2]
        config_path_attn = sys.argv[3]
        config_path_plot = sys.argv[4]

        if len(sys.argv)==6:
            try:
                batch_size = int(sys.argv[5])
            except ValueError:
                raise ValueError("Invalid batch size, must be integer")
        else:
            batch_size = 200
        load_and_plot_attn_maps(outputs_folder_path, config_path_data, config_path_attn, config_path_plot)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
