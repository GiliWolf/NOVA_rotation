import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
from collections import defaultdict

import logging
from src.models.architectures.NOVA_model import NOVAModel
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from NOVA_rotation.Configs.manuscript_attribution_config import AttributionConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
import torch
from src.common.utils import get_if_exists
from src.datasets.data_loader import get_dataloader
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.label_utils import map_labels, get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels,\
    edit_labels_by_config, get_batches_from_input_folders, get_reps_from_labels, get_conditions_from_labels, get_cell_lines_from_labels
from torch.utils.data import DataLoader
from collections import OrderedDict
from matplotlib import gridspec
from NOVA_rotation.load_files.load_data_from_npy import parse_paths, load_tile, load_paths_from_npy, Parse_Path_Item
import NOVA_rotation.attribution_maps.attribution_maps_utils.attribution_utils as attr_utils
import captum



def generate_attr_maps_with_model(outputs_folder_path:str, config_path_data:str, config_path_attr:str, batch_size:int=700)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_attr:AttributionConfig = load_config_file(config_path_attr, 'base')
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    attr_maps, labels, paths = generate_attr_maps(model, config_data, config_attr, batch_size=batch_size)
    
    # # OUTPUT 
    home_dir = "/home/projects/hornsteinlab/giliwo/NOVA_rotation"
    run_name = os.path.basename(config_path_data)
    outputs_folder_path = os.path.join(home_dir, "attribution_maps/attribution_maps_output", run_name)

    # if config_attr.SAMPLES_PATH is not None:
    #     # filter by path names 
    #     samples_indices = __extract_indices_to_plot(keep_samples_dir=config_attr.SAMPLES_PATH, paths = paths, data_config = config_data)
    #     attr_maps = __extract_samples_to_plot(attr_maps, samples_indices, data_config = config_data)
    #     labels = __extract_samples_to_plot(labels, samples_indices, data_config = config_data)
    #     paths = __extract_samples_to_plot(paths, samples_indices, data_config = config_data)

    # # save the raw attn_map (AFTER FILTERING)
    # save_attr_maps(attr_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "raw"))
 
    # # process and plot attr_maps (AFTER FILTERING)
    # proccesed_attr_maps, corr_data = plot_attr_maps(attr_maps, labels, paths, config_data, config_attr, output_folder_path=os.path.join(outputs_folder_path, "figures", config_attr.ATTN_METHOD))
    
    # # save the processed attn_map (AFTER FILTERING)
    # save_attr_maps(proccesed_attr_maps, labels, paths, config_data, output_folder_path=os.path.join(outputs_folder_path, "processed", config_attr.ATTN_METHOD))

    # # save the correlation data between the attn maps and input images
    # save_corr_data(corr_data, labels, config_data, output_folder_path=os.path.join(outputs_folder_path, "correlations", config_attr.ATTN_METHOD, config_attr.CORR_METHOD))

    # # save summary plots of the correlations
    # if config_attr.PLOT_CORR_SUMMARY:
    #     plot_corr_data(corr_data, labels, config_data, config_attr, output_folder_path=os.path.join(outputs_folder_path, "figures", config_attr.ATTN_METHOD))


def generate_attr_maps(model:NOVAModel, config_data:DatasetConfig, config_attr:AttributionConfig, 
                        batch_size:int=700, num_workers:int=6)->Tuple[List[np.ndarray[torch.Tensor]],
                                                                      List[np.ndarray[str]]]:
    logging.info(f"[generate_attr_maps] Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"[generate_attr_maps] Num GPUs Available: {torch.cuda.device_count()}")

    all_attr_maps, all_labels, all_paths = [], [], []

    train_paths:np.ndarray[str] = model.trainset_paths
    val_paths:np.ndarray[str] = model.valset_paths
    
    full_dataset = DatasetNOVA(config_data)
    full_paths = full_dataset.get_X_paths()
    full_labels = full_dataset.get_y()
    logging.info(f'[generate_attr_maps]: total files in dataset: {full_paths.shape[0]}')
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


        logging.info(f'[generate_attr_maps]: for set {set_type}, there are {new_set_paths.shape} paths and {new_set_labels.shape} labels')
        new_set_dataset = deepcopy(full_dataset)
        new_set_dataset.set_Xy(new_set_paths, new_set_labels)
        
        attr_maps, labels, paths = __generate_attr_maps_with_dataloader(new_set_dataset, model, config_data, config_attr, config_data.IMAGE_SIZE, batch_size, num_workers)
        
        all_attr_maps.append(attr_maps)
        all_labels.append(labels)
        all_paths.append(paths)

    return all_attr_maps, all_labels, all_paths


def save_attr_maps(attr_maps:List[np.ndarray[torch.Tensor]], 
                    labels:List[np.ndarray[str]], paths:List[np.ndarray[str]],
                    data_config:DatasetConfig, output_folder_path:str, attn_method:str = None)->None:
    """
        ** if attn_method is gover, process the attr_maps accordinly before saving 
    """

    unique_batches = get_unique_parts_from_labels(labels[0], get_batches_from_labels, data_config)
    logging.info(f'[save_attr_maps] unique_batches: {unique_batches}')
    
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        cur_attr_maps, cur_labels, cur_paths = attr_maps[i], labels[i], paths[i]
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
                    logging.warning(f"[save_attr_maps] SPLIT_DATA={data_config.SPLIT_DATA} BUT there exists trainset or valset in folder {batch_save_path}!! make sure you don't overwrite the testset!!")
            logging.info(f"[save_attr_maps] Saving {len(batch_indexes)} in {batch_save_path}")

            # process attn maps according to the attn_method
            if attn_method:
                cur_attr_maps = globals()[f"__attn_map_{attn_method}"](cur_attr_maps, attn_layer_dim = 1)

            np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(cur_labels[batch_indexes]))
            np.save(os.path.join(batch_save_path,f'{set_type}_attn.npy'), cur_attr_maps[batch_indexes])
            np.save(os.path.join(batch_save_path,f'{set_type}_paths.npy'), cur_paths[batch_indexes])

            logging.info(f'[save_attr_maps] Finished {set_type} set, saved in {batch_save_path}')



def _group_embeddings_by_labels(embeddings:np.ndarray[torch.Tensor], labels:np.ndarray[str], paths:np.ndarray[str], config_data, attr_config):
    # extract only the relevent parts from the labels to be served as classes
    class_labels = map_labels(labels, attr_config, config_data, config_function_name='CLASS_LABELS_FUNC')

    unique_labels, classes = np.unique(class_labels, return_inverse=True)
    label_to_class = {label: i for i, label in enumerate(unique_labels)}

    # Step 2: Initialize dictionary to hold grouped embeddings and paths
    grouped_embeddings = defaultdict(lambda: {'embeddings': []})

    # Step 3: Group by class label using index-based access
    for i in range(len(embeddings)):
        emb = embeddings[i]
        emb = torch.as_tensor(emb)
        cls_label = classes[i]
        grouped_embeddings[cls_label]['embeddings'].append(emb)

    # Step 4: Convert lists to tensors/arrays
    for cls_label in grouped_embeddings:
        grouped_embeddings[cls_label]['embeddings'] = torch.stack(grouped_embeddings[cls_label]['embeddings'])

    return grouped_embeddings, label_to_class

def _create_prototype(grouped_embeddings):
    prototypes = {
        label: torch.mean(data['embeddings'], dim=0)  # shape: (D,)
        for label, data in grouped_embeddings.items()
    }

    return prototypes

def __generate_attr_maps_with_dataloader(dataset:DatasetNOVA, model:NOVAModel, config_data:DatasetConfig, config_attr:AttributionConfig, input_size:tuple, batch_size:int=700, 
                                          num_workers:int=6)->Tuple[np.ndarray[torch.Tensor], np.ndarray[str]]:
    data_loader = get_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    
    logging.info(f"[generate_attr_maps_with_dataloader] Data loaded: there are {len(dataset)} images.")
    
    embeddings, labels, paths = model.infer(data_loader)

    # group by labels 
    grouped_embeddings, label_to_class = _group_embeddings_by_labels(embeddings, labels, paths, config_data, config_attr)

    # create prototypes 
    prototypes = _create_prototype(grouped_embeddings)


    # get forward func
    vit_model = model.get_model()
    def forward_func(inputs, labels):
        # get embeddings
        emb = vit_model(inputs)

        if emb.shape[0] > len(labels):
            labels = np.tile(labels, config_attr.N_STEPS)

        # match prototype to labels
        matched_prototypes = torch.stack([
        prototypes[label_to_class[label]].to(inputs.device) for label in labels
        ])

        return getattr(attr_utils, f"_{config_attr.FF_METHOD}_forward_func")(emb, matched_prototypes, config_attr)


    # initiate attribution pbject
    attr_object = getattr(captum.attr, config_attr.ATTR_METHOD)(forward_func)

    all_outputs:List[torch.Tensor] = []
    all_labels:np.ndarray[str] = np.array([])
    all_paths:np.ndarray[str] = np.array([])
    
    # Move model to cuda
    vit_model = vit_model.cuda()
    # Set model to eval mode
    vit_model.eval()

    if not config_attr.DISTINCT_BASE_LINE:
        shape = (config_data.NUM_CHANNELS, input_size[0], input_size[1])
        base_line = getattr(attr_utils, f"_{config_attr.BASE_LINE_METHOD}_base_line")(torch.zeros(shape))
        base_line = base_line.unsqueeze(0) 

    
    with torch.no_grad():
        for it, res in enumerate(data_loader):
            torch.cuda.empty_cache()

            logging.info(f"[Inference] Batch number: {it}/{len(data_loader)}")
            X, y, path = res
            X = X.cuda()
                
            # convert from indexesgeneral_base_line to the labels
            labels = data_loader.dataset.id2label(y)

            if config_attr.DISTINCT_BASE_LINE:
                base_line =  getattr(attr_utils, config_attr.BASE_LINE_METHOD)(X)
            
            device = X.device  # get device of input tensor
            base_line = base_line.to(device)  # move baseline to the same device
            labels = map_labels(labels, config_attr, config_data, config_function_name='CLASS_LABELS_FUNC')
            attributions = attr_object.attribute(X,
                                                 baselines=base_line,
                                                 method=config_attr.ATTR_ALGO,
                                                 additional_forward_args = labels,
                                                 n_steps = config_attr.N_STEPS)

            all_outputs.append(attributions)
            all_labels = np.append(all_labels, labels)
            all_paths = np.append(all_paths, path)
        
    all_outputs:np.ndarray[torch.Tensor] = np.vstack(all_outputs)
        

    # embeddings, labels, paths = model.infer(data_loader)
    # forward_func_class = globals()[f"_forward_func_{config_attr.ff_method}"](config_attr)
    # base_line_func = globals()[f"_base_line_{config_attr.base_line_method}"]
    # base_lines
    # captum_model = captum.attr[f"{config_attr.attr_method}"](forward_func_class)
    # attributions, approximation_error = captum_model.attribute(embeddings,
    #                                              baselines=base_line_func(embeddings),
    #                                              method=config.attr_algo)

    # logging.info(f'[generate_attr_maps_with_dataloader] total attr_maps: {attr_maps.shape}')
    
    # return attr_maps, labels, paths

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


if __name__ == "__main__":
    print("Starting generate attribution maps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply outputs folder path, data config and captum attribution config.")
        outputs_folder_path = sys.argv[1]
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        
        config_path_data = sys.argv[2]
        config_path_attr = sys.argv[3]
        if len(sys.argv)==5:
            try:
                batch_size = int(sys.argv[3])
            except ValueError:
                raise ValueError("Invalid batch size, must be integer")
        else:
            batch_size = 8
        generate_attr_maps_with_model(outputs_folder_path, config_path_data, config_path_attr, batch_size)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
