import os
import sys
import numpy as np

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file, get_if_exists
from src.embeddings.embeddings_utils import load_embeddings
from src.figures.umap_plotting import plot_umap
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig
from src.analysis.analyzer_umap_single_markers import AnalyzerUMAPSingleMarkers
from src.analysis.analyzer_umap_multiple_markers import AnalyzerUMAPMultipleMarkers
from src.analysis.analyzer_umap_multiplex_markers import AnalyzerUMAPMultiplexMarkers
from src.analysis.analyzer_umap import AnalyzerUMAP
from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_nparray
from NOVA_rotation.attention_maps.attention_maps_utils.generate_attention_maps import __extract_indices_to_plot, __extract_samples_to_plot

# Mapping between umap_type and corresponding Analyzer classes and plotting functions
analyzer_mapping = {
    0: (AnalyzerUMAPSingleMarkers, AnalyzerUMAP.UMAPType(0).name),
    1: (AnalyzerUMAPMultipleMarkers, AnalyzerUMAP.UMAPType(1).name),
    2: (AnalyzerUMAPMultiplexMarkers, AnalyzerUMAP.UMAPType(2).name)
}

def generate_umaps(input_folder_path:str, output_folder_path:str, config_path_umap:str, config_path_subset:str)->None:
    config_subset = load_config_file(config_path_subset,'data')
    config_umap:DatasetConfig = load_config_file(config_path_umap, 'data', args = config_subset)
    config_umap.OUTPUTS_FOLDER = output_folder_path
    config_path_plot = f"./NOVA/manuscript/manuscript_plot_config/{config_subset.UMAP_PLOT_CONFIG}"
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')

    # CHANGED: 
    emb_dir = os.path.join(input_folder_path, "embeddings", config_umap.EXPERIMENT_TYPE)
    subset_name:str = os.path.basename(config_path_subset)

    batches_names = [name for name in os.listdir(emb_dir)
              if os.path.isdir(os.path.join(emb_dir, name)) and name.lower().startswith("batch")]
    if not batches_names:
        logging.info(f"Error: No batches dirs found. exiting")
        sys.exit()
    
    for batch in batches_names:
        embeddings_folder = os.path.join(emb_dir, batch)

        umap_idx = get_if_exists(config_plot, 'UMAP_TYPE', None)
        if umap_idx not in analyzer_mapping:
            raise ValueError(f"Invalid UMAP index: {umap_idx}. Must be one of {list(analyzer_mapping.keys())}, and defined in plot config.")
        
        AnalyzerUMAPClass, UMAP_name = analyzer_mapping[umap_idx]
        logging.info(f"[Generate {UMAP_name} UMAP]")

        input_markers = config_umap.MARKERS
        samples_path = os.path.join(input_folder_path,"pairs", "euclidean", config_umap.EXPERIMENT_TYPE, batch)
        for marker in input_markers:
            config_umap.MARKERS = marker

            # Create the analyzer instance
            analyzer_UMAP:AnalyzerUMAP = AnalyzerUMAPClass(config_umap, output_folder_path)
            
            saveroot = os.path.join(output_folder_path, config_umap.EXPERIMENT_TYPE, batch, subset_name, marker)
            os.makedirs(saveroot, exist_ok=True)
            logging.info(f'saveroot: {saveroot}')
            
            # Calculate the UMAP embeddings for all samples
            embeddings, labels, paths  = load_embeddings(input_folder_path, config_umap)
            umap_embeddings, labels, paths, ari_scores = analyzer_UMAP.calculate(embeddings, labels, paths)
            plot_umap(umap_embeddings, labels, config_umap, config_plot, os.path.join(saveroot, "all_samples"), umap_idx, ari_scores, paths)

            # Calculate the UMAP embeddings for subset of the samples, using files from "pairs" by each marker
            temp_samples_path = os.path.join(samples_path, subset_name, marker)
            samples_indices = __extract_indices_to_plot(keep_samples_dirs=[temp_samples_path], paths = [paths], data_config = config_umap)
            marker_umap_embeddings = np.array(__extract_samples_to_plot([umap_embeddings], samples_indices, data_config = config_umap)[0])
            marker_labels = np.array(__extract_samples_to_plot([labels], samples_indices, data_config = config_umap)[0])
            marker_paths = np.array(__extract_samples_to_plot([paths], samples_indices, data_config = config_umap)[0])

            new_ari_scores = analyzer_UMAP._get_only_ari(embeddings = marker_umap_embeddings, labels = marker_labels) 
            # Plot the UMAP
            plot_umap(marker_umap_embeddings, marker_labels, config_umap, config_plot, os.path.join(saveroot, "subset"), umap_idx, new_ari_scores, marker_paths)

        

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        if len(sys.argv) < 5:
            raise ValueError("Invalid arguments. Must supply input_folder_path, output folder path, data config and plot config.")
        input_folder_path  = sys.argv[1]
        output_folder_path = sys.argv[2]
        config_path_umap = sys.argv[3]
        config_path_subset = sys.argv[4]

        generate_umaps(input_folder_path, output_folder_path, config_path_umap, config_path_subset)


    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
