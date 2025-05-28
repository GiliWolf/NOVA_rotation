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

def generate_umaps(output_folder_path:str, config_path_data:str, config_path_plot:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
    # CHANGED: 
    run_name = os.path.basename(config_path_data)
    input_dir = f"/home/projects/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/{run_name}"
    emb_dir = os.path.join(input_dir,"embeddings", config_data.EXPERIMENT_TYPE)
    batches_names = [name for name in os.listdir(emb_dir)
              if os.path.isdir(os.path.join(emb_dir, name)) and name.lower().startswith("batch")]
    if not batches_names:
        logging.info(f"Error: No batches dirs found. exiting")
        sys.exit()
    
    for batch in batches_names:
        embeddings_folder = os.path.join(emb_dir, batch)
        embeddings, labels, paths = load_npy_to_nparray(embeddings_folder, "testset.npy"), load_npy_to_nparray(embeddings_folder, "testset_labels.npy"), load_npy_to_nparray(embeddings_folder, "testset_paths.npy")

        umap_idx = get_if_exists(config_plot, 'UMAP_TYPE', None)
        if umap_idx not in analyzer_mapping:
            raise ValueError(f"Invalid UMAP index: {umap_idx}. Must be one of {list(analyzer_mapping.keys())}, and defined in plot config.")
        
        AnalyzerUMAPClass, UMAP_name = analyzer_mapping[umap_idx]
        logging.info(f"[Generate {UMAP_name} UMAP]")

        # Create the analyzer instance
        analyzer_UMAP:AnalyzerUMAP = AnalyzerUMAPClass(config_data, output_folder_path)
        
        # CHANGE: 
        umap_outdir = "/home/projects/hornsteinlab/giliwo/NOVA_rotation/UMAP/UMAP_output/from_embeddings"
        saveroot = os.path.join(umap_outdir, run_name)
        os.makedirs(saveroot, exist_ok=True)
        logging.info(f'saveroot: {saveroot}')
        
        # Calculate the UMAP embeddings for all samples
        umap_embeddings, labels, paths, ari_scores = analyzer_UMAP.calculate(embeddings, labels, paths)
        plot_umap(umap_embeddings, labels, config_data, config_plot, os.path.join(saveroot, "all_samples"), umap_idx, ari_scores, paths)

        # Calculate the UMAP embeddings for subset of the samples, using files from "pairs" by each marker
        samples_path = os.path.join(input_dir,"pairs", "euclidean", config_data.EXPERIMENT_TYPE, batch)
        markers_names = [name for name in os.listdir(samples_path)
              if os.path.isdir(os.path.join(samples_path, name))]
        if not markers_names:
            logging.info(f"Error: No markers dirs found. continuing")
            continue
        
        for marker in markers_names:
            temp_samples_path = os.path.join(samples_path, marker)
            samples_indices = __extract_indices_to_plot(keep_samples_dir=temp_samples_path, paths = [paths], data_config = config_data)
            marker_umap_embeddings = np.array(__extract_samples_to_plot([umap_embeddings], samples_indices, data_config = config_data)[0])
            marker_labels = np.array(__extract_samples_to_plot([labels], samples_indices, data_config = config_data)[0])
            marker_paths = np.array(__extract_samples_to_plot([paths], samples_indices, data_config = config_data)[0])

            new_ari_scores = analyzer_UMAP._get_only_ari(embeddings = marker_umap_embeddings, labels = marker_labels) 
            # Plot the UMAP
            plot_umap(marker_umap_embeddings, marker_labels, config_data, config_plot, os.path.join(saveroot, "subset"), umap_idx, new_ari_scores, marker_paths)

        

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and plot config.")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_plot = sys.argv[3]

        generate_umaps(output_folder_path, config_path_data, config_path_plot)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
