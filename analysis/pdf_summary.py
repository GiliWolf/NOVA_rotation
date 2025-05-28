import os
import sys
sys.path.insert(0, os.getenv("HOME"))
from NOVA.src.common.utils import load_config_file
from NOVA_rotation.Configs.subset_config import SubsetConfig
from fpdf import FPDF
from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_df, load_npy_to_nparray, load_paths_from_npy

"""

create a PDF that summarizes:
    (1) subset:
        -> Title: <mutual_attr>: <compare_attr_1> VS <compare_attr_2>, <marker>
        -> Input channels: marker+nucleus
        -> Distance matric
        -> N_pairs
        -> # selected samples
        -> distribution fid
        -> UMAP of all embeddings vs subset


    (2) attention maps:
        -> Title: <mutual_attr>: <compare_attr_1> VS <compare_attr_2>, <marker>
        -> Input channels: marker+nucleus
        -> # selected samples
        -> head_reduction_func
        -> layer_redusction
        -> min_attn-thershold
        -> correlation_method
        -> representive figures of min/max/middle
        -> correlation and entropy fig


"""

def get_title(data_config):
    mutual_attr:str = data_config.MUTUAL_ATTR
    compare_by_attr:str = data_config.COMPARE_BY_ATTR
    compare_by_attr_idx:list = data_config.COMPARE_BY_ATTR_IDX

    mutual_param = getattr(data_config, mutual_attr.upper())[0]
    compare_param_c1 = getattr(data_config,compare_by_attr.upper())[compare_by_attr_idx[0]]
    compare_param_c2 = getattr(data_config,compare_by_attr.upper())[compare_by_attr_idx[1]]

    title = f"{mutual_param.upper()}: {compare_param_c1.upper()} VS {compare_param_c2.upper()}"

    return title

def subet_pdf(input_folder_path, umap_folder_path, data_config, dataset_name, title):
    metric:str = data_config.METRIC
    num_pairs:int = data_config. NUM_PAIRS 

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        # get batche names by all subdirs that starts with "batch"
        temp_input_folder_path = os.path.join(input_folder_path, dataset_name, "pairs", metric, data_config.EXPERIMENT_TYPE)
        batches_names = [name for name in os.listdir(temp_input_folder_path)
              if os.path.isdir(os.path.join(temp_input_folder_path, name)) and name.lower().startswith("batch")]
        if not batches_names:
            logging.info(f"Error: No batches dirs found. exiting")
            sys.exit()

        for batch in batches_names:
            for marker in data_config.MARKERS:
                temp_input_folder_path = os.path.join(input_folder_path, dataset_name, "pairs", metric, data_config.EXPERIMENT_TYPE, batch, marker)
                paths = load_paths_from_npy(temp_input_folder_path, set_type)
                
                pdf = FPDF()
                pdf.add_page()

                # Title
                pdf.set_font("Arial", size=16)
                pdf.cell(200, 10, txt=title, ln=True, align='C')

                pdf.ln(10)  # Add space

                # Metadata
                pdf.set_font("Arial", size=12)
                metadata = (
                    f"-> Input channels: marker + nucleus\n"
                    f"-> Distance metric: {metric}\n"
                    f"-> N_pairs: {num_pairs}\n"
                    f"-> Marker: {marker}\n"
                    f"-> Subset Size: {paths.shape[0]}"
                )
                pdf.multi_cell(0, 10, txt=metadata)

                pdf.ln(5)  # Add space

                # Distance distribution figure
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt="Distance Distribution", ln=True)
                dist_fig_path = os.path.join(temp_input_folder_path, f"{metric}_distance_distribution.png")
                pdf.image(dist_fig_path, x=10, y=pdf.get_y(), w=150)
                pdf.ln(100)  # Adjust if needed based on image height

                # UMAP figures
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt="UMAP Plots", ln=True)
                pdf.ln(5)

                all_samples_path = os.path.join(umap_folder_path, dataset_name, "all_samples", f"{marker}.png")
                subset_path = os.path.join(umap_folder_path, dataset_name, "subset", f"{marker}.png")

                img_width = 90
                img_height = 60
                x_margin = 10
                y_position = pdf.get_y()

                # Captions
                pdf.set_font("Arial", size=10)
                pdf.text(x_margin, y_position - 2, "All Samples")
                pdf.text(x_margin + img_width + 10, y_position - 2, "Subset")

                # Images side-by-side
                pdf.image(all_samples_path, x=x_margin, y=y_position, w=img_width, h=img_height)
                pdf.image(subset_path, x=x_margin + img_width + 10, y=y_position, w=img_width, h=img_height)

                # Save
                pdf.output(f"{dataset_name}_subset_summary.pdf")

def attn_map_pdf(input_folder_path, dataset_name,data_config, plot_config, title):

    ATTN_METHOD:str = plot_config.ATTN_METHOD
    REDUCE_HEAD_FUNC:str = plot_config.REDUCE_HEAD_FUNC
    MIN_ATTN_THRESHOLD:float = plot_config.MIN_ATTN_THRESHOLD
    CORR_METHOD:str = plot_config.CORR_METHOD

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        # get batche names by all subdirs that starts with "batch"
        temp_input_folder_path = os.path.join(input_folder_path, dataset_name, "raw", data_config.EXPERIMENT_TYPE)
        batches_names = [name for name in os.listdir(temp_input_folder_path)
              if os.path.isdir(os.path.join(temp_input_folder_path, name)) and name.lower().startswith("batch")]
        if not batches_names:
            logging.info(f"Error: No batches dirs found. exiting")
            sys.exit()

        for batch in batches_names:
            for marker in data_config.MARKERS:
                fig_input_folder_path =os.path.join(input_folder_path, dataset_name, "figures", ATTN_METHOD,  data_config.EXPERIMENT_TYPE, batch, set_type)
                
                pdf = FPDF()
                pdf.add_page()

                # Title
                pdf.set_font("Arial", size=16)
                pdf.cell(200, 10, txt=title, ln=True, align='C')

                pdf.ln(10)  # Add space

                # Metadata
                pdf.set_font("Arial", size=12)
                metadata = (
                    f"-> Input channels: marker + nucleus\n"
                    f"-> ATTN_METHOD: {ATTN_METHOD}\n"
                    f"-> REDUCE_HEAD_FUNC: {REDUCE_HEAD_FUNC}\n"
                    f"-> MIN_ATTN_THRESHOLD: {MIN_ATTN_THRESHOLD}\n"
                    f"-> CORR_METHOD Size: {CORR_METHOD}"
                )
                pdf.multi_cell(0, 10, txt=metadata)

                # Save
                pdf.output(f"{dataset_name}_attn_summary.pdf")

    


def main():
    dataset_name = "EmbeddingsB9DatasetConfig"
    emb_folder_path = "./NOVA_rotation/embeddings/embedding_output"
    umap_folder_path = "./NOVA_rotation/UMAP/UMAP_output/from_embeddings"
    attn_folder_path  = "./NOVA_rotation/attention_maps/attention_maps_output"

    config_path_data = os.path.join("./NOVA_rotation/Configs/reps_dataset_config", dataset_name)
    config_path_subset = os.path.join("./NOVA_rotation/Configs/manuscript_subset_config", dataset_name)
    config_path_plot = os.path.join("./NOVA_rotation/Configs/manuscript_attn_plot_config", dataset_name)
    data_config:SubsetConfig = load_config_file(config_path_subset, "data", args = load_config_file(config_path_data, "data"))
    plot_config:PlotAttnMapConfig = load_config_file(config_path_plot, "plot")

    title = get_title(data_config)
    subet_pdf(emb_folder_path, umap_folder_path, data_config, dataset_name, title)
    print("created subet_pdf.")
    attn_map_pdf(attn_folder_path, dataset_name,data_config, plot_config, title)
    print("created attn_map_pdf.")


if __name__ == "__main__":
    main()