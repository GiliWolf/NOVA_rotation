import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
from NOVA.src.common.utils import load_config_file
from NOVA_rotation.Configs.subset_config import SubsetConfig
from fpdf import FPDF
from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_df, load_npy_to_nparray, load_paths_from_npy, parse_paths
import pandas as pd
from src.datasets.label_utils import get_markers_from_labels
from PIL import Image
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

CHANNELS_DICT = {
    1: "Only Marker",
    2: "Nucleus + Marker",
    3: "Nuckeus + Multiple Markers"
}

resample_methods = {
    Image.NEAREST: 'NEAREST',
    Image.BOX: 'BOX',
    Image.BILINEAR: 'BILINEAR',
    Image.HAMMING: 'HAMMING',
    Image.BICUBIC: 'BICUBIC',
    Image.LANCZOS: 'LANCZOS'
}


def get_title(data_config):
    mutual_param:str = data_config.MUTUAL_ATTR_VAL
    compare_by_attr_list:list = data_config.COMPARE_BY_ATTR_LIST
    compare_param_c1 = compare_by_attr_list[0]
    compare_param_c2 = compare_by_attr_list[1]


    title = f"{mutual_param.upper()}: {compare_param_c1.upper()} VS {compare_param_c2.upper()}"

    return title

def subet_pdf(input_folder_path, umap_folder_path, data_config, title, output_folder_path = "."):
    """
        input_folder_path: path to the embeddings output dir
        umap_folder_path: path to the umap output dir
        data_config: path to the subset config 
        title: title of the PDF 
        output_folder_path: directory to save the files
    """
    metric:str = data_config.METRIC
    num_pairs:int = data_config. NUM_PAIRS 

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        # get batche names by all subdirs that starts with "batch"
        temp_input_folder_path = os.path.join(input_folder_path, "pairs", metric, data_config.EXPERIMENT_TYPE)
        batches_names = [name for name in os.listdir(temp_input_folder_path)
              if os.path.isdir(os.path.join(temp_input_folder_path, name)) and name.lower().startswith("batch")]
        if not batches_names:
            logging.info(f"Error: No batches dirs found. exiting")
            sys.exit()

        for batch in batches_names:
            for marker in data_config.MARKERS:
                temp_input_folder_path = os.path.join(input_folder_path, "pairs", metric, data_config.EXPERIMENT_TYPE, batch, marker)
                paths = load_paths_from_npy(temp_input_folder_path, set_type)
                
                pdf = FPDF()
                pdf.add_page()

                # Title
                pdf.set_font("Arial", size=16)
                pdf.cell(200, 10, txt=title, ln=True, align='C')

                pdf.ln(10)  # Add space

                subset_size = len(paths["Path"].unique())
                # Metadata
                pdf.set_font("Arial", size=12,  style='B')
                metadata = (
                    f"-> Marker:    {marker}\n"
                    f"-> Input channels:    {CHANNELS_DICT[data_config.NUM_CHANNELS]}\n"
                    f"-> Distance metric:   {metric}\n"
                    f"-> N_pairs:   {num_pairs}\n"
                    f"-> Subset Size:   {subset_size}"
                )
                pdf.multi_cell(0, 10, txt=metadata)

                pdf.ln(5)  # Add space

                # Distance distribution figure
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt="Distance Distribution", ln=True)
                dist_fig_path = os.path.join(temp_input_folder_path, f"{metric}_distance_distribution.png")
                pdf.image(dist_fig_path, x=10, y=pdf.get_y(), w=140, h=90)
                pdf.ln(100)  # Adjust if needed based on image height

                # UMAP figures
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt="UMAP Plots", ln=True)
                pdf.ln(5)

                temp_umap_folder_path = os.path.join(umap_folder_path, data_config.EXPERIMENT_TYPE, batch, marker)
                all_samples_path = os.path.join(temp_umap_folder_path, "all_samples", f"{marker}.png")
                subset_path = os.path.join(temp_umap_folder_path, "subset", f"{marker}.png")

                set_adjacent_images(pdf, all_samples_path, subset_path, "All Samples", "Subset")

                # Save the PDF
                temp_output_folder_path = os.path.join(output_folder_path,data_config.EXPERIMENT_TYPE, batch, marker)
                os.makedirs(temp_output_folder_path, exist_ok=True)
                pdf.output(os.path.join(temp_output_folder_path, f"subset_summary.pdf"))

def set_adjacent_images(pdf, path1, path2, title1, title2, img_width = 95, img_height = 75, x_margin = 10,  buffer=10):
        y_position = pdf.get_y()

        # If the next content exceeds page height, add new page
        if y_position + img_height + buffer > 287:
            pdf.add_page()
            y_position = pdf.get_y()

        # Captions
        pdf.set_font("Arial", size=10)
        pdf.text(x_margin + (img_width//2) - 5, y_position, title1)
        pdf.text(x_margin + img_width + (img_width//2) - 5, y_position, title2)

        # Images side-by-side
        pdf.image(path1, x=x_margin, y=y_position+2, w=img_width, h=img_height)
        pdf.image(path2, x=x_margin + img_width + 5, y=y_position+2, w=img_width, h=img_height)



def attn_map_pdf(input_folder_path, subset_folder_path, dataset_name,data_config, plot_config, title, num_examples = 1, output_folder_path = "."):

    ATTN_METHOD:str = plot_config.ATTN_METHOD
    REDUCE_HEAD_FUNC:str = plot_config.REDUCE_HEAD_FUNC
    MIN_ATTN_THRESHOLD:float = plot_config.MIN_ATTN_THRESHOLD
    CORR_METHOD:str = plot_config.CORR_METHOD
    RESAMPLE_METHOD:int = resample_methods[plot_config.RESAMPLE_METHOD]

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
                temp_subset_folder_path = os.path.join(subset_folder_path, dataset_name, "pairs", data_config.METRIC, data_config.EXPERIMENT_TYPE, batch, marker)

                pdf = FPDF()
                pdf.add_page()

                # Title
                pdf.set_font("Arial", size=16)
                pdf.cell(200, 10, txt=title, ln=True, align='C')

                pdf.ln(10)  # Add space

                # Metadata
                pdf.set_font("Arial", size=12, style='B')
                metadata = (
                    f"-> Marker:    {marker}\n"
                    f"-> Input channels:    {CHANNELS_DICT[data_config.NUM_CHANNELS]}\n"
                    f"-> Attention Method:  {ATTN_METHOD}\n"
                    f"-> Head reduction Function:   {REDUCE_HEAD_FUNC}\n"
                    f"-> Minimum Attention Threshold:   {MIN_ATTN_THRESHOLD}\n"
                    f"-> Resampling Method: {RESAMPLE_METHOD}"
                    f"-> Correlation Method:    {CORR_METHOD}"
                )
                pdf.multi_cell(0, 10, txt=metadata)

                # correlation fig
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt="Correlation Plot", ln=True)
                dist_fig_path = os.path.join(fig_input_folder_path, f"{CORR_METHOD}_correlation.png")
                pdf.image(dist_fig_path, x=10, y=pdf.get_y(), w=120, h=140)
                pdf.ln(150)  # Adjust if needed based on image height

                # Attn maps examples
                distance_csv_path = os.path.join(temp_subset_folder_path, f"{data_config.METRIC}_distances.csv")
                dist_df = pd.read_csv(distance_csv_path)
                pdf.ln(10)
                pdf.set_font("Arial", style='B', size=14)
                pdf.cell(200, 10, txt="Attention Maps Examples", ln=True, align='C')
                for pair_type in ["min", "middle", "max"]:
                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt=f"--------------------------------------------------------------{pair_type}--------------------------------------------------------------", ln=True)

                    pair_list = extract_pairs(pair_type, dist_df, num_examples)
                    for dist, path_1, path_2, c1, c2 in pair_list:
                        parsed_paths = parse_paths([path_1, path_2])
                        file_name_1 = parsed_paths.iloc[0].File_Name
                        tile_1 = parsed_paths.iloc[0].Tile
                        file_name_2 = parsed_paths.iloc[1].File_Name
                        tile_2 = parsed_paths.iloc[1].Tile

                        set_adjacent_images(pdf,
                                            os.path.join(fig_input_folder_path, file_name_1, f"Tile{tile_1}.png"), 
                                            os.path.join(fig_input_folder_path, file_name_2, f"Tile{tile_2}.png"),
                                            title1= c1,
                                            title2= c2)
                        pdf.ln(80)

                # Save
                pdf.output(os.path.join(output_folder_path, f"{dataset_name}_attn_summary_{num_examples}.pdf"))

def extract_pairs(pair_type:str, dist_df:pd.DataFrame, num_examples =1):
    prev_path1 = set()
    prev_path2 = set()
    pair_list = []

    filtered_df = dist_df[dist_df["pair_type"] == pair_type]
    for index in range(len(filtered_df)):
        if len(pair_list) >= num_examples:
            break

        pair = filtered_df.iloc[index]

        # extract distance value
        dist = pair.filter(like="_distance").values[0]

        # # Extract values for the first and second "path_" columns
        path_cols = [col for col in pair.index if col.startswith("path_")]
        path1 = pair[path_cols[0]]
        path2 = pair[path_cols[1]]

        if path1 in prev_path1 or path2  in prev_path2:
            continue

        c1 = str(path_cols[0]).split("path_")[1]
        c2 = str(path_cols[1]).split("path_")[1]

        pair_list.append((dist, path1, path2, c1, c2))

        prev_path1.add(path1)
        prev_path2.add(path2)
    
    return pair_list


def main(subset_config_name, pdf_type):

    # path control
    emb_folder_path = "./NOVA_rotation/embeddings/embedding_output"
    umap_folder_path = "./NOVA_rotation/UMAP/UMAP_output/from_embeddings"
    attn_folder_path  = "./NOVA_rotation/attention_maps/attention_maps_output"
    output_folder_path = "./NOVA_rotation/analysis/output"
    os.makedirs(output_folder_path, exist_ok=True)

    # load configs
    config_path_subset = os.path.join("./NOVA_rotation/Configs/manuscript_subset_config", subset_config_name)
    config_path_plot = os.path.join("./NOVA_rotation/Configs/manuscript_attn_plot_config") #NEED TO CHECK
    data_config:SubsetConfig = load_config_file(config_path_subset, "data")
    #plot_config:PlotAttnMapConfig = load_config_file(config_path_plot, "plot")

    # run
    title = get_title(data_config)
    
    if pdf_type == "subset":
        subet_pdf(emb_folder_path, umap_folder_path, data_config, title, output_folder_path)
        print("created subet_pdf.")
    elif pdf_type == "attn_map":
        attn_map_pdf(attn_folder_path, emb_folder_path, dataset_name, data_config, plot_config, title, num_examples = 3, output_folder_path = output_folder_path)
        print("created attn_map_pdf.")
    else:
        print(f"[PDF summary: pdf_type <{pdf_type}> is not supported.")


if __name__ == "__main__":

    subset_config_name = sys.argv[1]
    pdf_type = sys.argv[2]
    main(subset_config_name, pdf_type)