

def corr_heatmap(input_folder_path, subset_folder_path ,data_config, attn_config,  output_folder_path = "."):

    ATTN_METHOD:str = attn_config.ATTN_METHOD
    REDUCE_HEAD_FUNC:str = attn_config.REDUCE_HEAD_FUNC
    MIN_ATTN_THRESHOLD:float = attn_config.MIN_ATTN_THRESHOLD
    CORR_METHOD:str = attn_config.CORR_METHOD
    RESAMPLE_METHOD:int = resample_methods[attn_config.RESAMPLE_METHOD]

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
        
    for i, set_type in enumerate(data_set_types):
        # get batche names by all subdirs that starts with "batch"
        temp_input_folder_path = os.path.join(input_folder_path, "raw", data_config.EXPERIMENT_TYPE)
        batches_names = [name for name in os.listdir(temp_input_folder_path)
              if os.path.isdir(os.path.join(temp_input_folder_path, name)) and name.lower().startswith("batch")]
        if not batches_names:
            logging.info(f"Error: No batches dirs found. exiting")
            sys.exit()

        for batch in batches_names:
            for marker in data_config.MARKERS:
                corr_input_folder_path =os.path.join(input_folder_path, "correlations", ATTN_METHOD,  CORR_METHOD ,data_config.EXPERIMENT_TYPE, batch, marker)
                for i in range(int(data_config.NUM_CHANNELS)):
                    corr_data = []
                    corr_path = os.path.join(corr_input_folder_path, f"{set_type}_corrs_ch{i}")
                    data = np.load(corr_path, allow_pickle=True)
                    avg = np.mean(data)
                    corr_data.append(avg)

def main(model_name, attn_config_name = "BaseAttnConfig"):

    # path control
    emb_folder_path = f"./NOVA_rotation/embeddings/embedding_output/{model_name}"
    attn_folder_path  = f"./NOVA_rotation/attention_maps/attention_maps_output/{model_name}"
    output_folder_path = f"./NOVA_rotation/analysis/output/{model_name}"
    os.makedirs(output_folder_path, exist_ok=True)

    # load configs
    config_path_subset = os.path.join("./NOVA_rotation/Configs/manuscript_subset_config", subset_config_name)
    config_path_plot = os.path.join("./NOVA_rotation/Configs/manuscript_attn_map_config", attn_config_name)
    data_config:SubsetConfig = load_config_file(config_path_subset, "data")
    attn_config:AttnConfig = load_config_file(config_path_plot, "data")

    corr_heatmap(attn_folder_path, emb_folder_path, data_config, attn_config, output_folder_path = output_folder_path)




if __name__ == "__main__":

    subset_config_name = sys.argv[1]
    pdf_type = sys.argv[2]
    model_name = sys.argv[3]
    main(subset_config_name, pdf_type, model_name)