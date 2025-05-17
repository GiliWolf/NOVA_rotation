def main(run_all=False, min_attn_threshold=None):

    input_dir = "./NOVA_rotation/attention_maps/attention_maps_output"
    run_name = "RotationDatasetConfig_Euc_Pairs_all_layers"
    attn_maps_dir = os.path.join(input_dir, run_name, "raw/attn_maps/neurons/batch9")
    save_dir =  os.path.join(input_dir, run_name,"layers_corr")

    img_input_dir = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk"
    path_name_to_plot = "batch9/WT/stress/G3BP1/rep1_R11_w3confCy5_s10_panelA_WT_processed.npy/7"
    path_to_plot = os.path.join(img_input_dir, path_name_to_plot)

    attn_maps = load_npy_to_nparray(attn_maps_dir, "testset_attn.npy") 
    labels = load_labels_from_npy(attn_maps_dir, "testset")
    paths = load_paths_from_npy(attn_maps_dir, "testset")

    init_globals(attn_maps)

    os.makedirs(save_dir, exist_ok=True)

    if run_all:
        run_all_samples(paths, attn_maps,labels, min_attn_threshold, save_dir=save_dir)
    else:
        run_one_sample(paths, path_to_plot, attn_maps,labels, min_attn_threshold, save_dir=save_dir)



if __name__ == "__main__":
    main(run_all=True, min_attn_threshold = 0.5)
    print("Done.")
