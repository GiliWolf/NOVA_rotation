import sys
import os
sys.path.insert(0, os.getenv("HOME"))

from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_nparray, load_paths_from_npy, load_labels_from_npy, load_tile, Parse_Path_Item
import matplotlib.pyplot as plt

# input paths
home_dir = os.getenv("HOME")
emb_out_dir = "NOVA_rotation/embeddings/embedding_output"
run_name = "RotationDatasetConfig"
emb_dir = os.path.join(home_dir, emb_out_dir, run_name)
metric = "euclidean"
set_type = "testset"
input_dir = os.path.join(emb_dir, "pairs", metric, set_type)

# output paths
img_outdir = "NOVA_rotation/load_files/images/output"
output_dir = os.path.join(home_dir, img_outdir, f"{run_name}_pairs_{metric}")
os.makedirs(output_dir, exist_ok = True)

# load data
paths_df = load_paths_from_npy(input_dir, set_type)
labels_df = load_labels_from_npy(input_dir, set_type)

# save all images 
for i in range(len(paths_df)):
    path_item = paths_df.iloc[i]
    label = labels_df.iloc[i].full_label
    file_name = path_item.File_Name

    img_path, tile, site = Parse_Path_Item(path_item)
    _, _, input_img = load_tile(img_path, tile)
    shape = input_img.shape

    height, width = input_img.shape[:2]
    dpi = 100

    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi = dpi)
    ax.set_title(f'{file_name}\n{label}', fontsize=12)
    ax.imshow(input_img)
    ax.set_axis_off()

    save_path = os.path.join(output_dir, f"{file_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()




    


