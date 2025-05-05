import re
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os


def load_labels_from_npy(embd_dir):
    labels_path = os.path.join(embd_dir, "testset_labels.npy")
    labels = np.load(labels_path, allow_pickle=True)
    labels_df = pd.DataFrame(labels, columns=['full_label'])
    labels_df[['protein', 'condition', 'treatment', 'batch', 'replicate']] = labels_df['full_label'].str.split('_', expand=True)

    return labels_df

def display_labels(df:pd.DataFrame, save_dir: str = None):
    grouped = df.groupby(['protein', 'condition', 'treatment'])
    print("labels_df:")
    print(df.shape)
    print(df.head())

    print("\nlabels groups:")
    print(grouped.size())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"labels.csv")
        df.to_csv(save_path, index=False)

def load_npy_to_df(input_dir, file_name):
    path = os.path.join(input_dir, file_name)
    data = np.load(path, allow_pickle=True)
    df = pd.DataFrame(data)
    return df

def load_npy_to_nparray(input_dir, file_name):
    path = os.path.join(input_dir, file_name)
    data = np.load(path, allow_pickle=True)
    return data


def load_embeddings_from_npy(embd_dir):
    embeddings_path = os.path.join(embd_dir, "testset.npy")
    embeddings = np.load(embeddings_path, allow_pickle=True)
    embeddings_df = pd.DataFrame(embeddings)
    return embeddings_df

def load_attn_maps_from_npy(embd_dir):
    attn_path = os.path.join(embd_dir, "testset_attn.npy")
    attn = np.load(attn_path, allow_pickle=True)
    attn_df = pd.DataFrame(attn)
    return attn_df

def display_embeddings(df:pd.DataFrame, save_dir: str = None):
    print("embeddings_df:")
    print(df.shape)
    print(df.head())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"embeddings.csv")
        df.to_csv(save_path, index=False)


def load_paths_from_npy(embd_dir):

    # Load data
    paths = np.load(os.path.join(embd_dir, "testset_paths.npy"), allow_pickle=True)

    df = parse_paths(paths)

    return df

def parse_paths(paths):
    """
    args:
        paths:  list/array of full path of format: 
                <path>/batchX/SCNA/Condition/Cell_Line/rep{X}_R11_w2confmCherry_s{X}_panel{X}_SCNA_processed.npy/{TILE}
    
    returns:
        df: parsed df woth columns: ["Batch", "Condition", "Rep", "Site", "Panel", "Cell_Line", "Tile", "Path"]
    """


    # Regex pattern to extract Batch, Condition, Rep, Raw Image Name, Panel, Cell Line, and Tile
    pattern = re.compile(
    r".*/[Bb]atch(\d+)/[^/]*/(Untreated|stress)/[^/]*/(rep\d+)_.*_(s\d+)_?(panel\w+)_([^_]+)_processed\.npy/(\d+)"
    )

    # Parsing the paths
    parsed_data = [pattern.match(path).groups() for path in paths if pattern.match(path)]

    if len(parsed_data) != len(paths):
        raise RuntimeError("in parse_paths: not all paths match the regex pattern.")
    # Convert metadata to DataFrame
    df = pd.DataFrame(parsed_data, columns=["Batch", "Condition", "Rep", "Site", "Panel", "Cell_Line", "Tile"])
    df['Path'] = paths
    df['File_Name'] = [os.path.basename(path.split('.npy')[0]) for path in paths]

    return df

def Parse_Path_Item(path_item):
    img_path = str(path_item.Path).split('.npy')[0]+'.npy'
    tile = int(path_item.Tile)
    Site = path_item.Site
    return img_path, tile, Site



def display_paths(df:pd.DataFrame, save_dir: str = None):
    grouped = paths_df.groupby(['Cell_Line', 'Condition', 'Site', 'Rep'])
    print("paths_df:")
    print(df.shape)
    print(df.head())
    print("\npaths groups:")
    print(grouped.size())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"paths.csv")
        df.to_csv(save_path, index=False)


def load_tile(path, tile):
    """
    args:
        path:   path of the original img. should be of size (num_tiles, H, W, num_ch)

    returns:
        marker:     normalized img matrix for the marker (ch0)
        nucleus:    normalized img matrix for the nucleus (ch1)
        overlay:    overlay of the marker on top of the nucleus (Red for marker, Green for nucleus)
    """
    # Load the image
    image = np.load(path)
    site_image = image[tile]
    marker = site_image[:, :, 0]
    nucleus = site_image[:, :, 1]

    # Normalize
    marker = np.clip(marker, 0, 1)
    nucleus = np.clip(nucleus, 0, 1)

    # Create RGB overlay: Red for marker, Green for nucleus
    overlay = np.zeros((*marker.shape, 3))
    overlay[..., 2] = marker      # blue channel = marker
    overlay[..., 1] = nucleus     # Green channel = nucleus

    return marker, nucleus, overlay

def display_tile(Site:str, tile:int, marker:np.array, nucleus:np.array, overlay:np.array, save_dir:str = None):

    # Plot target, nucleus, and overlay
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].set_title(f'{Site}/{tile} - Marker', fontsize=11)
    ax[0].imshow(marker, cmap='gray', vmin=0, vmax=1)
    ax[0].set_axis_off()
    ax[1].set_title(f'{Site}/{tile} - Nucleus', fontsize=11)
    ax[1].imshow(nucleus, cmap='gray', vmin=0, vmax=1)
    ax[1].set_axis_off()
    ax[2].set_title(f'{Site}/{tile} - Overlay', fontsize=11)
    ax[2].imshow(overlay)
    ax[2].set_axis_off()


    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Site}_tile{tile}.png")
        print(f"saving fig on {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()

 


if __name__ == "__main__":
    embd_dir = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/RotationDatasetConfig/pairs"
    output_dir = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/load_files/images/output"

    paths_df = load_paths_from_npy(embd_dir)
    display_paths(paths_df)
    print(paths_df.Path.iloc[0])

    # # load img
    # index = 0
    # path = paths_df.Path.loc[index]
    # tile = int(paths_df.Tile.loc[index])
    # Site = paths_df.Site.loc[index]
    # marker, nucleus, overlay = load_tile(path, tile)
    # display_tile(Site, tile, marker, nucleus, overlay)