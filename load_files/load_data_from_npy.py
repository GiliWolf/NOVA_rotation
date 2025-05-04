import re
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os


def load_labels_from_npy(embd_dir):
    abels_path = os.path.join(embd_dir, "testset_labels.npy")
    lablels = np.load(labels_path)
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

def load_embeddings_from_npy(embd_dir):
    embeddings_path = os.path.join(embd_dir, "testset.npy")
    embeddings = np.load(embeddings_path)
    embeddings_df = pd.DataFrame(embeddings)

    return embeddings_df

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

    # Regex pattern to extract Batch, Condition, Rep, Raw Image Name, Panel, Cell Line, and Tile
    pattern = re.compile(
    r".*/[Bb]atch(\d+)/[^/]*/(Untreated|stress)/[^/]*/(rep\d+)_.*_(s\d+)_?(panel\w+)_([^_]+)_processed\.npy/(\d+)"
    )

    # Parsing the paths
    parsed_data = [pattern.match(path).groups() for path in paths if pattern.match(path)]
    # Convert metadata to DataFrame
    df = pd.DataFrame(parsed_data, columns=["Batch", "Condition", "Rep", "Site", "Panel", "Cell_Line", "Tile"])
    df['Path'] = [path.split('.npy')[0]+'.npy' for path in paths]

    return df


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

def display_tile(df:pd.DataFrame, index:int, save_dir: str = None):
    path = df.Path.loc[index]
    tile = int(df.Tile.loc[index])
    Site = df.Site.loc[index]

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
    overlay[..., 0] = marker      # Red channel = marker
    overlay[..., 1] = nucleus     # Green channel = nucleus

    # Blue remains 0
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
    display_tile(paths_df, 2, save_dir = output_dir)