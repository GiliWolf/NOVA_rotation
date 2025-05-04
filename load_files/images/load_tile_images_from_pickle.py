import pickle

import re

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt


def load_tile_image_from_pickle(umaps_dir, path_to_umap):

    # Load data

    with open(umaps_dir + path_to_umap, "rb") as f:

        data = pickle.load(f)

    

    umap_embeddings = data["umap_embeddings"]

    label_data = data["label_data"]

    paths = data['paths']

    config_data = data["config_data"]

    config_plot = data["config_plot"]




    # Regex pattern to extract Batch, Condition, Rep, Raw Image Name, Panel, Cell Line, and Tile

#     pattern = re.compile(r".*/(Batch\d+)/.*/(Untreated|stress)/.*/(rep\d+)_(r\d+c\d+f\d+-ch\d+t\d+)_(panel\w+)_(.+)_processed\.npy/(\d+)")

    pattern = re.compile(

    r".*/[Bb]atch(\d+)/[^/]*/(Untreated|stress)/[^/]*/(rep\d+)_.*_(s\d+)_?(panel\w+)_([^_]+)_processed\.npy/(\d+)"

    )


    # Parsing the paths

    parsed_data = [pattern.match(path).groups() for path in paths if pattern.match(path)]

    

    # Convert metadata to DataFrame

    df = pd.DataFrame(parsed_data, columns=["Batch", "Condition", "Rep", "Image_Name", "Panel", "Cell_Line", "Tile"])

    df['Path'] = [path.split('.npy')[0]+'.npy' for path in paths]




    index = 0 ## For example, take the first tile

    path = df.Path.loc[index]

    tile = int(df.Tile.loc[index])

    image_name = df.Image_Name.loc[index]

    

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


    ax[0].set_title(f'{image_name}/{tile} - Marker', fontsize=11)

    ax[0].imshow(marker, cmap='gray', vmin=0, vmax=1)

    ax[0].set_axis_off()


    ax[1].set_title(f'{image_name}/{tile} - Nucleus', fontsize=11)

    ax[1].imshow(nucleus, cmap='gray', vmin=0, vmax=1)

    ax[1].set_axis_off()


    ax[2].set_title(f'{image_name}/{tile} - Overlay', fontsize=11)

    ax[2].imshow(overlay)

    ax[2].set_axis_off()


    plt.show()

    return df