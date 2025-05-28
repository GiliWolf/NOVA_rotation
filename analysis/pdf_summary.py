
"""

create a PDF that summarizes:
    (1) subset:
        -> Tile: <mutual_attr>: <compare_attr_1> VS <compare_attr_2>, <marker>
        -> Input channels: marker+nucleus
        -> Distance matric
        -> N_pairs
        -> # selected samples
        -> distribution fid
        -> UMAP of all embeddings vs subset :TODO- CHECK THIS WORKS "AUTOMATICLY"

"""


def subet_pdf():
    data_config:DatasetConfig = load_config_file(config_path_data, "data")
    data_config.OUTPUTS_FOLDER = output_folder_path


def main():
    pass


if __name__ == "__main__":
    main()