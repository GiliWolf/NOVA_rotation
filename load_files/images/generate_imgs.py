from load_data_from_npy import load_paths_from_npy, parse_paths, load_tile

input_dir = "/home/labs/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/RotationDatasetConfig/pairs"

paths = load_paths_from_npy(input_dir)
paths = parse_paths(paths)

for path in paths:
    load_tile()