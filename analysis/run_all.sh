# conda activate nova_nova
# get embeddings - 
# for all the configs in  - ./NOVA_rotation/Configs/reps_dataset_config
# run: ./NOVA/runnables/generate_embeddings.py $MODEL_PATH <dataset_config>

# get subset - 
# for all the configs in - /home/projects/hornsteinlab/giliwo/NOVA_rotation/Configs/manuscript_subset_config except BasicSubsetConfig
# run: ./NOVA_rotation/embeddings/embedding_utils/get_representitve_subset.py ./NOVA_rotation/embeddings/embedding_output <subset_config>

# run UMAP  - 
# for all the configs in - /home/projects/hornsteinlab/giliwo/NOVA_rotation/Configs/manuscript_subset_config except BasicSubsetConfig
# run ./NOVA_rotation/UMAP/UMAP_utils/generate_umaps_and_plot_test.py ./NOVA_rotation/UMAP/UMAP_output/from_embeddings ./NOVA_rotation/Configs/umap_config/UMAP_Subset_Config <subset_config>

# create subset PDF - 
# conda activate PDF 
# for all the configs in - /home/projects/hornsteinlab/giliwo/NOVA_rotation/Configs/manuscript_subset_config except BasicSubsetConfig
# run ./NOVA_rotation/analysis/pdf_summary.py <subset_config>(class name) subset


#!/bin/bash

# Set configs paths
CONFIG_DIR="./NOVA_rotation/Configs"
EMBEDDING_CONFIG_MODULE="reps_dataset_config"
EMBEDDING_CONFIG_PATH="${CONFIG_DIR}/${EMBEDDING_CONFIG_MODULE}.py"
SUBSET_CONFIG_MODULE="manuscript_subset_config" 
SUBSET_CONFIG_FILE="${CONFIG_DIR}/${SUBSET_CONFIG_MODULE}.py"


# ========================
# 1. Activate base conda env
# ========================
module load
ml miniconda
echo "Activating conda environment: nova_nova"
conda activate nova_nova

wait

# ========================
# 2. Generate embeddings
# ========================
echo "Generating embeddings..."

# Get class names from the Python file, excluding BasicSubsetConfig
CLASS_NAMES=$(grep -E '^class ' "$EMBEDDING_CONFIG_PATH" | \
              sed -E 's/^class ([A-Za-z0-9_]+)\(.*$/\1/')

# Loop through each class and run the commands
for class_name in $CLASS_NAMES; do
    CONFIG_REF="${CONFIG_DIR}/${EMBEDDING_CONFIG_MODULE}/${class_name}"

    echo "Creating embeddings for class: $class_name"
    echo python ./NOVA/runnables/generate_embeddings.py $MODEL_PATH $CONFIG_REF
done

wait

# ========================
# 3. Generate representative subsets
# ========================
echo "Generating representative subsets..."

# Get class names from the echo python file, excluding BasicSubsetConfig
CLASS_NAMES=$(grep -E '^class ' "$SUBSET_CONFIG_FILE" | \
              grep -v 'BasicSubsetConfig' | \
              sed -E 's/^class ([A-Za-z0-9_]+)\(.*$/\1/')

# Loop through each class and run the commands
for class_name in $CLASS_NAMES; do
    CONFIG_REF="${CONFIG_DIR}/${SUBSET_CONFIG_MODULE}/${class_name}"

    echo "Extracting subset for config class: $class_name"
    # Subset step
    echo python ./NOVA_rotation/embeddings/embedding_utils/get_representitve_subset.py \
        ./NOVA_rotation/embeddings/embedding_output "$CONFIG_REF"

    echo "Generating UMAP for config class: $class_name"
    # UMAP step
    echo python ./NOVA_rotation/UMAP/UMAP_utils/generate_umaps_and_plot_test.py \
        ./NOVA_rotation/UMAP/UMAP_output/from_embeddings \
        ./NOVA_rotation/Configs/umap_config/UMAP_Subset_Config \
        "$CONFIG_REF"
done

wait

# ========================
# 4. Create PDF summaries
# ========================

# Switch conda env to generate PDFs
conda activate pdf

# Loop again to create PDFs
for class_name in $CLASS_NAMES; do
    CONFIG_REF="${CONFIG_DIR}/${SUBSET_CONFIG_MODULE}/${class_name}"

    echo "Generating PDF for: $class_name"
    echo python ./NOVA_rotation/analysis/pdf_summary.py "$class_name" "subset"
done
