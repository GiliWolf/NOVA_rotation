# conda activate nova_nova
# get embeddings - 
# for all the configs in  - ./NOVA_rotation/Configs/reps_dataset_config
# run: ./NOVA/runnables/generate_embeddings.py $CURR_MODEL <dataset_config>

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

# set model paths
MODEL_DIR="/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local"


# Set configs paths
CONFIG_DIR="./NOVA_rotation/Configs"
EMBEDDING_CONFIG_MODULE="reps_dataset_config"
EMBEDDING_CONFIG_PATH="${CONFIG_DIR}/${EMBEDDING_CONFIG_MODULE}.py"
SUBSET_CONFIG_MODULE="manuscript_subset_config" 
SUBSET_CONFIG_PATH="${CONFIG_DIR}/${SUBSET_CONFIG_MODULE}.py"

MODEL_NAMES=('finetuned_model' 'finetuned_model_classification_with_batch_freeze' 'pretrained_model')

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    CURR_MODEL="$MODEL_DIR/$MODEL_NAME"
    echo "#########################################################################"
    echo "starting run for $MODEL_NAME"
    # ========================
    # 1. Activate base conda env
    # ========================
    echo "------------------------------------------------------------------"
    echo "Activating conda environment: nova_nova"
    module load
    ml miniconda
    conda activate nova_nova
    pid=$!
    wait $pid


    # ========================
    # 2. Generate embeddings
    # ========================
    echo "Generating embeddings..."

    # Get class names from the Python file
    CLASS_NAMES=$(grep -E '^class ' "$EMBEDDING_CONFIG_PATH" | \
                sed -E 's/^class ([A-Za-z0-9_]+)\(.*$/\1/')
    pid=$!
    wait $pid


    # Loop through each class and run the commands
    for class_name in $CLASS_NAMES; do
        CONFIG_REF="${CONFIG_DIR}/${EMBEDDING_CONFIG_MODULE}/${class_name}"

        echo "------------------------------------------------------------------"
        echo "Creating embeddings for class: $class_name"
        python ./NOVA/runnables/generate_embeddings.py $CURR_MODEL $CONFIG_REF
        pid=$!
        wait $pid
        echo "Finished embeddings for class: $class_name"

    done



    # ========================
    # 3. Generate representative subsets
    # ========================
    echo "Generating representative subsets..."

    # Get class names from the python file, excluding BasicSubsetConfig
    CLASS_NAMES=$(grep -E '^class ' "$SUBSET_CONFIG_PATH" | \
                sed -E 's/^class ([A-Za-z0-9_]+)\(.*$/\1/' | \
                grep -v 'BasicSubsetConfig')
    pid=$!
    wait $pid



    # Loop through each class and run the commands
    for class_name in $CLASS_NAMES; do
        CONFIG_REF="${CONFIG_DIR}/${SUBSET_CONFIG_MODULE}/${class_name}"

        echo "------------------------------------------------------------------"
        echo "Extracting subset for config class: $class_name"
        # Subset step
        python ./NOVA_rotation/embeddings/embedding_utils/get_representitve_subset.py \
            "./NOVA_rotation/embeddings/embedding_output/${MODEL_NAME}" "$CONFIG_REF"
        pid=$!
        wait $pid
        echo "Finished subset for config class: $class_name"
        
        echo "------------------------------------------------------------------"
        echo "Generating UMAP for config class: $class_name"
        # UMAP step
        python ./NOVA_rotation/UMAP/UMAP_utils/generate_umaps_and_plot_test.py \
            "./NOVA_rotation/embeddings/embedding_output/${MODEL_NAME}" \
            "./NOVA_rotation/UMAP/UMAP_output/from_embeddings/${MODEL_NAME}" \
            ./NOVA_rotation/Configs/umap_config/UMAP_Subset_Config \
            "$CONFIG_REF"
        pid=$!
        wait $pid
        echo "Finished UMAP for config class: $class_name"
    done

    pid=$!
    wait $pid

    # ========================
    # 4. Create PDF summaries
    # ========================

    # Switch conda env to generate PDFs
    echo "------------------------------------------------------------------"
    echo "Activating conda environment: pdf"
    module load
    ml miniconda
    conda activate pdf
    pid=$!
    wait $pid
    # Loop again to create PDFs
    for class_name in $CLASS_NAMES; do
        CONFIG_REF="${CONFIG_DIR}/${SUBSET_CONFIG_MODULE}/${class_name}"

        echo "------------------------------------------------------------------"
        echo "Generating PDF for: $class_name"
        python ./NOVA_rotation/analysis/utils/pdf_summary.py "$class_name" "subset" "$MODEL_NAME"
        pid=$!
        wait $pid
        echo "Finished PDF for: $class_name"
    done

    echo "------------------------------------------------------------------"
    echo "finished all part 1 for $MODEL_NAME"
    echo "#########################################################################"

    pid=$!
    wait $pid
done