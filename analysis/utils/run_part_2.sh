
# for all subset config in ./NOVA_rotation/Configs/manuscript_subset_config
# run - python ./NOVA_rotation/attention_maps/attention_maps_utils/generate_attention_maps.py $CURR_MODEL <subset_config> ./NOVA_rotation/Configs/manuscript_attn_plot_config/BaseAttnMapPlotConfig

#!/bin/bash
set -e
# set model paths
MODEL_DIR="/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local"
MODEL_NAME="pretrained_model"
CURR_MODEL="$MODEL_DIR/$MODEL_NAME"

# Set configs paths
CONFIG_DIR="./NOVA_rotation/Configs"
EMBEDDING_CONFIG_MODULE="reps_dataset_config"
EMBEDDING_CONFIG_PATH="${CONFIG_DIR}/${EMBEDDING_CONFIG_MODULE}.py"
SUBSET_CONFIG_MODULE="manuscript_subset_config" 
SUBSET_CONFIG_PATH="${CONFIG_DIR}/${SUBSET_CONFIG_MODULE}.py"
ATTN_PLOT_PATH="${CONFIG_DIR}/manuscript_attn_plot_config/BaseAttnMapPlotConfig"
ATTN_CONFIG_PATH="${CONFIG_DIR}/manuscript_attn_map_config/BaseAttnConfig"

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
# 2. Generate attention maps
# ========================
# Get class names from the Python file
CLASS_NAMES=$(grep -E '^class ' "$EMBEDDING_CONFIG_PATH" | \
              sed -E 's/^class ([A-Za-z0-9_]+)\(.*$/\1/')
pid=$!
wait $pid


# Loop through each class and run the commands
for class_name in $CLASS_NAMES; do
    CONFIG_REF="${CONFIG_DIR}/${EMBEDDING_CONFIG_MODULE}/${class_name}"

    echo "------------------------------------------------------------------"
    echo "Creating attention maps for class: $class_name"
    python ./NOVA_rotation/attention_maps/attention_maps_utils/generate_attention_maps.py $CURR_MODEL $CONFIG_REF $ATTN_CONFIG_PATH
    pid=$!
    wait $pid
    echo "Finished attention maps for class: $class_name"

done


# ========================
# 3. Plot attention maps
# ========================
echo "Generating attention maps..."

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
    echo "Generating attention maps for config class: $class_name"

    python ./NOVA_rotation/attention_maps/attention_maps_utils/plot_attention_maps.py $CURR_MODEL $CONFIG_REF $ATTN_CONFIG_PATH $ATTN_PLOT_PATH

    pid=$!
    wait $pid
    echo "Finished attention maps for config class: $class_name"
    
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
    python ./NOVA_rotation/analysis/utils/pdf_summary.py "$class_name" "attn_map" "$MODEL_NAME"
    pid=$!
    wait $pid
    echo "Finished PDF for: $class_name"
done

echo "------------------------------------------------------------------"
echo "finished all part 2"