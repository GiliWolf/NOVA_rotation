#!/bin/bash
# generate_embeddings_with_model(outputs_folder_path:str, config_path_data:str,batch_size:int=700)
# the model should be outputs_folder_path 
# running /home/labs/hornsteinlab/giliwo/NOVA_rotation/attention_maps/test/generate_embeddings.py  - costume 
script_path="$NOVA_HOME/runnables/generate_embeddings_test"
$NOVA_HOME/runnables/run.sh ${script_path} -g -m 20000 -b 10 -a /home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model $NOVA_HOME/src/embeddings/embedding_test_config -q short-gpu -j generate_embeddings