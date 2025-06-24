import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.base_config import BaseConfig
from src.embeddings.embeddings_config import EmbeddingsConfig


class EmbeddingsB9DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        
        super().__init__()
        
        # The path to the root of the processed folder
        self.PROCESSED_FOLDER_ROOT:str = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
        self.SHUFFLE:bool = False   
        #### from embedding config ####
        # The name for the experiment
        self.SETS:List[str] = ['testset']
        ################################

        #### from EmbeddingsB9DatasetConfig ####
        # CHANGED: to get specific sample (pairs)
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

        ################################
        # Which markers to include
        self.MARKERS:List[str]            =  ["G3BP1", "Phalloidin", "TDP43", "FUS", "ANXA11", "FMRP"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["WT", "TDP43", "FUSHomozygous", "FUSRevertant"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["stress", "Untreated"]
        ##


# TDP43: dox/untreated, TDP43
class EmbeddingsdNLSB4DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.SHUFFLE:bool = False   

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4"]]
                        
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'

        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

        self.MARKERS:List[str]            =  ["TDP43B", "DCP1A"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["TDP43", "WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["dox", "Untreated"]

        
      