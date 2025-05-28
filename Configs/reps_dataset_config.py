import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.base_config import BaseConfig
from src.embeddings.embeddings_config import EmbeddingsConfig


# WT: untreated/stress, G3BP1
class EmbeddingsB9DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        
        super().__init__()
        
        # The path to the root of the processed folder
        self.PROCESSED_FOLDER_ROOT:str = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
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
        ## CHANGED: verything except "G3BP1"    
        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "FMRP", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "FUS", "GM130", 
            "LAMP1", "NCL", "NONO", "Phalloidin", "PSD95", "SCNA", "TDP43", "TOMM20"
        ] 
        ##
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

        ################################

        #### from DatasetConfig ####
        ## CHANGED: get only WT-G3BP1-stress/untreated.
        # Which markers to include
        self.MARKERS:List[str]            =  ["G3BP1"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["stress", "Untreated"]
        ##

# WT: untreated/stress, Phalloidin
class EmbeddingsB9DatasetConfig_Phalloidin(EmbeddingsConfig):
    def __init__(self):
        
        super().__init__()
        
        # The path to the root of the processed folder
        self.PROCESSED_FOLDER_ROOT:str = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
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
        ## CHANGED: verything except "Phalloidin"    
        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "FMRP", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "FUS", "GM130", 
            "LAMP1", "NCL", "NONO", "G3BP1", "PSD95", "SCNA", "TDP43", "TOMM20"
        ] 
        ##
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

        ################################

        #### from DatasetConfig ####
        ## CHANGED: get only WT-G3BP1-stress/untreated.
        # Which markers to include
        self.MARKERS:List[str]            =  ["Phalloidin"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["stress", "Untreated"]


# FUSHomozygous/WT: Untreated, FUS
class EmbeddingsB9DatasetConfig_FUS(EmbeddingsConfig):
    def __init__(self):
        
        super().__init__()
        
        # The path to the root of the processed folder
        self.PROCESSED_FOLDER_ROOT:str = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
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
        ## CHANGED: verything except "FUS"    
        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "FMRP", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "G3BP1", "GM130", 
            "LAMP1", "NCL", "NONO", "Phalloidin", "PSD95", "SCNA", "TDP43", "TOMM20"
        ] 
        ##
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

        ################################

        #### from DatasetConfig ####
        ## CHANGED: get only WT-G3BP1-stress/untreated.
        # Which markers to include
        self.MARKERS:List[str]            =  ["FUS"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["FUSHomozygous", "WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["Untreated"]
        ##


# TDP43: dox/untreated, TDP43
class EmbeddingsdNLSB4DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "G3BP1", "FMRP", "FUS", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "GM130", 
            "LAMP1", "NCL", "NONO", "Phalloidin", "PSD95", "SCNA", "TOMM20"
        ] #everything except  "TDP43"

        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

        self.MARKERS:List[str]            =  ["TDP43B"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["TDP43"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["dox", "Untreated"]

        
      