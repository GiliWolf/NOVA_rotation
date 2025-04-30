import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.base_config import BaseConfig


class RotationDatasetConfig(BaseConfig):
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

        # Reps to include
        self.REPS:List[str]               = None

        # The percentage of the data that goes to the training set
        self.TRAIN_PCT:float              = 0.7

        # Should shuffle the data within each batch collected?
        ##Must be true whenever using SPLIT_DATA=True otherwise train,val,test set won't be the same as when shuffle was true

        # CHANGED: false so it will keep same order 
        self.SHUFFLE:bool                 = False     
        
        # Should add the cell line to the label?
        self.ADD_LINE_TO_LABEL:bool = True

        # Should add condition to the label?
        self.ADD_CONDITION_TO_LABEL:bool = True 

        # Number of channels per image
        self.NUM_CHANNELS:int = 2

        # The size of each image (width,height)
        self.IMAGE_SIZE:Tuple[int, int] = (100,100)

        ################################