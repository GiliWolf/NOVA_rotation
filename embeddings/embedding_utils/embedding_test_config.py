import os
import sys
# Get the NOVA_HOME environment variable
nova_home = os.getenv("NOVA_HOME")

sys.path.insert(1, nova_home)


from src.embeddings.embeddings_config import EmbeddingsConfig

"""
changed MARKERS_TO_EXCLUDE so it will only generate embedding for  G3BP1
"""

class EmbeddingsB9DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "FMRP", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "FUS", "GM130", 
            "LAMP1", "NCL", "NONO", "Phalloidin", "PSD95", "SCNA", "TDP43", "TOMM20"
        ] # enerything except "G3BP1"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True