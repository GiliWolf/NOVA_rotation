import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig
from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction



#UMAP0StressPlotConfig
class EmbeddingsB9DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # ADDED- 
        self.SPLIT_DATA = False

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    

        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "FMRP", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "FUS", "GM130", 
            "LAMP1", "NCL", "NONO", "Phalloidin", "PSD95", "SCNA", "TDP43", "TOMM20"
        ] 
        self.MARKERS:List[str]            =  ["G3BP1"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["stress", "Untreated"]
        ##
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ARI_LABELS_FUNC = "CONDITIONS"

# UMAP0StressPlotConfig
class EmbeddingsB9DatasetConfig_Phalloidin(FigureConfig):
    def __init__(self):
        super().__init__()

        # ADDED- 
        self.SPLIT_DATA = False

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    

        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "FMRP", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "FUS", "GM130", 
            "LAMP1", "NCL", "NONO", "G3BP1", "PSD95", "SCNA", "TDP43", "TOMM20"
        ] 
        ##
        self.MARKERS:List[str]            =  ["Phalloidin"]

        # Cell lines to include

        self.CELL_LINES:List[str]         = ["WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["stress", "Untreated"]
        ##
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ARI_LABELS_FUNC = "CONDITIONS"


class EmbeddingsB9DatasetConfig_FUS(FigureConfig):
    def __init__(self):
        super().__init__()

        # ADDED- 
        self.SPLIT_DATA = False

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    

        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "FMRP", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "G3BP1", "GM130", 
            "LAMP1", "NCL", "NONO", "Phalloidin", "PSD95", "SCNA", "TDP43", "TOMM20"
        ] 
        self.MARKERS:List[str]            =  ["FUS"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["FUSHomozygous", "WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["Untreated"]
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ARI_LABELS_FUNC = "CELL_LINES"

#UMAP0dNLSPlotConfig
class EmbeddingsdNLSB4DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # ADDED- 
        self.SPLIT_DATA = False

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS'    

        self.MARKERS_TO_EXCLUDE = [
            "ANXA11", "CD41", "DAPI", "G3BP1", "FMRP", "FUS", "KIF5A", "mitotracker", "NEMO", "PEX14", 
            "PML", "PURA", "SQSTM1", "TIA1", "Calreticulin", "CLTC", "DCP1A", "GM130", 
            "LAMP1", "NCL", "NONO", "Phalloidin", "PSD95", "SCNA", "TOMM20"
        ]

        self.MARKERS:List[str]            =  ["TDP43B"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["TDP43"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["dox", "Untreated"]

        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ARI_LABELS_FUNC = "CONDITIONS"