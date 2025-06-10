import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig
from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction


class UMAP_Subset_Config(FigureConfig):
    def __init__(self, subset_config):
        super().__init__()

        self.SPLIT_DATA = False

        # Batches used for model development
        self.INPUT_FOLDERS = subset_config.INPUT_FOLDERS
        
        self.EXPERIMENT_TYPE = subset_config.EXPERIMENT_TYPE

        self.MARKERS:List[str] = subset_config.MARKERS

        for attr in ["CELL_LINES", "CONDITIONS"]:
            mutual_attr = subset_config.MUTUAL_ATTR
            mutual_val = subset_config.MUTUAL_ATTR_VAL
            compare_attr = subset_config.COMPARE_BY_ATTR
            compare_list = subset_config.COMPARE_BY_ATTR_LIST

            if mutual_attr == attr:
                setattr(self, attr, [mutual_val])
            elif compare_attr == attr:
                setattr(self, attr, compare_list)
            else:
                setattr(self, attr, getattr(subset_config, attr))

        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ARI_LABELS_FUNC = subset_config.COMPARE_BY_ATTR




class EmbeddingsB9DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # ADDED- 
        self.SPLIT_DATA = False

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    

        self.MARKERS:List[str]            =  ["G3BP1"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["WT"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["stress", "Untreated"]

        
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


        self.MARKERS:List[str]            =  ["TDP43B"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["TDP43"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["dox", "Untreated"]

        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ARI_LABELS_FUNC = "CONDITIONS"