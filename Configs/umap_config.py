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

        # data used for UMAPing
        self.INPUT_FOLDERS = subset_config.INPUT_FOLDERS
        
        self.EXPERIMENT_TYPE = subset_config.EXPERIMENT_TYPE

        self.MARKERS:List[str] = subset_config.MARKERS

        self.CELL_LINES:List[str] = subset_config.CELL_LINES

        self.CONDITIONS:List[str] = subset_config.CONDITIONS

        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ARI_LABELS_FUNC = subset_config.COMPARE_BY_ATTR



