import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig
from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction

#vased on NeuronsUMAP0StressB9DAPIFigureConfig
class NeuronsUMAP0StressB9DAPIFigureConfigGili(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = None
        self.MARKERS = ['DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False