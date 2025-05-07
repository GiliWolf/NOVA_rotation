import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig
import cv2

class PlotAttnMapConfig(BaseConfig):
    """Config for Attention Maps plotting
    """
    
    def __init__(self):
        
        super().__init__()

        # Controls transparency of the attention overlay (higher alpha = more visible red)
        self.ALPHA:float = None

        # Controls layout size of the output figure.
        self.FIG_SIZE:tuple = None

        self.PLOT_SUPTITLE_FONTSIZE:int = None # main title font size

        self.PLOT_TITLE_FONTSIZE:int = None # each sub-figure font size

        self.PLOT_SAVEFIG_DPI:int = None # controls the resolution of saved figures.

        self.PLOT_HEATMAP_COLORMAP:int = None # cv2 int constanat that control colors of the heatmap

        self.SAVE_PLOT:bool = None

        self.SHOW_PLOT:bool = None

        # Path to the dir containing the samples  ("set_type_paths.npy" file) to be selectively plotted
        self.SAMPLES_PATH:str = None 

    
        
