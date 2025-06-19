import sys
import os
from typing import Dict, List, Tuple, Callable

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.base_config import BaseConfig
import cv2
from PIL import Image


class PlotAttnMapConfig(BaseConfig):
    """Config for Attention Maps plotting
    """
    
    def __init__(self):
        
        super().__init__()

        # Controls transparency of the attention overlay (higher alpha = more visible red)
        self.ALPHA:float = None

        # Controls layout size of the output figure.
        self.FIG_SIZE:tuple = None

        self.SAVE_SEPERATE_LAYERS:bool = None # for "all_layers" attention methods. if True saves each layer in distinct figure (in addition to all-layers-one-fig)

        self.ALL_LAYERS_FIG_SIZE:tuple = None

        self.PLOT_SUPTITLE_FONTSIZE:int = None # main title font size

        self.PLOT_TITLE_FONTSIZE:int = None # each sub-figure font size

        self.PLOT_LAYER_FONTSIZE:int = None # each layer, for SAVE_SEPERATE_LAYERS = True

        self.PLOT_SAVEFIG_DPI:int = None # controls the resolution of saved figures.

        self.PLOT_HEATMAP_COLORMAP:int = None # cv2 int constanat that control colors of the heatmap

        self.SAVE_PLOT:bool = None

        self.SHOW_PLOT:bool = None

        self.PLOT_CORR_SUMMARY:bool = None

        self.SAVE_CORR_SEPERATE_MARKERS:bool = None

        self.SAVE_CORR_ALL_MARKERS:bool = None
        
        self.PLOT_CORR_SEPERATE_MARKERS:bool = None

        self.PLOT_CORR_ALL_MARKERS:bool = None

