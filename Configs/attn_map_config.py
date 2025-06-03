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

        # Path to the dir containing the samples  ("set_type_paths.npy" file) to be selectively plotted
        self.SAMPLES_PATH:str = None 

        # attention method to be used: all_layers/rollout
        self.ATTN_METHOD:str = None 

        # integer represents PIL Image resampling method:
        # best methods: BICUBIC/LANCZOS, other (faster): NEAREST, BOX, BILINEAR, HAMMING
        self.RESAMPLE_METHOD:int = None

        # resuction of number of heads - min/max/mean (which is supported by numpy)
        self.REDUCE_HEAD_FUNC:str = None

        # min value to keep in the attentio maps (below is zeroed out)
        self.MIN_ATTN_THRESHOLD:float = None

        # correlation method to be used between attention maps and original image
        # options: ["pearsonr", "mutual_info", "ssim", "attn_overlap"]
        self.CORR_METHOD:str = None

        self.PLOT_CORR_SUMMARY:bool = None
        
        self.PLOT_CORR_SEPERATE_MARKERS:bool = None

        self.PLOT_CORR_ALL_MARKERS:bool = None


    
        
