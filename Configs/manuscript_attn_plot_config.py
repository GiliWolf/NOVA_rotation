import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA_rotation.Configs.attn_map_config import PlotAttnMapConfig
import cv2
import numpy as np


class InitialAttnMapPlotConfig(PlotAttnMapConfig):
    def __init__(self):
        super().__init__()

        # Controls transparency of the attention overlay (higher alpha = more visible red)
        self.ALPHA:float = 0.4

        # Controls layout size of the output figure.
        self.FIG_SIZE:tuple = (12, 4)

        self.SAVE_SEPERATE_LAYERS:bool = False

        self.ALL_LAYERS_FIG_SIZE:tuple = (13, 11)

        self.PLOT_SUPTITLE_FONTSIZE:int = 14 # main title font size

        self.PLOT_TITLE_FONTSIZE:int = 12 # each sub-figure font size

        self.PLOT_LAYER_FONTSIZE:int = 10 # each layer, for SAVE_SEPERATE_LAYERS = True

        self.PLOT_SAVEFIG_DPI:int = 300 # controls the resolution of saved figures.

        self.PLOT_HEATMAP_COLORMAP:int = cv2.COLORMAP_JET # cv2 int constanat that control colors of the heatmap

        self.SAVE_PLOT:bool = True

        self.SHOW_PLOT:bool = False

        self.SAMPLES_PATH:bool = "/home/projects/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/EmbeddingsB9DatasetConfig_Phalloidin/pairs/euclidean/neurons/batch9/Phalloidin"

        # attention method 
        self.ATTN_METHOD:str = "all_layers" #["rollout","all_layers"]

        self.REDUCE_HEAD_FUNC:str = "mean"

        self.MIN_ATTN_THRESHOLD:float = 0.25

        self.CORR_METHOD:str = "pearsonr" #options: ["pearsonr", "mutual_info", "ssim", "attn_overlap"]

        self.PLOT_CORR_SUMMARY:bool = True
        
        self.PLOT_CORR_SEPERATE_MARKERS:bool = True

        self.PLOT_CORR_ALL_MARKERS:bool = True