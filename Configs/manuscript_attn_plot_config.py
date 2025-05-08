import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA_rotation.Configs.attn_map_config import PlotAttnMapConfig
import cv2


class InitialAttnMapPlotConfig(PlotAttnMapConfig):
    def __init__(self):
        super().__init__()

        # Controls transparency of the attention overlay (higher alpha = more visible red)
        self.ALPHA:float = 0.45

        # Controls layout size of the output figure.
        self.FIG_SIZE:tuple = (12, 4)

        self.PLOT_SUPTITLE_FONTSIZE:int = 12 # main title font size

        self.PLOT_TITLE_FONTSIZE:int = 11 # each sub-figure font size

        self.PLOT_SAVEFIG_DPI:int = 300 # controls the resolution of saved figures.

        self.PLOT_HEATMAP_COLORMAP:int = cv2.COLORMAP_JET # cv2 int constanat that control colors of the heatmap

        self.SAVE_PLOT:bool = True

        self.SHOW_PLOT:bool = False

        self.SAMPLES_PATH:bool = "/home/projects/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/RotationDatasetConfig/pairs"

        # attention method 
        self.ATTN_METHOD:str = "rollout"