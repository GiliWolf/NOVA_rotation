import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA_rotation.Configs.attn_config import AttnConfig
import cv2
import numpy as np
from PIL import Image

class BaseAttnConfig(AttnConfig):
    def __init__(self):
        super().__init__()

        self.FILTER_BY_PAIRS = True
        # attention method 
        self.ATTN_METHOD:str = "rollout" #["rollout","all_layers"]

        self.RESAMPLE_METHOD:int = Image.BICUBIC 

        self.REDUCE_HEAD_FUNC:str = "mean"

        self.MIN_ATTN_THRESHOLD:float = 0.0

        self.CORR_METHOD:str = "soft_overlap" #options: ["pearsonr", "mutual_info", "ssim", "attn_overlap", "soft_overlap"]

