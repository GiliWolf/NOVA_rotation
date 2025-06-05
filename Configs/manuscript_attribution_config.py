import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from NOVA_rotation.Configs.attribution_config import AttributionConfig
from src.common.base_config import BaseConfig
DISTINCT_BASE_LINE_DICT = {
    "uniform": False, 
    "blurred": True, 
    "gaussian": True, 
    "blackout": False
}

class AttributionConfig(BaseConfig):
    """Config for creating attrubution maps with Captum
    """
    
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        self.FF_METHOD:str = None #forward function method to be used: {DistForwardFunc, DotProductForwardFunc, KNNForwardFunc}

        self.BASE_LINE_METHOD:str = "uniform" # base line methos to be used: {uniform, blurred, gaussian, blackout..}

        self.ATTR_METHOD:str = "IntegratedGradients" # attribution method from captum to be used: occlusion/IG .. 

        self.ATTR_ALGO:str = "gausslegendre" # specific algorithn of the attribition class ("method" attribute in .attribute())

        self.CLASS_LABELS_FUNC:str = "CONDITIONS"

        self.N_STEPS = 20

    @property
    def DISTINCT_BASE_LINE(self) -> bool:
        return DISTINCT_BASE_LINE_DICT.get(self.BASE_LINE_METHOD, False)

class Distance(AttributionConfig):
    """Config for creating attrubution maps with Captum
    """
    
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        self.FF_METHOD:str = "distance" #forward function method to be used: {DistForwardFunc, DotProductForwardFunc, KNNForwardFunc}

        self.DIST_METHOD:str = "euclidean" # or cosine 

class DotProduct(AttributionConfig):
    """Config for creating attrubution maps with Captum
    """
    
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        self.FF_METHOD:str = "dot_product" #forward function method to be used: {DistForwardFunc, DotProductForwardFunc, KNNForwardFunc}


