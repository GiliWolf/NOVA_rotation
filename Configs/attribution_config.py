import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.base_config import BaseConfig



class AttributionConfig(BaseConfig):
    """Config for creating attrubution maps with Captum
    """
    
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        self.FF_METHOD:str = None #forward function method to be used: {DistForwardFunc, DotProductForwardFunc, KNNForwardFunc}

        self.BASE_LINE_METHOD:str = None # base line methos to be used: {uniform, blurred, gaussian, blackout..}

        self.DISTINCT_BASE_LINE:bool = None # generate base line distinctanly for each input image. must be true for "blurred" BASE_LINE_METHOD 

        self.ATTR_METHOD:str = None # attribution method from captum to be used: occlusion/IG .. 

        self.ATTR_ALGO:str = None # specific algorithn of the attribition class ("method" attribute in .attribute())