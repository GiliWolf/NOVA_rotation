import os
import sys
from typing import List, Tuple
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.base_config import BaseConfig
from src.embeddings.embeddings_config import EmbeddingsConfig


class SubsetConfig(EmbeddingsConfig):
    """Config for extracting subset
    """
    
    def __init__(self, config: EmbeddingsConfig):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        # Copy attributes from the config given 
        self.__dict__.update(config.__dict__)

        #  metric for distance calculation
        self.METRIC:str = None

        #number of pairs from each type: min/middle/max
        self.NUM_PAIRS:int = None

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = None

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = None

        # the indeces for the comparison. should be a list of 2 indices
        self.COMPARE_BY_ATTR_IDX:list = None
