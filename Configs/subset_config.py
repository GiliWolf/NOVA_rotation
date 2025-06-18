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

        #number of pairs from each section
        self.NUM_PAIRS:int = None

        self.SUBSET_METHOD:str = None # the method to create the subset with :{sectional (min/max/middle), random}

        self.WITHOUT_REPEAT:bool = None # don't allow repeated samples

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = None

        # the value of the mutial attribute should be:
        # if - (1) single str/ list with of length 1 - shared between the COMPARE_BY_ATTR
        #      (2) list of length > 1,  try to match each index to the COMPARE_BY_ATTR_LIST index
        self.MUTUAL_ATTR_VAL = None

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = None

        # the names for the comparison. should be a list of 2.
        self.COMPARE_BY_ATTR_LIST:list = None
