import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA_rotation.Configs.subset_config import SubsetConfig
from src.embeddings.embeddings_config import EmbeddingsConfig
import numpy as np

class BasicSubsetConfig(SubsetConfig):
    """Config for extracting subset
    """
    
    def __init__(self, config):

        super().__init__(config)

        #  metric for distance calculation
        self.METRIC:str = "euclidean"

        self.NUM_PAIRS:int = 100

        self.SUBSET_METHOD:str = "sectional" # the method to create the subset with :{sectional (min/max/middle), random}

        self.WITHOUT_REPEAT:bool = True # don't allow repeated samples



class WTB9SubsetConfig(BasicSubsetConfig):
    def __init__(self, config):

        super().__init__(config)


        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = "CELL_LINES"

        self.MUTUAL_ATTR_VAL:str = "WT"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = "conditions"

        # the names for the comparison. should be a list of 2.
        self.COMPARE_BY_ATTR_LIST:list = ["stress", "Untreated"]



class FUSBB9SubsetConfig(BasicSubsetConfig):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = "CONDITIONS"

        self.MUTUAL_ATTR_VAL:str = "Untreated"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = "CELL_LINES"

        self.COMPARE_BY_ATTR_LIST:list = ["FUSHomozygous", "WT"]


class dNLSB4SubsetConfig(BasicSubsetConfig):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)


        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = "CELL_LINES"

        self.MUTUAL_ATTR_VAL:str = "TDP43"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = "conditions"

        self.COMPARE_BY_ATTR_LIST:list = ["dox", "Untreated"]