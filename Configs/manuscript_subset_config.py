import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA_rotation.Configs.subset_config import SubsetConfig
from src.embeddings.embeddings_config import EmbeddingsConfig
import numpy as np



class EmbeddingsB9DatasetConfig(SubsetConfig):
    def __init__(self, config):

        super().__init__(config)

        #  metric for distance calculation
        self.METRIC:str = "euclidean"

        #number of pairs from each type: min/middle/max
        self.NUM_PAIRS:int = 25

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = "CELL_LINES"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = "conditions"

        # the indeces for the comparison. should be a list of 2 indices
        self.COMPARE_BY_ATTR_IDX:list = [0,1]

class EmbeddingsB9DatasetConfig_Phalloidin(SubsetConfig):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)

        #  metric for distance calculation
        self.METRIC:str = "euclidean"

        #number of pairs from each type: min/middle/max
        self.NUM_PAIRS:int = 25

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = "CELL_LINES"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = "conditions"

        # the indeces for the comparison. should be a list of 2 indices
        self.COMPARE_BY_ATTR_IDX:list = [0,1]


class EmbeddingsB9DatasetConfig_FUS(SubsetConfig):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)

        #  metric for distance calculation
        self.METRIC:str = "euclidean"

        #number of pairs from each type: min/middle/max
        self.NUM_PAIRS:int = 25

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = "CONDITIONS"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = "CELL_LINES"

        # the indeces for the comparison. should be a list of 2 indices
        self.COMPARE_BY_ATTR_IDX:list = [0,1]


class EmbeddingsdNLSB4DatasetConfig(SubsetConfig):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)

        #  metric for distance calculation
        self.METRIC:str = "euclidean"

        #number of pairs from each type: min/middle/max
        self.NUM_PAIRS:int = 25

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR:str = "CELL_LINES"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR:str = "conditions"

        # the indeces for the comparison. should be a list of 2 indices
        self.COMPARE_BY_ATTR_IDX:list = [0,1]