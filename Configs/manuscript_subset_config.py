import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
from NOVA_rotation.Configs.reps_dataset_config import *
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

        self.NUM_PAIRS:int = 30

        self.SUBSET_METHOD:str = "sectional" # the method to create the subset with :{sectional (min/max/middle), random}

        self.WITHOUT_REPEAT:bool = True # don't allow repeated samples

#-------------------------------------------------------------------

# class WTB9SubsetConfigForPresentation(BasicSubsetConfig):
#     def __init__(self):
#         config = EmbeddingsB9DatasetConfigForPresentation()
#         super().__init__(config)

#         # the mutual attribute to be fixed when comparing
#         self.MUTUAL_ATTR: str = "CELL_LINES"
#         self.CELL_LINES: list = ["WT"]

#         # the attrubute the pair-wise distance comparison is calculated on
#         self.COMPARE_BY_ATTR: str = "CONDITIONS"
#         self.CONDITIONS: list = ["stress", "Untreated"]

#         # the names for the comparison. should be a list of 2.
#         self.UMAP_PLOT_CONFIG: str = "UMAP0StressPlotConfig"

# class dNLSB4TDP43SubsetConfigForPresentation(BasicSubsetConfig):
#     def __init__(self):
#         config = EmbeddingsdNLSB4DatasetConfigForPresentation()
#         super().__init__(config)

#         # the mutual attribute to be fixed when comparing
#         self.MUTUAL_ATTR: str = "CELL_LINES"
#         self.CELL_LINES: list = ["TDP43"]

#         # the attrubute the pair-wise distance comparison is calculated on
#         self.COMPARE_BY_ATTR: str = "CONDITIONS"
#         self.CONDITIONS: list = ["dox", "Untreated"]

#         # the names for the comparison. should be a list of 2.
#         self.UMAP_PLOT_CONFIG: str = "UMAP0dNLSPlotConfig"

#-------------------------------------------------------------------

class WTB9SubsetConfig(BasicSubsetConfig):
    def __init__(self):
        config = EmbeddingsB9DatasetConfig()
        super().__init__(config)

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR: str = "CELL_LINES"
        self.CELL_LINES: list = ["WT"]

        self.MARKERS: List[str] = ["G3BP1", "FMRP"] #"Phalloidin"

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR: str = "CONDITIONS"
        self.CONDITIONS: list = ["stress", "Untreated"]

        # the names for the comparison. should be a list of 2.
        self.UMAP_PLOT_CONFIG: str = "UMAP0StressPlotConfig"




class FUSB9WTSubsetConfig(BasicSubsetConfig):
    def __init__(self):
        config = EmbeddingsB9DatasetConfig()
        super().__init__(config)

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR: str = "CONDITIONS"
        self.CONDITIONS: list = ["Untreated"]

        self.MARKERS: List[str] = ["FUS", "ANXA11"]

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR: str = "CELL_LINES"
        self.CELL_LINES: list = ["FUSHomozygous", "WT"]

        # the names for the comparison. should be a list of 2.
        self.UMAP_PLOT_CONFIG: str = "UMAP0ALSPlotConfig"




class dNLSB4TDP43SubsetConfig(BasicSubsetConfig):
    def __init__(self):
        config = EmbeddingsdNLSB4DatasetConfig()
        super().__init__(config)

        # the mutual attribute to be fixed when comparing
        self.MUTUAL_ATTR: str = "CELL_LINES"
        self.CELL_LINES: list = ["TDP43"]

        self.MARKERS: List[str] = ["TDP43B", "DCP1A"]

        # the attrubute the pair-wise distance comparison is calculated on
        self.COMPARE_BY_ATTR: str = "CONDITIONS"
        self.CONDITIONS: list = ["dox", "Untreated"]

        # the names for the comparison. should be a list of 2.
        self.UMAP_PLOT_CONFIG: str = "UMAP0dNLSPlotConfig"

# ----------------------------------------------------

# class TDP43B9SubsetConfig(BasicSubsetConfig):
#     def __init__(self):
#         config = EmbeddingsB9DatasetConfig()
#         super().__init__(config)

#         # the mutual attribute to be fixed when comparing
#         self.MUTUAL_ATTR: str = "CONDITIONS"
#         self.CONDITIONS: list = ["Untreated"]

#         self.MARKERS: List[str] = ["TDP43"]

#         # the attrubute the pair-wise distance comparison is calculated on
#         self.COMPARE_BY_ATTR: str = "CELL_LINES"
#         self.CELL_LINES: list = ["TDP43", "WT"]

#         # the names for the comparison. should be a list of 2.
#         self.UMAP_PLOT_CONFIG: str = "UMAP0ALSPlotConfig"

# class FUSB9RevertantSubsetConfig(BasicSubsetConfig):
#     def __init__(self):
#         config = EmbeddingsB9DatasetConfig()
#         super().__init__(config)

#         # the mutual attribute to be fixed when comparing
#         self.MUTUAL_ATTR: str = "CONDITIONS"
#         self.CONDITIONS: list = ["Untreated"]

#         self.MARKERS: List[str] = ["FUS", "ANXA11"]

#         # the attrubute the pair-wise distance comparison is calculated on
#         self.COMPARE_BY_ATTR: str = "CELL_LINES"
#         self.CELL_LINES: list = ["FUSHomozygous", "FUSRevertant"]

#         # the names for the comparison. should be a list of 2.
#         self.UMAP_PLOT_CONFIG: str = "UMAP0ALSPlotConfig"


# class dNLSB4TDP43andWTSubsetConfig(BasicSubsetConfig):
#     def __init__(self):
#         config = EmbeddingsdNLSB4DatasetConfig()
#         super().__init__(config)

#         # the mutual attribute to be fixed when comparing
#         self.MUTUAL_ATTR: str = "CELL_LINES"
#         self.CELL_LINES: list = ["TDP43", "WT"]

#         self.MARKERS: List[str] = ["TDP43B", "DCP1A"]

#         # the attrubute the pair-wise distance comparison is calculated on
#         self.COMPARE_BY_ATTR: str = "CONDITIONS"
#         self.CONDITIONS: list = ["dox", "Untreated"]

#         # the names for the comparison. should be a list of 2.
#         self.UMAP_PLOT_CONFIG: str = "UMAP0dNLSPlotConfig"



