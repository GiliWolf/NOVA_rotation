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

class NoiseTunnelConfig:
    def __init__(self):
        self.USE_NOISE_TUNNEL:bool = True # apply NoiseTunnel (Adds gaussian noise to each input)

        self.nt_samples:int = 10 # The number of randomly generated examples per sample in the input batch.
        
        self.nt_type:str = 'smoothgrad' #Smoothing type of the attributions. smoothgrad, smoothgrad_sq or vargrad.

        self.nt_samples_batch_size:int = None #  The number of the nt_samples that will be processed together.

        self.stdevs:float = None # The standard deviation of gaussian noise with zero mean that is added to each input in the batch.

        self.draw_baseline_from_distrib:bool = None #Indicates whether to randomly draw baseline samples from the baselines distribution provided as an input tensor.
    
    def get_kwargs(self):
        # Exclude the flag itself
        return {
            k: v for k, v in vars(self).items()
            if k != "USE_NOISE_TUNNEL" and v is not None
        }

    def to_dict(self):
        return vars(self)


class AttributionConfig(BaseConfig):
    """Config for creating attrubution maps with Captum
    """
    
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        self.BASE_LINE_METHOD:str = "blackout" # base line methos to be used: {uniform, blurred, gaussian, blackout}

        self.SAMPLES_PATH:str = "/home/projects/hornsteinlab/giliwo/NOVA_rotation/embeddings/embedding_output/EmbeddingsB9DatasetConfig/pairs/euclidean/neurons/batch9/G3BP1"

        self.CLASS_LABELS_FUNC:str = "CONDITIONS"

        self.FF_METHOD:str = "distance"  #forward function method to be used: {DistForwardFunc, DotProductForwardFunc, KNNForwardFunc}

        self.DIST_METHOD:str = "euclidean" # or cosine. only for distance based FF_METHOD

        self.NOISE_TUNNEL_CONFIG = NoiseTunnelConfig()



    @property
    def DISTINCT_BASE_LINE(self) -> bool:
        return DISTINCT_BASE_LINE_DICT.get(self.BASE_LINE_METHOD, False)
    
    @property
    def USE_NOISE_TUNNEL(self) -> bool:
        return self.NOISE_TUNNEL_CONFIG.USE_NOISE_TUNNEL

    def get_kwargs(self):
        # Dynamically get the direct parent class (not self)
        base_class = type(self).__mro__[1]
        base_attrs = set(vars(base_class()).keys())
        all_attrs = set(vars(self).keys())
        
        # Subclass-specific attributes (excluding common ones like ATTR_METHOD)
        diff_attrs = all_attrs - base_attrs - {"ATTR_METHOD"}

        kwargs = {k: getattr(self, k) for k in diff_attrs}

        # add NOISE_TUNNEL parameters if spedified
        if self.NOISE_TUNNEL_CONFIG.USE_NOISE_TUNNEL:
            kwargs.update(self.NOISE_TUNNEL_CONFIG.get_kwargs())

        return kwargs





class IntegratedGradientsAttributionConfig(AttributionConfig):
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        self.ATTR_METHOD:str = "IntegratedGradients" # attribution method from captum to be used: occlusion/IG .. 

        self.n_steps = 100

        self.internal_batch_size = None

        self.return_convergence_delta = False

        self.method :str = "gausslegendre" # specific algorithn of the attribition class ("method" attribute in .attribute())


class OcclusionAttributionConfig(AttributionConfig):
    def __init__(self):
        # initi object with the data from the config given
        super().__init__()  # Initialize base class normally

        self.ATTR_METHOD:str = "Occlusion" # attribution method from captum to be used: occlusion/IG .. 

        self.sliding_window_shapes:tuple  = None

        self.strides:tuple  = None

        self.perturbations_per_eval:int  = None

        self.show_progress:bool = None



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


