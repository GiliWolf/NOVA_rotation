import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.distance import cdist

class AttributionHelperClass():
    def __init__(self, config):
        self.config = config


    def get_base_line(inputs):
        base_line_func = globals()[f"_{config.BASE_LINE_METHOD}_base_line"]
        return base_line_func(inputs)

    def _forward_func(self, inputs: Tensor, helper_emb: Tensor) -> np.ndarray[torch.Tensor]:
        forward_func = globals()[f"_base_line_{config.FF_METHOD}"]
        raise NotImplementedError
    
    def _initiate_attr_instance():
        pass

    def attribute():
        pass
    
    def __call__(self, inputs: np.ndarray[torch.Tensor], helper_emb: np.ndarray[torch.Tensor]) -> torch.Tensor:
        return self._forward_func(inputs, helper_emb)

class BaseForwardFunc():
    def __init__(self, ff_config):
        self.config = ff_config
        self.kwargs = self._set_kwargs()

    def _set_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _forward_func(self, inputs: Tensor, helper_emb: Tensor) -> np.ndarray[torch.Tensor]:
        raise NotImplementedError
    
    def __call__(self, inputs: np.ndarray[torch.Tensor], helper_emb: np.ndarray[torch.Tensor]) -> torch.Tensor:
        return self._forward_func(inputs, helper_emb)

# === Prototype Base Class ===
class PrototypeForwardFunc(BaseForwardFunc):
    def _get_prototype(self, class_embeddings: np.ndarray[torch.Tensor], dim=0) -> Tensor:
        return torch.mean(torch.stack(class_embeddings), dim=dim)

# === Distance-based forward function ===
class DistForwardFunc(PrototypeForwardFunc):
    def _set_kwargs(self) -> Dict[str, Any]:
        return {"dist_method": self.config.get("dist_method", "cosine")}

    def _forward_func(self,inputs: np.ndarray[torch.Tensor], helper_emb: np.ndarray[torch.Tensor]
    ) -> np.ndarray[torch.Tensor]:
        dist_method = self.kwargs["dist_method"]
        prototype = self._get_prototype(helper_emb)  # shape: (D,)

        if dist_method == "cosine":
            return F.cosine_similarity(inputs, prototype.unsqueeze(0), dim=1)
        elif dist_method == "euclidean":
            return -torch.norm(inputs - prototype, dim=1)  # more similar → higher
        else:
            raise ValueError(f"Unsupported distance method: {dist_method}")

class DotProductForwardFunc(BaseForwardFunc):
    def _set_kwargs(self) -> Dict[str, Any]:
        return {}

    def _forward_func(self, inputs: Tensor, helper_emb: Tensor) -> Tensor:
        # inputs: (B, D), helper (prototype): (D,)
        return torch.matmul(inputs, helper_emb)  # shape: (B,)

class KNNForwardFunc(PrototypeForwardFunc):
    def _set_kwargs(self) -> Dict[str, Any]:
        return {
            "dist_method": self.config.get("dist_method", "cosine"),
            "reduce": self.config.get("reduce", None),
            "pooling": self.config.get("pooling", "mean"),
            "K": self.config.get("K", 5),
        }

    def _forward_func(self, inputs: Tensor, helper_emb_list: List[Tensor]) -> Tensor:
        # list of index: class_ide, values: class_embeddings 
        from src.analysis.analyzer_umap import _compute_umap_embeddings

        dist_method = self.kwargs["dist_method"]
        reduce = self.kwargs["reduce"]
        K = self.kwargs["K"]

        i# Concatenate embeddings and create corresponding labels based on index in list
        helper_embeddings = []
        helper_labels = []
        for class_idx, emb in enumerate(helper_embeddings_list):
            helper_embeddings.append(emb)
            helper_labels.append(torch.full((emb.size(0),), class_idx, device=emb.device))
        
        helper_embeddings = torch.cat(helper_embeddings, dim=0)  # (N_total, D)
        helper_labels = torch.cat(helper_labels, dim=0)          # (N_total,)

        # reduce to umap space if specified
        if reduce:
            inputs = _compute_umap_embeddings(inputs)
            helper_embeddings = _compute_umap_embeddings(helper_embeddings)

        # Compute similarity between inputs and all of the neighbors
        if dist_method == "cosine":
            sim = F.cosine_similarity(inputs.unsqueeze(1), helper_embeddings.unsqueeze(0), dim=2)
            topk_idx = sim.topk(K, dim=1, largest=True).indices
        elif dist_method == "euclidean":
            dists = torch.cdist(inputs, helper_embeddings, p=2)
            topk_idx = dists.topk(K, dim=1, largest=False).indices
        else:
            raise ValueError(f"Unsupported dist_method: {dist_method}")

        # Get top-K labels based on the K nearest neighbors 
        topk_labels = helper_labels[topk_idx]  # (B, K)

        # get the mode (most common value) from the KNN labels
        predicted_labels = torch.mode(topk_labels, dim=1).values  # (B,)

        return predicted_labels

def _distance_forward_func(inputs: torch.Tensor, prototype:torch.Tensor, config) -> torch.Tensor:
        dist_method = config.DIST_METHOD

        if dist_method == "cosine":
            return F.cosine_similarity(inputs, prototype.unsqueeze(0), dim=1)
        elif dist_method == "euclidean":
            return -torch.norm(inputs - prototype, dim=1)  # more similar → higher
        else:
            raise ValueError(f"Unsupported distance method: {dist_method}")

def _dot_product_forward_func(inputs: torch.Tensor, prototype:torch.Tensor, config) -> torch.Tensor:

        return torch.matmul(inputs, prototype)


def _blackout_base_line(inputs: torch.Tensor) -> torch.Tensor:
    """
    Creates a uniform baseline tensor where every element is set to zero.
    Args:
        inputs: Input tensor of shape (B, D)
    Returns:
        Tensor of same shape as inputs filled with 0.
    """
    return torch.full_like(inputs, fill_value=0.0)

def _gaussian_base_line(inputs: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Gaussian baseline from Smilkov et al. (2017).
    Adds Gaussian noise centered on the input with standard deviation `sigma`.
    
    Args:
        inputs: Input tensor of shape (B, D)
        sigma: Standard deviation for the Gaussian noise
    Returns:
        Tensor with Gaussian noise added to each input (clamp for making sute its range is between [0,1])
    """
    noise = torch.randn_like(inputs) * sigma
    return torch.clamp(inputs + noise, 0.0, 1.0)

def _blurred_base_line(inputs):
    #https://hackernoon.com/how-to-implement-gaussian-blur-zw28312m
    pass

def _uniform_base_line(inputs: torch.Tensor, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    """
    Uniform baseline from Sturmfels et al. (2020).
    Generates a tensor of same shape as inputs using values drawn from U(low, high).
    
    Args:
        inputs: Tensor of shape (B, D)
        low: Lower bound of uniform distribution
        high: Upper bound of uniform distribution
    Returns:
        Random tensor of same shape from uniform distribution
    """
    return torch.empty_like(inputs).uniform_(low, high)

def _scaled_base_line(inputs: torch.Tensor, scale_value:float = 0.1) -> torch.Tensor:
    """
    Scaled version of the input
    Generates a tensor of same shape as inputs using values drawn from U(low, high).
    
    Args:
        inputs: Input tensor of shape (B, D)
        scale_value: float value to scale the input with 
    Returns:
        Tensor of same shape as inputs where its values are multiplied with scale_value
    """
    return inputs * scale_value