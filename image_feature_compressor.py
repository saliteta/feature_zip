import os
import torch
from featureHandler import FeatureHandler, INT8, INT16, FLOAT32


class feature_generator:
    """
        This is an abstract class, 
        The user must implement the following method
        1. forward method
    """
    def __init__(self) -> None:
        pass

    
    def forward(self, batch_images: torch.Tensor) -> torch.Tensor: 
        """
            This is an empty forward class, please implement this
        """
        H = 244 # height of feature map
        W = 244 # width pf feature map
        C = 512 # feature channel
        B = batch_images.shape[0]
        return torch.zeros(B,H,W,C) 

class image_feature_compressor:

    def __init__(self, feature_generator: feature_generator, precision: int = INT8, format: str = 'jpg') -> None:
        self.feature_generator = feature_generator
        self.img_buffer = None
        self.precision = precision
        self.format = format

        self.feature_handler = FeatureHandler()


    def forward(self, )
