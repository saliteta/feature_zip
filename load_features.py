from ast import Not
from ctypes.wintypes import INT
from pickletools import uint8
import torch
import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import torchvision.utils as vutils
import concurrent.futures
import cv2

"""
    This class will handle the input and output of feature.

    That means, the input of feature should be a torch.tensor, 
    and the output of the feature should be a special format encoded in the following way.
    Several PNG, and one npz file
    
    We will also decode it using this class, and we will make it as feature tensor. 
    
    FeatureHandler.save 
    FeatureHandler.read    
"""



INT8=0
INT16=1
FLOAT32=2

class FeatureHandler:
    def __init__(self, compressing_format='ftz'):
        """
            We create a standard for compressing feature called 'ftz' feature zip file
            normally the compressing protocal is lossless, that means we will use PNG to compress
            But for better compressing ration, using JPG, which means lossy is set to be true is also fine
        """
        
        self.compressing_format = compressing_format

    def save(self, feature_tensor: torch.Tensor, location: str, precision: str = INT16, format: str = 'png', lossy = False):

        assert precision == INT8 or precision == INT16 or precision==FLOAT32, f"we current only support INT8, INT16, or FLOAT32 format, you passed in a {precision} which we are not recognize"
        if format == 'png':
            if precision == INT8:
                self.save_png(feature_tensor, location, 8)
            elif precision == INT16:
                self.save_png(feature_tensor, location, 16)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def save_png(self, feature_tensor: torch.Tensor, location: str, precision) -> None:
        """
        Save a H,W,C tensor in a custom compressed folder format.
        The tensor is normalized per-channel (over H and W), then reshaped into
        C//4 images of shape (H, W, 4). Each image is saved as a 16-bit per channel PNG.
        The per-channel min and max (shape: 2 x C) are saved in a compressed .npz file.

        Args:
            feature_tensor (torch.Tensor): Input tensor of shape (H, W, C) with C multiple of 4.
            location (str): Folder name with a custom extension (e.g., 'output.ftz').
        """
        # Validate input shape.
        assert len(feature_tensor.shape) == 3, f"The input tensor must be H,W,C shape, but get shape: {feature_tensor.shape}"
        H, W, C = feature_tensor.shape
        assert C % 4 == 0, "The last dimension must be a multiple of 4"
        if not location.endswith('ftz'):
            location += '.ftz'
        os.makedirs(location, exist_ok=True)

        # Compute per-channel min and max over H and W.
        min_vals = feature_tensor.amin(dim=(0, 1))
        max_vals = feature_tensor.amax(dim=(0, 1))

        # Normalize to [0,1]: broadcasting over H, W.
        normalized = (feature_tensor - min_vals) / (max_vals - min_vals)

        # Convert to numpy.
        normalized_np = normalized.detach().cpu().numpy()  # shape (H, W, C)

        # Reshape to (num_images, H, W, 4).
        num_images = C // 4
        feature_split = normalized_np.reshape(H, W, num_images, 4).transpose(2, 0, 1, 3)

        # Create the output folder if it doesn't exist.
        os.makedirs(location, exist_ok=True)

        # Helper function to save a single image as 16-bit PNG.
        def save_concurent_wrapper(image_array, filename):
            # Convert the normalized [0, 1] image to 16-bit: [0, 65535]
            if precision == 16:
                img = (image_array * 65535).clip(0, 65535).astype(np.uint16)
            else:
                img = (image_array * 255).clip(0, 255).astype(np.uint8)
            # Create a PIL Image. For a 4-channel image, we use "RGBA" mode.

            # Save using OpenCV. This should preserve the bit depth.
            cv2.imwrite(filename, img)

        # Save all images in parallel.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(num_images):
                file_path = os.path.join(location, f"image_{i}.png")
                futures.append(executor.submit(save_concurent_wrapper, feature_split[i], file_path))
            concurrent.futures.wait(futures)

        # Save the per-channel min and max values (shape: 2 x C) into a compressed npz file.
        min_vals_np = min_vals.detach().cpu().numpy()  # shape (C,)
        max_vals_np = max_vals.detach().cpu().numpy()  # shape (C,)
        meta = np.stack([min_vals_np, max_vals_np], axis=0)  # shape (2, C)
        meta_file = os.path.join(location, "meta.npz")
        np.savez_compressed(meta_file, meta=meta)

    def load(self, location) -> torch.Tensor:
        """
        Load the feature tensor from the custom 'ftz' format using PyTorch operations.
        This method:
          1. Loads the per-channel min and max values from 'meta.npz'
          2. Loads each saved PNG image (assumed named as image_*.png)
          3. Converts the image pixels back to normalized [0, 1] values (using torch.from_numpy)
          4. Reassembles the images into the original (H, W, C) tensor using torch.stack, permute, and reshape
          5. De-normalizes the tensor using the stored min/max values

        Returns:
            torch.Tensor: The reconstructed tensor.
        """
        # Ensure the folder name ends with '.ftz'
        if not location.endswith('.ftz'):
            location += '.ftz'

        # Load metadata using NumPy and convert to torch.Tensor.
        meta_file = os.path.join(location, "meta.npz")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta file not found at {meta_file}")
        meta_data = np.load(meta_file)
        meta = meta_data["meta"]  # shape: (2, C)
        min_vals = torch.tensor(meta[0], dtype=torch.float32)  # shape: (C,)
        max_vals = torch.tensor(meta[1], dtype=torch.float32)  # shape: (C,)

        # Find and sort image files.
        image_files = [f for f in os.listdir(location) if f.startswith("image_") and f.endswith(".png")]
        if not image_files:
            raise FileNotFoundError("No image files found in the specified location.")
        image_files = sorted(image_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        images = []
        for file in image_files:
            file_path = os.path.join(location, file)
            # Load PNG preserving the original bit-depth
            np_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            np_type = np_img.dtype
            np_img = np_img.astype(np.float32)

            # Depending on the image dtype, choose the normalization factor.
            if np_type == np.uint8:
                tensor_img = torch.from_numpy(np_img).to(torch.float32) / 255.0
            elif np_type == np.uint16:
                tensor_img = torch.from_numpy(np_img).to(torch.float32) / 65535.0
            else:
                raise ValueError(f"Unexpected image data type: {np_img.dtype}")
            images.append(tensor_img)

        # Stack images into a tensor of shape (num_images, H, W, 4)
        images_stack = torch.stack(images, dim=0)
        # Reverse the permutation applied during saving.
        # During saving, the normalized array was reshaped as (H, W, num_images, 4)
        # and then transposed to (num_images, H, W, 4). Reverse that by permuting:
        normalized = images_stack.permute(1, 2, 0, 3)  # shape: (H, W, num_images, 4)
        H, W, num_images, channels_per_image = normalized.shape
        # Reshape to (H, W, C) where C = num_images * channels_per_image.
        normalized = normalized.reshape(H, W, num_images * channels_per_image)

        # Reverse the normalization using the stored min and max values.
        # Reshape min_vals and max_vals to (1, 1, C) for broadcasting.
        min_vals = min_vals.view(1, 1, -1)
        max_vals = max_vals.view(1, 1, -1)
        original = normalized * (max_vals - min_vals) + min_vals

        return original



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_file', type=str, help='PT file location') # test
    parser.add_argument('--output', type=str, help='ftz file location')
    args = parser.parse_args()
    
    tensor: torch.Tensor = torch.load(args.pt_file)
    tensor = tensor.squeeze().permute(1,2,0)

    handler = FeatureHandler()

    handler.save(feature_tensor = tensor, location=args.output, precision=INT8)

    tensor_back = handler.load('test.ftz')
    

    relative_error = torch.norm(tensor - tensor_back.cuda()) / torch.norm(tensor)
    print(f"Relative Error: {relative_error.item()}")

    # Relative Error: PNG INT8:  7.46*e-3  52M 
    # Relative Error: PNG INT16: 2.90*e-5  14M