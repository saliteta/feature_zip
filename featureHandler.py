import torch
import os
import numpy as np
import concurrent.futures
import cv2
import subprocess
import tempfile

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
        
        self.compressing_format = compressing_format # compressing format can be used mpf


    def save(self, feature_tensor: torch.Tensor, location: str = None, precision: str = INT16, format: str = 'png', lossy = False):

        assert precision == INT8 or precision == INT16 or precision==FLOAT32, f"we current only support INT8, INT16, or FLOAT32 format, you passed in a {precision} which we are not recognize"
        if location == None:
            assert self.compressing_format == 'mpf', f"if location is not given, we assume we will compress each image feature \
                finally to mpf, but get FeatureHandler compressing format {self.compressing_format}"
        if format == 'png' or format == 'jpg':
            if precision == INT8:
                self.save_png_jpg(feature_tensor, location, 8, format)
            elif precision == INT16:
                assert format == 'png', "only png support 16 bit encoding, jpg does not"
                self.save_png_jpg(feature_tensor, location, 16, format)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def save_png_jpg(self, feature_tensor: torch.Tensor, location: str, precision, format: str) -> None:
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
        if not location.endswith('ftz'):
            location += '.ftz'
        os.makedirs(location, exist_ok=True)





        # Compute per-channel min and max over H and W.
        min_vals = feature_tensor.amin(dim=(0, 1))
        max_vals = feature_tensor.amax(dim=(0, 1))

        # Normalize to [0,1]: broadcasting over H, W.
        normalized = (feature_tensor - min_vals) / (max_vals - min_vals)

        # Convert to numpy.

        # Reshape to (num_images, H, W, 4).
        if C%3 != 0:
            num_images = C // 3 + 1
            padding = torch.zeros(H,W,3-C%3).to(normalized.device)
            normalized = torch.concat([normalized, padding], dim=2)
        else:
            num_images = C//3

        normalized_np = normalized.detach().cpu().numpy()  # shape (H, W, C)
        feature_split = normalized_np.reshape(H, W, num_images, 3).transpose(2, 0, 1, 3)

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
                file_path = os.path.join(location, f"image_{i}.{format}")
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
        image_files = [f for f in os.listdir(location) if f.startswith("image_") and f.endswith((".png", ".jpg"))]
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
        normalized = normalized.reshape(H, W, num_images * channels_per_image)[:, :, :min_vals.shape[0]]

        # Reverse the normalization using the stored min and max values.
        # Reshape min_vals and max_vals to (1, 1, C) for broadcasting.
        min_vals = min_vals.view(1, 1, -1)
        max_vals = max_vals.view(1, 1, -1)
        original = normalized * (max_vals - min_vals) + min_vals

        return original
    
    #def save_mpf(ftzs_location: str, output_dir: str, framerate: int = 30, video_codec: str = 'libx264', container: str = 'mp4'):
    #    """
    #    Encode a sequence of FTZ folders into separate video files.
#
    #    Each FTZ folder contains PNG files (e.g. image_0.png, image_1.png, …)
    #    that represent a sequence of frames. This function uses ffmpeg to
    #    encode each FTZ folder's sequence into a video file in the specified
    #    container format (e.g., MP4 or AVI).
#
    #    Parameters:
    #        ftzs_location (str): Directory containing FTZ folders.
    #        output_dir (str): Directory where output video files will be saved.
    #        framerate (int): Frame rate of the output video (default is 30 fps).
    #        video_codec (str): Video codec to use (default 'libx264' for MP4).
    #        container (str): Video container format (e.g., 'mp4' or 'avi').
#
    #    Example:
    #        >>> save_mpf("/path/to/ftzs", "/path/to/output_videos")
    #    """
    #    # Ensure the output directory exists.
    #    os.makedirs(output_dir, exist_ok=True)
#
    #    # Get a sorted list of all FTZ folders.
    #    ftz_dirs = sorted([d for d in os.listdir(ftzs_location) if d.endswith('.ftz')]) ## how many frame we will have in each video
    #    if not ftz_dirs:
    #        raise ValueError("No FTZ folders found in the specified location.")
    #    feature_seperation_list = sorted([d for d in os.listdir(ftz_dirs[0]) if d.endswith('.png', '.jpg')])




def save_mpf(ftzs_location: str, output_dir: str, framerate: int = 30,
             video_codec: str = 'libx264', container: str = 'avi'):
    """
    Encode a sequence of FTZ folders into separate video files.
    
    Each FTZ folder contains one set of image files (e.g. "feature_0.png", "feature_1.jpg", …)
    that represent frames for a given feature separation. For each such image filename (feature),
    the script creates a video by taking that image from each FTZ folder in sorted order.
    
    The videos are encoded using ffmpeg's concat demuxer by writing a temporary file list
    containing full paths to each frame file. Videos are then produced in parallel.
    
    Parameters:
        ftzs_location (str): Base directory containing FTZ folders.
        output_dir (str): Directory where output video files will be saved.
        framerate (int): Frame rate of the output video (default: 30 fps).
        video_codec (str): Video codec to use (default 'mpeg4'; typically works with AVI).
        container (str): Video container format (e.g., 'avi').
        
    Example:
        >>> save_mpf("/path/to/ftzs", "/path/to/output_videos")
    """
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Get full paths for all FTZ folders (assuming they end with ".ftz").
    ftz_dirs = sorted([os.path.join(ftzs_location, d) 
                       for d in os.listdir(ftzs_location)
                       if os.path.isdir(os.path.join(ftzs_location, d)) and d.endswith('.ftz')])
    if not ftz_dirs:
        raise ValueError("No FTZ folders found in the specified location.")

    # Assume each FTZ folder contains the same set of images.
    # Get list of feature-separation filenames (supporting both PNG and JPG).
    sample_ftz = ftz_dirs[0]
    feature_files = sorted([f for f in os.listdir(sample_ftz)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not feature_files:
        raise ValueError("No image files found in the sample FTZ folder.")

    def process_feature(feature_filename: str):
        """
        For a given feature filename, form the sequence of frames from each FTZ folder,
        write a temporary ffmpeg concat file, and invoke ffmpeg to encode the video.
        """
        # Build list of frame file paths in order.
        frame_paths = [os.path.join(ftz_dir, feature_filename) for ftz_dir in ftz_dirs]

        # Create a temporary file to list frames for ffmpeg.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            for path in frame_paths:
                # ffmpeg expects each line to be in the form: file '/absolute/path/to/frame.png'
                temp_file.write(f"file '{os.path.abspath(path)}'\n")
            list_filename = temp_file.name
        
        # Create output video filename.
        base_feature = os.path.splitext(feature_filename)[0]
        output_video = os.path.join(output_dir, f"{base_feature}.{container}")

        # Build the ffmpeg command.
        # -f concat: use concat demuxer,
        # -safe 0: allow absolute paths,
        # -framerate: set input frame rate,
        # -c:v: video codec,
        # -y: overwrite output.
        command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-r", str(framerate),  # <---- put BEFORE -i
            "-i", list_filename,
            "-c:v", video_codec,
            "-crf", "0",
            "-preset", "veryslow",          # <- best compression efficiency
            output_video
        ]
        print("Encoding", output_video)
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error encoding {output_video}:\n", e.stderr.decode())
        finally:
            # Clean up the temporary list file.
            os.remove(list_filename)
        return output_video

    # Encode each feature's image sequence in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_feature, feature)
                   for feature in feature_files]
        results = [fut.result() for fut in concurrent.futures.as_completed(futures)]




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_file', type=str, help='PT file location') # test
    parser.add_argument('--output', type=str, help='ftz file location')
    args = parser.parse_args()
    
    tensor: torch.Tensor = torch.load(args.pt_file)
    tensor = tensor.squeeze().permute(1,2,0)

    handler = FeatureHandler()

    handler.save(feature_tensor = tensor, location=args.output, precision=INT8, format='png')

    tensor_back = handler.load('test.ftz')
    

    relative_error = torch.norm(tensor - tensor_back.cuda()) / torch.norm(tensor)
    print(f"Relative Error: {relative_error.item()}")

    # Relative Error: PNG INT16: 2.90*e-5  52.0M 
    # Relative Error: PNG INT8:  7.46*e-3  14.0M 
    # Relative Error: JPG INT8:  2.49*e-2  2.7M 