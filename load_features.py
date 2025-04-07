import torch
import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import torchvision.utils as vutils
import concurrent.futures

"""
    This class will handle the input and output of feature.

    That means, the input of feature should be a torch.tensor, 
    and the output of the feature should be a special format encoded in the following way.
    Several PNG, and one npz file
    
    We will also decode it using this class, and we will make it as feature tensor. 
    
    FeatureHandler.save 
    FeatureHandler.read    
"""


class FeatureHandler:
    def __init__(self, compressing_format='ftz', lossy = False):
        """
            We create a standard for compressing feature called 'ftz' feature zip file
            normally the compressing protocal is lossless, that means we will use PNG to compress
            But for better compressing ration, using JPG, which means lossy is set to be true is also fine
        """
        
        self.compressing_format = compressing_format
        self.lossy = lossy 
    
    
def save(feature_tensor: torch.Tensor, location: str) -> None:
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
    assert len(feature_tensor.shape) == 3, "The input tensor must be H,W,C shape"
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
    def save_png(image_array, filename):
        # Convert the normalized [0, 1] image to 16-bit: [0, 65535]
        img_uint16 = (image_array * 65535).clip(0, 65535).astype(np.uint16)
        # Create a PIL Image. For a 4-channel image, we use "RGBA" mode.
        im = Image.fromarray(img_uint16, mode="RGBA")
        im.save(filename, format="PNG")

    # Save all images in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_images):
            file_path = os.path.join(location, f"image_{i}.png")
            futures.append(executor.submit(save_png, feature_split[i], file_path))
        concurrent.futures.wait(futures)

    # Save the per-channel min and max values (shape: 2 x C) into a compressed npz file.
    min_vals_np = min_vals.detach().cpu().numpy()  # shape (C,)
    max_vals_np = max_vals.detach().cpu().numpy()  # shape (C,)
    meta = np.stack([min_vals_np, max_vals_np], axis=0)  # shape (2, C)
    meta_file = os.path.join(location, "meta.npz")
    np.savez_compressed(meta_file, meta=meta)
        
        

def save_features_as_png_from_tensor(features_path, output_folder, file_prefix="scene"):
    os.makedirs(output_folder, exist_ok=True)
    features = torch.load(features_path)
    if features.shape[0] == 1:
        features = features.squeeze(0)
    
    original_shape = list(features.shape)
    
    num_pngs = features.shape[0] // 4
    
    transform_params = {
        "original_shape": original_shape,
        "chunks": []
    }
    
    reshaped_features = features[:num_pngs*4].reshape(num_pngs, 4, features.shape[1], features.shape[2])
    
    min_vals = reshaped_features.view(num_pngs, -1).min(dim=1)[0]
    max_vals = reshaped_features.view(num_pngs, -1).max(dim=1)[0]
    

    for i in range(num_pngs):
        # 获取当前chunk
        chunk = reshaped_features[i]
        min_val = float(min_vals[i].item())
        max_val = float(max_vals[i].item())
        
        # 记录变换参数
        chunk_info = {
            "index": i,
            "channels": [i*4, i*4+1, i*4+2, i*4+3],
            "min_val": min_val,
            "max_val": max_val
        }
        transform_params["chunks"].append(chunk_info)
        
        # 归一化到0-1范围
        if min_val != max_val:  # 避免除以零
            normalized = (chunk - min_val) / (max_val - min_val)
        else:
            normalized = torch.zeros_like(chunk)
        
        # 使用torchvision.utils.save_image直接保存
        output_path = os.path.join(output_folder, f"{file_prefix}_{i:03d}.png")
        vutils.save_image(normalized, output_path)
    
    # 保存变换参数
    params_path = os.path.join(output_folder, f"{file_prefix}_transform_params.json")
    with open(params_path, 'w') as f:
        json.dump(transform_params, f, indent=2)
    
    return num_pngs


def restore_features_from_png(png_folder, output_pt_file):
    # 加载变换参数
    params_files = [f for f in os.listdir(png_folder) if f.endswith('_transform_params.json')]
    if not params_files:
        raise ValueError(f"在 {png_folder} 中找不到变换参数文件")
    
    params_path = os.path.join(png_folder, params_files[0])
    with open(params_path, 'r') as f:
        transform_params = json.load(f)
    
    original_shape = transform_params["original_shape"]
    
    restored_features = torch.zeros(original_shape, dtype=torch.float32)
    
    prefix = params_files[0].split('_transform_params.json')[0]
    
    for chunk_info in tqdm(transform_params["chunks"]):
        i = chunk_info["index"]
        min_val = chunk_info["min_val"]
        max_val = chunk_info["max_val"]
        channels = chunk_info["channels"]
        
        png_path = os.path.join(png_folder, f"{prefix}_{i:03d}.png")
        img = Image.open(png_path)
        
        # 将PNG转换为张量 [H, W, 4] -> [4, H, W]
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # 逆向变换：从0-255反向映射到原始特征范围
        denormalized = (img_tensor / 255.0) * (max_val - min_val) + min_val
        
        # 放回到对应的通道位置
        for j, channel in enumerate(channels):
            if channel < original_shape[0]:  # 安全检查
                restored_features[channel] = denormalized[j]
    
    # 如果原始特征有批次维度，则添加回来
    if len(transform_params.get("original_batch_shape", [])) > 0:
        restored_features = restored_features.unsqueeze(0)
    
    # 保存还原后的特征
    torch.save(restored_features, output_pt_file)
    print(f"还原的特征已保存到 {output_pt_file}")
    
    return restored_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['save', 'restore', 'compare', 'all'],
                        help='操作模式: save(保存为PNG), restore(从PNG还原), compare(比较原始和还原), all(全部流程)')
    parser.add_argument('--pt_file', type=str, help='输入的PT文件路径')
    parser.add_argument('--png_folder', type=str, help='PNG文件夹路径')
    parser.add_argument('--output', type=str, help='输出文件或文件夹路径')
    args = parser.parse_args()
    
    if args.mode == 'save' or args.mode == 'all':
        if not args.pt_file:
            raise ValueError("保存模式需要指定 --pt_file")
        if not args.output:
            raise ValueError("保存模式需要指定 --output 作为PNG输出文件夹")
        save_features_as_png_from_tensor(args.pt_file, args.output)
    
    if args.mode == 'restore' or args.mode == 'all':
        png_folder = args.png_folder if args.mode == 'restore' else args.output
        output_pt = args.output if args.mode == 'restore' else args.pt_file + ".restored.pt"
        if not png_folder:
            raise ValueError("还原模式需要指定 --png_folder")
        restore_features_from_png(png_folder, output_pt)
    