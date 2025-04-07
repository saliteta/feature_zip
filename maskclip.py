# import torch
# import torchvision.transforms as T
# from PIL import Image

# from featup.util import norm, unnorm
# import os

# import argparse
# from tqdm import tqdm



# input_size = 224
# image_path = "test.jpg"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_norm = True


# transform = T.Compose([
#     T.Resize((224,288)),
#     T.ToTensor(),
#     norm
# ])

# def parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image_folder', help="Colmap image files", type=str, required=True)
#     parser.add_argument('--output_feature_folder', help="Output feature folder location, everything is saved as .pt", type=str, required=True)
#     args = parser.parse_args()
#     return args



# if __name__ == "__main__":
#     args = parser()

#     image_folder = args.image_folder
#     feature_folder = args.output_feature_folder
#     allowed_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


#     os.makedirs(feature_folder, exist_ok=True)
#     images = [
#         f for f in os.listdir(image_folder)
#         if f.lower().endswith(allowed_extensions)
#     ]
#     upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)

#     for image in tqdm(images):
#         path = os.path.join(image_folder, image)
#         image_tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

#         hr_feats = upsampler(image_tensor)
#         lr_feats = upsampler.model(image_tensor)
#         torch.save(hr_feats, os.path.join(feature_folder, image.split('.')[0]+'.pt'))
import torch
import torchvision.transforms as T
from PIL import Image
from featup.util import norm, unnorm
import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import torchvision.utils as vutils
input_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_norm = True

transform = T.Compose([
    T.Resize((224,288)),
    T.ToTensor(),
    norm
])


def save_features_as_png_from_tensor(features, output_folder, file_prefix):
    os.makedirs(output_folder, exist_ok=True)
    
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
        chunk = reshaped_features[i]
        min_val = float(min_vals[i].item())
        max_val = float(max_vals[i].item())
        
        chunk_info = {
            "index": i,
            "channels": [i*4, i*4+1, i*4+2, i*4+3],
            "min_val": min_val,
            "max_val": max_val
        }
        transform_params["chunks"].append(chunk_info)
        
    
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

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', help="Colmap image files", type=str, required=True)
    parser.add_argument('--output_folder', help="Output feature folder location for PNG files", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()

    image_folder = args.image_folder
    output_base_folder = args.output_folder
    
    path_parts = os.path.normpath(image_folder).split(os.sep)
    path_parts = [p for p in path_parts if p]
    scene_name = path_parts[-2] if len(path_parts) >= 2 else "scene"
    

    scene_folder = os.path.join(output_base_folder, scene_name)
    os.makedirs(scene_folder, exist_ok=True)
    
    allowed_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


    images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(allowed_extensions)
    ]
    
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)

    for image in tqdm(images, desc="处理图片"):
        image_name = image.split('.')[0]
        
        image_output_folder = os.path.join(scene_folder, image_name)
        os.makedirs(image_output_folder, exist_ok=True)
        
        path = os.path.join(image_folder, image)
        image_tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        hr_feats = upsampler(image_tensor)
        
        num_pngs = save_features_as_png_from_tensor(hr_feats, image_output_folder, image_name)
        
        print(f"图片 {image} 已处理: 保存了 {num_pngs} 张PNG到 {image_output_folder}")