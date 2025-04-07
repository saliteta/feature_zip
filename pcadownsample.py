import cv2
import os
import torch
import numpy as np
import argparse
from sklearn.decomposition import PCA


def visualize_feature_with_pca(features):
    """
    Visualize a feature map by reducing its channel dimension using PCA.
    
    Given an input feature map of shape (H, W, C) where C is typically 512, 
    this function applies PCA to reduce the features to 3 channels. The resulting
    features are then normalized to the 0-255 range and converted to a BGR image 
    (cv2 standard) which can be directly saved using cv2.imwrite.
    
    Args:
        features (np.ndarray): Input feature map of shape (H, W, C).
        
    Returns:
        np.ndarray: A cv2-compatible BGR image of shape (H, W, 3) with dtype uint8.
    """

    
    H, W, C = features.shape
    
    # Reshape features to (H*W, C) for PCA
    flat_features = features.reshape(-1, C)
    
    # Perform PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    flat_features_pca = pca.fit_transform(flat_features)
    
    # Normalize the PCA output to the range [0, 255]
    min_val = flat_features_pca.min()
    max_val = flat_features_pca.max()
    flat_features_norm = (flat_features_pca - min_val) / (max_val - min_val)
    flat_features_norm = (flat_features_norm * 255).astype(np.uint8)
    
    # Reshape back to (H, W, 3)
    image_rgb = flat_features_norm.reshape(H, W, 3)
    
    # Convert RGB to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    return image_bgr

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_location', help="lifted feature location, feature.pt", type=str, required=True)
    parser.add_argument('--output', help="segmented result", type=str, required=True)
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parser()
    feature_location = args.feature_location
    output_folder = args.output
    for feature_folder in os.listdir(feature_location):
        current_parents_path = os.path.join(feature_location, feature_folder) 
        os.makedirs(os.path.join(output_folder, feature_folder), exist_ok= True)
        for feature_name in os.listdir(current_parents_path):
            feature_file = os.path.join(current_parents_path, feature_name)
            if feature_name.endswith('.pt'):
                feature:torch.Tensor = torch.load(feature_file).detach().cpu().squeeze().permute((1,2,0))
                feature = feature.numpy()
            else:
                feature = np.load(feature_name)
            img = visualize_feature_with_pca(feature)
            cv2.imwrite(os.path.join(output_folder, feature_folder, feature_name.split('.')[0]+'.jpg'), img)
