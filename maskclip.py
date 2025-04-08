from featureHandler import FeatureHandler, INT16, save_mpf

import torch 
from tqdm import tqdm

import os

if __name__ == "__main__":
    file_names = os.listdir('/data/lerf_ovs/figurines/feature')
    count = 0
    file_names.sort()
    featuireHandler = FeatureHandler()
    for file in tqdm(file_names):  
        tensor: torch.Tensor = torch.load(os.path.join('/data/lerf_ovs/figurines/feature', file))
        tensor = tensor.squeeze().permute(1,2,0)
        featuireHandler.save(tensor, f'test/{count:05d}' , precision=INT16)
        count+=1
    
    #save_mpf('/home/ICT2000/bxiong/workspace/semanticLifting/feature_zip/test', 
    #'/home/ICT2000/bxiong/workspace/semanticLifting/feature_zip/video', framerate=10)