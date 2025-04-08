# Feature Zip
In current package, we provide a simple compression method for compressing feature in picture and video. We reached at 10 times compression for image, and 15 times compression for video for lossless compression.

| Precision| Format   | FPS      | Relative Err| Ratio    | 
|----------|----------|----------|-------------|----------|
| INT 16   | PNG      |   2      |    2.90e-5  | 2.40     |
| INT 8    | PNG      |   4      |    7.46e-3  | 9.70     |
| INT 8    | JPG      |   5      |    2.31e-2  | 47.0     |


## Format
- We can compress a H,W,C channel tensor using our feature_handler, and you can select compressing method in detail
- We can also compress using mp4 file in the main.py, but not accomplished yet

## ftz format:
After we compressing using ftz format, we have a sequence of RGB images, and a meta data
```
- |img.png
  |img.png
  |img.png
   ...
  |meta.npz 
```

Each image is store the information of 3 channel
![Example](assets/image_0.jpg)

Example and API:
```
output = 'place to save ftz'

tensor: torch.Tensor = torch.random((H,W,C))

handler = FeatureHandler()
handler.save(feature_tensor = tensor, location=output, precision=INT8, format='jpg')

tensor_back = handler.load(output)

relative_error = torch.norm(tensor - tensor_back.cuda()) / torch.norm(tensor)
print(f"Relative Error: {relative_error.item()}")
```