## Dependencies
* python 3.6
```bash
pip install -r requirements.txt
```

## Download Pretrained Model
download the whole folder in pretrained model's link and put in a folder name 'pretrained_models'
* Places2 ([Center Mask](https://drive.google.com/drive/folders/1T7uLBwXHRKJWUHVNACkYutvpAyDlbJdD?usp=sharing), [Random Mask](https://drive.google.com/drive/folders/1dk1zSm1FxZVaafOtvoud8aAdZ6Ubs4oU?usp=sharing))

## Usage
```bash
python single_test.py --image image/[image name] --mask mask/[mask name] --output output/[output file name] --pretrained_model_dir pretrained_models/[model dir] --checkpoint_prefix [model prefix]
```
example
```bash
python single_test.py --image image/002.png --mask mask/center_mask.png --output output/002_random.png --pretrained_model_dir pretrained_models/places2_random --checkpoint_prefix places2_256x256_random_mask
````

## Reference
[Hypergraph Image Inpainting](https://github.com/GouravWadhwa/Hypergraphs-Image-Inpainting)