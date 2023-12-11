# Image Super-Resolution via Iterative Refinement for Split Image Experiment

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) |  [Project](https://iterative-refinement.github.io/ )

## Brief

Code Submission for UCLA CS245: Big Data Analytics course offered in Fall 2023

This is an unofficial implementation of **Image Super-Resolution via Iterative Refinement(SR3)** by **PyTorch** and based on the git repository https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement. The following are instructions on how to setup this repository for the Split Image Experiment. The code in this branch is only set up to perform 16x16 --> 128x128 super-resolution, spliting the image into 4 quadrants. Splitting the image into more parts would require changes to the edge consistency term in the loss function in model/sr3_modules/diffusion.py


## Usage
### Environment
```python
pip install -r requirement.txt
```

### Pretrained Model

The following is the weights for the pretrained model for the super-resoltion task 16x16 -> 128x128 that was trained on the FFHQ dataset.

| Tasks                             | Platform（Code：qwer)                                        | 
| --------------------------------- | ------------------------------------------------------------ |
| 16×16 -> 128×128 on FFHQ-CelebaHQ | [Google Drive](https://drive.google.com/drive/folders/12jh0K8XoM1FqpeByXvugHHAF3oAZ8KRu?usp=sharing)\|[Baidu Yun](https://pan.baidu.com/s/1OzsGZA2Vmq1ZL_VydTbVTQ) |  
 

```python
# Download the pretrained model and edit [sr|sample]_[ddpm|sr3]_[resolution option].json about "resume_state":
"resume_state": [your pretrained model's path]
```

### Data Prepare

#### New Start

If you didn't have the data, you can prepare it by following steps:

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256)

Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128
```

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "img", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
```

#### Own Data

You also can use your image data by following steps, and we have some examples in dataset folder.

At first, you should organize the images layout like this, this step can be finished by `data/prepare_data.py` automatically:

```shell
# set the high/low resolution images, bicubic interpolation images path 
dataset/celebahq_16_128/
├── hr_128 # it's same with sr_16_128 directory if you don't have ground-truth images.
├── lr_16 # vinilla low resolution images
└── sr_16_128 # images ready to super resolution
```

```python
# super resolution from 16 to 128
python data/prepare_data.py  --path [dataset root]  --out celebahq --size 16,128
```

*Note: Above script can be used whether you have the vanilla high-resolution images or not.*

then you need to change the dataset config to your data path and image resolution: 

```json
"datasets": {
    "train|val": { // train and validation part
        "dataroot": "dataset/celebahq_16_128",
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "img", //lmdb or img, path of img files
    }
},
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/split.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/split.json

# Quantitative evaluation alone using SSIM/PSNR/Consistency metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c [config file]
```

## Acknowledgements

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)

Furthermore, we are benefitting a lot from the following projects:

- https://github.com/bhushan23/BIG-GAN
- https://github.com/lmnt-com/wavegrad
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/hejingwenhejingwen/AdaFM
- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
