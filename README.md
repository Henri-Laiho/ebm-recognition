# Recognition as Navigation in Energy-Based Models

Pytorch code for the [thesis](cs.ut.ee),
based on the [code](https://github.com/yilundu/improved_contrastive_divergence) 
for the paper [Improved Contrastive Divergence Training of Energy Based Models](https://arxiv.org/abs/2012.01316)

## Installation

1. Create a new environment and install the requirements file:

```
pip install -r requirements.txt
```

2. Download the dataset, [CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)

3. Download the [pretrained models](https://www.dropbox.com/sh/4p43o1kgt804kwg/AADZF89qY89UdwzYJYvVzVmha?dl=0) 
that were shared in [this repository](https://github.com/yilundu/improved_contrastive_divergence).

4. Place the files like this:
```
./
    README.md
    ./CelebA/
        ./CelebA/Anno/
            identity_CelebA.txt
            list_attr_celeba.txt
            ...
        ./CelebA/img_align_celeba/
                000001.jpg
                ...
                202599.jpg
    ./celeba_combine/
        ./celeba_combine/celeba_128_male_2/
            model_latest.pth
        ./celeba_combine/celeba_128_old_2/
            model_6000.pth
        ./celeba_combine/celeba_128_smiling_2/
            model_13000.pth
        ./celeba_combine/celeba_128_wavy_hair_2/
            model_9000.pth
```

## Running the experiments

The following script is used to run the experiments. The command line arguments can be used to change the output 
behaviour and some dataset options.
```
 python celeba_combine/walk_visualisation.py
```
The file ```celeba_combine/experiment_conf.py``` is used to define more detailed experiment configurations.


