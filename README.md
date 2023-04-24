# Projected_VAE
In this project we explore with a new type of VAE called Projected VAE (PVAE for short).  
the advantge of PAVE is ability to perform beter imgae generation of small datasets.
In this project we have a training and infering (creating image) enviroment for the PVAE.  
this projected is based of [Projected GAN](https://github.com/autonomousvision/projected-gan) project
## training:
you can train are own PVAE this your selected dataset.
### data preperation  
the dataset folder format must be:  
  - dataset_name  
    - resized_images  
      - all_images.jpg
    - weights  
   - **notice that you should resize the dataset images to 256x256 therefore the folder "resized_images"**
### training  
you can train the PVAE by running "train.py"  
**notice you adapt the paths for the right places in your computer**  

## Generating Images  
run the script "generate_images.py"  
**notice you adapt the paths for the right places in your computer**  

## Datasets
we used datasets : a, b,c. all can be downloaded [in this link](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view)
### Dataset a
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/flowers1.jpg)
![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/flowers2.jpg)
### Dataset b
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/pokemon1.jpg) ![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/pokemon2.jpg)
### Dataset c
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/obama1.jpg)![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/obama2.jpg)

## Results on decreasing size dataset - Flowers VAE vs PVAE:
### 8000 images
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/8000vae.jpg)
![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/8000pvae.jpg)
### 2000 images
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/2000vae.jpg)
![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/2000pvae.jpg)
### 500 images
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/1000vae.jpg)
![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/1000pvae.jpg)
