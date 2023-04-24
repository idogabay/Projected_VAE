# Projected_VAE
  - In this project we explore with a new type of VAE called Projected VAE (PVAE for short).  
  - The advantge of PAVE is the ability to perform better image generation of small datasets.  
  - In this project we have a training and infering (creating image) enviroment for the PVAE.  
  
    
![1](https://github.com/idogabay/Projected_VAE/blob/270df427193767f2d8dd28a60b0b3a59bb3b21b1/readme_imgs/top.jpg)
![1](https://github.com/idogabay/Projected_VAE/blob/270df427193767f2d8dd28a60b0b3a59bb3b21b1/readme_imgs/architecture.jpg)  
this projected is based of [Projected GAN](https://github.com/autonomousvision/projected-gan) project
## Requirements to run  
To install the required libraries, run:
```bibtex
python -m pip install -r requirments.txt
```


## training
you can train are own PVAE this your selected dataset.
### data preperation  
the dataset folder format must be:  
  - dataset_name  
    - resized_images  
      - all_images.jpg
    - weights  
   - **notice that you should resize the dataset images to 256x256 therefore the folder "resized_images"**
### training  
  - you can train the PVAE by running "train.py".  
  
tested in VScode.  
**notice you adapt the paths for the right places in your computer**  

## Generating Images  
  - Weights for flowers dataset can be downloaded [here](https://drive.google.com/drive/folders/13E3UjUSg8k6vPrz3NapMZaDaNuEhZTz5?usp=sharing)
  - Run the script "generate_images.py"  
  
tested in VScode.  
**notice you adapt the paths for the right places in your computer**  
## Datasets
we used datasets : Flowers, Pokemons,Obama. all can be downloaded [in this link](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view)
### FLowers ~8000 images
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/flowers1.jpg)
![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/flowers2.jpg)  
### Pokemon ~800 images
![1](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/pokemon1.jpg) ![2](https://github.com/idogabay/Projected_VAE/blob/975751538a1a202ed438a7af5d7a7b9f8b83ad58/readme_imgs/pokemon2.jpg)  
### Obama 100 images
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

## Results  
 - In most cases PVAE performed better then VAE.  
 - We compared the two using FID and the Flowers dataset in a decreasing size:
 ![1](https://github.com/idogabay/Projected_VAE/blob/a9ab72267143858219b89d3c61d287f98d9f5c43/readme_imgs/graph.jpg)
