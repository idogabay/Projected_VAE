# Projected_VAE  
> *Abstract:* *In recent years, generative models have gained significant attention for their ability to learn the distribution of a given dataset and generate new examples. However, the training process usually involves a large dataset, which is a costly and time-consuming process. This poses a challenge for generative models when training on small datasets.  
To address this issue, we proposed a new architecture called Projected-VAE, which uses a pre-trained classification network to extract features from an image and then trains the model on these feature maps. this method is based on the Projected-GAN model. With this model we aim to improve the performance of VAE models on small datasets.  
While working on this project, we tested our model on various datasets of different sizes and trained it with different parameters to find the optimal weights. We evaluated our projected VAE model using the FID score, which measures the similarity between generated images and real images from the dataset.  
> To verify whether the project's goal was achieved, we compared the performance of our model to that of a regular VAE model with the same architecture. The results of our model outperformed the regular VAE model in most cases, demonstrating the effectiveness of our approach.*
  
  - In this project we explore with a new type of VAE called Projected VAE (PVAE for short).  
  - The advantge of PAVE is the ability to perform better image generation of small datasets.  
  - In this project we allow both training and generation of images using PVAE.
  
  
![1](https://github.com/idogabay/Projected_VAE/blob/a1db156c8a253d61994b7ba57bed1716c4ec0cda/readme_imgs/architecture.jpg)  
![1](https://github.com/idogabay/Projected_VAE/blob/a1db156c8a253d61994b7ba57bed1716c4ec0cda/readme_imgs/top1.jpg)
![1](https://github.com/idogabay/Projected_VAE/blob/a1db156c8a253d61994b7ba57bed1716c4ec0cda/readme_imgs/top2.jpg)
this projected is based of [Projected GAN](https://github.com/autonomousvision/projected-gan) project
## Requirements to run  
To install the required libraries, run:
```bibtex
python -m pip install -r requirments.txt
```


## training
you can train your own PVAE with your selected dataset.
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
### Flowers ~8000 images
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
