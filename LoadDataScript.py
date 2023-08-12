# --- Load Data --- #
import pandas as pd
import torchvision.transforms as T
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import PIL
import os

# --- Function to transform data --- #

transform = T.Compose([T.ToTensor(), #Change image format to tensor
                       T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #Normalise pixel values
                       T.Resize(size = (100,100)) #Reduce the image size
                       ])

# --- File Directories --- #
image_dir = '/Data/images/images/' 
labels_csv = '/Data/annotations.csv' 
dataset_annotations = pd.read_csv(labels_csv)

# Names of example images to elicit distributions for
image_names_of_testing = np.array(["MHIST_dog.png","MHIST_cnh.png","MHIST_aai.png","MHIST_ahg.png"])
rslt_df = dataset_annotations[dataset_annotations['Image Name'].isin(image_names_of_testing)]
# Remove images that we want to elicit distributions for from model building datasets
dataset_without_testing = dataset_annotations[~dataset_annotations['Image Name'].isin(image_names_of_testing)]

# ---- Function to create datasets ---- #
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, pd_dataset, image_dir, transform = None):
        self.df = pd_dataset
        self.image_dir = image_dir
        self.transform = transform
        self.class2index = {"SSA":1, "HP":0}

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.iloc[index]['Image Name']
        agree = self.df.iloc[index]['Number of Annotators who Selected SSA (Out of 7)']
        label = self.class2index[self.df.iloc[index]['Majority Vote Label']] 
        image = PIL.Image.open(os.path.join(self.image_dir, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label, filename, agree
    
# --- Obtain datasets --- #
# Full training and testing dataset
dataset = CustomDataset(dataset_without_testing, image_dir, transform)
# Dataset containing only the examples used to elicit distributions
dataset_testing = CustomDataset(rslt_df, image_dir, transform)