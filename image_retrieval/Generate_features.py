"""
Author:Purnasai
Description:This file generates image features from
        Database of images & stores them h5py file.
"""
import os
import h5py
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32" 
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def get_labels(files):
    labels = []
    for file_path in files:
        directory, filename = file_path.split("\\")
        directory_parts = directory.split("/")
        label = directory_parts[-1]
        if label not in labels:
            labels.append(label)
    return labels
    
def list_files(dir):
    images = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            images.append(os.path.join(root, name))
    return images


class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list_files(self.img_dir)[:1000]
        random.choices(self.images, k=5)
        self.transform =  transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                [0.229, 0.224, 0.225])
                        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path
    

dir_path = "./Data/"
dataset = CustomImageDataset(dir_path)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True,)


final_img_features = []
final_img_filepaths = []
for image_tensors, file_paths in tqdm(train_dataloader):
    try:
        image_features = model.get_image_features(image_tensors) #512
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.tolist()
        final_img_features.extend(image_features)
        final_img_filepaths.extend((list(file_paths)))
    except Exception as e:
        print("Exception occurred: ",e)
        break


with h5py.File('features/Jewellery_features.h5','w') as h5f:
    h5f.create_dataset("jewellery_features", data= np.array(final_img_features))
    # to save file names strings in byte format. 
    h5f.create_dataset("jewellery_filenames", data= np.array(final_img_filepaths,
                                                             dtype=object))