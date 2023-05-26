"""
Author:Purnasai
Description:This file loads h5py file and 
           searches for match image.
"""
import os
import random
import h5py
import faiss

from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
tokenizer= AutoTokenizer.from_pretrained(MODEL_NAME)


transform =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])


features_file_path = "features/Jewellery_features.h5"
print(f"The Features File size:", round(os.path.getsize(features_file_path)/1000000,2),"MB \n")

# Open the HDF5 file for reading
with h5py.File(features_file_path, 'r') as h5f:
    # Read the dataset named "jewellery_features"
    jewellery_features = np.array(h5f['jewellery_features'])
    # Read the dataset named "jewellery_names"
    jewellery_filenames = np.array(h5f['jewellery_filenames'])


# Print the shape of the arrays to verify the data
print("jewellery_features shape:", type(jewellery_features), jewellery_features.shape)
print("jewellery_names shape:", jewellery_filenames.shape, type(jewellery_filenames))
print("sample:", random.choices(jewellery_filenames,k =5))

# The Inner Product similarity is often used in scenarios 
# where vectors represent semantic or conceptual features.
faiss_index = faiss.IndexFlatIP(jewellery_features.shape[1])
faiss_index.add(jewellery_features)

# L2norm is Euclidean distance, i.e disimilarity. 100-disimarlity
# faiss_index =  faiss.IndexFlatL2(jewellery_features.shape[1])
# faiss_index.add(jewellery_features)

# "gold ring with flower design on top"
text = ["A Long thick heavy necklace with goddess lakshmi on top of it"]


# both gives same. can use any
# text_processed = processor(text=text, return_tensors="pt")
text_tokenized = tokenizer(text=text, padding=True, return_tensors="pt")

querry_features = model.get_text_features(**text_tokenized)
querry_features /= querry_features.norm(dim=-1, keepdim=True)
querry_features = querry_features.detach().numpy()

K_neighbours = 3  # number of neighbors to retrieve
distances, indices = faiss_index.search(querry_features, K_neighbours)
for index in range(K_neighbours):
    print(jewellery_filenames[indices[0][index]], distances[0][index]*100)
