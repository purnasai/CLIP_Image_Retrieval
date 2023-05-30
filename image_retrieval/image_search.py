"""
Author:Purnasai
Description:This file loads h5py file and 
           searches for match image.
"""
import random
import h5py
import faiss

from PIL import Image
import numpy as np

from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

transform =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])

# Open the HDF5 file for reading
with h5py.File('features/Jewellery_features.h5', 'r') as h5f:
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

image = Image.open("./Data/Rings\\C023064_32.jpeg")
t_image = transform(image).unsqueeze(dim=0)
querry_features = model.get_image_features(t_image)
querry_features /= querry_features.norm(dim=-1, keepdim=True)
querry_features = querry_features.detach().numpy()

K_neighbours = 10  # number of neighbors to retrieve
distances, indices = faiss_index.search(querry_features, K_neighbours)
for index in range(K_neighbours):
    score = max(0, round(distances[0][index]*100))
    print(jewellery_filenames[indices[0][index]], score)
