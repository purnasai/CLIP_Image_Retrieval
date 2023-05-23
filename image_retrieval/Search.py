import random
import h5py
import faiss

from PIL import Image
import numpy as np

from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

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
print("sample:", random.choices(jewellery_filenames,k=5))

# faiss.indexFlatIP()# inner product
faiss_index =  faiss.IndexFlatL2(jewellery_features.shape[1])
faiss_index.add(jewellery_features)

image = Image.open("sampleImagePath")
t_image = transform(image).unsqueeze(dim=0)
querry_features = model.get_image_features(t_image).detach().numpy()

k = 10  # number of neighbors to retrieve
distances, indices = faiss_index.search(querry_features, k)
for index in range(k):
    print(jewellery_filenames[indices[0][index]], distances[0][index])


