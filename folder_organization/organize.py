import os
import torch
import transformers
from PIL import Image
import shutil

from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

folder_path = "Data1/unorganized_data"
relocate_folder_path = "Data1/organized_data"
files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

folder_names = ["rings","chain","earrings","necklace"]

transform =  transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])

def copy_image(folder_name, file_name):
    old_path = file_name
    file = file_name.split("\\")[-1]
    new_folder = os.path.join(relocate_folder_path, 
                            folder_name)
    new_file_path = os.path.join(new_folder, file)
    
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    shutil.move(old_path, new_file_path)

for file in files:
    img = Image.open(file)
    # t_img = transform(img)
    inputs = processor(text = folder_names,
                       images= img,
                       return_tensors="pt",
                       padding=True)

    outputs = model(**inputs)
    logits_per_img = outputs.logits_per_image
    probs = logits_per_img.softmax(dim=1)
    decided_folder = folder_names[torch.argmax(probs)]
    copy_image(decided_folder, file)


