# Imports

# General imports
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

# Pytorch and transformers (for LLM)
import transformers, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
transformers.logging.set_verbosity_info()

# For loading documents from a path
from pathlib import Path

# For the embedding module
from sentence_transformers import SentenceTransformer


# Load device

if torch.backends.mps.is_available():
    # MPS is the GPU model in Mac technology
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device =torch.device("cpu")

print (torch.ones(1, device=device))

import os
import clip
from PIL import Image
from pathlib import Path
from annoy import AnnoyIndex

#load device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#load du model
model, preprocess = clip.load("ViT-B/32", device=device)

# We are going to load and collect the images which are all in subfolders of a root folder 'root_dir' (choose the path where the images of the project are located)

root_dir = Path.cwd()/"/Users/zoe/Desktop/5A/AIF/content"
print(root_dir)

image_paths = []
# collects all the paths of subdirectories

for root, dirs, files in os.walk(root_dir):
    print("ROOT:", root)
    print("FILES:", files)
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))

#load les images 
images = []
for path in image_paths:
    img = preprocess(Image.open(path).convert("RGB"))
    images.append(img)

images = torch.stack(images).to(device)
print(images.shape)

# We create embeddings for the images

with torch.no_grad():
    image_embeddings = model.encode_image(images)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

image_embeddings = image_embeddings.cpu().numpy()
print(type(image_embeddings),image_embeddings.shape)
