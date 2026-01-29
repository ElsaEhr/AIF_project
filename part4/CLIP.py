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

import os
import clip
from PIL import Image
from pathlib import Path
from annoy import AnnoyIndex

# Load device

if torch.backends.mps.is_available():
    # MPS is the GPU model in Mac technology
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device =torch.device("cpu")

#load device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

###########
#Impornts from our own files

from embed_images import embed_images

#load du model
model, preprocess = clip.load("ViT-B/32", device=device)

# We are going to load and collect the images which are all in subfolders of a root folder 'root_dir' (choose the path where the images of the project are located)

root_dir = "../MovieGenre/content"
image_embeddings, image_paths = embed_images(directory_images=root_dir, model=model,
                                            preprocess=preprocess, device=device)

image_embeddings = image_embeddings.cpu().numpy()
print(type(image_embeddings),image_embeddings.shape)
