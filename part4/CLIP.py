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
from pathlib import Path


###########
#Impornts from our own files
from embed import embed_plot
from annoy_builder import index_builder

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


#load du model
model, preprocess = clip.load("ViT-B/32", device=device)

# We are going to load and collect the images which are all in subfolders of a root folder 'root_dir' (choose the path where the images of the project are located)

BASE_DIR = Path(__file__).resolve().parent
plots_path = BASE_DIR.parent / "movie_plots.csv"
image_folder = BASE_DIR.parent / "MovieGenre" / "content"

DIM=512
annoy_index, metadata = index_builder(plots_path,image_folder,DIM,
                  model,preprocess,device)

user_input="a film with samurais and swords in Japan."
query_emb = embed_plot(user_input, model,device)

results = []
movie_ids = annoy_index.get_nns_by_vector(query_emb, n=1)

for movie_id in movie_ids:
    movie = metadata[movie_id]
    results.append({
        "poster": movie["movie_poster_path"],
        "plot": movie["movie_plot"],
        "category": movie["movie_category"]
    })

