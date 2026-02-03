import os
import torch
from PIL import Image
import clip
import pandas as pd
import numpy as np

def embed_images(directory_images,model, preprocess,device):
    """
    Function that embeds the images present in directory_images according to a model
    :param directory_images:
    :return:
    """

    image_paths = []
    # collects all the paths of subdirectories
    print("Retrieving images from directory")
    for root, dirs, files in os.walk(directory_images):
        #print("ROOT:", root)
        #print("FILES:", files)
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    print("All image files found")
    # load les images
    images = []
    for path in image_paths:
        img = preprocess(Image.open(path).convert("RGB"))
        images.append(img)

    images = torch.stack(images).to(device)
    print("All images in memory")
    # We create embeddings for the images
    print("Creating embeddings")
    with torch.no_grad():
        image_embeddings = model.encode_image(images)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    print("Images have been emebedded, and their embeddings have been normalized")

    return image_embeddings, image_paths


def embed_poster(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path).convert("RGB"))
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=1, keepdim=True)

    return emb.cpu().numpy()[0]


def embed_plot(plot, model, device):
    tokens = clip.tokenize([plot]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.cpu().numpy()[0]

def embed_plots(plots_file, model, preprocess, device):
    plots_dataframe = pd.read_csv(plots_file)
    plots = plots_dataframe["movie_plot"]

    embeds = []
    for plot in plots:
        embeds.append(embed_plot(plot, model, preprocess, device))
    return embeds


def chunk_text(text, max_chars=500):
    """Splits text into chunks of max_chars each"""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def embed_movie(plot, poster_path, model, preprocess, device):
    # chunk plot
    plot_chunks = chunk_text(plot, max_chars=78)

    # embed each chunk
    chunk_embeddings = [embed_plot(chunk, model, device) for chunk in plot_chunks]

    # average chunk embeddings
    plot_emb = np.mean(chunk_embeddings, axis=0)

    # embed poster
    poster_emb = embed_poster(poster_path, model, preprocess, device)

    # combine text + image embeddings
    movie_emb = (plot_emb + poster_emb) / 2
    return movie_emb