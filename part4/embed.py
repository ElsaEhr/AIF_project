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
    tokens = clip.tokenize([plot], truncate=True).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.cpu().numpy()[0]


def embed_plots(plots_file, model, device):
    plots_dataframe = pd.read_csv(plots_file)
    plots = plots_dataframe["movie_plot"]

    embeds = []
    for plot in plots:
        embeds.append(embed_plot(plot, model, device))
    return embeds


def embed_movie(plot, poster_path, model, preprocess, device, alpha=0.5):
    # embed plot
    plot_emb = embed_plot(plot, model, device).astype(np.float32) #annoy needs float32

    # embed poster
    poster_emb = embed_poster(poster_path, model, preprocess, device).astype(np.float32)

    # combine (weighted average)
    movie_emb = alpha * plot_emb + (1 - alpha) * poster_emb

    # (optionnel) renormaliser pour rester comparable en cosinus/angular
    norm = np.linalg.norm(movie_emb)
    if norm > 0:
        movie_emb = movie_emb / norm

    return movie_emb
