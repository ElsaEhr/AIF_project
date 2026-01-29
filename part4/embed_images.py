import os
import torch
from PIL import Image

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