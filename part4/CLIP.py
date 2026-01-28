import os
import clip
from PIL import Image
from pathlib import Path
from annoy import AnnoyIndex

model, preprocess = clip.load("ViT-B/32", device=device)

# We are going to load and collect the images which are all in subfolders of a root folder 'root_dir' (choose the path where the images of the project are located)

root_dir = Path.cwd()
print(root_dir)
exit()
image_paths = []
# collects all the paths of subdirectories

for root, dirs, files in os.walk(root_dir):
    print("ROOT:", root)
    print("FILES:", files)
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))

