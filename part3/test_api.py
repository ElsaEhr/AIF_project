import requests
import io
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from dknn import testset

to_pil = transforms.ToPILImage()
dataset_path = '../content/'


transform = transforms.Compose([
    transforms.Resize((224,224)),  # adapte selon ton besoin
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


anomalies = ImageFolder(root="../flickr8k", transform=transform)

print("\n test with poster images \n")

for i in range(20):
    k=np.random.randint(len(testset))
    image, label = testset[k]
    # Convert image to bytes
    if image.is_cuda:
        image = image.cpu()
        
    
    pil_image = to_pil(image)
    
    # 2. Now the PIL Image object has the .save() method
    img_binary = io.BytesIO()
    # Use the PIL image's save method
    pil_image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://127.0.0.1:5075/validate_poster", data=img_binary.getvalue())

    print("Is a poster:", response.json()["is a poster"], "// True Label: True")

print("\n test with random images \n")

for i in range(20):
    k=np.random.randint(len(anomalies))
    image, label = anomalies[k]
    # Convert image to bytes
    if image.is_cuda:
        image = image.cpu()
        
    
    pil_image = to_pil(image)
    
    # 2. Now the PIL Image object has the .save() method
    img_binary = io.BytesIO()
    # Use the PIL image's save method
    pil_image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://127.0.0.1:5075/validate_poster", data=img_binary.getvalue())
    print("Is a poster:", response.json()["is a poster"], "// True Label: False")

print("\n test for batch prediction \n")

images = []
labels = []

for i in range(20):
    k=np.random.randint(len(testset))
    image, label = testset[k]
    # Convert image to bytes
    if image.is_cuda:
        image = image.cpu()
        
    
    pil_image = to_pil(image)
    
    # 2. Now the PIL Image object has the .save() method
    img_binary = io.BytesIO()
    # Use the PIL image's save method
    pil_image.save(img_binary, format="PNG")
    images.append(('images[]', (f"image_{i}.png", img_binary.getvalue(), 'image/png')))
    labels.append(label)
    # Send request to the API for batch prediction
response = requests.post("http://127.0.0.1:5075/batch_validate_poster", files=images)
predictions = response.json()["is a poster"]

for i, (pred, true_label) in enumerate(zip(predictions, labels)):
    print(f"Image {i+1} - is a poster :", pred, "| True Label: True")