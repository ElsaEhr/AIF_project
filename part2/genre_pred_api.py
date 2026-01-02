import argparse
import json
import io

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

from test_functions import DKNN
from model_resNet import poster_classifier

import pickle



try:
    with open("part2/dknn.pkl", "rb") as f: 
        data = pickle.load(f) 
    print("Chargement dknn")    
    dknn = data["dknn"]
    threshold = data["threshold"]
except FileNotFoundError:
    print("ERREUR : Le fichier dknn.pkl est introuvable.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="weights_resNet/poster_classifier.pth", help="model path")
parser.add_argument("--labels_path", type=str, default="content/labels.json", help="labels json path")
args = parser.parse_args()

# ---- Load labels (id -> genre) ----
with open(args.labels_path, "r") as f:
    labels = json.load(f)  # e.g. ["Action", "Comedy", "Drama", ...]

# ---- Load model ----
model = poster_classifier().to(device)
state_dict = torch.load(args.model_path, map_location=device,weights_only=True)  # state_dict pur
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),  # adapte selon ton besoin
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



@app.route("/predict", methods=["POST"])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary)).convert("RGB")

    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)                 # shape (1, C)
        probs = F.softmax(outputs, dim=1)[0]    # shape (C,)
        pred_id = int(torch.argmax(probs).item())
        pred_label = labels[pred_id]

    return jsonify({
        "prediction": pred_id,
        "label": pred_label,
        "confidence": float(probs[pred_id].item()),
        # "probs": probs.tolist(),  # décommente si tu veux toutes les probas
    })

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    images_binary = request.files.getlist("images[]")
    if len(images_binary) == 0:
        return jsonify({"error": "No files provided. Use key images[]"}), 400

    tensors = []
    for img_file in images_binary:
        img_pil = Image.open(img_file.stream).convert("RGB")
        tensors.append(transform(img_pil))

    batch_tensor = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)                 # (B, C)
        probs = F.softmax(outputs, dim=1)             # (B, C)
        pred_ids = torch.argmax(probs, dim=1).tolist()
        pred_labels = [labels[i] for i in pred_ids]
        confidences = [float(probs[j, i].item()) for j, i in enumerate(pred_ids)]

    return jsonify({
        "predictions": pred_ids,
        "labels": pred_labels,
        "confidences": confidences,
        # "probs": probs.tolist(),  # optionnel
    })

@app.route("/validate_poster", methods=["POST"])
def validate_poster():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary)).convert("RGB")


    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor,return_features=True)                 # shape (1, C)
    
    score=dknn.compute_scores(features)

    return jsonify({"is a poster": (score < threshold).item()})


@app.route("/batch_validate_poster", methods=["POST"])
def batch_validate_poster():
    images_binary = request.files.getlist("images[]")
    if len(images_binary) == 0:
        return jsonify({"error": "No files provided. Use key images[]"}), 400
    
    tensors = []
    for img_file in images_binary:
        img_pil = Image.open(img_file.stream).convert("RGB")
        tensors.append(transform(img_pil))

    batch_tensor = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        features = model(batch_tensor,return_features=True) 
        scores=dknn.compute_scores(features)
    
    is_poster = scores < threshold
    poster_val = is_poster.tolist()

    return jsonify({"is a poster": poster_val})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5075, debug=True)
