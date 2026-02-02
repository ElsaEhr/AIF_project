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
import os

import pandas as pd
from annoy import AnnoyIndex
from transformers import DistilBertTokenizerFast



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#DKNN related paths
DKNN_PATH = os.path.join(BASE_DIR, "dknn.pkl")

#Plot classification related paths
PLOT_MODEL_PATH = os.path.join(BASE_DIR, "weights/text_classifier.pth")
PLOT_LABELS_PATH = os.path.join(BASE_DIR, "weights/plot_labels.json")

#ANNOY related paths
ANNOY_PATH = os.path.join(BASE_DIR, "embeddings/movie_plots.ann")
ANNOY_META_PATH = os.path.join(BASE_DIR, "embeddings/movie_plots_index.csv")

PLOT_MAX_LEN = 256
PLOT_EMBED_DIM = 300


try:
    with open(DKNN_PATH, "rb") as f:
        data = pickle.load(f)
    print("Chargement dknn")
    dknn = data["dknn"]
    threshold = data["threshold"]
except FileNotFoundError:
    print(f"ERREUR : Fichier introuvable : {DKNN_PATH}")


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

# ANNOY and NLP model loading
# ---- Load tokenizer ----
plot_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# ---- Load plot labels ----
with open(PLOT_LABELS_PATH, "r") as f:
    plot_labels = json.load(f)  # list of genres in the correct order

# ---- Load NLP model ----
plot_model = TextClassifier(
    vocab_size=plot_tokenizer.vocab_size,
    embedding_dim=PLOT_EMBED_DIM,
    num_classes=len(plot_labels)
).to(device)

plot_state = torch.load(PLOT_MODEL_PATH, map_location=device, weights_only=True)
plot_model.load_state_dict(plot_state)
plot_model.eval()

# ---- Load Annoy index + metadata ----
annoy_index = AnnoyIndex(PLOT_EMBED_DIM, "angular")
annoy_index.load(ANNOY_PATH)

annoy_meta = pd.read_csv(ANNOY_META_PATH)  # columns: annoy_id, label, plot

# Function to convert plot to embedding
def plot_to_embedding(plot: str):
    enc = plot_tokenizer(
        str(plot),
        truncation=True,
        padding=False,
        max_length=PLOT_MAX_LEN,
        return_tensors=None
    )
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([input_ids.shape[1]], device=device)

    with torch.no_grad():
        emb = plot_model.get_embeddings(input_ids, lengths)  # (1, 300)
    return emb

# Function to predict genre from plot
def predict_genre_from_plot(plot: str):
    enc = plot_tokenizer(
        str(plot),
        truncation=True,
        padding=False,
        max_length=PLOT_MAX_LEN,
        return_tensors=None
    )
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([input_ids.shape[1]], device=device)

    with torch.no_grad():
        logits = plot_model(input_ids, lengths)        # (1, C)
        probs = F.softmax(logits, dim=1)[0]            # (C,)
        pred_id = int(torch.argmax(probs).item())
        pred_label = plot_labels[pred_id]
        conf = float(probs[pred_id].item())
    return pred_id, pred_label, conf


#ROUTES
#Genre prediction route
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

#Batch genre prediction route
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

# Plot validation route
@app.route("/validate_poster", methods=["POST"])
def validate_poster():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary)).convert("RGB")


    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor,return_features=True)                 # shape (1, C)
    
    score=dknn.compute_scores(features)

    return jsonify({"is a poster": (score < threshold).item()})

# Batch validation route
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

# Plot genre prediction route // prédiction de genre à partir du synopsis
@app.route("/predict_plot", methods=["POST"])
def predict_plot():
    payload = request.get_json(silent=True)
    if not payload or "plot" not in payload:
        return jsonify({"error": "Missing JSON field 'plot'"}), 400

    pred_id, pred_label, conf = predict_genre_from_plot(payload["plot"])

    return jsonify({
        "prediction": pred_id,
        "label": pred_label,
        "confidence": conf
    })

# Nearest neighbors route // recherche des synopsis similaires
@app.route("/recommend_from_plot", methods=["POST"])
def recommend_from_plot():
    payload = request.get_json(silent=True)
    if not payload or "plot" not in payload:
        return jsonify({"error": "Missing JSON field 'plot'"}), 400

    k = int(payload.get("k", 5))
    k = max(1, min(k, 20))  # garde-fou

    emb = plot_to_embedding(payload["plot"]).squeeze(0).detach().cpu().numpy()

    neigh_ids, dists = annoy_index.get_nns_by_vector(emb, k, include_distances=True)

    recs = []
    for annoy_id, dist in zip(neigh_ids, dists):
        row = annoy_meta[annoy_meta["annoy_id"] == annoy_id].iloc[0]
        recs.append({
            "annoy_id": int(annoy_id),
            "label_id": int(row["label"]),
            "label": plot_labels[int(row["label"])],
            "plot": str(row["plot"])[:400],     # éviter d’envoyer des pavés énormes
            "distance": float(dist)
        })
    return jsonify({"recommendations": recs})
