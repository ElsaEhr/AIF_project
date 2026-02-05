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

# Partie 3
from nlp_model import TextClassifier

# Partie 4
from retriever import load_assets, discover_movies
from movie_llm import MovieChatLLM


parser = argparse.ArgumentParser()
parser.add_argument("--poster_classifier_path", type=str, default="weights/poster_classifier_resnet.pth", help="model path") 
parser.add_argument("--text_classifier_path", type=str, default="weights/text_classifier.pth", help="text classifier path")
parser.add_argument("--dknn_path", type=str, default="weights/dknn.pkl", help="dknn path")

parser.add_argument("--annoy_path", type=str, default="embeddings/movie_plots.ann", help="annoy_path")
parser.add_argument("--annoy_meta_path", type=str, default="embeddings/movie_plots_index.csv", help="annoy_meta_path")

parser.add_argument("--labels_path", type=str, default="./labels.json", help="labels json path")

args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#classifier path
POSTER_CLASSIFIER_PATH = args.poster_classifier_path

#DKNN related paths
DKNN_PATH = args.dknn_path

#Plot classification related paths
PLOT_MODEL_PATH = args.text_classifier_path

#ANNOY related paths
ANNOY_PATH = args.annoy_path
ANNOY_META_PATH = args.annoy_meta_path

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



# ---- Load labels (id -> genre) ----
with open(args.labels_path, "r") as f:
    labels = json.load(f)  # e.g. ["Action", "Comedy", "Drama", ...]

# ---- Load model ----
model = poster_classifier().to(device)
state_dict = torch.load(POSTER_CLASSIFIER_PATH, map_location=device,weights_only=True)  # state_dict pur
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


# ---- Load NLP model ----
plot_model = TextClassifier(
    vocab_size=plot_tokenizer.vocab_size,
    embedding_dim=PLOT_EMBED_DIM,
    num_classes=len(labels)
).to(device)

plot_state = torch.load(PLOT_MODEL_PATH, map_location=device, weights_only=True)
plot_model.load_state_dict(plot_state)
plot_model.eval()

# ---- Load Annoy index + metadata ----
annoy_index = AnnoyIndex(PLOT_EMBED_DIM, "angular")
annoy_index.load(ANNOY_PATH)

annoy_meta = pd.read_csv(ANNOY_META_PATH)  # columns: annoy_id, label, plot


# ---- Load CLIP+Annoy assets (Part 4 - multimodal retrieval) ----
try:
    mm_annoy_index, mm_metadata, mm_clip_model, mm_preprocess, mm_device = load_assets()
    print("Loaded multimodal CLIP+Annoy assets for /discover_movies")
except Exception as e:
    mm_annoy_index = None
    mm_metadata = None
    mm_clip_model = None
    mm_preprocess = None
    mm_device = None
    print(f"WARNING: Could not load multimodal assets: {e}")


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
        pred_label = labels[pred_id]
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
            "label": labels[int(row["label"])],
            "plot": str(row["plot"])[:400],     # éviter d’envoyer des pavés énormes
            "distance": float(dist)
        })
    return jsonify({"recommendations": recs})

# ---------- Movie discovery (CLIP + Annoy) ----------
@app.route("/discover_movies", methods=["POST"])
def discover_movies_route():
    if mm_annoy_index is None:
        return jsonify({"error": "Multimodal index not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing JSON field 'query'"}), 400

    k = int(payload.get("k", 5))
    k = max(1, min(k, 20))

    results = discover_movies(query, k, mm_annoy_index, mm_metadata, mm_clip_model, mm_device)
    return jsonify({"query": query, "k": k, "results": results})

# --------- LLM chat ---------
try:
    movie_llm = MovieChatLLM(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=180)
    print("Loaded MovieChatLLM for /chat_movies")
except Exception as e:
    movie_llm = None
    print(f"WARNING: Could not load MovieChatLLM: {e}")

@app.route("/chat_movies", methods=["POST"])
def chat_movies_route():
    if mm_annoy_index is None:
        return jsonify({"error": "Multimodal index not loaded"}), 500
    if movie_llm is None:
        return jsonify({"error": "LLM not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Missing JSON field 'message'"}), 400

    k = int(payload.get("k", 5))
    k = max(1, min(k, 20))

    # history attendu: liste de paires [ ["user","assistant"], ... ]
    history = payload.get("history") or []
    history = [(h[0], h[1]) for h in history if isinstance(h, list) and len(h) == 2]

    # 1) retrieval
    results = discover_movies(message, k, mm_annoy_index, mm_metadata, mm_clip_model, mm_device)

    # 2) génération LLM (on limite le contexte)
    reasons = movie_llm.reply(user_msg=message, movies=results, k=k)

   # On renvoie reasons + results
    return jsonify({
    "reasons": reasons,
    "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5075, debug=True)