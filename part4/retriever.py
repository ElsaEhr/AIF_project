import json
from pathlib import Path

import torch
import clip
from annoy import AnnoyIndex

from embed import embed_plot

DIM = 512
METRIC = "angular"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_assets(annoy_path=None, metadata_path=None):
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "assets"

    index_path = Path(annoy_path) if annoy_path else (assets_dir / "movies.ann")
    metadata_path = Path(metadata_path) if metadata_path else (assets_dir / "metadata.json")

    if not index_path.exists():
        raise FileNotFoundError(f"Missing Annoy index: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    device = pick_device()
    model, preprocess = clip.load("ViT-B/32", device=device)

    annoy_index = AnnoyIndex(DIM, METRIC)
    annoy_index.load(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return annoy_index, metadata, model, preprocess, device



def discover_movies(query: str, k: int, annoy_index, metadata, model, device):
    """
    Returns top-k movies for a natural language query using CLIP-text embedding + Annoy retrieval.
    """
    query_emb = embed_plot(query, model, device)
    ids, dists = annoy_index.get_nns_by_vector(query_emb, k, include_distances=True)

    results = []
    for movie_id, dist in zip(ids, dists):
        m = metadata[str(movie_id)]
        results.append({
            "id": int(movie_id),
            "category": m.get("movie_category"),
            "poster_path": m.get("movie_poster_path"),
            "plot": m.get("movie_plot"),
            "distance": float(dist),
        })
    return results


