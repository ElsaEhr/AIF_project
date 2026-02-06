from pathlib import Path
import torch
import clip

from annoy_builder import index_builder

"""
Offline script used to build the Annoy index for the movie discovery system.

This script computes multimodal CLIP embeddings (plot + poster) for each movie,
builds a single Annoy index for fast similarity search, and saves the resulting
index and metadata to disk. The generated files are meant to be loaded at
application startup and should not be rebuilt at runtime.
"""

# Utility function to pick device
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Main script
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    ROOT = BASE_DIR.parent  # adapte si ton CSV est ailleurs
    # Define paths
    plots_path = ROOT / "movie_plots.csv"
    image_folder = ROOT / "content"

    assets_dir = BASE_DIR / "assets"
    assets_dir.mkdir(exist_ok=True)

    index_path = assets_dir / "movies.ann"
    metadata_path = assets_dir / "metadata.json"

    # Load CLIP model + preprocessors + device
    device = pick_device()
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Define embedding dimension
    DIM = 512

    # Build the Annoy index
    print("Building annoy index...")
    index, metadata, failed = index_builder(
        plots_adress=plots_path,
        image_folder=image_folder,
        DIM=DIM,
        model=model,
        preprocess=preprocess,
        device=device,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        n_trees=50,
        alpha=0.5
    )
    
    # Summary
    print(f"Done. Items: {index.get_n_items()} | Failed: {len(failed)}")
    print("Saved:", index_path)
    print("Saved:", metadata_path)
