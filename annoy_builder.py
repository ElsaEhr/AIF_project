import json
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

from embed import embed_movie

def index_builder(
    plots_adress,
    image_folder,
    DIM,
    model,
    preprocess,
    device,
    index_path="movies.ann",
    metadata_path="metadata.json",
    n_trees=50,
    alpha=0.5,
):
    df = pd.read_csv(plots_adress)

    index = AnnoyIndex(DIM, metric="angular")
    metadata = {}
    failed = []

    for i, row in df.iterrows():
        try:
            poster_path = image_folder / row["movie_poster_path"]
            emb = embed_movie(
                plot=row["movie_plot"],
                poster_path=poster_path,
                model=model,
                preprocess=preprocess,
                device=device,
                alpha=alpha
            )

            # Annoy likes python lists of floats
            index.add_item(int(i), emb.astype(np.float32).tolist())
            metadata[int(i)] = row.to_dict()

        except Exception as e:
            failed.append((int(i), str(e)))

    # build + save
    index.build(n_trees)
    index.save(index_path)

    # save metadata + failures
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    if failed:
        with open("failed_rows.txt", "w", encoding="utf-8") as f:
            for i, err in failed:
                f.write(f"{i}\t{err}\n")

    return index, metadata, failed
