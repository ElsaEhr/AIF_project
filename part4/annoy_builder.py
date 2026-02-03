

import pandas as pd
from annoy import AnnoyIndex

from embed import embed_movie

def index_builder(plots_adress,
                  image_folder,
                  DIM,
                  model,
                  preprocess,
                  device):
    df = pd.read_csv(plots_adress)

    index = AnnoyIndex(DIM, metric="angular")
    metadata = {}

    for i, row in df.iterrows():
        try:
            emb = embed_movie(
                plot=row["movie_plot"],
                poster_path=image_folder/row["movie_poster_path"],
                model=model,
                preprocess=preprocess,
                device=device
            )

            index.add_item(i, emb)
            metadata[i] = row.to_dict()

        except Exception as e:
            print(f"Row {i} was impossible to embed due to the following error:\n {e}")
            print("Exiting\n")
            exit()

    index.build(50)
    index.save("movies.ann")
    return index, metadata
