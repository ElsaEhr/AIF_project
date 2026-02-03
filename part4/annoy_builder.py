from annoy_builder import AnnoyIndex
import torch
import clip


def build_annoy(image_embeddings):
    dim = image_embeddings.shape[1]
    index = AnnoyIndex(dim, metric="angular")

    for i, emb in enumerate(image_embeddings):
        index.add_item(i, emb)

    # The index search is based on random trees method and one has to initialize the size of the index (20 trees in the example below)
    index.build(20)
