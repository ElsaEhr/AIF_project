import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from annoy import AnnoyIndex
import pandas as pd

from transformers import DistilBertTokenizerFast
from train_nlp import TextClassifier, MovieDataset, collate_batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


embed_dim = 300

model = TextClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=300,
    num_classes=10
).to(device)

model.load_state_dict(torch.load("./weights/text_classifier.pth", map_location=device))
model.eval()


dataset_path = "./data/movie_plots.csv"
dataset = MovieDataset(dataset_path, tokenizer)

loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)


annoy_index = AnnoyIndex(embed_dim, "angular")
id_to_metadata = {}

rows = [] 
current_id = 0

with torch.no_grad():
    for batch_idx, (text, labels, text_lengths) in enumerate(loader):

        embeddings = model.get_embeddings(text, text_lengths)
        embeddings = embeddings.cpu().numpy()

        for i in range(embeddings.shape[0]):
            annoy_index.add_item(current_id, embeddings[i])

            rows.append({
                "annoy_id": current_id,
                "label": int(labels[i].cpu()),
                "plot": dataset.data.iloc[current_id]["plot"]
            })

            current_id += 1

os.makedirs("./embeddings", exist_ok=True)
# The index search is based on random trees method, it is necessary to initialize the size of the index (20 trees in this case)

annoy_index.build(20)  # 10–50 is typical
annoy_index.save("./embeddings/movie_plots.ann")

df = pd.DataFrame(rows)
df.to_csv("./embeddings/movie_plots_index.csv", index=False)

print(f"Saved {len(df)} embeddings")


""" 
        To make a query for the n closest neighbours

index = AnnoyIndex(300, "cosine")
index.load("movie_plots.ann")

metadata = pd.read_csv("movie_plots_metadata.csv")

neighbors, distances = index.get_nns_by_vector(query_embedding, n, include_distances=True)

for k, d in zip(neighbors, distances):
    row = metadata.iloc[k]
    print(row["label"], row["plot"], d)
"""