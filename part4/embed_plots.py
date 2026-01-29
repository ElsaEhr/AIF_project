import torch
from sentence_transformers import SentenceTransformer

#This file contains classes that we will use when pipelining our RAG

class EmbeddingModelText():
    """
    Class whose instances are models for embedding texts
    """
    def __init__(self, EMBEDD_MODEL_PATH):
        # EMBEDD_MODEL_PATH is the name of the embedding model used within the SentenceTransformer lib
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Embedmodel = SentenceTransformer(EMBEDD_MODEL_PATH).to(device)
        self.dim = SentenceTransformer(EMBEDD_MODEL_PATH).get_sentence_embedding_dimension()

    def get_embeddings(self, texts):
        # texts is a list of strings (which is supposed to be the list of chinks; without the source)
        # we return embeddings of torch type with shape (len(texts),self.dim)

        embeddings = self.Embedmodel.encode(texts, convert_to_tensor=True, normalize_embeddings=True).to(device)
        return embeddings

    def compute_cos_sim_embed(self, embed1, embed2):
        # embed1,embeds2 are two embeddings of shape (1,dim)
        # We compute the cos-similarity of two texts (it is returned as a float)

        embed1 = embed1.view(-1)
        embed2 = embed2.view(-1)

        norm1 = torch.norm(embed1, p=2, dim=0)
        norm2 = torch.norm(embed2, p=2, dim=0)

        scal = torch.dot(embed1, embed2)

        return scal.item() / (norm1.item() * norm2.item())

    def compute_cos_sim_texts(self, text_1, text_2):
        # text1,text2 are two str
        # We compute the cos-similarity of two texts (it is returned as a float)

        embeds = self.get_embeddings(texts=[text_1, text_2])

        return self.compute_cos_sim_embed(embeds[0], embeds[1])
