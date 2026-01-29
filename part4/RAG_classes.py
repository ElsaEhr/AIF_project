class FoundationModel():

    def __init__(self, FOUND_MODEL_PATH, TEMPERATURE=None, MAX_NEW_TOKENS=10000):

        self.model = AutoModelForCausalLM.from_pretrained(FOUND_MODEL_PATH,
                                                          # device_map=mps_device,
                                                          # device_map=cuda,
                                                          torch_dtype="auto",
                                                          trust_remote_code=True,
                                                          ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(FOUND_MODEL_PATH)

        self.model.generation_config.temperature = TEMPERATURE  # Config of the temperature
        self.model.generation_config.top_p = None  # Config parameter related to the type of generation (like greedy decoding for instance)

        self.llm = pipeline("text-generation",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            return_full_text=False,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True
                            )

        self.num_parameters = self.model.num_parameters()

        print('Number of parameters in my model', '{:.2e}'.format(self.num_parameters))

    def generate_response(self, prompt):

        messages = [
            {'role': 'user', 'content': prompt}
        ]

        output = self.llm(messages)
        # Note that the output is a list of len 1 which is a dict with key 'generated_text'
        return output

    # We anticipate the use of RAG and create a generate response taking into account the context

    def generate_response_with_context(self, prompt, context):

        # The context is a list of str

        messages = []

        if context:
            for i, ctx in enumerate(context):
                messages.append({'role': 'system', 'content': f"context {i + 1}: {ctx}"})

        messages.append({'role': 'user', 'content': prompt})

        output = self.llm(messages)
        return output


# Some format printing functions

def extract_response(output):
    # output is a list of len 1 as a dict with key 'generated_text'

    return output[0]['generated_text']


def short_response(output):
    response = extract_response(output=output)
    # text = "Before <think>to delete</think> After"
    short = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return short.strip()


class EmbeddingModel():

    def __init__(self, EMBEDD_MODEL_PATH):
        # EMBEDD_MODEL_PATH is the name of the embedding model used within the SentenceTransformer lib

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


class Chunk():

    def __init__(self, source, content, embed_model: EmbeddingModel):
        self.embedding_model = embed_model

        # dim is the common dimension of the embeddings
        dim = self.embedding_model.dim

        # A chunk is defined by its source (str); its content (str); its embedding (a torch which shape (1,dim))

        self.source = str(source)
        self.content = str(content)
        self.embedding = self.embedding_model.get_embeddings(texts=[content]).reshape(1, dim)

    def print_chunk(self):
        print('source:', self.source, 'content:', self.content, 'embedding shape:', self.embedding.shape)


class Splitter():

    def __init__(self, embed_model: EmbeddingModel):

        self.embedding_model = embed_model

        self.docs = []
        # We store the original documents as a list of .txt files (format is {"source":'File_name',"content_page":(str)})
        self.chunks = []
        # This will be the list of chunks

    def get_documents(self, path_doc):
        # PATH_DOC is the Path form where the documents will be found (each document is a.txt file).
        docs = []

        for file in Path(path_doc).rglob("*.txt"):
            name = file.name
            with open(file, "r", encoding="utf-8") as file:
                resource = file.read().strip()
                if resource:
                    # print(resource,len(resource))
                    docs.append({"source": name, "content_page": resource})

        self.docs = docs

    def get_chunks_contents_from_1_doc(self, file_name, content_page, chunk_size, overlap, sentence_split=False):

        if chunk_size < overlap:
            raise Exception('Careful overlap must be smaller than chunk_size')

        # Now we chunk according to chunk size and overlap

        if sentence_split:

            content = content_page.split(".")

            for text in content:

                text = text.lstrip()

                if not text == "":
                    self.chunks.append(Chunk(source=file_name,
                                             content=text, embed_model=self.embedding_model))

        else:

            current = 0

            while current < len(content_page):
                end = min(len(content_page), current + chunk_size)
                content = content_page[current:end]

                self.chunks.append(Chunk(source=file_name,
                                         content=content, embed_model=self.embedding_model))

                current += chunk_size - overlap

    def get_chunks(self, path_doc, chunk_size, overlap, sentence_split=False):

        self.get_documents(path_doc=path_doc)

        docs = self.docs

        for doc in docs:
            self.get_chunks_contents_from_1_doc(file_name=doc["source"],
                                                content_page=doc["content_page"],
                                                chunk_size=chunk_size,
                                                overlap=overlap,
                                                sentence_split=sentence_split)

    def reset_splitter(self):

        self.docs = []
        self.chunks = []


class Retriever():

    def __init__(self, embed_model: EmbeddingModel):

        self.embedding_model = embed_model

        # The index is a list of (Id(int),chunk); chunk needs the size DIM for the Embeddings
        self.index = []

    def add_elements_to_index(self, chunks):

        # chunks is a list of chunk

        num = len(self.index)

        for chunk in chunks:
            self.index.append([num, chunk])
            num += 1

    def search_best(self, query, number_of_hits=3, adapt=False):

        # query is a str

        query_embed = self.embedding_model.get_embeddings(texts=[query]).to(device).reshape(1, self.embedding_model.dim)

        results = []

        index = self.index

        scores = []

        for item in index:
            id, chunk = item

            sim = self.embedding_model.compute_cos_sim_embed(embed1=query_embed, embed2=chunk.embedding)

            scores.append((id, chunk, sim))

        results = sorted(scores, key=lambda x: x[2], reverse=True)[:min(number_of_hits, len(index))]

        # We can also add a criterion to exclude the worst hits; here we choose an arbitrary criterion (we exclude a hit if the similarity is smaller than half of the previous one among the number_of_hits chunks)

        if adapt:

            i = 1
            go = True
            while go and i < len(results):
                if results[i][2] < results[i - 1][2] * 0.5:
                    go = False
                else:
                    i += 1

            results = results[:i]

        return results

    def reset_Retriever_index(self):

        self.index = []