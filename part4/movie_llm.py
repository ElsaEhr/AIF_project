"""
movie_llm.py

The MovieChatLLM class integrates an LLM with retrieved movie candidates obtained from a CLIP + Annoy retrieval pipeline.
For each user message, the LLM receives a compact context composed of the
top retrieved movies (genre and plot) and generates a response that recommends or discusses these movies.

Key characteristics:
- The LLM does NOT perform retrieval itself.
- Retrieval is handled separately using multimodal CLIP embeddings and Annoy.
- The LLM is strictly constrained to use only the retrieved movies as context
  in order to avoid hallucinated recommendations.
- A short chat history is optionally injected to support conversational
  interactions.

This design follows a retrieval-augmented generation (RAG) pattern, where
retrieval and generation are cleanly separated:
    User query → CLIP embedding → Annoy retrieval → Movie context → LLM response

The module is intended to be used in interactive applications such as a
Gradio chatbot for natural language movie discovery.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class MovieChatLLM:
    def __init__(self, model_name: str, temperature: float = 0.7, max_new_tokens: int = 250):
        device = pick_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True
        )

        if device == "cuda":
            self.model = self.model.to("cuda")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def reply(self, user_msg: str, history: list[tuple[str, str]], movies: list[dict]) -> str:
        # Keep history short to avoid huge prompts
        history = history[-4:] if history else []

        # Build compact movie context
        ctx_lines = []
        for i, m in enumerate(movies, start=1):
            plot = (m.get("plot") or "").replace("\n", " ")[:220]
            cat = m.get("category") or "Unknown"
            ctx_lines.append(f"{i}. Genre: {cat}. Plot: {plot}")

        movies_ctx = "\n".join(ctx_lines) if ctx_lines else "No candidates."

        # Build a simple chat prompt
        chat_ctx = ""
        for u, a in history:
            chat_ctx += f"User: {u}\nAssistant: {a}\n"

        prompt = (
            "You are a friendly movie discovery assistant.\n"
            "You must recommend movies based on the user's request.\n"
            "Use ONLY the candidate movies provided below (do not invent titles).\n"
            "If the user asks for something not covered, ask a clarifying question.\n\n"
            f"Candidate movies:\n{movies_ctx}\n\n"
            f"{chat_ctx}"
            f"User: {user_msg}\n"
            "Assistant:"
        )

        out = self.pipe(prompt)
        return out[0]["generated_text"].strip()
