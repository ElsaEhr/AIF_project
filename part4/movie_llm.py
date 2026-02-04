import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def _extract_json_array(text: str):
    # Cherche le premier bloc JSON qui ressemble à {"reasons":[...]}
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


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

        # Set up text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            do_sample=False,
            temperature=0.0,
            max_new_tokens= 180,
            repetition_penalty=1.1,
        )

    def reply(self, user_msg: str, movies: list[dict], k: int = 5) -> list[str]:
        """
        Returns ONLY the reasons (one short sentence per candidate), as a list of strings.
        You display genre/dist yourself in the UI.
        """
        k = min(k, len(movies))
        candidates = movies[:k]

        # Context ONLY: plots (and maybe genre to guide)
        ctx_lines = []
        for i, m in enumerate(candidates, start=1):
            plot = (m.get("plot") or "").replace("\n", " ")[:220]
            cat = m.get("category") or "Unknown"
            ctx_lines.append(f"{i}) Genre: {cat}. Plot: {plot}")

        movies_ctx = "\n".join(ctx_lines)

        prompt = (
            "You are given a user request and k candidate movies.\n"
            "Write ONE short reason (max 18 words) for EACH candidate.\n"
            "Return ONLY valid JSON, nothing else.\n"
            "JSON schema: {\"reasons\": [\"...\", \"...\", ...]} with exactly k strings.\n\n"
            f"User request: {user_msg}\n\n"
            f"Candidates:\n{movies_ctx}\n"
        )

        out = self.pipe(prompt)
        raw = out[0]["generated_text"].strip()

        data = _extract_json_array(raw)
        if data and isinstance(data.get("reasons"), list):
            reasons = [str(r).strip() for r in data["reasons"]][:k]
            # fallback if wrong length
            if len(reasons) == k:
                return reasons

        # Fallback ultra simple si JSON raté : take first k non-empty lines
        lines = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
        return (lines + [""] * k)[:k]



