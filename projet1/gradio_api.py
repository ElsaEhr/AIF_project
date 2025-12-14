import gradio as gr
from PIL import Image
import requests
import io

API_URL = "http://127.0.0.1:5075/predict"  # Flask en local

def recognize_genre(image: Image.Image):
    # image est un PIL.Image car on met type="pil" dans Gradio
    image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="PNG")  # PNG ou JPEG, les deux ok

    r = requests.post(API_URL, data=buf.getvalue(), timeout=10)
    r.raise_for_status()
    data = r.json()

    # Gradio Label accepte str/int ; si tu as un mapping id->genre, tu peux l’afficher ici
    return data["label"]  # ou data["prediction"] pour l’id numérique)

if __name__ == "__main__":
    interface = gr.Interface(
        fn=recognize_genre,
        inputs=gr.Image(type="pil", label="Movie poster"),
        outputs=gr.Label(label="Predicted genre (id)"),
        live=False,
        description="Upload a movie poster and the API will predict its genre.",
    )

    print("Starting Gradio app... Open: http://127.0.0.1:7860")
    interface.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
