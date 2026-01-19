import gradio as gr
from PIL import Image
import requests
import io

#API_URL = "http://127.0.0.1:5075/predict"  # Flask en local
API_URL = "http://api:5075/predict" # Docker compose
API_VALIDATE_URL = "http://api:5075/validate_poster"



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


# New button

def validate_poster(image: Image.Image):
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    r = requests.post(API_VALIDATE_URL, data=buf.getvalue(), timeout=10)
    r.raise_for_status()
    result = r.json()

    return "It is a movie poster" if result["is a poster"] else "It is not a movie poster"


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## Movie Poster Analyzer")

        image_input = gr.Image(type="pil", label="Upload an image")

        with gr.Row():
            btn_genre = gr.Button("Predict genre")
            btn_validate = gr.Button("Validate poster")

        genre_output = gr.Label(label="Predicted genre")
        validation_output = gr.Textbox(label="Poster validation")

        btn_genre.click(recognize_genre, image_input, genre_output)
        btn_validate.click(validate_poster, image_input, validation_output)

    demo.launch(server_name="0.0.0.0", server_port=7860)
