import gradio as gr
from PIL import Image
import requests
import io
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

#EN LOCAL
POSTERS_ROOT = BASE_DIR.parent / "content"  
API_URL = "http://127.0.0.1:5075/predict"
API_VALIDATE_URL = "http://127.0.0.1:5075/validate_poster"
API_PREDICT_PLOT = "http://127.0.0.1:5075/predict_plot"
API_RECO_PLOT = "http://127.0.0.1:5075/recommend_from_plot"
API_DISCOVER_MOVIES = "http://127.0.0.1:5075/discover_movies"

#EN DOCKER
#API_URL = "http://api:5075/predict"
#API_VALIDATE_URL = "http://api:5075/validate_poster"
#API_PREDICT_PLOT = "http://api:5075/predict_plot"
#API_RECO_PLOT = "http://api:5075/recommend_from_plot"
#API_DISCOVER_MOVIES = "http://api:5075/discover_movies"





def recognize_genre(image):
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")  # PNG ou JPEG, les deux ok

        r = requests.post(API_URL, data=buf.getvalue(), timeout=10)
        r.raise_for_status()
        # Gradio Label accepte str/int ou dict pour les labels
        return r.json()["label"]
    
    except Exception as e:
        return f"ERROR: {e}"


# New button function for poster validation
def validate_poster(image):
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")  # PNG ou JPEG, les deux ok

        r = requests.post(API_VALIDATE_URL, data=buf.getvalue(), timeout=10)
        r.raise_for_status()
        result = r.json()
        return "It is a movie poster" if result["is a poster"] else "It is not a movie poster"
    except Exception as e:
        return f"ERROR: {e}"

# New functions for plot genre prediction and recommendation

def predict_genre_from_plot(plot: str):
    try:
        r = requests.post(API_PREDICT_PLOT, json={"plot": plot}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return f"{data['label']} (conf={data['confidence']:.2f})"
    except Exception as e:
        return f"ERROR: {e}"

def recommend_from_plot(plot: str, k: int):
    try:
        r = requests.post(API_RECO_PLOT, json={"plot": plot, "k": int(k)}, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Format d'affichage simple
        lines = []
        for rec in data["recommendations"]:
            lines.append(
                f"- {rec['label']} | dist={rec['distance']:.3f}\n  {rec['plot'][:200]}..."
            )
        return "\n".join(lines) if lines else "No recommendations."
    except Exception as e:
        return f"ERROR: {e}"


def discover_movies_nl(query: str, k: int):
    try:
        r = requests.post(API_DISCOVER_MOVIES, json={"query": query, "k": int(k)}, timeout=30)
        r.raise_for_status()
        data = r.json()

        lines = []
        images = []

        for m in data["results"]:
            lines.append(
                f"- {m.get('category')} | dist={m['distance']:.3f}\n"
                f"  {m.get('plot','')[:200]}..."
            )

            poster_rel = m.get("poster_path")
            if poster_rel:
                poster_path = POSTERS_ROOT / poster_rel
                if poster_path.exists():
                    images.append(Image.open(poster_path).convert("RGB"))

        text = "\n".join(lines) if lines else "No results."
        return text, images

    except Exception as e:
        return f"ERROR: {e}", []



if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# AIF project: Movie Platform")

        with gr.Tabs():

            # -------------------------
            # TAB 1 — Poster Analyzer
            # -------------------------
            with gr.Tab("Poster Analyzer"):
                gr.Markdown("## Movie Poster Analyzer")

                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload an image")
                    with gr.Column():
                        btn_genre = gr.Button("Predict genre")
                        btn_validate = gr.Button("Validate poster")

                        genre_output = gr.Label(label="Predicted genre")
                        validation_output = gr.Textbox(label="Poster validation")

                btn_genre.click(recognize_genre, image_input, genre_output)
                btn_validate.click(validate_poster, image_input, validation_output)

            # -----------------------------------------
            # TAB 2 — Plot-based genre + recommendations
            # -----------------------------------------
            with gr.Tab("Plot Tools"):
                gr.Markdown("## Plot-based genre + recommendations")

                plot_input = gr.Textbox(
                    label="Paste a movie plot / synopsis",
                    placeholder="Type or paste a synopsis here...",
                    lines=8
                )

                with gr.Row():
                    btn_plot_genre = gr.Button("Predict genre (plot)")
                    k_input = gr.Slider(1, 20, value=5, step=1, label="Number of recommendations (k)")
                    btn_plot_reco = gr.Button("Recommend similar plots")

                with gr.Row():
                    plot_genre_output = gr.Textbox(label="Plot genre prediction")
                    plot_reco_output = gr.Textbox(label="Recommendations", lines=12)

                btn_plot_genre.click(predict_genre_from_plot, plot_input, plot_genre_output)
                btn_plot_reco.click(recommend_from_plot, inputs=[plot_input, k_input], outputs=plot_reco_output)

            # -----------------------------------------
            # TAB 3 — Natural Language Movie Discovery
            # -----------------------------------------
            with gr.Tab("Natural Language Discovery"):
                gr.Markdown("## Natural Language Movie Discovery")
                gr.Markdown("Describe what you want to watch — we retrieve similar movies using CLIP + Annoy.")

                nl_query = gr.Textbox(
                    label="Your request",
                    placeholder="e.g. a film with samurais and swords in Japan",
                    lines=3
                )

                with gr.Row():
                    nl_k = gr.Slider(1, 20, value=5, step=1, label="Top-k")
                    btn_nl = gr.Button("Find movies")

                with gr.Row():
                    nl_out = gr.Textbox(label="Results", lines=12)
                    nl_gallery = gr.Gallery(label="Recommended posters", columns=5, height=250)

                btn_nl.click(discover_movies_nl, inputs=[nl_query, nl_k], outputs=[nl_out, nl_gallery])

    demo.launch(server_name="0.0.0.0", server_port=7860)








    




