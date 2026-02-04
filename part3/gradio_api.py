import gradio as gr
from PIL import Image
import requests
import io


#EN LOCAL
API_URL = "http://127.0.0.1:5075/predict"
API_VALIDATE_URL = "http://127.0.0.1:5075/validate_poster"
API_PREDICT_PLOT = "http://127.0.0.1:5075/predict_plot"
API_RECO_PLOT = "http://127.0.0.1:5075/recommend_from_plot"


#EN DOCKER
#API_URL = "http://api:5075/predict"
#API_VALIDATE_URL = "http://api:5075/validate_poster"
#API_PREDICT_PLOT = "http://api:5075/predict_plot"
#API_RECO_PLOT = "http://api:5075/recommend_from_plot"



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


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## Movie Poster Analyzer")

        image_input = gr.Image(type="pil", label="Upload an image")

        # SECTION PREDICTION GENRE + VALIDATION FROM POSTER
        with gr.Row():
            btn_genre = gr.Button("Predict genre")
            btn_validate = gr.Button("Validate poster")

        genre_output = gr.Label(label="Predicted genre")
        validation_output = gr.Textbox(label="Poster validation")

        btn_genre.click(recognize_genre, image_input, genre_output)
        btn_validate.click(validate_poster, image_input, validation_output)

        gr.Markdown("---")
        gr.Markdown("## Plot-based genre + recommendations")

        # SECTION PREDICTION GENRE + RECOMMENDATIONS FROM PLOT
        plot_input = gr.Textbox(
            label="Paste a movie plot / synopsis",
            placeholder="Type or paste a synopsis here...",
            lines=6
        )

        with gr.Row():
            btn_plot_genre = gr.Button("Predict genre (plot)")
            k_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of recommendations (k)")
            btn_plot_reco = gr.Button("Recommend similar plots")

        plot_genre_output = gr.Textbox(label="Plot genre prediction")
        plot_reco_output = gr.Textbox(label="Recommendations", lines=10)

        btn_plot_genre.click(predict_genre_from_plot, plot_input, plot_genre_output)
        btn_plot_reco.click(recommend_from_plot, inputs=[plot_input, k_input], outputs=plot_reco_output)

    demo.launch(server_name="0.0.0.0", server_port=7860)

    



