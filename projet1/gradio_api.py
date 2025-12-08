
import gradio as gr
from PIL import Image
import requests
import io
import numpy as np
 
def recognize_genre(image):
    # the sketchpad returns a dict with a 'composite' key corresponding to the image
    image = image['composite']
    # By default the image is a 4 channels image, we need to convert it to a 1 channel image since the API expects a 1 channel image
    image = image[:, :, 0]
    # invert the image
    image = (image - 255)*-1
    # convert numpy to uint8
    image = image.astype(np.uint8)
    # Convert the image to a PIL Image
    image = Image.fromarray(image)
    # Convert the image to a binary file
    img_binary = io.BytesIO()
    image.save(img_binary, format="JPEG")
    # Send request to the API
    response = requests.post("http://api:5075/predict", data=img_binary.getvalue())
    return response.json()["prediction"]
 
if __name__=='__main__':
 
    interface = gr.Interface(fn=recognize_genre, 
                inputs="image", 
                outputs='label',
                live=False,
                description="Upload an image of a movie poster and this beatifully gorgeously amazingly smart model will get you the genre",
                )
    print("Starting Gradio app...")
    interface.launch(server_name="0.0.0.0", server_port=7860)