import gradio as gr
from transformers import pipeline

# Initialize the model pipeline
model = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

# Define the prediction function
def predict(image):
    predictions = model(image)
    return {p["label"]: p["score"] for p in predictions}

# Create Gradio interface using updated syntax
gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.Label(num_top_classes=2),
    title="Hot Dog? Or Not?",
    allow_flagging="manual"
).launch()
