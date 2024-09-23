from flask import Flask, request, send_file
from PIL import Image, ImageDraw
import torch
import io
from torchvision import transforms
from ultralytics import YOLO  # Adjust this import based on your YOLO library
import cv2


app = Flask(__name__)

# Load your YOLO model
model = YOLO('fire.pt')

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((416, 416)),  # Resize to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def process_image(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)

    # Example output processing
    boxes = [[50, 50, 200, 200]]  # Replace with actual output processing logic

    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box[:4], outline='red', width=3)
    return image
    
@app.route('/')
def index():
    return "Welcome to the Fire Detection API. Use the /detect endpoint to upload images."


@app.route('/detect-fire', methods=['POST'])
def detect_fire():
    file = request.files['image']
    img = Image.open(file)

    # Process the image and get results
    result_image = process_image(img)

    # Save the result to a buffer
    img_io = io.BytesIO()
    result_image.save(img_io, format='JPEG')  # Specify format here
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
