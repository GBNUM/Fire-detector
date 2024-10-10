from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, db
import base64
from io import BytesIO
import cv2
import numpy as np
from ultralytics import YOLO
import math


# Initialize the Flask app
app = Flask(__name__)

# Initialize the Firebase app
cred = credentials.Certificate('C:\\Users\\micke\\OneDrive\\Desktop\\machinelearning\\FIRE_DETECTION-main\\FIRE_DETECTION-main\\esp32-cam-32ed7-firebase-adminsdk-6wleo-c7ccb4ac98.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://esp32-cam-32ed7-default-rtdb.firebaseio.com/'
})

# Reference to your database

# Load YOLO model
model = YOLO('fire.pt')
classnames = ['fire']

def predict_image(frame):
    result = model(frame, stream=True)
    predictions = []
    
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 40:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=1)
                predictions.append({
                    "class": classnames[Class],
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })
    
    return frame, predictions

@app.route('/predict', methods=['GET'])
def predict():
    camId = request.args.get('camId')
    # Retrieve the Base64 image from Firebase
    ref = db.reference('' + str(camId) + '/photo')
    base64_image = ref.get()

    if base64_image:
        # If the Base64 string includes metadata, remove it
        if base64_image.startswith('data:image/png;base64,'):
            base64_image = base64_image.split(',')[1]

        # Decode the image and convert it to an OpenCV format
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (640, 480))  # Resize if necessary

        # Predict and return results
        frame, predictions = predict_image(frame)

        # Encode image back to Base64 for response (optional)
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Return predictions and the image as Base64 (optional)
        return jsonify({
            "predictions": predictions,
            "image": f"data:image/jpeg;base64,{encoded_image}"  # If you want to return the image
        })

    return jsonify({"error": "No image found in Firebase"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
