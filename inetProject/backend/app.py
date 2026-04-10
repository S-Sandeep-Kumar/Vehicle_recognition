from flask import Flask, request, jsonify
from cnn.predict import predict_image
from nlp.generate import generate_response
import os
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global state for the current session (single-user mode)
current_vehicle = None

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    global current_vehicle

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(TEMP_DIR, filename)
    file.save(image_path)

    try:
        # CNN prediction (Fixed to catch both label and confidence)
        vehicle_label, confidence = predict_image(image_path)
        current_vehicle = vehicle_label

        # Get initial description
        description = generate_response(vehicle_label)

        return jsonify({
            'vehicle': vehicle_label,
            'confidence': f"{confidence*100:.2f}%",
            'description': description
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not current_vehicle:
        return jsonify({'error': 'Please upload an image first'}), 400
    
    question = data.get('question')
    answer = generate_response(current_vehicle, question)

    return jsonify({
        'vehicle': current_vehicle,
        'answer': answer
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)