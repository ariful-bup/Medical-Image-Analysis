import os
import shutil
from flask import Flask, request, render_template, jsonify
from PIL import Image
from model import load_model, predict, fine_tune_model
import threading

app = Flask(__name__)

# Configure upload and feedback folders
UPLOAD_FOLDER = 'static/uploads'
TEMP_FOLDER = 'static/temp'
FEEDBACK_FOLDER = 'feedback'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

# Load the model
model = load_model()

# Counter for feedback samples
feedback_counter = 0
FINE_TUNE_THRESHOLD = 100 # Fine-tune after 10 feedback samples

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Check if image was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the uploaded image temporarily
        temp_filename = os.path.join(TEMP_FOLDER, file.filename)
        file.save(temp_filename)
        
        # Make prediction
        image = Image.open(temp_filename).convert('RGB')
        result = predict(model, image)
        
        # Return the result along with the image filename
        result['temp_filename'] = file.filename
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def save_feedback():
    global feedback_counter, model
    
    try:
        data = request.get_json()
        
        # Check if feedback is provided
        if not data.get('feedback', '').strip():
            return jsonify({'error': 'Feedback is required'}), 400
        
        # Determine the correct label from feedback
        feedback_text = data['feedback'].lower()
        if "positive" in feedback_text:
            correct_label = "positive"
        elif "negative" in feedback_text:
            correct_label = "negative"
        else:
            return jsonify({'error': 'Feedback must specify "positive" or "negative"'}), 400
        
        # Create the appropriate feedback folder if it doesn't exist
        feedback_label_folder = os.path.join(FEEDBACK_FOLDER, correct_label)
        os.makedirs(feedback_label_folder, exist_ok=True)
        
        # Move the image to the correct feedback folder
        temp_filename = os.path.join(TEMP_FOLDER, data['imageFileName'])
        feedback_image_filename = os.path.join(feedback_label_folder, data['imageFileName'])
        shutil.move(temp_filename, feedback_image_filename)
        
        # Save feedback to a file in the correct folder
        feedback_file = os.path.join(feedback_label_folder, f"{data['imageFileName']}_feedback.txt")
        with open(feedback_file, 'w') as f:
            f.write(f"Image: {data['imageFileName']}\n")
            f.write(f"Prediction: {data['prediction']}\n")
            f.write(f"Feedback: {data['feedback']}\n")
        
        # Increment feedback counter
        feedback_counter += 1
        
        # Trigger fine-tuning if threshold is reached
        if feedback_counter >= FINE_TUNE_THRESHOLD:
            feedback_counter = 0  # Reset counter
            threading.Thread(target=fine_tune_model, args=(model, FEEDBACK_FOLDER)).start()
            print("Fine-tuning started in the background...")
        
        return jsonify({'message': 'Feedback saved successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fine-tune', methods=['POST'])
def trigger_fine_tuning():
    try:
        # Trigger fine-tuning
        threading.Thread(target=fine_tune_model, args=(model, FEEDBACK_FOLDER)).start()
        return jsonify({'message': 'Fine-tuning started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)