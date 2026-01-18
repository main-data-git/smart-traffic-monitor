from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model once at startup
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

# Vehicle class IDs in COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

def analyze_traffic(image_path):
    """
    Analyze traffic density from uploaded image
    Returns: dict with vehicle counts, density, and green light duration
    """
    # Run YOLO inference
    results = model(image_path, verbose=False)
    
    # Count vehicles by type
    vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id in VEHICLE_CLASSES:
                vehicle_type = VEHICLE_CLASSES[class_id]
                vehicle_counts[vehicle_type] += 1
    
    # Calculate total vehicles
    total_vehicles = sum(vehicle_counts.values())
    
    # Determine traffic density
    if total_vehicles <= 5:
        density = 'Low'
        density_color = '#4CAF50'  # Green
        green_duration = 30
    elif total_vehicles <= 15:
        density = 'Medium'
        density_color = '#FF9800'  # Orange
        green_duration = 45
    else:
        density = 'High'
        density_color = '#F44336'  # Red
        green_duration = 60
    
    return {
        'total_vehicles': total_vehicles,
        'vehicle_counts': vehicle_counts,
        'density': density,
        'density_color': density_color,
        'green_duration': green_duration
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/map')
def map_page():
    """Map visualization page"""
    return render_template('map.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and process with YOLO"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG'}), 400
    
    try:
        # Save uploaded file
        filename = 'latest_upload.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze traffic
        results = analyze_traffic(filepath)
        
        # Add image path to results
        results['image_url'] = f'/uploads/{filename}'
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("\n🚦 Smart Traffic Monitor Starting...")
    print("📍 Open: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
