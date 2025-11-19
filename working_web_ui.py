#!/usr/bin/env python3
"""
Complete Working Web UI for Fruit Label Placement
Supports upload and analysis of new images with single placement results.
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pandas as pd
import json
import time
import threading
from pathlib import Path
from single_placement_pipeline import SinglePlacementAnalyzer

app = Flask(__name__)
app.secret_key = 'fruit_analysis_working_ui'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables to track processing status
processing_status = {
    'running': False,
    'progress': 0,
    'message': 'Ready to analyze images',
    'log': [],
    'results': None,
    'current_image': None
}

def log_message(message):
    """Add message to processing log."""
    processing_status['log'].append(f"{time.strftime('%H:%M:%S')} - {message}")
    if len(processing_status['log']) > 50:
        processing_status['log'] = processing_status['log'][-50:]

@app.route('/')
def index():
    """Main page with upload and results."""
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>🍎 Fruit Label Placement Analysis</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .upload-section { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .file-input { margin: 10px 0; }
        input, button { padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px; }
        button { background: #3498db; color: white; border: none; cursor: pointer; }
        button:hover { background: #2980b9; }
        button:disabled { background: #bdc3c7; cursor: not-allowed; }
        .status-box { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .progress { width: 100%; height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-bar { height: 100%; background: #3498db; transition: width 0.3s; }
        .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; height: 200px; overflow-y: scroll; font-family: monospace; font-size: 12px; }
        .result-section { margin: 30px 0; }
        .result-image { max-width: 100%; border: 2px solid #ddd; border-radius: 5px; margin: 20px 0; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert-success { background: #d4edda; color: #155724; }
        .alert-error { background: #f8d7da; color: #721c24; }
        .alert-info { background: #d1ecf1; color: #0c5460; }
        .summary { background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .fruit-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin: 20px 0; }
        .fruit-item { background: #f0f8ff; padding: 10px; border-radius: 5px; border-left: 4px solid #3498db; }
        .download-btn { background: #27ae60; color: white; padding: 8px 16px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 5px; }
        .download-btn:hover { background: #229954; }
    </style>
    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('progress-bar').style.width = data.progress + '%';
                    document.getElementById('status-message').textContent = data.message;
                    document.getElementById('log-content').textContent = data.log.join('\\n');
                    document.getElementById('log-content').scrollTop = document.getElementById('log-content').scrollHeight;
                    
                    const submitButton = document.getElementById('submit-btn');
                    if (data.running) {
                        submitButton.disabled = true;
                        submitButton.textContent = '🔄 Analyzing...';
                    } else {
                        submitButton.disabled = false;
                        submitButton.textContent = '🚀 Analyze Images';
                    }
                    
                    // Show results if available
                    if (data.results && !data.running) {
                        showResults(data.results);
                    }
                });
        }
        
        function showResults(results) {
            const resultsDiv = document.getElementById('results-section');
            resultsDiv.style.display = 'block';
            
            // Update summary
            const summaryDiv = document.getElementById('results-summary');
            summaryDiv.innerHTML = `
                <h3>✅ Analysis Complete</h3>
                <p><strong>${results.total_placements} fruits detected</strong> with <strong>1 optimal placement each</strong></p>
                <p>Processing time: ${results.processing_time.toFixed(3)}s</p>
            `;
            
            // Update fruit list
            const fruitListDiv = document.getElementById('fruit-list');
            fruitListDiv.innerHTML = '';
            
            results.fruits.forEach(fruit => {
                const placement = fruit.optimal_placement;
                const fruitDiv = document.createElement('div');
                fruitDiv.className = 'fruit-item';
                fruitDiv.innerHTML = `
                    <strong>Fruit ${fruit.fruit_id}:</strong><br>
                    Label at (${placement.center[0]}, ${placement.center[1]})<br>
                    Score: ${placement.confidence.toFixed(3)}<br>
                    Radius: ${placement.radius}px
                `;
                fruitListDiv.appendChild(fruitDiv);
            });
        }
        
        setInterval(updateStatus, 2000);
        window.onload = updateStatus;
    </script>
</head>
<body>
    <div class="container">
        <h1>🍎 Fruit Label Placement Analysis</h1>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="upload-section">
            <h2>📤 Upload RGB-D Image Pair</h2>
            <p><strong>Upload your color and depth images for analysis:</strong></p>
            <form method="POST" action="/analyze" enctype="multipart/form-data" id="upload-form">
                <div class="file-input">
                    <label><strong>Color Image:</strong></label><br>
                    <input type="file" name="color_file" accept="image/*" required>
                </div>
                <div class="file-input">
                    <label><strong>Depth Image:</strong></label><br>
                    <input type="file" name="depth_file" accept="image/*" required>
                </div>
                <button type="submit" id="submit-btn">🚀 Analyze Images</button>
            </form>
        </div>
        
        <div class="status-box">
            <h3>📊 Processing Status</h3>
            <div id="status-message">Ready to analyze images</div>
            <div class="progress">
                <div id="progress-bar" class="progress-bar" style="width: 0%;"></div>
            </div>
            <div id="log-content" class="log"></div>
        </div>
        
        <div id="results-section" class="result-section" style="display: none;">
            <h2>🎯 Analysis Results</h2>
            
            <div id="results-summary" class="summary">
                <!-- Results summary will be populated by JavaScript -->
            </div>
            
            <div>
                <h3>📷 Result Visualization</h3>
                <img src="/result_image" class="result-image" alt="Analysis Results" id="result-image">
            </div>
            
            <div>
                <h3>📊 Individual Placements</h3>
                <div id="fruit-list" class="fruit-list">
                    <!-- Fruit placements will be populated by JavaScript -->
                </div>
            </div>
            
            <div>
                <h3>💾 Download Results</h3>
                <a href="/download/csv" class="download-btn">📄 Download CSV</a>
                <a href="/download/image" class="download-btn">🖼️ Download Image</a>
                <a href="/download/json" class="download-btn">📋 Download JSON</a>
            </div>
        </div>
    </div>
</body>
</html>
    """)

@app.route('/analyze', methods=['POST'])
def analyze_images():
    """Handle image upload and analysis."""
    if processing_status['running']:
        flash("Analysis already running!", "error")
        return redirect(url_for('index'))
    
    # Check files
    if 'color_file' not in request.files or 'depth_file' not in request.files:
        flash("Please select both color and depth images!", "error")
        return redirect(url_for('index'))
    
    color_file = request.files['color_file']
    depth_file = request.files['depth_file']
    
    if color_file.filename == '' or depth_file.filename == '':
        flash("Please select both images!", "error")
        return redirect(url_for('index'))
    
    if not (allowed_file(color_file.filename) and allowed_file(depth_file.filename)):
        flash("Invalid file format! Use PNG, JPG, etc.", "error")
        return redirect(url_for('index'))
    
    # Save files first, then pass paths to thread
    upload_dir = "uploaded_images"
    os.makedirs(upload_dir, exist_ok=True)
    
    color_filename = secure_filename(color_file.filename)
    depth_filename = secure_filename(depth_file.filename)
    
    color_path = os.path.join(upload_dir, color_filename)
    depth_path = os.path.join(upload_dir, depth_filename)
    
    color_file.save(color_path)
    depth_file.save(depth_path)
    
    # Start analysis in background thread with file paths
    thread = threading.Thread(target=analyze_images_thread, args=(color_path, depth_path, color_filename))
    thread.daemon = True
    thread.start()
    
    flash("Analysis started! Check progress below.", "info")
    return redirect(url_for('index'))

def analyze_images_thread(color_path, depth_path, color_filename):
    """Run analysis in background thread."""
    global processing_status
    
    try:
        processing_status['running'] = True
        processing_status['progress'] = 0
        processing_status['message'] = 'Starting analysis...'
        processing_status['log'] = []
        processing_status['results'] = None
        
        log_message("🚀 Starting fruit label placement analysis")
        processing_status['progress'] = 20
        
        # Load images
        log_message("🖼️ Loading images...")
        color_image = cv2.imread(color_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        processing_status['progress'] = 30
        
        log_message(f"✅ Images loaded: color {color_image.shape}, depth {depth_image.shape}")
        
        # Create analyzer
        log_message("⚙️ Initializing analyzer...")
        analyzer = SinglePlacementAnalyzer()
        processing_status['progress'] = 40
        
        # Process images
        log_message("🔬 Analyzing fruit placements...")
        processing_status['message'] = 'Analyzing fruit placements...'
        results = analyzer.process_image_single(color_image, depth_image)
        processing_status['progress'] = 70
        
        # Create visualization
        log_message("🎨 Creating visualization...")
        vis_image = analyzer.create_marked_visualization(color_image, results)
        
        # Save results
        log_message("💾 Saving results...")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        vis_path = os.path.join(output_dir, "latest_result.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Save data
        analyzer.save_results(results, output_dir)
        
        # Update with filename for current image
        results['current_image'] = color_filename
        processing_status['results'] = results
        processing_status['current_image'] = vis_path
        processing_status['progress'] = 100
        processing_status['message'] = f'✅ Analysis complete! Found {len(results["fruits"])} fruits'
        
        log_message(f"🎉 Analysis complete!")
        log_message(f"📍 Detected {len(results['fruits'])} fruits with optimal placements")
        log_message(f"⏱️ Processing time: {results['processing_time']:.3f}s")
        
    except Exception as e:
        processing_status['message'] = f'❌ Error: {str(e)}'
        log_message(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        processing_status['running'] = False

@app.route('/status')
def get_status():
    """Get current processing status."""
    return jsonify(processing_status)

@app.route('/result_image')
def get_result_image():
    """Get the latest result image."""
    if processing_status['current_image'] and os.path.exists(processing_status['current_image']):
        return send_file(processing_status['current_image'], mimetype='image/png')
    else:
        # Return placeholder or existing result
        default_path = "output/latest_result.png"
        if os.path.exists(default_path):
            return send_file(default_path, mimetype='image/png')
        else:
            return "No result image available", 404

@app.route('/download/<file_type>')
def download_file(file_type):
    """Download result files."""
    output_dir = "output"
    
    if file_type == 'csv':
        file_path = os.path.join(output_dir, "single_placement_results.csv")
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='fruit_placements.csv')
    elif file_type == 'json':
        file_path = os.path.join(output_dir, "single_placement_results.json")
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='fruit_analysis.json')
    elif file_type == 'image':
        file_path = os.path.join(output_dir, "latest_result.png")
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='fruit_analysis_result.png')
    
    flash(f"File not found: {file_type}", "error")
    return redirect(url_for('index'))

def render_template_string(template):
    """Render template string with Flask context."""
    from flask import render_template_string as rts
    return rts(template)

if __name__ == '__main__':
    print("🌐 Starting Complete Fruit Analysis Web UI...")
    print("📱 Features:")
    print("   - Upload color and depth images")
    print("   - Single placement analysis per fruit")
    print("   - Real-time progress tracking")
    print("   - Download results (CSV, JSON, Image)")
    print("🔗 Open: http://localhost:5002")
    print("🛑 Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5002, debug=False)