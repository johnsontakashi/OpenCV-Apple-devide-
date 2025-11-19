#!/usr/bin/env python3
"""
Web-based User Interface for Fruit Label Placement Analysis

Simple web interface to run the pipeline easily through a browser.
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import subprocess
import json
import threading
import time
import sys
from pathlib import Path
import tempfile
import shutil
import cv2
import numpy as np
from realtime_pipeline import RealTimeFruitAnalyzer

app = Flask(__name__)
app.secret_key = 'fruit_analysis_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables to track processing status
processing_status = {
    'running': False,
    'progress': 0,
    'message': 'Ready',
    'log': [],
    'results': None
}

# Initialize real-time analyzer
realtime_analyzer = None
try:
    realtime_analyzer = RealTimeFruitAnalyzer()
    print("Real-time analyzer initialized successfully")
except Exception as e:
    print(f"Failed to initialize real-time analyzer: {e}")

def log_message(message):
    """Add message to processing log."""
    processing_status['log'].append(f"{time.strftime('%H:%M:%S')} - {message}")
    if len(processing_status['log']) > 100:  # Keep only last 100 messages
        processing_status['log'] = processing_status['log'][-100:]

@app.route('/')
def index():
    """Main page."""
    # Check available directories
    dirs = []
    if os.path.exists('test_data'):
        dirs.append('test_data')
    
    # Add uploaded directory if it exists
    if os.path.exists('uploaded_images'):
        dirs.insert(0, 'uploaded_images')  # Put at top of list
    
    # Look for other directories with images
    for item in os.listdir('.'):
        if os.path.isdir(item) and item not in ['output', '__pycache__', '.git', 'templates', 'uploaded_images']:
            dirs.append(item)
    
    return render_template('index.html', directories=dirs, status=processing_status)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload."""
    if 'color_file' not in request.files or 'depth_file' not in request.files:
        flash("Please select both color and depth images!", "error")
        return redirect(url_for('index'))
    
    color_file = request.files['color_file']
    depth_file = request.files['depth_file']
    
    if color_file.filename == '' or depth_file.filename == '':
        flash("Please select both color and depth images!", "error")
        return redirect(url_for('index'))
    
    if not (allowed_file(color_file.filename) and allowed_file(depth_file.filename)):
        flash("Invalid file format! Please use PNG, JPG, JPEG, BMP, TIFF files.", "error")
        return redirect(url_for('index'))
    
    try:
        # Create uploads directory
        upload_dir = "uploaded_images"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Clean up previous uploads
        for file in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, file))
        
        # Get base name from color image (remove extension)
        base_name = secure_filename(color_file.filename).rsplit('.', 1)[0]
        
        # Save files with consistent naming
        color_ext = secure_filename(color_file.filename).rsplit('.', 1)[1].lower()
        depth_ext = secure_filename(depth_file.filename).rsplit('.', 1)[1].lower()
        
        color_filename = f"{base_name}.{color_ext}"
        depth_filename = f"{base_name}_depth.{depth_ext}"
        
        color_path = os.path.join(upload_dir, color_filename)
        depth_path = os.path.join(upload_dir, depth_filename)
        
        color_file.save(color_path)
        depth_file.save(depth_path)
        
        log_message(f"✓ Uploaded: {color_filename} and {depth_filename}")
        flash(f"Images uploaded successfully! Starting analysis: {base_name}", "success")
        
        # Automatically start analysis with uploaded images
        thread = threading.Thread(target=run_analysis_thread, args=(upload_dir, "config.yaml"))
        thread.daemon = True
        thread.start()
        
    except Exception as e:
        log_message(f"❌ Upload error: {e}")
        flash(f"Upload failed: {e}", "error")
    
    return redirect(url_for('index'))

@app.route('/create_test_data', methods=['POST'])
def create_test_data():
    """Create synthetic test data."""
    try:
        log_message("Creating synthetic test data...")
        result = subprocess.run([sys.executable, "create_test_data.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            log_message("✓ Test data created successfully!")
            flash("Test data created successfully!", "success")
        else:
            log_message(f"❌ Error: {result.stderr}")
            flash(f"Error creating test data: {result.stderr}", "error")
            
    except Exception as e:
        log_message(f"❌ Exception: {e}")
        flash(f"Error: {e}", "error")
    
    return redirect(url_for('index'))

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Start analysis process."""
    if processing_status['running']:
        flash("Analysis is already running!", "warning")
        return redirect(url_for('index'))
    
    input_dir = request.form.get('input_dir')
    config_file = request.form.get('config_file', 'config.yaml')
    analysis_mode = request.form.get('analysis_mode', 'standard')
    
    if not input_dir or not os.path.exists(input_dir):
        flash("Please select a valid input directory!", "error")
        return redirect(url_for('index'))
    
    if analysis_mode == 'realtime':
        # Use real-time pipeline
        if realtime_analyzer is None:
            flash("Real-time analyzer not available!", "error")
            return redirect(url_for('index'))
        
        thread = threading.Thread(target=run_realtime_analysis_thread, args=(input_dir,))
        thread.daemon = True
        thread.start()
        flash("Real-time analysis started! Check progress below.", "info")
    else:
        # Use standard pipeline
        if not os.path.exists(config_file):
            flash(f"Configuration file not found: {config_file}", "error")
            return redirect(url_for('index'))
        
        thread = threading.Thread(target=run_analysis_thread, args=(input_dir, config_file))
        thread.daemon = True
        thread.start()
        flash("Analysis started! Check progress below.", "info")
    
    return redirect(url_for('index'))

def run_analysis_thread(input_dir, config_file):
    """Run analysis in background thread."""
    global processing_status
    
    try:
        processing_status['running'] = True
        processing_status['progress'] = 0
        processing_status['message'] = 'Starting analysis...'
        processing_status['log'] = []
        processing_status['results'] = None
        
        log_message("🚀 Starting fruit label placement analysis")
        log_message(f"Input directory: {input_dir}")
        log_message(f"Configuration: {config_file}")
        
        # Build command
        cmd = [
            sys.executable, "main.py",
            "--input", input_dir,
            "--config", config_file,
            "--output", "output"
        ]
        
        processing_status['progress'] = 10
        processing_status['message'] = 'Running pipeline...'
        
        # Run the pipeline
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                log_message(line)
                
                # Update progress based on log messages
                if "Step 1:" in line:
                    processing_status['progress'] = 20
                    processing_status['message'] = 'Preprocessing images...'
                elif "Step 2:" in line:
                    processing_status['progress'] = 35
                    processing_status['message'] = 'Segmenting fruits...'
                elif "Step 3:" in line:
                    processing_status['progress'] = 50
                    processing_status['message'] = 'Analyzing surface planarity...'
                elif "Step 4:" in line:
                    processing_status['progress'] = 70
                    processing_status['message'] = 'Finding label candidates...'
                elif "Step 5:" in line:
                    processing_status['progress'] = 85
                    processing_status['message'] = 'Applying constraints...'
                elif "Step 6:" in line:
                    processing_status['progress'] = 95
                    processing_status['message'] = 'Creating visualizations...'
        
        # Wait for completion
        process.wait()
        
        if process.returncode == 0:
            processing_status['progress'] = 100
            processing_status['message'] = '✅ Analysis completed successfully!'
            log_message("🎉 Analysis completed successfully!")
            
            # Load results
            try:
                results_file = "output/results.json"
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        processing_status['results'] = json.load(f)
                    log_message("📊 Results loaded successfully")
            except Exception as e:
                log_message(f"⚠️ Could not load results: {e}")
        else:
            processing_status['message'] = '❌ Analysis failed'
            log_message("❌ Analysis failed")
            
    except Exception as e:
        processing_status['message'] = f'❌ Error: {str(e)}'
        log_message(f"❌ Exception during analysis: {e}")
    
    finally:
        processing_status['running'] = False

def run_realtime_analysis_thread(input_dir):
    """Run real-time analysis in background thread."""
    global processing_status
    
    try:
        processing_status['running'] = True
        processing_status['progress'] = 0
        processing_status['message'] = 'Starting real-time analysis...'
        processing_status['log'] = []
        processing_status['results'] = None
        
        log_message("⚡ Starting real-time fruit label placement analysis")
        log_message(f"Input directory: {input_dir}")
        
        # Find image pairs
        image_files = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                if 'depth' not in filename.lower():
                    # Look for corresponding depth image
                    base_name = os.path.splitext(filename)[0]
                    depth_candidates = [
                        f"{base_name}_depth.png", f"{base_name}_depth.jpg",
                        f"{base_name}_depth.jpeg", f"{base_name}_d.png"
                    ]
                    
                    depth_file = None
                    for depth_name in depth_candidates:
                        if os.path.exists(os.path.join(input_dir, depth_name)):
                            depth_file = depth_name
                            break
                    
                    if depth_file:
                        image_files.append((filename, depth_file))
                    else:
                        log_message(f"⚠️ No depth image found for {filename}")
        
        if not image_files:
            processing_status['message'] = '❌ No image pairs found'
            log_message("❌ No valid image pairs found in directory")
            return
        
        log_message(f"🖼️ Found {len(image_files)} image pairs")
        processing_status['progress'] = 20
        
        # Process images with real-time pipeline
        all_results = []
        os.makedirs("output", exist_ok=True)
        
        for i, (color_file, depth_file) in enumerate(image_files):
            log_message(f"⚡ Processing {color_file}...")
            
            # Load images
            color_path = os.path.join(input_dir, color_file)
            depth_path = os.path.join(input_dir, depth_file)
            
            color_image = cv2.imread(color_path)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            # Process with real-time analyzer
            start_time = time.time()
            results = realtime_analyzer.process_image_realtime(color_image, depth_image)
            process_time = time.time() - start_time
            
            # Add metadata
            results['image_name'] = color_file
            results['depth_image'] = depth_file
            all_results.append(results)
            
            # Save visualization
            vis_image = realtime_analyzer.visualize_results_fast(color_image, results)
            vis_path = f"output/{os.path.splitext(color_file)[0]}_realtime.png"
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            fps = results['performance_stats']['fps_estimate']
            candidates = results['total_candidates']
            fruits = len(results['fruits'])
            
            log_message(f"✅ {color_file}: {process_time:.3f}s ({fps:.1f} FPS), {fruits} fruits, {candidates} candidates")
            
            # Update progress
            processing_status['progress'] = 20 + (70 * (i + 1) // len(image_files))
        
        # Save combined results
        summary_results = {
            'total_images': len(image_files),
            'total_processing_time': sum(r['processing_time'] for r in all_results),
            'average_fps': sum(r['performance_stats']['fps_estimate'] for r in all_results) / len(all_results),
            'total_fruits': sum(len(r['fruits']) for r in all_results),
            'total_candidates': sum(r['total_candidates'] for r in all_results),
            'images': all_results
        }
        
        with open("output/realtime_results.json", 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        processing_status['results'] = summary_results
        processing_status['progress'] = 100
        processing_status['message'] = '✅ Real-time analysis completed!'
        
        avg_time = summary_results['total_processing_time'] / len(image_files)
        avg_fps = summary_results['average_fps']
        
        log_message(f"🎉 Analysis complete!")
        log_message(f"📊 Average: {avg_time:.3f}s per image ({avg_fps:.1f} FPS)")
        log_message(f"🍎 Total: {summary_results['total_fruits']} fruits, {summary_results['total_candidates']} candidates")
        
    except Exception as e:
        processing_status['message'] = f'❌ Real-time analysis error: {str(e)}'
        log_message(f"❌ Exception during real-time analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        processing_status['running'] = False

@app.route('/status')
def get_status():
    """Get current processing status (for AJAX updates)."""
    return jsonify(processing_status)

@app.route('/results')
def view_results():
    """View analysis results."""
    results_file = "output/results.json"
    if not os.path.exists(results_file):
        flash("No results found. Please run analysis first.", "warning")
        return redirect(url_for('index'))
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Get list of visualization files
        viz_dir = "output/visualizations"
        visualizations = []
        if os.path.exists(viz_dir):
            for file in os.listdir(viz_dir):
                if file.endswith('.png'):
                    visualizations.append(file)
        
        return render_template('results.html', 
                             results=results, 
                             visualizations=sorted(visualizations))
    
    except Exception as e:
        flash(f"Error loading results: {e}", "error")
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """Download result files."""
    # Security: only allow downloading from output directory
    safe_path = os.path.join("output", filename)
    if not os.path.exists(safe_path):
        safe_path = os.path.join("output", "visualizations", filename)
    
    if os.path.exists(safe_path):
        return send_file(safe_path, as_attachment=True)
    else:
        flash(f"File not found: {filename}", "error")
        return redirect(url_for('view_results'))

# HTML Templates as strings (to avoid needing template files)
@app.route('/templates/index.html')
def get_index_template():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>🍎 Fruit Label Placement Analysis</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }
        select, input, button { padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        select, input { width: 100%; max-width: 400px; }
        button { background: #3498db; color: white; border: none; cursor: pointer; margin: 5px; }
        button:hover { background: #2980b9; }
        button.success { background: #27ae60; }
        button.danger { background: #e74c3c; }
        .status-box { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .progress { width: 100%; height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; }
        .progress-bar { height: 100%; background: #3498db; transition: width 0.3s; }
        .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; height: 300px; overflow-y: scroll; font-family: monospace; font-size: 12px; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .alert-warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .hidden { display: none; }
    </style>
    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusBox = document.getElementById('status-box');
                    const progressBar = document.getElementById('progress-bar');
                    const statusMessage = document.getElementById('status-message');
                    const logContent = document.getElementById('log-content');
                    const runButton = document.getElementById('run-button');
                    const resultsButton = document.getElementById('results-button');
                    
                    // Update progress
                    progressBar.style.width = data.progress + '%';
                    statusMessage.textContent = data.message;
                    
                    // Update log
                    logContent.textContent = data.log.join('\\n');
                    logContent.scrollTop = logContent.scrollHeight;
                    
                    // Update buttons
                    if (data.running) {
                        runButton.disabled = true;
                        runButton.textContent = '🔄 Running...';
                        statusBox.style.display = 'block';
                    } else {
                        runButton.disabled = false;
                        runButton.textContent = '🚀 Run Analysis';
                        if (data.results) {
                            resultsButton.style.display = 'inline-block';
                        }
                    }
                });
        }
        
        // Update status every 2 seconds when page loads
        setInterval(updateStatus, 2000);
        window.onload = updateStatus;
    </script>
</head>
<body>
    <div class="container">
        <h1>🍎 Fruit Label Placement Analysis</h1>
        
        <!-- Flash messages -->
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form method="POST" action="/run_analysis">
            <div class="form-group">
                <label for="input_dir">Select Input Directory:</label>
                <select name="input_dir" id="input_dir" required>
                    <option value="">-- Select Directory --</option>
                    {% for dir in directories %}
                        <option value="{{ dir }}">{{ dir }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <div class="form-group">
                <label for="config_file">Configuration File:</label>
                <input type="text" name="config_file" id="config_file" value="config.yaml" readonly>
            </div>
            
            <div class="form-group">
                <button type="submit" id="run-button">🚀 Run Analysis</button>
                <button type="submit" formaction="/create_test_data" class="success">📊 Create Test Data</button>
                <button type="button" id="results-button" onclick="location.href='/results'" style="display: none;">📁 View Results</button>
            </div>
        </form>
        
        <div id="status-box" class="status-box" style="display: none;">
            <h3>Processing Status</h3>
            <div id="status-message">Ready</div>
            <div class="progress">
                <div id="progress-bar" class="progress-bar" style="width: 0%;"></div>
            </div>
        </div>
        
        <div class="form-group">
            <label>Processing Log:</label>
            <div id="log-content" class="log"></div>
        </div>
        
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px;">
            <h4>Instructions:</h4>
            <ol>
                <li><strong>Create Test Data:</strong> Click "Create Test Data" to generate synthetic fruit images for testing</li>
                <li><strong>Select Directory:</strong> Choose a directory containing RGB-D image pairs (color + depth images)</li>
                <li><strong>Run Analysis:</strong> Click "Run Analysis" to start processing</li>
                <li><strong>View Results:</strong> Once complete, click "View Results" to see visualizations and download data</li>
            </ol>
            <p><strong>Image Naming:</strong> Depth images should contain "depth" in the filename (e.g., apple1.jpg + apple1_depth.png)</p>
        </div>
    </div>
</body>
</html>
    '''

# Create templates directory and write templates
def create_templates():
    """Create template files."""
    os.makedirs('templates', exist_ok=True)
    
    # Index template
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>🍎 Fruit Label Placement Analysis</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; }
        h1 { text-align: center; margin-bottom: 30px; }
        h2 { margin-top: 30px; margin-bottom: 15px; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }
        select, input, button { padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        select, input { width: 100%; max-width: 400px; }
        button { background: #3498db; color: white; border: none; cursor: pointer; margin: 5px; }
        button:hover { background: #2980b9; }
        button.success { background: #27ae60; }
        button.upload { background: #e67e22; }
        .status-box { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .progress { width: 100%; height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; }
        .progress-bar { height: 100%; background: #3498db; transition: width 0.3s; }
        .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; height: 300px; overflow-y: scroll; font-family: monospace; font-size: 12px; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert-success { background: #d4edda; color: #155724; }
        .alert-error { background: #f8d7da; color: #721c24; }
        .alert-info { background: #d1ecf1; color: #0c5460; }
        .alert-warning { background: #fff3cd; color: #856404; }
        .upload-section { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .file-input { margin: 10px 0; }
        .tabs { display: flex; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #ecf0f1; border: 1px solid #bdc3c7; cursor: pointer; }
        .tab.active { background: #3498db; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('progress-bar').style.width = data.progress + '%';
                    document.getElementById('status-message').textContent = data.message;
                    document.getElementById('log-content').textContent = data.log.join('\\n');
                    document.getElementById('log-content').scrollTop = document.getElementById('log-content').scrollHeight;
                    
                    const runButton = document.getElementById('run-button');
                    if (data.running) {
                        runButton.disabled = true;
                        runButton.textContent = '🔄 Running...';
                    } else {
                        runButton.disabled = false;
                        runButton.textContent = '🚀 Run Analysis';
                        if (data.results) {
                            document.getElementById('results-button').style.display = 'inline-block';
                        }
                    }
                });
        }
        setInterval(updateStatus, 2000);
        window.onload = function() {
            updateStatus();
            showTab('upload-tab');
            document.querySelector('.tab').classList.add('active');
        };
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
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('upload-tab')">📤 Upload Images</div>
            <div class="tab" onclick="showTab('directory-tab')">📁 Use Directory</div>
        </div>
        
        <!-- Upload Images Tab -->
        <div id="upload-tab" class="tab-content active">
            <div class="upload-section">
                <h2>Upload RGB-D Image Pair</h2>
                <p><strong>Upload your color and depth images directly:</strong></p>
                <form method="POST" action="/upload" enctype="multipart/form-data">
                    <div class="file-input">
                        <label for="color_file">Color Image (RGB):</label>
                        <input type="file" name="color_file" id="color_file" accept=".png,.jpg,.jpeg,.bmp,.tiff,.tif" required>
                        <small>Supported formats: PNG, JPG, JPEG, BMP, TIFF</small>
                    </div>
                    
                    <div class="file-input">
                        <label for="depth_file">Depth Image:</label>
                        <input type="file" name="depth_file" id="depth_file" accept=".png,.tiff,.tif" required>
                        <small>Supported formats: PNG, TIFF (16-bit recommended)</small>
                    </div>
                    
                    <button type="submit" class="upload">📤 Upload & Analyze</button>
                </form>
            </div>
        </div>
        
        <!-- Directory Tab -->
        <div id="directory-tab" class="tab-content">
            <form method="POST" action="/run_analysis">
                <div class="form-group">
                    <label for="input_dir">Select Input Directory:</label>
                    <select name="input_dir" required>
                        <option value="">-- Select Directory --</option>
                        {% for dir in directories %}
                            <option value="{{ dir }}">{{ dir }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <button type="submit" id="run-button">🚀 Run Analysis</button>
                    <button type="submit" formaction="/create_test_data" class="success">📊 Create Test Data</button>
                    <button type="button" id="results-button" onclick="location.href='/results'" style="display: none;">📁 View Results</button>
                </div>
            </form>
        </div>
        
        <div class="status-box">
            <h3>Processing Status</h3>
            <div id="status-message">Ready</div>
            <div class="progress">
                <div id="progress-bar" class="progress-bar" style="width: 0%;"></div>
            </div>
        </div>
        
        <div class="form-group">
            <label>Processing Log:</label>
            <div id="log-content" class="log"></div>
        </div>
        
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px;">
            <h4>Instructions:</h4>
            <ol>
                <li><strong>Upload Method:</strong> Select "Upload Images" tab and choose your color + depth image files</li>
                <li><strong>Directory Method:</strong> Select "Use Directory" tab and choose a folder with RGB-D pairs</li>
                <li><strong>Test Data:</strong> Click "Create Test Data" to generate synthetic fruit images</li>
                <li><strong>View Results:</strong> After analysis, click "View Results" for visualizations</li>
            </ol>
        </div>
    </div>
</body>
</html>''')
    
    # Results template
    with open('templates/results.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Analysis Results - Fruit Label Placement</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        h1, h2 { color: #2c3e50; }
        .summary { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .visualizations { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .viz-card { border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }
        .viz-card img { width: 100%; height: auto; }
        .viz-card h3 { margin: 0; padding: 10px; background: #34495e; color: white; font-size: 14px; }
        button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer; }
        button:hover { background: #2980b9; }
        .back-btn { background: #95a5a6; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
        th { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Analysis Results</h1>
        
        <button onclick="location.href='/'" class="back-btn">← Back to Main</button>
        <button onclick="location.href='/download/results.json'">💾 Download JSON</button>
        <button onclick="location.href='/download/results.csv'">📊 Download CSV</button>
        
        <div class="summary">
            <h2>Summary</h2>
            {% if results.processing_summary %}
            <p><strong>Images processed:</strong> {{ results.processing_summary.successful_images }}/{{ results.processing_summary.total_images }}</p>
            <p><strong>Fruits detected:</strong> {{ results.processing_summary.total_fruits }}</p>
            <p><strong>Label candidates found:</strong> {{ results.processing_summary.total_candidates }}</p>
            {% endif %}
        </div>
        
        <h2>🖼️ Visualizations</h2>
        <div class="visualizations">
            {% for viz in visualizations %}
            <div class="viz-card">
                <h3>{{ viz }}</h3>
                <img src="/download/{{ viz }}" alt="{{ viz }}">
                <div style="padding: 10px;">
                    <button onclick="window.open('/download/{{ viz }}')">🔍 View Full Size</button>
                    <button onclick="location.href='/download/{{ viz }}'">💾 Download</button>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if results.results %}
        <h2>📋 Detailed Results</h2>
        {% for result in results.results %}
        <h3>Image: {{ result.image_name }}</h3>
        <p><strong>Processing time:</strong> {{ "%.2f"|format(result.processing_time) }} seconds</p>
        <p><strong>Fruits found:</strong> {{ result.num_fruits }}</p>
        
        {% for fruit in result.fruits %}
        <h4>Fruit {{ fruit.fruit_id }}</h4>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Area</td><td>{{ fruit.properties.area }} pixels</td></tr>
            <tr><td>Mean Depth</td><td>{{ "%.3f"|format(fruit.properties.mean_depth) }} meters</td></tr>
            <tr><td>Circularity</td><td>{{ "%.3f"|format(fruit.properties.circularity) }}</td></tr>
            <tr><td>Flat Surface %</td><td>{{ "%.1f"|format(fruit.planarity_statistics.flat_percentage) }}%</td></tr>
            <tr><td>Candidates Found</td><td>{{ fruit.num_candidates }}</td></tr>
        </table>
        {% endfor %}
        {% endfor %}
        {% endif %}
    </div>
</body>
</html>''')

if __name__ == '__main__':
    # Check if we're in the right directory
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found. Please run this from the project directory.")
        sys.exit(1)
    
    # Create templates
    create_templates()
    
    print("🌐 Starting Fruit Analysis Web UI...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Web UI stopped!")