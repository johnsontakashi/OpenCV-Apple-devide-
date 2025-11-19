#!/usr/bin/env python3
"""
Simple Results Viewer - Shows your latest analysis results directly
"""

from flask import Flask, send_file, render_template_string
import os
import json

app = Flask(__name__)

@app.route('/')
def show_results():
    """Show the latest results."""
    
    # Check what result files exist
    output_dir = "output"
    
    # Check for latest result image
    latest_image = None
    if os.path.exists(f"{output_dir}/latest_result.png"):
        latest_image = "latest_result.png"
    elif os.path.exists(f"{output_dir}/SINGLE_RESULT_IMAGE.png"):
        latest_image = "SINGLE_RESULT_IMAGE.png"
    
    # Check for CSV data
    csv_data = []
    if os.path.exists(f"{output_dir}/single_placement_results.csv"):
        with open(f"{output_dir}/single_placement_results.csv", 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # More than just header
                for line in lines[1:]:
                    if line.strip():
                        csv_data.append(line.strip().split(','))
    
    if not csv_data and os.path.exists(f"{output_dir}/SINGLE_RESULTS.csv"):
        with open(f"{output_dir}/SINGLE_RESULTS.csv", 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    if line.strip():
                        parts = line.strip().split(',')
                        csv_data.append([parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]])
    
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>🍎 Your Fruit Analysis Results</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #f8f9fa; 
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        h1 { 
            color: #28a745; 
            text-align: center; 
            margin-bottom: 30px;
        }
        .image-section {
            text-align: center;
            margin: 30px 0;
        }
        .result-image { 
            max-width: 100%; 
            border: 3px solid #28a745; 
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .data-section {
            margin: 30px 0;
        }
        .fruit-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .fruit-card {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            border: 2px solid #28a745;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .fruit-title {
            font-size: 18px;
            font-weight: bold;
            color: #155724;
            margin-bottom: 10px;
        }
        .fruit-detail {
            margin: 5px 0;
            color: #155724;
        }
        .download-section {
            background: #e9ecef;
            padding: 20px;
            border-radius: 10px;
            margin: 30px 0;
            text-align: center;
        }
        .download-btn {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px;
            font-weight: bold;
        }
        .download-btn:hover {
            background: #0056b3;
        }
        .status {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍎 Your Fruit Label Placement Results</h1>
        
        {% if latest_image %}
        <div class="status">
            ✅ <strong>Analysis Complete!</strong> Results from your uploaded images are ready.
        </div>
        
        <div class="image-section">
            <h2>📷 Result Visualization</h2>
            <img src="/image/{{ latest_image }}" class="result-image" alt="Your Analysis Results">
        </div>
        {% endif %}
        
        <div class="data-section">
            <h2>📊 Placement Details</h2>
            {% if csv_data %}
            <div class="fruit-grid">
                {% for fruit in csv_data %}
                <div class="fruit-card">
                    <div class="fruit-title">🍎 Fruit {{ fruit[1] }}</div>
                    <div class="fruit-detail">📍 Position: ({{ fruit[2] }}, {{ fruit[3] }})</div>
                    <div class="fruit-detail">⭐ Score: {{ fruit[4] }}</div>
                    <div class="fruit-detail">📏 Radius: {{ fruit[6] }}px</div>
                    <div class="fruit-detail">🔧 Planarity: {{ fruit[5] }}</div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="status">
                ℹ️ No detailed placement data found. Your analysis may still be processing or check the download files below.
            </div>
            {% endif %}
        </div>
        
        <div class="download-section">
            <h3>💾 Download Your Results</h3>
            <a href="/download/latest_image" class="download-btn">🖼️ Download Image</a>
            <a href="/download/csv" class="download-btn">📄 Download CSV</a>
            <a href="/download/json" class="download-btn">📋 Download JSON</a>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #6c757d;">
            <p>Need to analyze new images? Go back to: <a href="http://localhost:5002">🔗 Upload Interface</a></p>
        </div>
    </div>
</body>
</html>
    """, latest_image=latest_image, csv_data=csv_data)

@app.route('/image/<filename>')
def get_image(filename):
    """Serve result images."""
    return send_file(f"output/{filename}", mimetype='image/png')

@app.route('/download/<file_type>')
def download_file(file_type):
    """Download result files."""
    if file_type == 'latest_image':
        if os.path.exists("output/latest_result.png"):
            return send_file("output/latest_result.png", as_attachment=True, download_name='your_fruit_analysis.png')
        elif os.path.exists("output/SINGLE_RESULT_IMAGE.png"):
            return send_file("output/SINGLE_RESULT_IMAGE.png", as_attachment=True, download_name='your_fruit_analysis.png')
    elif file_type == 'csv':
        if os.path.exists("output/single_placement_results.csv"):
            return send_file("output/single_placement_results.csv", as_attachment=True, download_name='fruit_placements.csv')
        elif os.path.exists("output/SINGLE_RESULTS.csv"):
            return send_file("output/SINGLE_RESULTS.csv", as_attachment=True, download_name='fruit_placements.csv')
    elif file_type == 'json':
        if os.path.exists("output/single_placement_results.json"):
            return send_file("output/single_placement_results.json", as_attachment=True, download_name='fruit_analysis.json')
    
    return "File not found", 404

if __name__ == '__main__':
    print("🍎 Starting Simple Results Viewer...")
    print("📊 View your results at: http://localhost:5003")
    print("🔄 Upload new images at: http://localhost:5002") 
    print("🛑 Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=5003, debug=False)