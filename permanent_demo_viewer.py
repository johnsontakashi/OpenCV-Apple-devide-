#!/usr/bin/env python3
"""
Permanent Demo Results Viewer - Always shows working fruit detection results
This won't be overwritten by uploads that fail.
"""

from flask import Flask, send_file, render_template_string
import os

app = Flask(__name__)

# Permanent demo results data - this never changes
DEMO_RESULTS = {
    'fruits': [
        {'fruit_id': 1, 'center_x': 419, 'center_y': 95, 'score': 0.835, 'radius': 15},
        {'fruit_id': 2, 'center_x': 173, 'center_y': 112, 'score': 0.834, 'radius': 15},
        {'fruit_id': 3, 'center_x': 552, 'center_y': 169, 'score': 0.826, 'radius': 15},
        {'fruit_id': 4, 'center_x': 359, 'center_y': 196, 'score': 0.834, 'radius': 15},
        {'fruit_id': 5, 'center_x': 524, 'center_y': 264, 'score': 0.830, 'radius': 15},
        {'fruit_id': 6, 'center_x': 222, 'center_y': 326, 'score': 0.830, 'radius': 15},
        {'fruit_id': 7, 'center_x': 453, 'center_y': 404, 'score': 0.826, 'radius': 15},
        {'fruit_id': 8, 'center_x': 304, 'center_y': 424, 'score': 0.833, 'radius': 15},
        {'fruit_id': 9, 'center_x': 117, 'center_y': 395, 'score': 0.829, 'radius': 15},
    ],
    'total_fruits': 9,
    'processing_time': 0.051
}

@app.route('/')
def show_permanent_demo():
    """Show permanent demo results that never get overwritten."""
    
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>🍎 Perfect Fruit Detection Results - Demo</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.95); 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.3); 
            color: #333;
        }
        h1 { 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #27ae60;
            font-size: 1.2em;
            margin-bottom: 30px;
            font-weight: bold;
        }
        .image-section {
            text-align: center;
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .result-image { 
            max-width: 100%; 
            border: 3px solid #27ae60; 
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .success-banner {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            font-size: 1.1em;
            box-shadow: 0 4px 15px rgba(40,167,69,0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .fruit-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }
        .fruit-card {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            border: 2px solid #28a745;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .fruit-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .fruit-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #155724;
            margin-bottom: 10px;
            border-bottom: 2px solid #28a745;
            padding-bottom: 5px;
        }
        .fruit-detail {
            margin: 8px 0;
            color: #155724;
            display: flex;
            justify-content: space-between;
        }
        .download-section {
            background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
            text-align: center;
            color: white;
        }
        .download-btn {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 25px;
            margin: 10px;
            font-weight: bold;
            border: 2px solid rgba(255,255,255,0.3);
            transition: all 0.3s;
        }
        .download-btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        .upload-link {
            background: #fd7e14;
            color: white;
            padding: 15px 30px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
            margin: 20px;
            box-shadow: 0 4px 15px rgba(253,126,20,0.3);
            transition: all 0.3s;
        }
        .upload-link:hover {
            background: #e8680e;
            transform: translateY(-2px);
        }
        .note {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍎 Perfect Fruit Detection Results</h1>
        <div class="subtitle">✨ This is what successful analysis looks like ✨</div>
        
        <div class="success-banner">
            🎉 <strong>ANALYSIS SUCCESSFUL!</strong> - This demonstrates the full capabilities of our fruit label placement system
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ total_fruits }}</div>
                <div>Fruits Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ processing_time }}s</div>
                <div>Processing Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">100%</div>
                <div>Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">0.83</div>
                <div>Avg Score</div>
            </div>
        </div>
        
        <div class="image-section">
            <h2>📷 Result Visualization</h2>
            <img src="/demo_image" class="result-image" alt="Perfect Fruit Detection Results">
            <p style="margin-top: 15px; color: #666;">
                <strong>Green circles show optimal label placements on detected red/orange fruits</strong>
            </p>
        </div>
        
        <div>
            <h2>🎯 Individual Fruit Placements</h2>
            <div class="fruit-grid">
                {% for fruit in fruits %}
                <div class="fruit-card">
                    <div class="fruit-title">🍎 Fruit {{ fruit.fruit_id }}</div>
                    <div class="fruit-detail">
                        <span>📍 Position:</span>
                        <span>({{ fruit.center_x }}, {{ fruit.center_y }})</span>
                    </div>
                    <div class="fruit-detail">
                        <span>⭐ Confidence:</span>
                        <span>{{ "%.3f"|format(fruit.score) }}</span>
                    </div>
                    <div class="fruit-detail">
                        <span>📏 Label Radius:</span>
                        <span>{{ fruit.radius }}px</span>
                    </div>
                    <div class="fruit-detail">
                        <span>🎯 Quality:</span>
                        <span>{% if fruit.score > 0.83 %}Excellent{% elif fruit.score > 0.82 %}Very Good{% else %}Good{% endif %}</span>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="download-section">
            <h3>💾 Download Demo Results</h3>
            <p>Get the data from this successful analysis</p>
            <a href="/download/demo_csv" class="download-btn">📄 Download CSV</a>
            <a href="/download/demo_image" class="download-btn">🖼️ Download Image</a>
            <a href="/download/demo_json" class="download-btn">📋 Download JSON</a>
        </div>
        
        <div class="note">
            <strong>💡 Note:</strong> These are permanent demo results showing successful fruit detection. 
            For the detection to work on your images, you need RGB-D images with clearly visible red/orange fruits (apples, tomatoes).
        </div>
        
        <div style="text-align: center;">
            <h3>🔄 Try With Your Own Images</h3>
            <a href="http://localhost:5002" class="upload-link">📤 Upload Your RGB-D Images</a>
        </div>
    </div>
</body>
</html>
    """, **DEMO_RESULTS)

@app.route('/demo_image')
def get_demo_image():
    """Serve the permanent demo result image."""
    # Use ONLY the protected demo result that upload interface cannot touch
    if os.path.exists("protected_demo/perfect_result.png"):
        return send_file("protected_demo/perfect_result.png", mimetype='image/png')
    else:
        return "Protected demo image not available", 404

@app.route('/download/<file_type>')
def download_demo_file(file_type):
    """Download demo result files."""
    if file_type == 'demo_image':
        if os.path.exists("protected_demo/perfect_result.png"):
            return send_file("protected_demo/perfect_result.png", as_attachment=True, download_name='perfect_fruit_analysis.png')
    elif file_type == 'demo_csv':
        # Generate CSV from hardcoded data since it's always the same
        csv_content = "fruit_id,center_x,center_y,score,radius\n"
        for fruit in DEMO_RESULTS['fruits']:
            csv_content += f"{fruit['fruit_id']},{fruit['center_x']},{fruit['center_y']},{fruit['score']:.3f},{fruit['radius']}\n"
        
        from flask import Response
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=demo_fruit_placements.csv"}
        )
    elif file_type == 'demo_json':
        if os.path.exists("protected_demo/perfect_results.json"):
            return send_file("protected_demo/perfect_results.json", as_attachment=True, download_name='demo_fruit_analysis.json')
    
    return "File not found", 404

if __name__ == '__main__':
    print("🍎 Starting Permanent Demo Results Viewer...")
    print("✨ This viewer always shows perfect fruit detection results")
    print("🔗 Perfect Results: http://localhost:5004")
    print("📤 Upload Interface: http://localhost:5002")
    print("🛑 Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5004, debug=False)