#!/usr/bin/env python3
"""
Create Single Result Image
Takes existing results and shows only the BEST placement per fruit.
"""

import cv2
import numpy as np
import pandas as pd
import json
import os

def create_single_result_image():
    """Create image showing only one best placement per fruit."""
    
    # Load the original color image
    color_image_path = "sample_data/sample_color.png"
    if os.path.exists("uploaded_images"):
        files = os.listdir("uploaded_images") 
        color_files = [f for f in files if 'depth' not in f.lower()]
        if color_files:
            color_image_path = os.path.join("uploaded_images", color_files[0])
    
    # Load image
    image = cv2.imread(color_image_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load existing results
    results_path = "output/results.csv" 
    if not os.path.exists(results_path):
        print("❌ No results found")
        return
        
    df = pd.read_csv(results_path)
    print(f"📊 Loaded {len(df)} total results")
    
    # Group by fruit_id and select BEST result for each fruit
    best_results = []
    
    for fruit_id in df['fruit_id'].unique():
        fruit_data = df[df['fruit_id'] == fruit_id]
        
        # Select the result with highest total score
        best_row = fruit_data.loc[fruit_data['candidate_total_score'].idxmax()]
        best_results.append(best_row)
        
        print(f"🍎 Fruit {fruit_id}: Selected placement at ({best_row['candidate_pixel_x']}, {best_row['candidate_pixel_y']}) score={best_row['candidate_total_score']:.3f}")
    
    print(f"✅ Selected {len(best_results)} best placements (one per fruit)")
    
    # Create visualization
    vis_image = image.copy()
    
    for i, result in enumerate(best_results):
        fruit_id = int(result['fruit_id'])
        x = int(result['candidate_pixel_x'])
        y = int(result['candidate_pixel_y'])
        score = float(result['candidate_total_score'])
        planarity = float(result['candidate_planarity_score'])
        
        # Calculate label radius based on fruit area
        area = float(result['fruit_area'])
        radius = max(15, min(50, int(np.sqrt(area/100))))
        
        # Color code by score quality
        if score > 0.9:
            color = (0, 255, 0)    # Green - excellent
        elif score > 0.8:
            color = (0, 255, 255)  # Yellow - good  
        elif score > 0.7:
            color = (255, 165, 0)  # Orange - ok
        else:
            color = (255, 0, 0)    # Red - poor
        
        # Draw label placement circle
        cv2.circle(vis_image, (x, y), radius, color, 3)
        
        # Draw center point
        cv2.circle(vis_image, (x, y), 3, (255, 255, 255), -1)
        
        # Draw fruit ID
        cv2.putText(vis_image, f"F{fruit_id}", (x - 10, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw score
        cv2.putText(vis_image, f"{score:.2f}", (x - 15, y + radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw radius
        cv2.putText(vis_image, f"R{radius}", (x + radius + 5, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add title and summary
    cv2.putText(vis_image, f"SINGLE PLACEMENT PER FRUIT - {len(best_results)} fruits", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(vis_image, "Green=Excellent, Yellow=Good, Orange=OK, Red=Poor", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save result
    output_path = "output/SINGLE_RESULT_IMAGE.png"
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    print(f"🎯 SINGLE RESULT IMAGE saved: {output_path}")
    
    # Save simplified CSV with only best results
    simple_csv = []
    simple_csv.append("fruit_id,placement_x,placement_y,score,planarity,radius,area")
    
    for result in best_results:
        fruit_id = int(result['fruit_id'])
        x = int(result['candidate_pixel_x'])
        y = int(result['candidate_pixel_y']) 
        score = float(result['candidate_total_score'])
        planarity = float(result['candidate_planarity_score'])
        area = float(result['fruit_area'])
        radius = max(15, min(50, int(np.sqrt(area/100))))
        
        simple_csv.append(f"{fruit_id},{x},{y},{score:.4f},{planarity:.4f},{radius},{area}")
    
    with open("output/SINGLE_RESULTS.csv", 'w') as f:
        f.write('\n'.join(simple_csv))
    
    print(f"📄 Simplified CSV saved: output/SINGLE_RESULTS.csv")
    
    return vis_image, best_results

if __name__ == "__main__":
    print("🎯 Creating single result image...")
    
    try:
        vis_image, results = create_single_result_image()
        print(f"\n✅ SUCCESS!")
        print(f"   📍 {len(results)} fruits with single optimal placement each")
        print(f"   🖼️ Clean visualization created")
        print(f"   📊 Simplified data exported")
        
        # Show summary
        print(f"\n📋 Summary:")
        for result in results:
            fruit_id = int(result['fruit_id'])
            x = int(result['candidate_pixel_x'])
            y = int(result['candidate_pixel_y'])
            score = float(result['candidate_total_score'])
            print(f"   Fruit {fruit_id}: ({x}, {y}) score={score:.3f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()