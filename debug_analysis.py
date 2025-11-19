#!/usr/bin/env python3
"""
Debug script to test fruit detection on uploaded images
"""

import cv2
import numpy as np
import os
from single_placement_pipeline import SinglePlacementAnalyzer

def debug_fruit_detection():
    """Test fruit detection with uploaded images."""
    
    print("🔍 Debugging Fruit Detection...")
    
    # Check uploaded images
    upload_dir = "uploaded_images"
    files = os.listdir(upload_dir)
    print(f"📁 Found uploaded files: {files}")
    
    # Try different combinations
    color_candidates = [f for f in files if not 'depth' in f.lower()]
    depth_candidates = [f for f in files if 'depth' in f.lower()]
    
    print(f"🎨 Color candidates: {color_candidates}")
    print(f"📏 Depth candidates: {depth_candidates}")
    
    # Try to find a working pair
    if color_candidates and depth_candidates:
        color_path = os.path.join(upload_dir, color_candidates[0])
        depth_path = os.path.join(upload_dir, depth_candidates[0])
        test_pair(color_path, depth_path)
    elif len(color_candidates) >= 2:
        # Try using two color images (second as depth)
        color_path = os.path.join(upload_dir, color_candidates[0])
        depth_path = os.path.join(upload_dir, color_candidates[1])
        print("⚠️ Using second color image as depth (may not work well)")
        test_pair(color_path, depth_path)
    else:
        print("❌ Need at least 2 images to proceed")
        return
    
def test_pair(color_path, depth_path):
    """Test analysis with a specific image pair."""
    
    print(f"\n🧪 Testing pair:")
    print(f"   Color: {color_path}")
    print(f"   Depth: {depth_path}")
    
    try:
        # Load images
        print("📖 Loading images...")
        color_image = cv2.imread(color_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        print(f"✅ Color shape: {color_image.shape}")
        print(f"✅ Depth shape: {depth_image.shape}")
        print(f"✅ Color type: {color_image.dtype}")
        print(f"✅ Depth type: {depth_image.dtype}")
        print(f"✅ Depth range: {np.min(depth_image)} - {np.max(depth_image)}")
        
        # Test fruit detection step by step
        analyzer = SinglePlacementAnalyzer()
        
        print("\n🍎 Testing fruit segmentation...")
        
        # Convert to HSV for fruit detection
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        
        # Red/orange fruit detection (apples/tomatoes)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        fruit_mask = cv2.bitwise_or(mask1, mask2)
        
        # Count fruit pixels
        fruit_pixels = np.sum(fruit_mask > 0)
        total_pixels = fruit_mask.shape[0] * fruit_mask.shape[1]
        fruit_percentage = (fruit_pixels / total_pixels) * 100
        
        print(f"🎯 Fruit pixels: {fruit_pixels}")
        print(f"📊 Fruit percentage: {fruit_percentage:.2f}%")
        
        if fruit_pixels < 1000:
            print("⚠️ Very few fruit pixels detected - adjusting thresholds...")
            
            # Try more lenient thresholds
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([160, 30, 30])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            fruit_mask = cv2.bitwise_or(mask1, mask2)
            
            fruit_pixels = np.sum(fruit_mask > 0)
            fruit_percentage = (fruit_pixels / total_pixels) * 100
            print(f"🔄 Adjusted fruit pixels: {fruit_pixels}")
            print(f"📊 Adjusted fruit percentage: {fruit_percentage:.2f}%")
        
        # Save debug mask
        cv2.imwrite("output/debug_fruit_mask.png", fruit_mask)
        print("💾 Saved fruit mask to output/debug_fruit_mask.png")
        
        # Now try full analysis
        print("\n🚀 Running full analysis...")
        results = analyzer.process_image_single(color_image, depth_image)
        
        print(f"📊 Results: {len(results['fruits'])} fruits detected")
        print(f"⏱️ Processing time: {results['processing_time']:.3f}s")
        
        if len(results['fruits']) > 0:
            print("✅ SUCCESS! Fruits detected:")
            for i, fruit in enumerate(results['fruits']):
                print(f"   🍎 Fruit {i+1}: {fruit}")
        else:
            print("❌ No fruits detected")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fruit_detection()