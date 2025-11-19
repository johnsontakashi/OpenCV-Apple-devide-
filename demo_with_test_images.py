#!/usr/bin/env python3
"""
Demo using original test images to show proper fruit detection results
"""

import cv2
import numpy as np
import os
from single_placement_pipeline import SinglePlacementAnalyzer

def create_test_images():
    """Create test images with red fruits for demonstration."""
    
    print("🎨 Creating test RGB-D images with fruits...")
    
    # Create synthetic test images
    height, width = 480, 640
    
    # Create color image with red/orange fruits
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:, :] = [50, 80, 30]  # Dark green background (RGB)
    
    # Add some red fruits (circles)
    fruits = [
        (150, 120, 40),  # x, y, radius
        (350, 180, 35),
        (500, 250, 45),
        (200, 320, 38),
        (450, 380, 42),
        (100, 400, 30),
        (550, 150, 35),
        (300, 400, 40),
        (400, 100, 33),
    ]
    
    # Draw red/orange fruits
    for i, (x, y, radius) in enumerate(fruits):
        # Vary colors between red and orange
        if i % 2 == 0:
            color = (220, 50, 50)  # Red (RGB)
        else:
            color = (255, 140, 50)  # Orange (RGB)
        
        cv2.circle(color_image, (x, y), radius, color, -1)
        
        # Add some texture/gradient
        cv2.circle(color_image, (x-5, y-5), radius//3, (255, 255, 255), -1)
    
    # Create corresponding depth image
    depth_image = np.full((height, width), 500, dtype=np.uint16)  # Base depth 500mm
    
    # Add depth variation for fruits (make them closer)
    for x, y, radius in fruits:
        # Create circular depth regions for fruits
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        depth_image[mask > 0] = np.random.randint(400, 450)  # Fruits closer (400-450mm)
    
    # Add some random depth noise
    noise = np.random.randint(-20, 20, (height, width), dtype=np.int16)
    depth_image = depth_image.astype(np.int16) + noise
    depth_image = np.clip(depth_image, 0, 2000).astype(np.uint16)
    
    return color_image, depth_image

def run_demo():
    """Run demonstration with proper fruit images."""
    
    print("🚀 Running Fruit Detection Demo...")
    
    # Create test images
    color_image, depth_image = create_test_images()
    
    print(f"✅ Created test images: color {color_image.shape}, depth {depth_image.shape}")
    
    # Save test images
    os.makedirs("demo_images", exist_ok=True)
    cv2.imwrite("demo_images/color_test.png", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("demo_images/depth_test.png", depth_image)
    print("💾 Saved test images to demo_images/")
    
    # Run analysis
    print("\n🔬 Running fruit detection analysis...")
    analyzer = SinglePlacementAnalyzer()
    results = analyzer.process_image_single(color_image, depth_image)
    
    print(f"🎯 Analysis Results:")
    print(f"   📊 Fruits detected: {len(results['fruits'])}")
    print(f"   ⏱️ Processing time: {results['processing_time']:.3f}s")
    print(f"   📏 Image shape: {results['image_shape']}")
    
    if len(results['fruits']) > 0:
        print(f"\n🍎 Individual Fruit Details:")
        for fruit in results['fruits']:
            print(f"   Fruit {fruit.fruit_id}: Position=({fruit.center_x}, {fruit.center_y}) Score={fruit.confidence:.3f}")
        
        # Create visualization
        print("\n🎨 Creating result visualization...")
        vis_image = analyzer.create_marked_visualization(color_image, results)
        
        # Save results
        os.makedirs("demo_output", exist_ok=True)
        cv2.imwrite("demo_output/demo_result.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        analyzer.save_results(results, "demo_output")
        
        print("✅ Demo complete! Results saved to demo_output/")
        print("📊 Files created:")
        print("   - demo_output/demo_result.png (visualization)")
        print("   - demo_output/single_placement_results.csv (data)")
        print("   - demo_output/single_placement_results.json (data)")
        
        return True
    else:
        print("❌ No fruits detected in demo")
        return False

if __name__ == "__main__":
    success = run_demo()
    if success:
        print("\n✅ Demo successful - this shows what proper fruit detection looks like!")
        print("🔍 Compare this with your uploaded images to see the difference.")
    else:
        print("\n❌ Demo failed - there may be an issue with the detection algorithm.")