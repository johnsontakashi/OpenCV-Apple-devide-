"""
Create synthetic test data for the fruit label placement pipeline.
Generates simple RGB and depth images for testing purposes.
"""

import numpy as np
import cv2
import os

def create_test_data():
    """Create simple test images for pipeline testing."""
    
    # Create test data directory
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Image dimensions
    width, height = 640, 480
    
    # Create a simple color image with circular "fruits"
    color_image = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark background
    
    # Add some circular "apples" with red/orange colors
    fruits = [
        {"center": (200, 150), "radius": 60, "color": (220, 80, 40)},   # Red fruit
        {"center": (450, 200), "radius": 50, "color": (200, 100, 60)},  # Orange fruit
        {"center": (300, 350), "radius": 55, "color": (180, 90, 50)},   # Another fruit
    ]
    
    # Create corresponding depth image
    depth_image = np.zeros((height, width), dtype=np.uint16)
    
    for fruit in fruits:
        center, radius, color = fruit["center"], fruit["radius"], fruit["color"]
        
        # Draw colored circle on RGB image
        cv2.circle(color_image, center, radius, color, -1)
        
        # Add some variation to make it look more natural
        for i in range(10):
            noise_center = (
                center[0] + np.random.randint(-radius//2, radius//2),
                center[1] + np.random.randint(-radius//2, radius//2)
            )
            noise_radius = np.random.randint(5, 15)
            noise_color = tuple(max(0, min(255, c + np.random.randint(-30, 30))) for c in color)
            cv2.circle(color_image, noise_center, noise_radius, noise_color, -1)
        
        # Create depth data (fruit appears closer than background)
        y, x = np.ogrid[:height, :width]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Set fruit depth (closer = smaller depth value)
        fruit_depth = 800 + np.random.randint(-50, 50)  # Around 0.8 meters
        depth_image[mask] = fruit_depth
        
        # Add some depth variation to make surface not perfectly flat
        for dy in range(-radius, radius, 5):
            for dx in range(-radius, radius, 5):
                if dx*dx + dy*dy <= radius*radius:
                    px, py = center[0] + dx, center[1] + dy
                    if 0 <= px < width and 0 <= py < height:
                        variation = np.random.randint(-20, 20)
                        new_depth = max(100, int(depth_image[py, px]) + variation)
                        depth_image[py, px] = min(65535, new_depth)
    
    # Add background depth (tray/table)
    background_mask = depth_image == 0
    depth_image[background_mask] = 1200  # 1.2 meters - further than fruits
    
    # Save test images
    color_path = os.path.join(test_dir, "test_fruits.jpg")
    depth_path = os.path.join(test_dir, "test_fruits_depth.png")
    
    cv2.imwrite(color_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth_image)
    
    print(f"Created test data:")
    print(f"  Color: {color_path}")
    print(f"  Depth: {depth_path}")
    print(f"  Image size: {width}x{height}")
    print(f"  Number of fruits: {len(fruits)}")
    
    return test_dir

if __name__ == "__main__":
    create_test_data()