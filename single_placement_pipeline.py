#!/usr/bin/env python3
"""
Single Placement Pipeline for Fruit Label Placement
Generates exactly one optimal placement per fruit with visual marking.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OptimalPlacement:
    """Single optimal placement per fruit."""
    fruit_id: int
    center_x: int
    center_y: int
    confidence: float
    planarity_score: float
    depth: float
    world_coords: Tuple[float, float, float]
    radius: int  # Label radius in pixels


class SinglePlacementAnalyzer:
    """Optimized analyzer that finds one best placement per fruit."""
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """Initialize with optimized configuration."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Camera parameters
        self.fx = self.config['camera']['fx']
        self.fy = self.config['camera']['fy'] 
        self.cx = self.config['camera']['cx']
        self.cy = self.config['camera']['cy']
        
        # Placement parameters
        self.min_radius = 15  # Minimum label radius
        self.max_radius = 40  # Maximum label radius
        self.edge_margin = 10  # Minimum distance from fruit edge
        
        logger.info("Single placement analyzer initialized")
    
    def process_image_single(self, color_image: np.ndarray, 
                           depth_image: np.ndarray) -> Dict:
        """Process image and return one placement per fruit."""
        start_time = time.time()
        
        results = {
            'processing_time': 0,
            'fruits': [],
            'total_placements': 0,
            'image_shape': color_image.shape
        }
        
        try:
            # 1. Fast fruit segmentation
            fruits = self._segment_fruits(color_image, depth_image)
            
            # 2. Find optimal placement for each fruit
            for fruit in fruits:
                optimal_placement = self._find_optimal_placement(
                    fruit, color_image, depth_image
                )
                
                if optimal_placement:
                    fruit_result = {
                        'fruit_id': int(fruit['id']),
                        'area': int(fruit['area']),
                        'centroid': [float(fruit['centroid'][0]), float(fruit['centroid'][1])],
                        'bbox': fruit['bbox'],
                        'optimal_placement': {
                            'center': [int(optimal_placement.center_x), int(optimal_placement.center_y)],
                            'confidence': float(optimal_placement.confidence),
                            'planarity_score': float(optimal_placement.planarity_score),
                            'depth': float(optimal_placement.depth),
                            'world_coords': [float(x) for x in optimal_placement.world_coords],
                            'radius': int(optimal_placement.radius)
                        }
                    }
                    results['fruits'].append(fruit_result)
                    results['total_placements'] += 1
            
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            
            logger.info(f"Processed {len(fruits)} fruits with {results['total_placements']} placements in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
        
        return results
    
    def _segment_fruits(self, color_image: np.ndarray, 
                       depth_image: np.ndarray) -> List[Dict]:
        """Fast fruit segmentation using color and depth."""
        # Convert to HSV for color segmentation
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        
        # Combined mask for red/orange fruits
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 50, 50]) 
        red_upper2 = np.array([180, 255, 255])
        orange_lower = np.array([5, 50, 50])
        orange_upper = np.array([25, 255, 255])
        
        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask3 = cv2.inRange(hsv, orange_lower, orange_upper)
        color_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
        
        # Resize depth image to match color image if needed
        if depth_image.shape != color_image.shape[:2]:
            depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))
        
        # Depth filtering
        depth_mask = (depth_image > 100) & (depth_image < 2000)  # 10cm to 2m
        combined_mask = color_mask & (depth_mask.astype(np.uint8) * 255)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(combined_mask)
        
        fruits = []
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            area = np.sum(mask)
            
            if area < 500:  # Filter small areas
                continue
            
            # Calculate fruit properties
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0 or len(x_coords) == 0:
                continue  # Skip invalid masks
                
            centroid_y = float(np.mean(y_coords))
            centroid_x = float(np.mean(x_coords))
            bbox = [int(np.min(y_coords)), int(np.min(x_coords)), 
                   int(np.max(y_coords)), int(np.max(x_coords))]
            mean_depth = float(np.mean(depth_image[mask]) / 1000.0)  # Convert to meters
            
            fruits.append({
                'id': label_id,
                'mask': mask,
                'area': area,
                'centroid': (centroid_y, centroid_x),
                'bbox': bbox,
                'mean_depth': mean_depth
            })
        
        return fruits
    
    def _find_optimal_placement(self, fruit: Dict, color_image: np.ndarray, 
                              depth_image: np.ndarray) -> Optional[OptimalPlacement]:
        """Find the single best placement for this fruit."""
        mask = fruit['mask']
        bbox = fruit['bbox']
        
        # Get distance transform to find points away from edges
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Find the point furthest from edges (most central)
        max_dist = np.max(dist_transform)
        if max_dist < self.edge_margin:
            return None  # Fruit too small for label
        
        # Find all points at maximum distance (center region)
        center_points = np.where(dist_transform >= max_dist * 0.9)
        
        if len(center_points[0]) == 0:
            return None
        
        # Evaluate each center point and pick the best
        best_score = -1
        best_placement = None
        
        for i in range(0, len(center_points[0]), max(1, len(center_points[0])//5)):  # Sample max 5 points
            y, x = center_points[0][i], center_points[1][i]
            
            # Calculate placement quality
            score = self._evaluate_placement(x, y, fruit, color_image, depth_image)
            
            if score > best_score:
                best_score = score
                
                # Calculate optimal label radius
                available_radius = int(dist_transform[y, x])
                label_radius = max(self.min_radius, 
                                 min(self.max_radius, available_radius - 2))
                
                depth_value = depth_image[y, x] / 1000.0  # Convert to meters
                world_coords = self._pixel_to_world(x, y, depth_value)
                
                best_placement = OptimalPlacement(
                    fruit_id=fruit['id'],
                    center_x=x,
                    center_y=y,
                    confidence=score,
                    planarity_score=self._estimate_planarity(x, y, mask, depth_image),
                    depth=depth_value,
                    world_coords=world_coords,
                    radius=label_radius
                )
        
        return best_placement
    
    def _evaluate_placement(self, x: int, y: int, fruit: Dict,
                          color_image: np.ndarray, depth_image: np.ndarray) -> float:
        """Evaluate the quality of a placement location."""
        mask = fruit['mask']
        
        # 1. Color uniformity in local region
        r = 8  # Local region radius
        y_min, y_max = max(0, y-r), min(color_image.shape[0], y+r+1)
        x_min, x_max = max(0, x-r), min(color_image.shape[1], x+r+1)
        
        local_color = color_image[y_min:y_max, x_min:x_max]
        color_std = np.mean(np.std(local_color.reshape(-1, 3), axis=0))
        color_uniformity = max(0, 1.0 - color_std / 50.0)
        
        # 2. Depth consistency (planarity)
        local_depth = depth_image[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max]
        
        valid_depths = local_depth[local_mask]
        if len(valid_depths) > 5:
            depth_std = np.std(valid_depths)
            planarity = max(0, 1.0 - depth_std / 20.0)
        else:
            planarity = 0.5
        
        # 3. Distance from stem/contact areas (avoid top and bottom)
        bbox = fruit['bbox']
        fruit_height = bbox[2] - bbox[0]
        relative_y = (y - bbox[0]) / fruit_height
        
        # Prefer middle region, avoid top 20% and bottom 20%
        if relative_y < 0.2 or relative_y > 0.8:
            position_score = 0.3
        else:
            position_score = 1.0
        
        # Combined score
        total_score = (0.4 * color_uniformity + 
                      0.3 * planarity + 
                      0.3 * position_score)
        
        return total_score
    
    def _estimate_planarity(self, x: int, y: int, mask: np.ndarray, 
                          depth_image: np.ndarray) -> float:
        """Estimate surface planarity at given location."""
        r = 10
        y_min, y_max = max(0, y-r), min(depth_image.shape[0], y+r+1)
        x_min, x_max = max(0, x-r), min(depth_image.shape[1], x+r+1)
        
        local_depth = depth_image[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max]
        
        valid_depths = local_depth[local_mask]
        if len(valid_depths) < 5:
            return 0.5
        
        depth_std = np.std(valid_depths)
        return max(0.0, 1.0 - depth_std / 30.0)
    
    def _pixel_to_world(self, x: int, y: int, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates."""
        if depth <= 0:
            return (0.0, 0.0, 0.0)
        
        world_x = (x - self.cx) * depth / self.fx
        world_y = (y - self.cy) * depth / self.fy
        world_z = depth
        
        return (world_x, world_y, world_z)
    
    def create_marked_visualization(self, color_image: np.ndarray, 
                                  results: Dict) -> np.ndarray:
        """Create visualization with single placement marked per fruit."""
        vis_image = color_image.copy()
        
        if 'error' in results:
            cv2.putText(vis_image, f"Error: {results['error']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return vis_image
        
        # Draw each fruit with its optimal placement
        for i, fruit in enumerate(results['fruits']):
            bbox = fruit['bbox']
            placement = fruit['optimal_placement']
            
            # Draw fruit bounding box
            cv2.rectangle(vis_image, 
                         (bbox[1], bbox[0]), 
                         (bbox[3], bbox[2]), 
                         (0, 255, 0), 2)
            
            # Draw fruit ID
            cv2.putText(vis_image, f"Fruit {fruit['fruit_id']}", 
                       (bbox[1], bbox[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw optimal placement
            center = placement['center']
            radius = placement['radius']
            confidence = placement['confidence']
            planarity = placement['planarity_score']
            
            # Color code by quality: green (good) > yellow (ok) > red (poor)
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow  
            else:
                color = (255, 0, 0)  # Red
            
            # Draw label circle
            cv2.circle(vis_image, (center[0], center[1]), radius, color, 3)
            
            # Draw center point
            cv2.circle(vis_image, (center[0], center[1]), 3, (255, 255, 255), -1)
            
            # Draw quality scores
            score_text = f"C:{confidence:.2f} P:{planarity:.2f}"
            cv2.putText(vis_image, score_text, 
                       (center[0] - 40, center[1] - radius - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw radius
            cv2.putText(vis_image, f"R:{radius}px", 
                       (center[0] - 20, center[1] + radius + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add summary info
        num_fruits = len(results['fruits'])
        processing_time = results['processing_time']
        
        cv2.putText(vis_image, f"Fruits: {num_fruits} | Processing: {processing_time:.3f}s", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Single placement per fruit", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def save_results(self, results: Dict, output_path: str = "output"):
        """Save results in simplified format."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save JSON results
        with open(f"{output_path}/single_placement_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV in simplified format
        csv_data = []
        csv_data.append("image_name,fruit_id,fruit_area,center_x,center_y,confidence,planarity_score,depth,world_x,world_y,world_z,label_radius")
        
        for fruit in results['fruits']:
            placement = fruit['optimal_placement']
            csv_data.append(
                f"single_image,{fruit['fruit_id']},{fruit['area']},"
                f"{placement['center'][0]},{placement['center'][1]},"
                f"{placement['confidence']:.4f},{placement['planarity_score']:.4f},"
                f"{placement['depth']:.4f},"
                f"{placement['world_coords'][0]:.4f},{placement['world_coords'][1]:.4f},{placement['world_coords'][2]:.4f},"
                f"{placement['radius']}"
            )
        
        with open(f"{output_path}/single_placement_results.csv", 'w') as f:
            f.write('\n'.join(csv_data))
        
        logger.info(f"Results saved to {output_path}")


def process_single_image(color_path: str, depth_path: str, 
                        output_path: str = "output") -> Dict:
    """Process a single image and return one placement per fruit."""
    
    # Load images
    color_image = cv2.imread(color_path)
    if color_image is None:
        raise ValueError(f"Could not load color image: {color_path}")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    
    # Create analyzer
    analyzer = SinglePlacementAnalyzer()
    
    # Process image
    results = analyzer.process_image_single(color_image, depth_image)
    
    # Create visualization
    vis_image = analyzer.create_marked_visualization(color_image, results)
    
    # Save visualization
    os.makedirs(output_path, exist_ok=True)
    vis_path = f"{output_path}/single_placement_visualization.png"
    cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Save results
    analyzer.save_results(results, output_path)
    
    print(f"✅ Processing complete!")
    print(f"   Fruits detected: {len(results['fruits'])}")
    print(f"   Processing time: {results['processing_time']:.3f}s")
    print(f"   Visualization saved: {vis_path}")
    print(f"   Results saved: {output_path}/single_placement_results.json")
    
    return results


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Use uploaded images or sample data
    if len(sys.argv) > 2:
        color_path = sys.argv[1]
        depth_path = sys.argv[2]
    else:
        # Check for uploaded images
        if os.path.exists("uploaded_images"):
            files = os.listdir("uploaded_images")
            color_files = [f for f in files if 'depth' not in f.lower()]
            if color_files:
                color_path = os.path.join("uploaded_images", color_files[0])
                depth_path = color_path.replace('.', '_depth.')
                if not os.path.exists(depth_path):
                    # Try other depth naming conventions
                    depth_candidates = [
                        color_path.replace('.png', '_depth.png'),
                        color_path.replace('.jpg', '_depth.jpg'),
                        color_path.replace('.jpeg', '_depth.jpeg')
                    ]
                    for candidate in depth_candidates:
                        if os.path.exists(candidate):
                            depth_path = candidate
                            break
            else:
                print("No uploaded images found, using sample data")
                color_path = "sample_data/sample_color.png"
                depth_path = "sample_data/sample_depth.png"
        else:
            print("Using sample data")
            color_path = "sample_data/sample_color.png" 
            depth_path = "sample_data/sample_depth.png"
    
    try:
        results = process_single_image(color_path, depth_path)
        
        # Print summary
        print(f"\n📊 Summary:")
        for i, fruit in enumerate(results['fruits']):
            placement = fruit['optimal_placement']
            print(f"   Fruit {fruit['fruit_id']}: "
                  f"Label at ({placement['center'][0]}, {placement['center'][1]}) "
                  f"radius {placement['radius']}px, "
                  f"confidence {placement['confidence']:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()