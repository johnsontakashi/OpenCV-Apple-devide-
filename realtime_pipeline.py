#!/usr/bin/env python3
"""
Real-time Optimized Fruit Label Placement Pipeline

High-performance version designed for real-time applications.
Target: <1 second total processing time per image.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LabelCandidate:
    """Lightweight candidate data structure."""
    x: int
    y: int
    score: float
    depth: float
    planarity_score: float
    

class RealTimeFruitAnalyzer:
    """Real-time optimized fruit analysis pipeline."""
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """Initialize with optimized configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Optimization parameters
        self.target_width = self.config['image']['max_width']
        self.target_height = self.config['image']['max_height']
        self.max_fruits = self.config['performance']['max_fruits_per_image']
        self.candidates_per_fruit = self.config['candidate_selection']['max_candidates_per_fruit']
        self.downsample_factor = self.config['pointcloud']['downsample_factor']
        self.grid_size = self.config['planarity']['grid_size']
        
        # Camera parameters
        self.fx = self.config['camera']['fx']
        self.fy = self.config['camera']['fy']
        self.cx = self.config['camera']['cx']
        self.cy = self.config['camera']['cy']
        
        # Segmentation parameters
        self.red_lower = np.array(self.config['segmentation']['color_ranges']['red_lower'])
        self.red_upper = np.array(self.config['segmentation']['color_ranges']['red_upper'])
        self.orange_lower = np.array(self.config['segmentation']['color_ranges']['orange_lower'])
        self.orange_upper = np.array(self.config['segmentation']['color_ranges']['orange_upper'])
        self.depth_min = 0.1  # 10cm minimum
        self.depth_max = 2.0  # 2m maximum
        self.min_area = self.config['segmentation']['min_fruit_area']
        
        logger.info(f"Real-time pipeline initialized. Target: {self.target_width}x{self.target_height}")
    
    def preprocess_fast(self, color_image: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fast preprocessing with minimal operations."""
        start_time = time.time()
        
        # Resize images for faster processing
        if color_image.shape[:2] != (self.target_height, self.target_width):
            color_resized = cv2.resize(color_image, (self.target_width, self.target_height))
            depth_resized = cv2.resize(depth_image, (self.target_width, self.target_height))
        else:
            color_resized = color_image.copy()
            depth_resized = depth_image.copy()
        
        # Normalize depth
        depth_normalized = depth_resized.astype(np.float32) / 1000.0  # Convert to meters
        
        # Simple hole filling - replace zeros with nearby values
        mask = depth_normalized == 0
        if np.any(mask):
            # Fast morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            depth_filled = cv2.morphologyEx(depth_normalized, cv2.MORPH_CLOSE, kernel)
            depth_normalized[mask] = depth_filled[mask]
        
        logger.debug(f"Preprocessing: {time.time() - start_time:.3f}s")
        return color_resized, depth_normalized
    
    def segment_fast(self, color_image: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Fast fruit segmentation using optimized algorithms."""
        start_time = time.time()
        
        # HSV color segmentation
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        
        # Red and orange fruit detection
        red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        color_mask = cv2.bitwise_or(red_mask, orange_mask)
        
        # Depth-based foreground
        valid_depth = (depth_image > self.depth_min) & (depth_image < self.depth_max)
        
        # Combine masks
        combined_mask = color_mask & (valid_depth.astype(np.uint8) * 255)
        
        # Fast morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Fast connected components
        num_labels, labels = cv2.connectedComponents(combined_mask)
        
        # Extract fruit properties quickly
        fruits = []
        for label_id in range(1, min(num_labels, self.max_fruits + 1)):
            mask = (labels == label_id)
            area = np.sum(mask)
            
            if area < self.min_area:
                continue
            
            # Fast centroid calculation
            y_coords, x_coords = np.where(mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            
            # Bounding box
            bbox = [np.min(y_coords), np.min(x_coords), np.max(y_coords), np.max(x_coords)]
            
            # Mean depth
            mean_depth = np.mean(depth_image[mask])
            
            fruit_data = {
                'id': label_id,
                'mask': mask,
                'area': area,
                'centroid': (centroid_y, centroid_x),
                'bbox': bbox,
                'mean_depth': mean_depth
            }
            fruits.append(fruit_data)
        
        logger.debug(f"Segmentation: {time.time() - start_time:.3f}s, {len(fruits)} fruits")
        return labels, fruits
    
    def analyze_planarity_fast(self, depth_image: np.ndarray, mask: np.ndarray, 
                             sample_points: List[Tuple[int, int]]) -> np.ndarray:
        """Fast planarity analysis using sampling."""
        start_time = time.time()
        
        planarity_scores = []
        
        for y, x in sample_points:
            if not mask[y, x]:
                planarity_scores.append(1.0)  # Not on fruit = bad planarity
                continue
            
            # Small local window for speed
            r = 5  # Small radius for speed
            y_min, y_max = max(0, y-r), min(depth_image.shape[0], y+r+1)
            x_min, x_max = max(0, x-r), min(depth_image.shape[1], x+r+1)
            
            local_depth = depth_image[y_min:y_max, x_min:x_max]
            local_mask = mask[y_min:y_max, x_min:x_max]
            
            valid_depths = local_depth[local_mask]
            
            if len(valid_depths) < 5:
                planarity_scores.append(1.0)
                continue
            
            # Simple planarity measure: standard deviation of depths
            depth_std = np.std(valid_depths)
            planarity_score = min(depth_std * 100, 1.0)  # Normalize roughly
            planarity_scores.append(planarity_score)
        
        logger.debug(f"Planarity analysis: {time.time() - start_time:.3f}s")
        return np.array(planarity_scores)
    
    def select_candidates_fast(self, color_image: np.ndarray, depth_image: np.ndarray,
                             fruit: Dict) -> List[LabelCandidate]:
        """Fast candidate selection using sparse sampling."""
        start_time = time.time()
        
        mask = fruit['mask']
        bbox = fruit['bbox']
        
        # Sample candidate locations sparsely using grid
        sample_points = []
        for y in range(bbox[0], bbox[2], self.grid_size):
            for x in range(bbox[1], bbox[3], self.grid_size):
                if mask[y, x]:
                    sample_points.append((y, x))
        
        if len(sample_points) == 0:
            return []
        
        # Fast planarity analysis
        planarity_scores = self.analyze_planarity_fast(depth_image, mask, sample_points)
        
        # Fast scoring
        candidates = []
        for i, (y, x) in enumerate(sample_points):
            # Edge distance (fast approximation)
            edge_dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)[y, x]
            
            if edge_dist < 15:  # Too close to edge
                continue
            
            # Color uniformity (simple local std)
            r = 5
            y_min, y_max = max(0, y-r), min(color_image.shape[0], y+r+1)
            x_min, x_max = max(0, x-r), min(color_image.shape[1], x+r+1)
            local_color = color_image[y_min:y_max, x_min:x_max]
            color_std = np.mean(np.std(local_color.reshape(-1, 3), axis=0))
            color_uniformity = max(0, 1.0 - color_std / 50.0)
            
            # Combined score
            planarity_score = 1.0 - planarity_scores[i]
            edge_score = min(edge_dist / 30.0, 1.0)
            
            total_score = (0.6 * planarity_score + 
                          0.3 * color_uniformity + 
                          0.1 * edge_score)
            
            if total_score > 0.4:  # Threshold for valid candidates
                candidate = LabelCandidate(
                    x=x, y=y, 
                    score=total_score,
                    depth=depth_image[y, x],
                    planarity_score=planarity_score
                )
                candidates.append(candidate)
        
        # Sort by score and take top candidates
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:self.candidates_per_fruit]
        
        logger.debug(f"Candidate selection: {time.time() - start_time:.3f}s, {len(candidates)} candidates")
        return candidates
    
    def process_image_realtime(self, color_image: np.ndarray, depth_image: np.ndarray) -> Dict:
        """Process image pair for real-time analysis."""
        total_start = time.time()
        
        results = {
            'processing_time': 0,
            'fruits': [],
            'total_candidates': 0,
            'performance_stats': {}
        }
        
        try:
            # 1. Fast preprocessing
            step_start = time.time()
            color_proc, depth_proc = self.preprocess_fast(color_image, depth_image)
            preprocess_time = time.time() - step_start
            
            # 2. Fast segmentation  
            step_start = time.time()
            labels, fruits = self.segment_fast(color_proc, depth_proc)
            segmentation_time = time.time() - step_start
            
            # 3. Fast candidate selection
            step_start = time.time()
            all_candidates = []
            
            for fruit in fruits:
                candidates = self.select_candidates_fast(color_proc, depth_proc, fruit)
                
                # Convert to export format
                fruit_result = {
                    'fruit_id': fruit['id'],
                    'area': fruit['area'],
                    'centroid': fruit['centroid'],
                    'bbox': fruit['bbox'],
                    'mean_depth': fruit['mean_depth'],
                    'candidates': [
                        {
                            'pixel_location': [c.y, c.x],
                            'depth': c.depth,
                            'total_score': c.score,
                            'planarity_score': c.planarity_score,
                            'world_coordinates': self._pixel_to_world(c.x, c.y, c.depth)
                        }
                        for c in candidates
                    ],
                    'num_candidates': len(candidates)
                }
                
                results['fruits'].append(fruit_result)
                all_candidates.extend(candidates)
            
            candidate_time = time.time() - step_start
            
            # Performance statistics
            total_time = time.time() - total_start
            results['processing_time'] = total_time
            results['total_candidates'] = len(all_candidates)
            results['performance_stats'] = {
                'preprocessing_time': preprocess_time,
                'segmentation_time': segmentation_time,
                'candidate_time': candidate_time,
                'total_time': total_time,
                'fps_estimate': 1.0 / total_time if total_time > 0 else 0,
                'num_fruits': len(fruits)
            }
            
            logger.info(f"Real-time processing: {total_time:.3f}s ({1.0/total_time:.1f} FPS), "
                       f"{len(fruits)} fruits, {len(all_candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")
            results['error'] = str(e)
            results['processing_time'] = time.time() - total_start
        
        return results
    
    def _pixel_to_world(self, x: int, y: int, depth: float) -> List[float]:
        """Convert pixel coordinates to world coordinates."""
        if depth <= 0:
            return [0.0, 0.0, 0.0]
        
        world_x = (x - self.cx) * depth / self.fx
        world_y = (y - self.cy) * depth / self.fy
        world_z = depth
        
        return [world_x, world_y, world_z]
    
    def visualize_results_fast(self, color_image: np.ndarray, results: Dict) -> np.ndarray:
        """Fast visualization for real-time display."""
        vis_image = color_image.copy()
        
        if 'error' in results:
            cv2.putText(vis_image, f"Error: {results['error']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis_image
        
        # Draw fruit bounding boxes and candidates
        for fruit in results['fruits']:
            bbox = fruit['bbox']
            
            # Draw bounding box
            cv2.rectangle(vis_image, 
                         (bbox[1], bbox[0]), 
                         (bbox[3], bbox[2]), 
                         (0, 255, 0), 2)
            
            # Draw fruit ID
            cv2.putText(vis_image, f"F{fruit['fruit_id']}", 
                       (bbox[1], bbox[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw candidates
            for i, candidate in enumerate(fruit['candidates']):
                y, x = candidate['pixel_location']
                score = candidate['total_score']
                
                # Color code by score
                if score > 0.8:
                    color = (0, 255, 0)  # Green for high score
                elif score > 0.6:
                    color = (0, 255, 255)  # Yellow for medium score
                else:
                    color = (255, 0, 0)  # Red for low score
                
                cv2.circle(vis_image, (x, y), 8, color, 2)
                cv2.putText(vis_image, f"{score:.2f}", (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Performance info
        stats = results.get('performance_stats', {})
        fps = stats.get('fps_estimate', 0)
        total_time = stats.get('total_time', 0)
        
        cv2.putText(vis_image, f"FPS: {fps:.1f} | Time: {total_time:.3f}s", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Fruits: {len(results['fruits'])} | Candidates: {results['total_candidates']}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image


def benchmark_performance(analyzer: RealTimeFruitAnalyzer, 
                         color_image: np.ndarray, 
                         depth_image: np.ndarray, 
                         num_runs: int = 10) -> Dict:
    """Benchmark the real-time analyzer performance."""
    times = []
    
    print(f"Benchmarking performance over {num_runs} runs...")
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=' ')
        
        start_time = time.time()
        results = analyzer.process_image_realtime(color_image, depth_image)
        run_time = time.time() - start_time
        
        times.append(run_time)
        print(f"{run_time:.3f}s ({1/run_time:.1f} FPS)")
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_fps': np.mean([1/t for t in times]),
        'times': times
    }


if __name__ == "__main__":
    import sys
    import os
    
    # Test with uploaded image or create sample
    if len(sys.argv) > 1:
        color_path = sys.argv[1]
        depth_path = sys.argv[2] if len(sys.argv) > 2 else color_path.replace('.', '_depth.')
    else:
        # Use uploaded image if available
        if os.path.exists("uploaded_images"):
            files = os.listdir("uploaded_images")
            color_files = [f for f in files if 'depth' not in f.lower()]
            if color_files:
                color_path = os.path.join("uploaded_images", color_files[0])
                depth_path = color_path.replace('.', '_depth.')
            else:
                print("No uploaded images found. Using sample data.")
                color_path = "sample_data/sample_color.png"
                depth_path = "sample_data/sample_depth.png"
        else:
            print("Using sample data.")
            color_path = "sample_data/sample_color.png"
            depth_path = "sample_data/sample_depth.png"
    
    # Load images
    try:
        color_image = cv2.imread(color_path)
        if color_image is None:
            print(f"Could not load color image: {color_path}")
            sys.exit(1)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Could not load depth image: {depth_path}")
            sys.exit(1)
        
        print(f"Loaded images: {color_image.shape}, {depth_image.shape}")
        
        # Initialize analyzer
        analyzer = RealTimeFruitAnalyzer()
        
        # Single test run
        print("\nSingle test run:")
        results = analyzer.process_image_realtime(color_image, depth_image)
        
        # Performance benchmark
        benchmark_stats = benchmark_performance(analyzer, color_image, depth_image, num_runs=5)
        
        print(f"\nPerformance Benchmark:")
        print(f"Mean time: {benchmark_stats['mean_time']:.3f}s ± {benchmark_stats['std_time']:.3f}s")
        print(f"Mean FPS: {benchmark_stats['mean_fps']:.1f}")
        print(f"Range: {benchmark_stats['min_time']:.3f}s - {benchmark_stats['max_time']:.3f}s")
        
        target_time = 1.0
        if benchmark_stats['mean_time'] <= target_time:
            print(f"✅ Real-time target achieved! ({benchmark_stats['mean_time']:.3f}s <= {target_time}s)")
        else:
            print(f"❌ Real-time target missed. ({benchmark_stats['mean_time']:.3f}s > {target_time}s)")
        
        # Save visualization
        vis_image = analyzer.visualize_results_fast(color_image, results)
        output_path = "output/realtime_result.png"
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()