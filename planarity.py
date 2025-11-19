"""
Surface Planarity and Curvature Analysis Module

Analyzes fruit surfaces for flatness to identify suitable label placement locations.
Computes local planarity scores using SVD plane fitting and PCA analysis.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List
from scipy import ndimage
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class PlanarityAnalyzer:
    """Analyzes surface planarity and curvature for label placement."""
    
    def __init__(self, config: dict):
        """
        Initialize planarity analyzer with configuration parameters.
        
        Args:
            config: Dictionary containing analysis parameters
        """
        self.config = config
        self.label_diameter_mm = config.get('label_diameter_mm', 20.0)
        self.label_diameter_pixels = config.get('label_diameter_pixels', 40)
        self.window_scale_factor = config.get('window_scale_factor', 1.2)
        self.planarity_threshold = config.get('planarity_threshold', 0.01)  # RMS error threshold
        
        # Camera parameters for pixel-to-metric conversion
        self.fx = config.get('camera_fx', 525.0)
        self.depth_scale = config.get('depth_scale', 1000.0)
        
    def compute_window_size(self, depth_value: float) -> int:
        """
        Compute analysis window size based on depth and desired physical size.
        
        Args:
            depth_value: Depth at the location in meters
            
        Returns:
            Window radius in pixels
        """
        if depth_value <= 0:
            return self.label_diameter_pixels // 2
        
        # Convert physical label size to pixels at given depth
        pixels_per_mm = self.fx / (depth_value * 1000)  # Convert to mm
        radius_pixels = int((self.label_diameter_mm * pixels_per_mm * self.window_scale_factor) / 2)
        
        # Clamp to reasonable range
        return max(10, min(radius_pixels, 50))
    
    def fit_plane_svd(self, points_3d: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit a plane to 3D points using SVD (least squares).
        
        Args:
            points_3d: Array of 3D points (N x 3)
            
        Returns:
            Tuple of (plane_normal, rms_error)
        """
        if len(points_3d) < 3:
            return np.array([0, 0, 1]), float('inf')
        
        # Center the points
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        
        # SVD to find the plane
        try:
            U, S, Vt = np.linalg.svd(centered_points)
            # The plane normal is the last row of V (smallest singular value)
            normal = Vt[-1]
            
            # Compute RMS fitting error
            distances = np.abs(np.dot(centered_points, normal))
            rms_error = np.sqrt(np.mean(distances ** 2))
            
            return normal, rms_error
        
        except np.linalg.LinAlgError:
            return np.array([0, 0, 1]), float('inf')
    
    def compute_pca_planarity(self, points_3d: np.ndarray) -> float:
        """
        Compute planarity score using PCA eigenvalue analysis.
        
        Args:
            points_3d: Array of 3D points (N x 3)
            
        Returns:
            Planarity score (0 = perfectly flat, 1 = highly curved)
        """
        if len(points_3d) < 3:
            return 1.0
        
        try:
            pca = PCA()
            pca.fit(points_3d)
            
            eigenvalues = pca.explained_variance_
            
            # Sort eigenvalues in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            if eigenvalues[0] == 0:
                return 1.0
            
            # Planarity measure: ratio of smallest to largest eigenvalue
            # Low ratio indicates planar surface
            planarity_ratio = eigenvalues[-1] / eigenvalues[0]
            
            return planarity_ratio
        
        except:
            return 1.0
    
    def extract_local_points(self, depth_image: np.ndarray, 
                           center_y: int, center_x: int, 
                           window_radius: int) -> np.ndarray:
        """
        Extract 3D points from a local window around a center pixel.
        
        Args:
            depth_image: Normalized depth image in meters
            center_y: Y coordinate of center pixel
            center_x: X coordinate of center pixel
            window_radius: Radius of analysis window
            
        Returns:
            Array of 3D points in the window
        """
        height, width = depth_image.shape
        
        # Define window bounds
        y_min = max(0, center_y - window_radius)
        y_max = min(height, center_y + window_radius + 1)
        x_min = max(0, center_x - window_radius)
        x_max = min(width, center_x + window_radius + 1)
        
        # Extract depth values
        local_depth = depth_image[y_min:y_max, x_min:x_max]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Filter valid depth pixels
        valid_mask = local_depth > 0
        
        if np.sum(valid_mask) < 3:
            return np.array([]).reshape(0, 3)
        
        # Convert to 3D coordinates
        z = local_depth[valid_mask]
        x = (x_coords[valid_mask] - self.config.get('camera_cx', 320.0)) * z / self.fx
        y = (y_coords[valid_mask] - self.config.get('camera_cy', 240.0)) * z / self.config.get('camera_fy', 525.0)
        
        points_3d = np.stack([x, y, z], axis=1)
        
        return points_3d
    
    def analyze_local_planarity(self, depth_image: np.ndarray,
                              center_y: int, center_x: int) -> Dict:
        """
        Analyze planarity in a local neighborhood around a pixel.
        
        Args:
            depth_image: Normalized depth image in meters
            center_y: Y coordinate of center pixel
            center_x: X coordinate of center pixel
            
        Returns:
            Dictionary with planarity analysis results
        """
        # Get depth at center pixel
        center_depth = depth_image[center_y, center_x]
        
        if center_depth <= 0:
            return {
                'rms_error': float('inf'),
                'pca_planarity': 1.0,
                'valid_points': 0,
                'coverage': 0.0,
                'plane_normal': np.array([0, 0, 1])
            }
        
        # Compute window size
        window_radius = self.compute_window_size(center_depth)
        
        # Extract local 3D points
        points_3d = self.extract_local_points(depth_image, center_y, center_x, window_radius)
        
        if len(points_3d) < 3:
            return {
                'rms_error': float('inf'),
                'pca_planarity': 1.0,
                'valid_points': 0,
                'coverage': 0.0,
                'plane_normal': np.array([0, 0, 1])
            }
        
        # Fit plane using SVD
        plane_normal, rms_error = self.fit_plane_svd(points_3d)
        
        # Compute PCA-based planarity
        pca_planarity = self.compute_pca_planarity(points_3d)
        
        # Compute coverage (percentage of valid pixels in window)
        total_pixels = (2 * window_radius + 1) ** 2
        coverage = len(points_3d) / total_pixels
        
        return {
            'rms_error': rms_error,
            'pca_planarity': pca_planarity,
            'valid_points': len(points_3d),
            'coverage': coverage,
            'plane_normal': plane_normal,
            'window_radius': window_radius
        }
    
    def compute_planarity_map(self, depth_image: np.ndarray, 
                            instance_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute planarity map for an entire fruit instance.
        
        Args:
            depth_image: Normalized depth image in meters
            instance_mask: Binary mask for the fruit instance
            
        Returns:
            Tuple of (planarity_map, rms_error_map)
        """
        height, width = depth_image.shape
        planarity_map = np.ones((height, width), dtype=np.float32)  # 1 = highly curved
        rms_error_map = np.full((height, width), float('inf'), dtype=np.float32)
        
        # Find pixels to analyze
        mask_indices = np.where(instance_mask > 0)
        
        logger.info(f"Analyzing planarity for {len(mask_indices[0])} pixels")
        
        # Analyze each pixel in the instance
        for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
            if i % 1000 == 0:
                logger.debug(f"Analyzed {i}/{len(mask_indices[0])} pixels")
            
            analysis = self.analyze_local_planarity(depth_image, y, x)
            
            planarity_map[y, x] = analysis['pca_planarity']
            rms_error_map[y, x] = analysis['rms_error']
        
        # Normalize planarity map
        valid_mask = instance_mask > 0
        if np.sum(valid_mask) > 0:
            valid_planarity = planarity_map[valid_mask]
            valid_planarity = valid_planarity[np.isfinite(valid_planarity)]
            
            if len(valid_planarity) > 0:
                # Normalize to 0-1 range
                p_min, p_max = np.percentile(valid_planarity, [5, 95])
                if p_max > p_min:
                    planarity_map = np.clip(
                        (planarity_map - p_min) / (p_max - p_min),
                        0, 1
                    )
        
        logger.info("Planarity analysis complete")
        
        return planarity_map, rms_error_map
    
    def identify_flat_regions(self, planarity_map: np.ndarray,
                            rms_error_map: np.ndarray,
                            instance_mask: np.ndarray) -> np.ndarray:
        """
        Identify regions suitable for label placement based on planarity.
        
        Args:
            planarity_map: Map of planarity scores (0 = flat, 1 = curved)
            rms_error_map: Map of RMS fitting errors
            instance_mask: Binary mask for the fruit instance
            
        Returns:
            Binary mask of suitable regions
        """
        # Create combined criteria
        flat_regions = (
            (planarity_map < 0.3) &  # Low curvature
            (rms_error_map < self.planarity_threshold) &  # Good plane fit
            (instance_mask > 0) &  # Within fruit
            (np.isfinite(planarity_map)) &  # Valid computation
            (np.isfinite(rms_error_map))
        )
        
        # Apply morphological closing to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        flat_regions = cv2.morphologyEx(
            flat_regions.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        ).astype(bool)
        
        # Remove small regions
        from skimage import morphology
        flat_regions = morphology.remove_small_objects(flat_regions, min_size=100)
        
        return flat_regions.astype(np.uint8) * 255
    
    def compute_curvature_heatmap(self, planarity_map: np.ndarray,
                                instance_mask: np.ndarray) -> np.ndarray:
        """
        Create a curvature heatmap for visualization.
        
        Args:
            planarity_map: Map of planarity scores
            instance_mask: Binary mask for the fruit instance
            
        Returns:
            RGB heatmap image
        """
        # Create color map
        heatmap = np.zeros((*planarity_map.shape, 3), dtype=np.uint8)
        
        # Apply mask
        masked_planarity = planarity_map.copy()
        masked_planarity[instance_mask == 0] = 0
        
        # Convert to colormap (blue = flat, red = curved)
        valid_mask = (instance_mask > 0) & np.isfinite(masked_planarity)
        
        if np.sum(valid_mask) > 0:
            # Normalize for visualization
            vis_planarity = masked_planarity.copy()
            vis_planarity = np.clip(vis_planarity * 255, 0, 255).astype(np.uint8)
            
            # Apply jet colormap
            colored = cv2.applyColorMap(vis_planarity, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            # Mask invalid regions
            heatmap[~valid_mask] = 0
        
        return heatmap
    
    def get_planarity_statistics(self, planarity_map: np.ndarray,
                               instance_mask: np.ndarray) -> Dict:
        """
        Compute statistics about planarity for a fruit instance.
        
        Args:
            planarity_map: Map of planarity scores
            instance_mask: Binary mask for the fruit instance
            
        Returns:
            Dictionary with planarity statistics
        """
        valid_mask = (instance_mask > 0) & np.isfinite(planarity_map)
        
        if np.sum(valid_mask) == 0:
            return {
                'mean_planarity': 1.0,
                'std_planarity': 0.0,
                'min_planarity': 1.0,
                'max_planarity': 1.0,
                'flat_percentage': 0.0
            }
        
        valid_values = planarity_map[valid_mask]
        
        # Count flat regions (planarity < 0.3)
        flat_count = np.sum(valid_values < 0.3)
        flat_percentage = flat_count / len(valid_values) * 100
        
        return {
            'mean_planarity': np.mean(valid_values),
            'std_planarity': np.std(valid_values),
            'min_planarity': np.min(valid_values),
            'max_planarity': np.max(valid_values),
            'flat_percentage': flat_percentage
        }