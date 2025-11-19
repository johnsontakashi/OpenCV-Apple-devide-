"""
Candidate Label Location Selection Module

Identifies optimal locations for placing circular label stickers on fruit surfaces
based on planarity, color uniformity, depth validity, and edge distance.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class CandidateSelector:
    """Selects optimal label placement candidates based on multiple criteria."""
    
    def __init__(self, config: dict):
        """
        Initialize candidate selector with configuration parameters.
        
        Args:
            config: Dictionary containing selection parameters
        """
        self.config = config
        self.label_diameter_pixels = config.get('label_diameter_pixels', 40)
        self.min_valid_depth_fraction = config.get('min_valid_depth_fraction', 0.95)
        self.max_rms_threshold = config.get('max_rms_threshold', 0.01)
        self.max_candidates_per_fruit = config.get('max_candidates_per_fruit', 5)
        self.min_edge_distance = config.get('min_edge_distance', 20)
        
        # Scoring weights
        self.weight_planarity = config.get('weight_planarity', 0.5)
        self.weight_depth_validity = config.get('weight_depth_validity', 0.25)
        self.weight_color_uniformity = config.get('weight_color_uniformity', 0.15)
        self.weight_edge_distance = config.get('weight_edge_distance', 0.10)
        
    def compute_valid_depth_fraction(self, depth_image: np.ndarray,
                                   center_y: int, center_x: int,
                                   radius: int) -> float:
        """
        Compute fraction of valid depth pixels in a circular region.
        
        Args:
            depth_image: Normalized depth image in meters
            center_y: Y coordinate of center
            center_x: X coordinate of center
            radius: Radius of circular region
            
        Returns:
            Fraction of valid depth pixels (0.0 to 1.0)
        """
        height, width = depth_image.shape
        
        # Create circular mask
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Count valid depth pixels in the circle
        circle_pixels = np.sum(mask)
        if circle_pixels == 0:
            return 0.0
        
        valid_depth_pixels = np.sum(mask & (depth_image > 0))
        
        return valid_depth_pixels / circle_pixels
    
    def compute_color_uniformity(self, color_image: np.ndarray,
                               center_y: int, center_x: int,
                               radius: int) -> float:
        """
        Compute color uniformity (inverse of standard deviation) in a circular region.
        
        Args:
            color_image: RGB color image
            center_y: Y coordinate of center
            center_x: X coordinate of center
            radius: Radius of circular region
            
        Returns:
            Color uniformity score (higher = more uniform)
        """
        height, width = color_image.shape[:2]
        
        # Create circular mask
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        if not np.any(mask):
            return 0.0
        
        # Extract colors in the circular region
        masked_colors = color_image[mask]  # Shape: (N, 3)
        
        if len(masked_colors) == 0:
            return 0.0
        
        # Compute standard deviation across all color channels
        color_std = np.mean(np.std(masked_colors, axis=0))
        
        # Convert to uniformity score (lower std = higher uniformity)
        # Normalize assuming std can range from 0 to 255
        uniformity = 1.0 - min(color_std / 255.0, 1.0)
        
        return uniformity
    
    def compute_edge_distance(self, instance_mask: np.ndarray,
                            center_y: int, center_x: int) -> float:
        """
        Compute distance to nearest edge of the fruit instance.
        
        Args:
            instance_mask: Binary mask of the fruit instance
            center_y: Y coordinate of center
            center_x: X coordinate of center
            
        Returns:
            Distance to nearest edge in pixels
        """
        # Compute distance transform from edges
        distance_map = cv2.distanceTransform(instance_mask, cv2.DIST_L2, 5)
        
        if center_y < 0 or center_y >= instance_mask.shape[0] or \
           center_x < 0 or center_x >= instance_mask.shape[1]:
            return 0.0
        
        return distance_map[center_y, center_x]
    
    def check_depth_discontinuities(self, depth_image: np.ndarray,
                                  center_y: int, center_x: int,
                                  radius: int) -> float:
        """
        Check for depth discontinuities in a circular region.
        
        Args:
            depth_image: Normalized depth image in meters
            center_y: Y coordinate of center
            center_x: X coordinate of center
            radius: Radius of circular region
            
        Returns:
            Discontinuity score (0 = smooth, 1 = discontinuous)
        """
        height, width = depth_image.shape
        
        # Create circular mask
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        if not np.any(mask):
            return 1.0
        
        # Extract depth values in the region
        region_depths = depth_image[mask]
        valid_depths = region_depths[region_depths > 0]
        
        if len(valid_depths) < 2:
            return 1.0
        
        # Compute depth gradient magnitude
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average gradient in the circular region
        avg_gradient = np.mean(gradient_magnitude[mask])
        
        # Normalize gradient (assuming max reasonable gradient is 0.1 m/pixel)
        discontinuity_score = min(avg_gradient / 0.1, 1.0)
        
        return discontinuity_score
    
    def compute_combined_score(self, depth_image: np.ndarray,
                             color_image: np.ndarray,
                             instance_mask: np.ndarray,
                             planarity_map: np.ndarray,
                             rms_error_map: np.ndarray,
                             center_y: int, center_x: int) -> Dict:
        """
        Compute combined score for a candidate location.
        
        Args:
            depth_image: Normalized depth image in meters
            color_image: RGB color image
            instance_mask: Binary mask of the fruit instance
            planarity_map: Map of planarity scores
            rms_error_map: Map of RMS fitting errors
            center_y: Y coordinate of candidate center
            center_x: X coordinate of candidate center
            
        Returns:
            Dictionary with detailed scoring information
        """
        radius = self.label_diameter_pixels // 2
        
        # Check if center is within the fruit
        if center_y < 0 or center_y >= instance_mask.shape[0] or \
           center_x < 0 or center_x >= instance_mask.shape[1] or \
           instance_mask[center_y, center_x] == 0:
            return {
                'total_score': 0.0,
                'planarity_score': 0.0,
                'depth_validity_score': 0.0,
                'color_uniformity_score': 0.0,
                'edge_distance_score': 0.0,
                'valid': False,
                'rejection_reason': 'outside_fruit'
            }
        
        # Planarity score (lower is better, so invert)
        planarity_value = planarity_map[center_y, center_x]
        rms_error = rms_error_map[center_y, center_x]
        
        if not np.isfinite(planarity_value) or not np.isfinite(rms_error):
            planarity_score = 0.0
        else:
            planarity_score = max(0.0, 1.0 - planarity_value)
        
        # Check RMS threshold
        if rms_error > self.max_rms_threshold:
            return {
                'total_score': 0.0,
                'planarity_score': planarity_score,
                'depth_validity_score': 0.0,
                'color_uniformity_score': 0.0,
                'edge_distance_score': 0.0,
                'valid': False,
                'rejection_reason': 'high_rms_error'
            }
        
        # Depth validity score
        depth_fraction = self.compute_valid_depth_fraction(depth_image, center_y, center_x, radius)
        if depth_fraction < self.min_valid_depth_fraction:
            return {
                'total_score': 0.0,
                'planarity_score': planarity_score,
                'depth_validity_score': depth_fraction,
                'color_uniformity_score': 0.0,
                'edge_distance_score': 0.0,
                'valid': False,
                'rejection_reason': 'insufficient_depth_coverage'
            }
        
        # Color uniformity score
        color_uniformity = self.compute_color_uniformity(color_image, center_y, center_x, radius)
        
        # Edge distance score
        edge_distance = self.compute_edge_distance(instance_mask, center_y, center_x)
        if edge_distance < self.min_edge_distance:
            return {
                'total_score': 0.0,
                'planarity_score': planarity_score,
                'depth_validity_score': depth_fraction,
                'color_uniformity_score': color_uniformity,
                'edge_distance_score': 0.0,
                'valid': False,
                'rejection_reason': 'too_close_to_edge'
            }
        
        # Normalize edge distance score
        max_edge_distance = 100.0  # Assume max reasonable distance
        edge_distance_score = min(edge_distance / max_edge_distance, 1.0)
        
        # Check depth discontinuities
        discontinuity_score = self.check_depth_discontinuities(depth_image, center_y, center_x, radius)
        if discontinuity_score > 0.5:  # Too many discontinuities
            return {
                'total_score': 0.0,
                'planarity_score': planarity_score,
                'depth_validity_score': depth_fraction,
                'color_uniformity_score': color_uniformity,
                'edge_distance_score': edge_distance_score,
                'valid': False,
                'rejection_reason': 'depth_discontinuities'
            }
        
        # Compute weighted total score
        total_score = (
            self.weight_planarity * planarity_score +
            self.weight_depth_validity * depth_fraction +
            self.weight_color_uniformity * color_uniformity +
            self.weight_edge_distance * edge_distance_score
        )
        
        return {
            'total_score': total_score,
            'planarity_score': planarity_score,
            'depth_validity_score': depth_fraction,
            'color_uniformity_score': color_uniformity,
            'edge_distance_score': edge_distance_score,
            'edge_distance_pixels': edge_distance,
            'discontinuity_score': discontinuity_score,
            'rms_error': rms_error,
            'valid': True,
            'rejection_reason': None
        }
    
    def find_candidates(self, depth_image: np.ndarray,
                       color_image: np.ndarray,
                       instance_mask: np.ndarray,
                       planarity_map: np.ndarray,
                       rms_error_map: np.ndarray,
                       sampling_step: int = 5) -> List[Dict]:
        """
        Find candidate label locations for a fruit instance.
        
        Args:
            depth_image: Normalized depth image in meters
            color_image: RGB color image
            instance_mask: Binary mask of the fruit instance
            planarity_map: Map of planarity scores
            rms_error_map: Map of RMS fitting errors
            sampling_step: Step size for candidate sampling
            
        Returns:
            List of candidate dictionaries sorted by score
        """
        logger.info("Finding label placement candidates")
        
        candidates = []
        height, width = instance_mask.shape
        
        # Find bounding box of the instance
        mask_indices = np.where(instance_mask > 0)
        if len(mask_indices[0]) == 0:
            return candidates
        
        min_y, max_y = np.min(mask_indices[0]), np.max(mask_indices[0])
        min_x, max_x = np.min(mask_indices[1]), np.max(mask_indices[1])
        
        # Sample candidate locations
        radius = self.label_diameter_pixels // 2
        
        for y in range(min_y + radius, max_y - radius + 1, sampling_step):
            for x in range(min_x + radius, max_x - radius + 1, sampling_step):
                # Skip if not in instance
                if instance_mask[y, x] == 0:
                    continue
                
                # Compute score
                score_info = self.compute_combined_score(
                    depth_image, color_image, instance_mask,
                    planarity_map, rms_error_map, y, x
                )
                
                if score_info['valid'] and score_info['total_score'] > 0.1:
                    candidate = {
                        'center_y': y,
                        'center_x': x,
                        'pixel_location': [y, x],
                        'depth': depth_image[y, x] if y < depth_image.shape[0] and x < depth_image.shape[1] else 0.0,
                        **score_info
                    }
                    candidates.append(candidate)
        
        # Sort candidates by total score (descending)
        candidates.sort(key=lambda c: c['total_score'], reverse=True)
        
        # Apply non-maximum suppression to avoid overlapping candidates
        final_candidates = self.non_maximum_suppression(candidates)
        
        # Limit number of candidates
        final_candidates = final_candidates[:self.max_candidates_per_fruit]
        
        logger.info(f"Found {len(final_candidates)} valid candidates")
        
        return final_candidates
    
    def non_maximum_suppression(self, candidates: List[Dict],
                              min_distance: Optional[int] = None) -> List[Dict]:
        """
        Apply non-maximum suppression to remove overlapping candidates.
        
        Args:
            candidates: List of candidate dictionaries
            min_distance: Minimum distance between candidates (default: label diameter)
            
        Returns:
            Filtered list of candidates
        """
        if not candidates:
            return candidates
        
        if min_distance is None:
            min_distance = self.label_diameter_pixels
        
        # Sort by score (already done in find_candidates)
        selected = []
        
        for candidate in candidates:
            y1, x1 = candidate['center_y'], candidate['center_x']
            
            # Check if too close to any selected candidate
            too_close = False
            for selected_candidate in selected:
                y2, x2 = selected_candidate['center_y'], selected_candidate['center_x']
                distance = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(candidate)
        
        return selected
    
    def refine_candidate_location(self, depth_image: np.ndarray,
                                color_image: np.ndarray,
                                instance_mask: np.ndarray,
                                planarity_map: np.ndarray,
                                rms_error_map: np.ndarray,
                                initial_y: int, initial_x: int,
                                search_radius: int = 3) -> Dict:
        """
        Refine candidate location by local search for optimal position.
        
        Args:
            depth_image: Normalized depth image
            color_image: RGB color image
            instance_mask: Binary mask
            planarity_map: Planarity score map
            rms_error_map: RMS error map
            initial_y: Initial Y coordinate
            initial_x: Initial X coordinate
            search_radius: Radius for local search
            
        Returns:
            Refined candidate dictionary
        """
        best_score = 0.0
        best_candidate = None
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                y = initial_y + dy
                x = initial_x + dx
                
                score_info = self.compute_combined_score(
                    depth_image, color_image, instance_mask,
                    planarity_map, rms_error_map, y, x
                )
                
                if score_info['valid'] and score_info['total_score'] > best_score:
                    best_score = score_info['total_score']
                    best_candidate = {
                        'center_y': y,
                        'center_x': x,
                        'pixel_location': [y, x],
                        'depth': depth_image[y, x] if y < depth_image.shape[0] and x < depth_image.shape[1] else 0.0,
                        **score_info
                    }
        
        return best_candidate if best_candidate else {
            'center_y': initial_y,
            'center_x': initial_x,
            'pixel_location': [initial_y, initial_x],
            'depth': 0.0,
            'total_score': 0.0,
            'valid': False,
            'rejection_reason': 'refinement_failed'
        }
    
    def add_3d_coordinates(self, candidates: List[Dict], 
                         camera_config: Dict) -> List[Dict]:
        """
        Add 3D world coordinates to candidates.
        
        Args:
            candidates: List of candidate dictionaries
            camera_config: Camera intrinsic parameters
            
        Returns:
            Candidates with added 3D coordinates
        """
        fx = camera_config.get('camera_fx', 525.0)
        fy = camera_config.get('camera_fy', 525.0)
        cx = camera_config.get('camera_cx', 320.0)
        cy = camera_config.get('camera_cy', 240.0)
        
        for candidate in candidates:
            y, x = candidate['center_y'], candidate['center_x']
            depth = candidate['depth']
            
            if depth > 0:
                # Convert to 3D coordinates
                world_x = (x - cx) * depth / fx
                world_y = (y - cy) * depth / fy
                world_z = depth
                
                candidate['world_coordinates'] = [world_x, world_y, world_z]
            else:
                candidate['world_coordinates'] = [0.0, 0.0, 0.0]
        
        return candidates