"""
Constraint Filtering Module

Filters candidate label locations based on various constraints:
- Stem region detection and avoidance
- Tray boundary detection and filtering
- Fruit-to-fruit contact area identification
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import RANSACRegressor, LinearRegression
import logging

logger = logging.getLogger(__name__)


class ConstraintFilter:
    """Applies various constraints to filter candidate label locations."""
    
    def __init__(self, config: dict):
        """
        Initialize constraint filter with configuration parameters.
        
        Args:
            config: Dictionary containing constraint parameters
        """
        self.config = config
        
        # Stem detection parameters
        self.stem_color_ranges = {
            'brown': {'lower': np.array([10, 50, 20]), 'upper': np.array([20, 255, 200])},
            'green': {'lower': np.array([40, 40, 40]), 'upper': np.array([80, 255, 255])}
        }
        self.stem_depth_spike_threshold = config.get('stem_depth_spike_threshold', 0.02)
        self.stem_region_radius = config.get('stem_region_radius', 15)
        
        # Tray boundary parameters
        self.tray_boundary_margin = config.get('tray_boundary_margin', 20)
        self.tray_ransac_threshold = config.get('tray_ransac_threshold', 0.01)
        self.tray_min_samples = config.get('tray_min_samples', 100)
        
        # Contact area parameters
        self.contact_gradient_threshold = config.get('contact_gradient_threshold', 0.05)
        self.contact_dilation_kernel = config.get('contact_dilation_kernel', 5)
        
    def detect_stem_regions(self, color_image: np.ndarray, 
                          depth_image: np.ndarray,
                          instance_mask: np.ndarray) -> np.ndarray:
        """
        Detect stem regions using color signature and depth spike detection.
        
        Args:
            color_image: RGB color image
            depth_image: Normalized depth image in meters
            instance_mask: Binary mask for the fruit instance
            
        Returns:
            Binary mask of detected stem regions
        """
        logger.debug("Detecting stem regions")
        
        height, width = instance_mask.shape
        stem_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        
        # Detect brown/green stem colors
        color_mask = np.zeros((height, width), dtype=np.uint8)
        
        for color_name, color_range in self.stem_color_ranges.items():
            color_region = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            color_mask = cv2.bitwise_or(color_mask, color_region)
        
        # Detect depth spikes (stems often protrude)
        depth_gradient = self._compute_depth_gradient(depth_image)
        depth_spikes = depth_gradient > self.stem_depth_spike_threshold
        
        # Combine color and depth cues
        potential_stem = cv2.bitwise_and(color_mask, depth_spikes.astype(np.uint8) * 255)
        
        # Only consider regions within the fruit
        potential_stem = cv2.bitwise_and(potential_stem, instance_mask)
        
        # Find connected components and filter by size
        contours, _ = cv2.findContours(potential_stem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Reasonable stem size
                cv2.fillPoly(stem_mask, [contour], 255)
        
        # Expand stem regions
        if np.sum(stem_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.stem_region_radius * 2, self.stem_region_radius * 2))
            stem_mask = cv2.dilate(stem_mask, kernel, iterations=1)
        
        logger.debug(f"Detected stem regions covering {np.sum(stem_mask)} pixels")
        
        return stem_mask
    
    def _compute_depth_gradient(self, depth_image: np.ndarray) -> np.ndarray:
        """Compute depth gradient magnitude."""
        # Compute gradients
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_magnitude
    
    def detect_tray_boundary(self, depth_image: np.ndarray, 
                           segmentation_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect tray boundary using RANSAC plane fitting.
        
        Args:
            depth_image: Normalized depth image in meters
            segmentation_mask: Overall segmentation mask (all fruits)
            
        Returns:
            Tray boundary mask or None if detection fails
        """
        logger.debug("Detecting tray boundary")
        
        height, width = depth_image.shape
        
        # Find potential tray points (background with valid depth)
        background_mask = (segmentation_mask == 0) & (depth_image > 0)
        
        if np.sum(background_mask) < self.tray_min_samples:
            logger.warning("Insufficient background points for tray detection")
            return None
        
        # Extract 3D points from background
        y_coords, x_coords = np.where(background_mask)
        depths = depth_image[background_mask]
        
        # Convert to 3D coordinates
        fx = self.config.get('camera_fx', 525.0)
        fy = self.config.get('camera_fy', 525.0)
        cx = self.config.get('camera_cx', 320.0)
        cy = self.config.get('camera_cy', 240.0)
        
        points_3d = np.column_stack([
            (x_coords - cx) * depths / fx,
            (y_coords - cy) * depths / fy,
            depths
        ])
        
        # Use only X and Z coordinates for horizontal plane fitting
        points_xz = points_3d[:, [0, 2]]
        y_values = points_3d[:, 1]  # Y is the vertical coordinate
        
        try:
            # RANSAC to fit a line in XZ plane (horizontal tray)
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=self.tray_min_samples // 10,
                residual_threshold=self.tray_ransac_threshold,
                max_trials=100
            )
            
            ransac.fit(points_xz, y_values)
            
            # Get inlier mask
            inlier_mask = ransac.inlier_mask_
            
            if np.sum(inlier_mask) < self.tray_min_samples // 2:
                logger.warning("RANSAC found too few inliers for tray plane")
                return None
            
            # Create tray boundary mask
            tray_mask = np.zeros((height, width), dtype=np.uint8)
            tray_points_y = y_coords[inlier_mask]
            tray_points_x = x_coords[inlier_mask]
            tray_mask[tray_points_y, tray_points_x] = 255
            
            # Expand tray region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tray_mask = cv2.dilate(tray_mask, kernel, iterations=2)
            
            logger.debug(f"Detected tray boundary with {np.sum(inlier_mask)} inlier points")
            
            return tray_mask
        
        except Exception as e:
            logger.warning(f"Tray boundary detection failed: {e}")
            return None
    
    def detect_contact_areas(self, depth_image: np.ndarray,
                           instance_labels: np.ndarray) -> np.ndarray:
        """
        Detect fruit-to-fruit contact areas using depth gradients.
        
        Args:
            depth_image: Normalized depth image in meters
            instance_labels: Instance label map
            
        Returns:
            Binary mask of contact areas
        """
        logger.debug("Detecting fruit-to-fruit contact areas")
        
        height, width = depth_image.shape
        contact_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Find boundaries between different instances
        gradient = self._compute_depth_gradient(depth_image)
        
        # Strong gradients indicate potential contact areas
        strong_gradients = gradient > self.contact_gradient_threshold
        
        # Find pixels near instance boundaries
        for instance_id in range(1, instance_labels.max() + 1):
            instance_mask = (instance_labels == instance_id)
            
            # Erode to find boundary pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(instance_mask.astype(np.uint8), kernel, iterations=1)
            boundary = instance_mask & (~eroded.astype(bool))
            
            # Contact areas are boundaries with strong gradients
            contact_region = boundary & strong_gradients
            contact_mask[contact_region] = 255
        
        # Dilate contact areas to create exclusion zones
        if np.sum(contact_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.contact_dilation_kernel, self.contact_dilation_kernel))
            contact_mask = cv2.dilate(contact_mask, kernel, iterations=2)
        
        logger.debug(f"Detected contact areas covering {np.sum(contact_mask)} pixels")
        
        return contact_mask
    
    def apply_tray_boundary_constraint(self, candidates: List[Dict],
                                     tray_mask: Optional[np.ndarray]) -> List[Dict]:
        """
        Filter candidates that are too close to tray boundary.
        
        Args:
            candidates: List of candidate dictionaries
            tray_mask: Tray boundary mask (None to skip this constraint)
            
        Returns:
            Filtered list of candidates
        """
        if tray_mask is None:
            logger.debug("Skipping tray boundary constraint (no tray detected)")
            return candidates
        
        filtered_candidates = []
        
        for candidate in candidates:
            y, x = candidate['center_y'], candidate['center_x']
            
            # Check distance to tray boundary
            if y < tray_mask.shape[0] and x < tray_mask.shape[1]:
                # Compute distance to nearest tray pixel
                if np.sum(tray_mask) > 0:
                    distance_map = cv2.distanceTransform(
                        (tray_mask == 0).astype(np.uint8), cv2.DIST_L2, 5
                    )
                    distance_to_tray = distance_map[y, x]
                    
                    if distance_to_tray >= self.tray_boundary_margin:
                        filtered_candidates.append(candidate)
                    else:
                        candidate['valid'] = False
                        candidate['rejection_reason'] = 'too_close_to_tray'
                        logger.debug(f"Candidate at ({y}, {x}) rejected: too close to tray")
                else:
                    filtered_candidates.append(candidate)
            else:
                filtered_candidates.append(candidate)
        
        logger.debug(f"Tray boundary filtering: {len(candidates)} -> {len(filtered_candidates)}")
        
        return filtered_candidates
    
    def apply_stem_constraint(self, candidates: List[Dict],
                            stem_mask: np.ndarray) -> List[Dict]:
        """
        Filter candidates that overlap with stem regions.
        
        Args:
            candidates: List of candidate dictionaries
            stem_mask: Binary mask of stem regions
            
        Returns:
            Filtered list of candidates
        """
        filtered_candidates = []
        
        for candidate in candidates:
            y, x = candidate['center_y'], candidate['center_x']
            
            if y < stem_mask.shape[0] and x < stem_mask.shape[1]:
                if stem_mask[y, x] == 0:  # Not in stem region
                    filtered_candidates.append(candidate)
                else:
                    candidate['valid'] = False
                    candidate['rejection_reason'] = 'in_stem_region'
                    logger.debug(f"Candidate at ({y}, {x}) rejected: in stem region")
            else:
                filtered_candidates.append(candidate)
        
        logger.debug(f"Stem filtering: {len(candidates)} -> {len(filtered_candidates)}")
        
        return filtered_candidates
    
    def apply_contact_constraint(self, candidates: List[Dict],
                               contact_mask: np.ndarray) -> List[Dict]:
        """
        Filter candidates that overlap with fruit contact areas.
        
        Args:
            candidates: List of candidate dictionaries
            contact_mask: Binary mask of contact areas
            
        Returns:
            Filtered list of candidates
        """
        filtered_candidates = []
        
        for candidate in candidates:
            y, x = candidate['center_y'], candidate['center_x']
            
            if y < contact_mask.shape[0] and x < contact_mask.shape[1]:
                if contact_mask[y, x] == 0:  # Not in contact area
                    filtered_candidates.append(candidate)
                else:
                    candidate['valid'] = False
                    candidate['rejection_reason'] = 'in_contact_area'
                    logger.debug(f"Candidate at ({y}, {x}) rejected: in contact area")
            else:
                filtered_candidates.append(candidate)
        
        logger.debug(f"Contact filtering: {len(candidates)} -> {len(filtered_candidates)}")
        
        return filtered_candidates
    
    def apply_all_constraints(self, candidates_per_fruit: List[List[Dict]],
                            color_image: np.ndarray,
                            depth_image: np.ndarray,
                            instance_labels: np.ndarray,
                            instance_masks: List[np.ndarray]) -> Tuple[List[List[Dict]], Dict]:
        """
        Apply all constraints to filter candidates for all fruits.
        
        Args:
            candidates_per_fruit: List of candidate lists (one per fruit)
            color_image: RGB color image
            depth_image: Normalized depth image
            instance_labels: Instance label map
            instance_masks: List of binary masks for each fruit
            
        Returns:
            Tuple of (filtered_candidates_per_fruit, constraint_masks)
        """
        logger.info("Applying constraint filters")
        
        # Generate constraint masks
        overall_mask = (instance_labels > 0).astype(np.uint8) * 255
        
        # Detect tray boundary
        tray_mask = self.detect_tray_boundary(depth_image, overall_mask)
        
        # Detect contact areas between fruits
        contact_mask = self.detect_contact_areas(depth_image, instance_labels)
        
        # Process each fruit individually
        filtered_candidates_per_fruit = []
        stem_masks = []
        
        for i, (candidates, instance_mask) in enumerate(zip(candidates_per_fruit, instance_masks)):
            logger.debug(f"Processing constraints for fruit {i+1}")
            
            # Detect stem regions for this fruit
            stem_mask = self.detect_stem_regions(color_image, depth_image, instance_mask)
            stem_masks.append(stem_mask)
            
            # Apply all constraints
            filtered_candidates = candidates.copy()
            filtered_candidates = self.apply_stem_constraint(filtered_candidates, stem_mask)
            filtered_candidates = self.apply_tray_boundary_constraint(filtered_candidates, tray_mask)
            filtered_candidates = self.apply_contact_constraint(filtered_candidates, contact_mask)
            
            filtered_candidates_per_fruit.append(filtered_candidates)
            
            logger.debug(f"Fruit {i+1}: {len(candidates)} -> {len(filtered_candidates)} candidates")
        
        # Prepare constraint masks for visualization
        constraint_masks = {
            'tray_boundary': tray_mask,
            'contact_areas': contact_mask,
            'stem_regions': stem_masks
        }
        
        total_original = sum(len(candidates) for candidates in candidates_per_fruit)
        total_filtered = sum(len(candidates) for candidates in filtered_candidates_per_fruit)
        
        logger.info(f"Constraint filtering complete: {total_original} -> {total_filtered} candidates")
        
        return filtered_candidates_per_fruit, constraint_masks
    
    def get_rejection_statistics(self, candidates_per_fruit: List[List[Dict]]) -> Dict:
        """
        Compute statistics about candidate rejections.
        
        Args:
            candidates_per_fruit: List of candidate lists (including rejected ones)
            
        Returns:
            Dictionary with rejection statistics
        """
        rejection_counts = {
            'in_stem_region': 0,
            'too_close_to_tray': 0,
            'in_contact_area': 0,
            'other': 0,
            'valid': 0
        }
        
        for candidates in candidates_per_fruit:
            for candidate in candidates:
                if candidate.get('valid', True):
                    rejection_counts['valid'] += 1
                else:
                    reason = candidate.get('rejection_reason', 'other')
                    if reason in rejection_counts:
                        rejection_counts[reason] += 1
                    else:
                        rejection_counts['other'] += 1
        
        total = sum(rejection_counts.values())
        
        # Convert to percentages
        if total > 0:
            rejection_percentages = {k: v/total*100 for k, v in rejection_counts.items()}
        else:
            rejection_percentages = {k: 0.0 for k in rejection_counts}
        
        return {
            'counts': rejection_counts,
            'percentages': rejection_percentages,
            'total_candidates': total
        }