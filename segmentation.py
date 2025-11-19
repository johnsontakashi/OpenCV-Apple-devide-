"""
Fruit Segmentation Module

Provides classical computer vision and optional deep learning approaches 
for segmenting apples/tomatoes from RGB-D images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from skimage import measure, morphology
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class FruitSegmenter:
    """Handles fruit segmentation using multiple approaches."""
    
    def __init__(self, config: dict):
        """
        Initialize segmenter with configuration parameters.
        
        Args:
            config: Dictionary containing segmentation parameters
        """
        self.config = config
        self.method = config.get('segmentation_method', 'classical')
        
        # Classical CV parameters
        self.hsv_lower = np.array(config.get('hsv_lower', [0, 50, 50]))
        self.hsv_upper = np.array(config.get('hsv_upper', [25, 255, 255]))
        self.depth_fg_threshold = config.get('depth_fg_threshold', 0.3)  # meters
        self.depth_bg_threshold = config.get('depth_bg_threshold', 1.5)  # meters
        self.min_area = config.get('min_fruit_area', 1000)
        
        # Morphological parameters
        self.morph_kernel_size = config.get('morph_kernel_size', 5)
        self.opening_iterations = config.get('opening_iterations', 2)
        self.closing_iterations = config.get('closing_iterations', 3)
        
    def segment_classical(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        Classical computer vision segmentation using HSV and depth thresholding.
        
        Args:
            color_image: RGB color image
            depth_image: Normalized depth image in meters
            
        Returns:
            Binary segmentation mask
        """
        logger.info("Running classical segmentation")
        
        # HSV color segmentation
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Also include red fruits (wrapped around hue)
        hsv_lower_red2 = np.array([160, 50, 50])
        hsv_upper_red2 = np.array([180, 255, 255])
        color_mask_red2 = cv2.inRange(hsv, hsv_lower_red2, hsv_upper_red2)
        color_mask = cv2.bitwise_or(color_mask, color_mask_red2)
        
        # Depth-based foreground extraction
        valid_depth = depth_image > 0
        depth_mask = (depth_image > self.depth_fg_threshold) & (depth_image < self.depth_bg_threshold)
        depth_mask = depth_mask & valid_depth
        
        # Combine color and depth masks
        combined_mask = cv2.bitwise_and(color_mask, depth_mask.astype(np.uint8) * 255)
        
        # Morphological operations
        combined_mask = self._apply_morphology(combined_mask)
        
        return combined_mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morph_kernel_size, self.morph_kernel_size))
        
        # Opening (erosion followed by dilation) to remove noise
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                                iterations=self.opening_iterations)
        
        # Closing (dilation followed by erosion) to fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, 
                                iterations=self.closing_iterations)
        
        return closed
    
    def separate_instances(self, mask: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Separate touching fruit instances using distance transform and watershed.
        
        Args:
            mask: Binary segmentation mask
            depth_image: Normalized depth image
            
        Returns:
            Tuple of (instance_labels, num_instances)
        """
        logger.info("Separating fruit instances")
        
        # Remove small objects
        cleaned_mask = morphology.remove_small_objects(
            mask.astype(bool), min_size=self.min_area
        ).astype(np.uint8) * 255
        
        if np.sum(cleaned_mask) == 0:
            return np.zeros_like(mask), 0
        
        # Distance transform
        dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        
        # Find local maxima as seeds using a simple approach
        # Dilate the distance transform and find where original equals dilated (local maxima)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # min_distance=20 -> kernel=21
        dilated = cv2.dilate(dist_transform, kernel)
        local_maxima = (dist_transform == dilated) & (dist_transform > 10)  # threshold_abs=10
        
        seeds = measure.label(local_maxima)
        
        # If no seeds found, treat as single object
        if seeds.max() == 0:
            return cleaned_mask, 1
        
        # Watershed segmentation
        # Use negative distance transform as elevation map
        elevation_map = -dist_transform
        
        # Apply watershed
        from skimage.segmentation import watershed
        labels = watershed(elevation_map, seeds, mask=cleaned_mask.astype(bool))
        
        # Remove background label (0) from count
        num_instances = labels.max()
        
        logger.info(f"Found {num_instances} fruit instances")
        
        return labels.astype(np.int32), num_instances
    
    def refine_with_depth_gradients(self, labels: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        Refine instance boundaries using depth discontinuities.
        
        Args:
            labels: Instance label map
            depth_image: Normalized depth image
            
        Returns:
            Refined label map
        """
        if labels.max() <= 1:
            return labels
        
        # Compute depth gradients
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find strong depth edges
        edge_threshold = np.percentile(gradient_magnitude[gradient_magnitude > 0], 85)
        strong_edges = gradient_magnitude > edge_threshold
        
        # Erode instance boundaries at strong edges
        refined_labels = labels.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        for instance_id in range(1, labels.max() + 1):
            instance_mask = (labels == instance_id)
            
            # Find intersection with strong edges
            edge_intersection = instance_mask & strong_edges
            
            if np.sum(edge_intersection) > 0:
                # Erode the instance slightly at edge locations
                eroded = cv2.erode(instance_mask.astype(np.uint8), kernel, iterations=1)
                refined_labels[instance_mask & ~eroded.astype(bool)] = 0
        
        return refined_labels
    
    def segment(self, color_image: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Complete segmentation pipeline.
        
        Args:
            color_image: RGB color image
            depth_image: Normalized depth image in meters
            
        Returns:
            Tuple of (instance_labels, num_instances)
        """
        # Primary segmentation
        if self.method == 'classical':
            mask = self.segment_classical(color_image, depth_image)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
        
        # Instance separation
        labels, num_instances = self.separate_instances(mask, depth_image)
        
        # Refine with depth information
        if num_instances > 1:
            labels = self.refine_with_depth_gradients(labels, depth_image)
        
        return labels, num_instances
    
    def get_instance_masks(self, labels: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual binary masks for each instance.
        
        Args:
            labels: Instance label map
            
        Returns:
            List of binary masks for each instance
        """
        masks = []
        for instance_id in range(1, labels.max() + 1):
            mask = (labels == instance_id).astype(np.uint8) * 255
            if np.sum(mask) > 0:
                masks.append(mask)
        
        return masks
    
    def get_instance_properties(self, labels: np.ndarray, 
                              color_image: np.ndarray, 
                              depth_image: np.ndarray) -> List[Dict]:
        """
        Extract properties for each segmented instance.
        
        Args:
            labels: Instance label map
            color_image: RGB color image
            depth_image: Normalized depth image
            
        Returns:
            List of dictionaries containing instance properties
        """
        properties = []
        
        for instance_id in range(1, labels.max() + 1):
            mask = labels == instance_id
            
            if np.sum(mask) == 0:
                continue
            
            # Geometric properties
            props = measure.regionprops(mask.astype(int))[0]
            
            # Color statistics
            masked_color = color_image[mask]
            mean_color = np.mean(masked_color, axis=0)
            std_color = np.std(masked_color, axis=0)
            
            # Depth statistics
            masked_depth = depth_image[mask & (depth_image > 0)]
            if len(masked_depth) > 0:
                mean_depth = np.mean(masked_depth)
                std_depth = np.std(masked_depth)
                min_depth = np.min(masked_depth)
                max_depth = np.max(masked_depth)
            else:
                mean_depth = std_depth = min_depth = max_depth = 0.0
            
            instance_props = {
                'instance_id': instance_id,
                'area': props.area,
                'centroid': props.centroid,
                'bbox': props.bbox,
                'circularity': 4 * np.pi * props.area / (props.perimeter ** 2) if props.perimeter > 0 else 0,
                'mean_color': mean_color.tolist(),
                'std_color': std_color.tolist(),
                'mean_depth': mean_depth,
                'std_depth': std_depth,
                'depth_range': [min_depth, max_depth]
            }
            
            properties.append(instance_props)
        
        logger.info(f"Extracted properties for {len(properties)} instances")
        
        return properties


class MaskRCNNSegmenter:
    """Optional deep learning segmentation using Mask R-CNN."""
    
    def __init__(self, config: dict):
        """Initialize Mask R-CNN segmenter (placeholder for future implementation)."""
        self.config = config
        logger.warning("Mask R-CNN segmentation not implemented in this PoC")
    
    def segment(self, color_image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Placeholder for Mask R-CNN segmentation."""
        logger.warning("Falling back to classical segmentation")
        # Fallback to classical method
        classical_segmenter = FruitSegmenter(self.config)
        return classical_segmenter.segment_classical(color_image, np.ones_like(color_image[:,:,0]))