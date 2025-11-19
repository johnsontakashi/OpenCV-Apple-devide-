"""
Color and Depth Image Preprocessing Module

Handles loading, alignment, and preprocessing of synchronized RGB and depth images.
Includes depth hole filling, noise reduction, and reflection reduction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles preprocessing of aligned color and depth image pairs."""
    
    def __init__(self, config: dict):
        """
        Initialize preprocessor with configuration parameters.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config
        self.depth_scale = config.get('depth_scale', 1000.0)  # Convert to meters
        self.max_depth = config.get('max_depth', 2.0)  # Maximum depth in meters
        self.bilateral_d = config.get('bilateral_d', 9)
        self.bilateral_sigma_color = config.get('bilateral_sigma_color', 75)
        self.bilateral_sigma_space = config.get('bilateral_sigma_space', 75)
        
    def load_image_pair(self, color_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load aligned color and depth image pair.
        
        Args:
            color_path: Path to color image
            depth_path: Path to depth image
            
        Returns:
            Tuple of (color_image, depth_image)
        """
        # Load color image
        color = cv2.imread(color_path)
        if color is None:
            raise ValueError(f"Could not load color image from {color_path}")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        # Load depth image (assuming 16-bit depth)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not load depth image from {depth_path}")
            
        # Ensure both images have the same dimensions
        if color.shape[:2] != depth.shape[:2]:
            logger.warning(f"Image size mismatch: color {color.shape[:2]}, depth {depth.shape[:2]}")
            # Resize depth to match color
            depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        return color, depth
    
    def fill_depth_holes(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Fill holes in depth image using inpainting and interpolation.
        
        Args:
            depth_image: Input depth image with holes (zeros)
            
        Returns:
            Depth image with filled holes
        """
        # Create mask for holes (zero depth values)
        mask = (depth_image == 0).astype(np.uint8)
        
        # If no holes, return original
        if np.sum(mask) == 0:
            return depth_image
        
        # Convert to float for processing
        depth_float = depth_image.astype(np.float32)
        
        # Use OpenCV inpainting for small holes
        if np.sum(mask) < 0.1 * mask.size:  # Less than 10% holes
            filled = cv2.inpaint(depth_float, mask, 3, cv2.INPAINT_NS)
        else:
            # For large holes, use nearest neighbor interpolation
            filled = self._nearest_neighbor_fill(depth_float, mask)
        
        return filled.astype(depth_image.dtype)
    
    def _nearest_neighbor_fill(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill holes using nearest neighbor interpolation."""
        from scipy.ndimage import distance_transform_edt
        
        # Find indices of valid pixels
        valid_mask = (mask == 0)
        
        if np.sum(valid_mask) == 0:
            return depth
        
        # Distance transform to find nearest valid pixel
        indices = distance_transform_edt(mask, return_indices=True, return_distances=False)
        
        # Fill holes with nearest valid values
        filled = depth.copy()
        filled[mask > 0] = depth[indices[0][mask > 0], indices[1][mask > 0]]
        
        return filled
    
    def reduce_depth_noise(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Reduce noise in depth image using bilateral filtering.
        
        Args:
            depth_image: Input noisy depth image
            
        Returns:
            Denoised depth image
        """
        # Convert to float for filtering
        depth_float = depth_image.astype(np.float32)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(
            depth_float,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        
        return filtered.astype(depth_image.dtype)
    
    def reduce_reflections(self, color_image: np.ndarray) -> np.ndarray:
        """
        Reduce specular highlights/reflections in color image using HSV processing.
        
        Args:
            color_image: Input RGB color image
            
        Returns:
            Color image with reduced reflections
        """
        # Convert to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        
        # Create mask for very bright pixels (potential reflections)
        _, _, v_channel = cv2.split(hsv)
        bright_mask = v_channel > 200  # Threshold for bright pixels
        
        # Reduce brightness of specular highlights
        hsv_corrected = hsv.copy()
        hsv_corrected[bright_mask, 2] = np.clip(
            hsv_corrected[bright_mask, 2] * 0.7,  # Reduce brightness by 30%
            0, 255
        )
        
        # Convert back to RGB
        corrected_rgb = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2RGB)
        
        return corrected_rgb
    
    def normalize_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Normalize depth values to meters and clip to max range.
        
        Args:
            depth_image: Raw depth image
            
        Returns:
            Normalized depth image in meters
        """
        # Convert to meters
        depth_meters = depth_image.astype(np.float32) / self.depth_scale
        
        # Clip to maximum depth
        depth_meters = np.clip(depth_meters, 0, self.max_depth)
        
        # Set invalid depths (beyond max) to 0
        depth_meters[depth_image == 0] = 0
        
        return depth_meters
    
    def preprocess_pair(self, color_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for an image pair.
        
        Args:
            color_path: Path to color image
            depth_path: Path to depth image
            
        Returns:
            Tuple of (preprocessed_color, preprocessed_depth)
        """
        logger.info(f"Preprocessing image pair: {color_path}, {depth_path}")
        
        # Load images
        color, depth = self.load_image_pair(color_path, depth_path)
        
        # Preprocess depth
        depth_filled = self.fill_depth_holes(depth)
        depth_denoised = self.reduce_depth_noise(depth_filled)
        depth_normalized = self.normalize_depth(depth_denoised)
        
        # Preprocess color
        color_corrected = self.reduce_reflections(color)
        
        logger.info(f"Preprocessing complete. Image size: {color.shape[:2]}")
        
        return color_corrected, depth_normalized
    
    def create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Create a colormap visualization of depth image.
        
        Args:
            depth_image: Normalized depth image in meters
            
        Returns:
            RGB colormap of depth
        """
        # Normalize to 0-255 for visualization
        valid_mask = depth_image > 0
        if np.sum(valid_mask) == 0:
            return np.zeros((*depth_image.shape, 3), dtype=np.uint8)
        
        depth_viz = np.zeros_like(depth_image)
        depth_viz[valid_mask] = (depth_image[valid_mask] / self.max_depth * 255)
        depth_viz = depth_viz.astype(np.uint8)
        
        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        
        # Set invalid regions to black
        depth_colormap[~valid_mask] = 0
        
        return depth_colormap