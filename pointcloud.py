"""
Point Cloud Generation and Processing Module

Converts depth images to 3D point clouds and computes surface normals.
Provides utilities for 3D analysis and visualization.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional
import cv2
import logging

logger = logging.getLogger(__name__)


class PointCloudProcessor:
    """Handles 3D point cloud generation and processing."""
    
    def __init__(self, config: dict):
        """
        Initialize point cloud processor with camera parameters.
        
        Args:
            config: Dictionary containing camera intrinsics and processing parameters
        """
        self.config = config
        
        # Camera intrinsics
        self.fx = config.get('camera_fx', 525.0)
        self.fy = config.get('camera_fy', 525.0)
        self.cx = config.get('camera_cx', 320.0)
        self.cy = config.get('camera_cy', 240.0)
        
        # Processing parameters
        self.normal_radius = config.get('normal_radius', 0.03)
        self.normal_max_nn = config.get('normal_max_nn', 30)
        self.outlier_nb_neighbors = config.get('outlier_nb_neighbors', 20)
        self.outlier_std_ratio = config.get('outlier_std_ratio', 2.0)
        
    def depth_to_pointcloud(self, depth_image: np.ndarray, 
                          color_image: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Convert depth image to 3D point cloud.
        
        Args:
            depth_image: Normalized depth image in meters
            color_image: Optional RGB color image for coloring points
            
        Returns:
            Open3D point cloud object
        """
        height, width = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth pixels
        valid_mask = depth_image > 0
        
        # Convert to 3D coordinates
        z = depth_image[valid_mask]
        x = (u[valid_mask] - self.cx) * z / self.fx
        y = (v[valid_mask] - self.cy) * z / self.fy
        
        # Create point cloud
        points = np.stack([x, y, z], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if provided
        if color_image is not None:
            colors = color_image[valid_mask] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Generated point cloud with {len(points)} points")
        
        return pcd
    
    def compute_normals(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Compute surface normals for point cloud.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Array of normal vectors (N x 3)
        """
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )
        
        # Orient normals consistently (towards viewpoint)
        pcd.orient_normals_consistent_tangent_plane(k=10)
        
        normals = np.asarray(pcd.normals)
        
        logger.info(f"Computed normals for {len(normals)} points")
        
        return normals
    
    def remove_outliers(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove statistical outliers from point cloud.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Cleaned point cloud
        """
        cleaned_pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio
        )
        
        logger.info(f"Removed outliers: {len(pcd.points) - len(cleaned_pcd.points)} points")
        
        return cleaned_pcd
    
    def project_to_image(self, points_3d: np.ndarray, 
                        image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points back to 2D image coordinates.
        
        Args:
            points_3d: 3D points (N x 3)
            image_shape: Target image shape (height, width)
            
        Returns:
            Tuple of (pixel_coordinates, valid_mask)
        """
        # Project to image plane
        u = (points_3d[:, 0] * self.fx / points_3d[:, 2]) + self.cx
        v = (points_3d[:, 1] * self.fy / points_3d[:, 2]) + self.cy
        
        # Round to integer pixel coordinates
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        # Check valid bounds
        height, width = image_shape
        valid_mask = (
            (u_int >= 0) & (u_int < width) &
            (v_int >= 0) & (v_int < height) &
            (points_3d[:, 2] > 0)  # Valid depth
        )
        
        pixel_coords = np.stack([v_int, u_int], axis=1)
        
        return pixel_coords, valid_mask
    
    def normals_to_image(self, points_3d: np.ndarray, normals: np.ndarray, 
                        image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Map 3D normals back to 2D image space.
        
        Args:
            points_3d: 3D points (N x 3)
            normals: Surface normals (N x 3)
            image_shape: Target image shape (height, width)
            
        Returns:
            Normal map image (H x W x 3)
        """
        height, width = image_shape
        normal_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Project points to image
        pixel_coords, valid_mask = self.project_to_image(points_3d, image_shape)
        
        # Map normals to valid pixels
        valid_coords = pixel_coords[valid_mask]
        valid_normals = normals[valid_mask]
        
        if len(valid_coords) > 0:
            normal_image[valid_coords[:, 0], valid_coords[:, 1]] = valid_normals
        
        return normal_image
    
    def create_instance_pointcloud(self, depth_image: np.ndarray, 
                                 color_image: np.ndarray,
                                 instance_mask: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Create point cloud for a specific instance.
        
        Args:
            depth_image: Normalized depth image in meters
            color_image: RGB color image
            instance_mask: Binary mask for the instance
            
        Returns:
            Point cloud for the instance
        """
        # Mask the depth and color images
        masked_depth = depth_image.copy()
        masked_depth[instance_mask == 0] = 0
        
        masked_color = color_image.copy()
        masked_color[instance_mask == 0] = 0
        
        # Generate point cloud
        pcd = self.depth_to_pointcloud(masked_depth, masked_color)
        
        # Clean up
        if len(pcd.points) > 100:  # Only clean if enough points
            pcd = self.remove_outliers(pcd)
        
        return pcd
    
    def compute_instance_normals(self, depth_image: np.ndarray,
                               instance_mask: np.ndarray) -> np.ndarray:
        """
        Compute surface normals for a specific fruit instance.
        
        Args:
            depth_image: Normalized depth image in meters
            instance_mask: Binary mask for the instance
            
        Returns:
            Normal map for the instance (H x W x 3)
        """
        # Create instance point cloud
        pcd = self.create_instance_pointcloud(
            depth_image, 
            np.ones_like(depth_image)[..., None].repeat(3, axis=2) * 255,
            instance_mask
        )
        
        if len(pcd.points) < 10:
            logger.warning("Too few points for normal computation")
            return np.zeros((*depth_image.shape, 3))
        
        # Compute normals
        normals = self.compute_normals(pcd)
        points_3d = np.asarray(pcd.points)
        
        # Map back to image
        normal_image = self.normals_to_image(points_3d, normals, depth_image.shape)
        
        return normal_image
    
    def estimate_fruit_size(self, pcd: o3d.geometry.PointCloud) -> dict:
        """
        Estimate physical size properties of a fruit from its point cloud.
        
        Args:
            pcd: Point cloud of the fruit
            
        Returns:
            Dictionary with size estimates
        """
        if len(pcd.points) == 0:
            return {'diameter': 0, 'volume': 0, 'surface_area': 0}
        
        points = np.asarray(pcd.points)
        
        # Estimate diameter as maximum distance between points
        if len(points) > 1:
            # Use subset for efficiency
            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                subset_points = points[indices]
            else:
                subset_points = points
            
            distances = np.linalg.norm(
                subset_points[:, None] - subset_points[None, :], 
                axis=2
            )
            diameter = np.max(distances)
        else:
            diameter = 0
        
        # Estimate volume using convex hull
        try:
            hull = pcd.compute_convex_hull()[0]
            volume = hull.get_volume()
        except:
            volume = 0
        
        # Estimate surface area
        try:
            # Create mesh and compute area
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            surface_area = mesh.get_surface_area()
        except:
            surface_area = 0
        
        return {
            'diameter': diameter,
            'volume': volume,
            'surface_area': surface_area
        }
    
    def visualize_pointcloud(self, pcd: o3d.geometry.PointCloud, 
                           window_name: str = "Point Cloud") -> None:
        """
        Visualize point cloud (optional utility).
        
        Args:
            pcd: Point cloud to visualize
            window_name: Window title
        """
        logger.info(f"Visualizing point cloud with {len(pcd.points)} points")
        o3d.visualization.draw_geometries([pcd], window_name=window_name)