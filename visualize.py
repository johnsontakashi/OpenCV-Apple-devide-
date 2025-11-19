"""
Visualization Module

Creates comprehensive visualizations of the fruit label placement analysis results.
Includes segmentation masks, depth maps, planarity heatmaps, and candidate locations.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Creates visualizations for fruit label placement analysis."""
    
    def __init__(self, config: dict, output_dir: str):
        """
        Initialize visualizer with configuration and output directory.
        
        Args:
            config: Dictionary containing visualization parameters
            output_dir: Directory to save visualization images
        """
        self.config = config
        self.output_dir = output_dir
        self.label_diameter_pixels = config.get('label_diameter_pixels', 40)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualization parameters
        self.figure_size = config.get('figure_size', (15, 10))
        self.dpi = config.get('visualization_dpi', 150)
        
    def create_instance_colormap(self, num_instances: int) -> List[Tuple[int, int, int]]:
        """
        Create distinct colors for each fruit instance.
        
        Args:
            num_instances: Number of fruit instances
            
        Returns:
            List of RGB color tuples
        """
        if num_instances == 0:
            return [(0, 0, 0)]
        
        # Use matplotlib's tab colormap for distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_instances, 20)))
        rgb_colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        
        # If more than 20 instances, cycle through colors
        while len(rgb_colors) < num_instances:
            rgb_colors.extend(rgb_colors[:min(20, num_instances - len(rgb_colors))])
        
        return rgb_colors[:num_instances]
    
    def draw_segmentation_overlay(self, color_image: np.ndarray,
                                instance_labels: np.ndarray,
                                alpha: float = 0.5) -> np.ndarray:
        """
        Create segmentation overlay on color image.
        
        Args:
            color_image: RGB color image
            instance_labels: Instance label map
            alpha: Overlay transparency
            
        Returns:
            RGB image with segmentation overlay
        """
        overlay = color_image.copy()
        
        num_instances = instance_labels.max()
        if num_instances == 0:
            return overlay
        
        # Create colored masks
        colors = self.create_instance_colormap(num_instances)
        
        for instance_id in range(1, num_instances + 1):
            mask = (instance_labels == instance_id)
            if np.sum(mask) > 0:
                color = colors[instance_id - 1]
                overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
        
        return overlay.astype(np.uint8)
    
    def create_depth_visualization(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Create colorized depth visualization.
        
        Args:
            depth_image: Normalized depth image in meters
            
        Returns:
            RGB depth colormap
        """
        # Handle invalid depths
        valid_mask = depth_image > 0
        
        if not np.any(valid_mask):
            return np.zeros((*depth_image.shape, 3), dtype=np.uint8)
        
        # Normalize depth for visualization
        depth_viz = np.zeros_like(depth_image)
        valid_depths = depth_image[valid_mask]
        
        if len(valid_depths) > 0:
            depth_min, depth_max = np.percentile(valid_depths, [5, 95])
            if depth_max > depth_min:
                depth_viz[valid_mask] = np.clip(
                    (depth_image[valid_mask] - depth_min) / (depth_max - depth_min),
                    0, 1
                )
        
        # Apply colormap
        depth_colored = plt.cm.viridis(depth_viz)[:, :, :3]  # Remove alpha channel
        depth_colored = (depth_colored * 255).astype(np.uint8)
        
        # Set invalid regions to black
        depth_colored[~valid_mask] = 0
        
        return depth_colored
    
    def create_planarity_heatmap(self, planarity_map: np.ndarray,
                               instance_mask: np.ndarray) -> np.ndarray:
        """
        Create planarity heatmap visualization.
        
        Args:
            planarity_map: Planarity score map (0 = flat, 1 = curved)
            instance_mask: Binary mask for the instance
            
        Returns:
            RGB heatmap image
        """
        # Create masked planarity map
        masked_planarity = planarity_map.copy()
        masked_planarity[instance_mask == 0] = 0
        
        # Apply colormap (blue = flat, red = curved)
        heatmap = plt.cm.jet(masked_planarity)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Set background to black
        heatmap[instance_mask == 0] = 0
        
        return heatmap
    
    def draw_candidates_on_image(self, base_image: np.ndarray,
                               candidates: List[Dict],
                               show_scores: bool = True) -> np.ndarray:
        """
        Draw candidate locations on base image.
        
        Args:
            base_image: Base RGB image
            candidates: List of candidate dictionaries
            show_scores: Whether to display scores as text
            
        Returns:
            Image with drawn candidates
        """
        result_image = base_image.copy()
        radius = self.label_diameter_pixels // 2
        
        for i, candidate in enumerate(candidates):
            y, x = candidate['center_y'], candidate['center_x']
            score = candidate.get('total_score', 0.0)
            valid = candidate.get('valid', True)
            
            # Choose color based on validity and score
            if not valid:
                color = (128, 128, 128)  # Gray for invalid
                thickness = 1
            elif score > 0.7:
                color = (0, 255, 0)  # Green for high score
                thickness = 3
            elif score > 0.4:
                color = (255, 255, 0)  # Yellow for medium score
                thickness = 2
            else:
                color = (255, 0, 0)  # Red for low score
                thickness = 2
            
            # Draw circle
            cv2.circle(result_image, (x, y), radius, color, thickness)
            
            # Draw center point
            cv2.circle(result_image, (x, y), 2, color, -1)
            
            # Add score text
            if show_scores and valid:
                text = f'{score:.2f}'
                font_scale = 0.4
                font_thickness = 1
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = x - text_size[0] // 2
                text_y = y + radius + text_size[1] + 5
                
                # Add background rectangle for text
                cv2.rectangle(result_image, 
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (255, 255, 255), -1)
                
                cv2.putText(result_image, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        return result_image
    
    def create_comprehensive_visualization(self, image_name: str,
                                         color_image: np.ndarray,
                                         depth_image: np.ndarray,
                                         instance_labels: np.ndarray,
                                         planarity_maps: List[np.ndarray],
                                         instance_masks: List[np.ndarray],
                                         candidates_per_fruit: List[List[Dict]],
                                         constraint_masks: Optional[Dict] = None) -> str:
        """
        Create comprehensive visualization with all analysis results.
        
        Args:
            image_name: Base name for the image
            color_image: Original RGB color image
            depth_image: Normalized depth image
            instance_labels: Instance segmentation labels
            planarity_maps: List of planarity maps for each fruit
            instance_masks: List of instance masks
            candidates_per_fruit: List of candidates for each fruit
            constraint_masks: Optional constraint visualization masks
            
        Returns:
            Path to saved visualization image
        """
        logger.info(f"Creating comprehensive visualization for {image_name}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=self.figure_size, dpi=self.dpi)
        axes = axes.flatten()
        
        # 1. Original image
        axes[0].imshow(color_image)
        axes[0].set_title('Original RGB Image')
        axes[0].axis('off')
        
        # 2. Depth visualization
        depth_vis = self.create_depth_visualization(depth_image)
        axes[1].imshow(depth_vis)
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        
        # 3. Instance segmentation
        seg_overlay = self.draw_segmentation_overlay(color_image, instance_labels)
        axes[2].imshow(seg_overlay)
        axes[2].set_title(f'Segmentation ({instance_labels.max()} fruits)')
        axes[2].axis('off')
        
        # 4. Planarity heatmap (combined)
        if planarity_maps and instance_masks:
            combined_planarity = np.zeros_like(planarity_maps[0])
            combined_mask = np.zeros_like(instance_masks[0])
            
            for planarity_map, instance_mask in zip(planarity_maps, instance_masks):
                combined_planarity += planarity_map * (instance_mask > 0)
                combined_mask = combined_mask | (instance_mask > 0)
            
            heatmap = self.create_planarity_heatmap(combined_planarity, combined_mask)
            axes[3].imshow(heatmap)
            axes[3].set_title('Surface Planarity (Blue=Flat, Red=Curved)')
        else:
            axes[3].text(0.5, 0.5, 'No planarity data', ha='center', va='center')
            axes[3].set_title('Surface Planarity')
        axes[3].axis('off')
        
        # 5. Constraint visualization
        if constraint_masks:
            constraint_vis = color_image.copy()
            
            # Overlay tray boundary
            if constraint_masks.get('tray_boundary') is not None:
                tray_mask = constraint_masks['tray_boundary']
                constraint_vis[tray_mask > 0] = [255, 0, 255]  # Magenta
            
            # Overlay contact areas
            if constraint_masks.get('contact_areas') is not None:
                contact_mask = constraint_masks['contact_areas']
                constraint_vis[contact_mask > 0] = [0, 255, 255]  # Cyan
            
            # Overlay stem regions
            stem_masks = constraint_masks.get('stem_regions', [])
            for stem_mask in stem_masks:
                if stem_mask is not None:
                    constraint_vis[stem_mask > 0] = [255, 255, 0]  # Yellow
            
            axes[4].imshow(constraint_vis)
            axes[4].set_title('Constraints (Magenta=Tray, Cyan=Contact, Yellow=Stem)')
        else:
            axes[4].imshow(color_image)
            axes[4].set_title('Constraints')
        axes[4].axis('off')
        
        # 6. Final candidates
        candidates_vis = color_image.copy()
        
        # Draw all candidates
        for candidates in candidates_per_fruit:
            candidates_vis = self.draw_candidates_on_image(candidates_vis, candidates)
        
        axes[5].imshow(candidates_vis)
        
        total_candidates = sum(len(candidates) for candidates in candidates_per_fruit)
        valid_candidates = sum(len([c for c in candidates if c.get('valid', True)]) 
                             for candidates in candidates_per_fruit)
        
        axes[5].set_title(f'Label Candidates ({valid_candidates}/{total_candidates} valid)')
        axes[5].axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{image_name}_analysis.png')
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comprehensive visualization to {output_path}")
        
        return output_path
    
    def create_individual_fruit_visualizations(self, image_name: str,
                                             color_image: np.ndarray,
                                             instance_masks: List[np.ndarray],
                                             planarity_maps: List[np.ndarray],
                                             candidates_per_fruit: List[List[Dict]]) -> List[str]:
        """
        Create individual visualizations for each fruit.
        
        Args:
            image_name: Base name for the image
            color_image: Original RGB color image
            instance_masks: List of instance masks
            planarity_maps: List of planarity maps
            candidates_per_fruit: List of candidates for each fruit
            
        Returns:
            List of paths to saved visualization images
        """
        output_paths = []
        
        for i, (instance_mask, planarity_map, candidates) in enumerate(
            zip(instance_masks, planarity_maps, candidates_per_fruit)
        ):
            logger.debug(f"Creating visualization for fruit {i+1}")
            
            # Create figure for this fruit
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=self.dpi)
            
            # Crop to fruit region
            mask_indices = np.where(instance_mask > 0)
            if len(mask_indices[0]) == 0:
                continue
            
            min_y, max_y = np.min(mask_indices[0]), np.max(mask_indices[0])
            min_x, max_x = np.min(mask_indices[1]), np.max(mask_indices[1])
            
            # Add padding
            padding = 20
            min_y = max(0, min_y - padding)
            max_y = min(color_image.shape[0], max_y + padding)
            min_x = max(0, min_x - padding)
            max_x = min(color_image.shape[1], max_x + padding)
            
            # Crop images
            cropped_color = color_image[min_y:max_y, min_x:max_x]
            cropped_mask = instance_mask[min_y:max_y, min_x:max_x]
            cropped_planarity = planarity_map[min_y:max_y, min_x:max_x]
            
            # 1. Original fruit
            axes[0].imshow(cropped_color)
            axes[0].set_title(f'Fruit {i+1}')
            axes[0].axis('off')
            
            # 2. Planarity heatmap
            heatmap = self.create_planarity_heatmap(cropped_planarity, cropped_mask)
            axes[1].imshow(heatmap)
            axes[1].set_title('Surface Planarity')
            axes[1].axis('off')
            
            # 3. Candidates
            candidates_vis = cropped_color.copy()
            
            # Adjust candidate coordinates for cropped image
            adjusted_candidates = []
            for candidate in candidates:
                adjusted_candidate = candidate.copy()
                adjusted_candidate['center_y'] = candidate['center_y'] - min_y
                adjusted_candidate['center_x'] = candidate['center_x'] - min_x
                adjusted_candidates.append(adjusted_candidate)
            
            candidates_vis = self.draw_candidates_on_image(candidates_vis, adjusted_candidates)
            axes[2].imshow(candidates_vis)
            
            valid_count = len([c for c in candidates if c.get('valid', True)])
            axes[2].set_title(f'Candidates ({valid_count}/{len(candidates)} valid)')
            axes[2].axis('off')
            
            # Save individual fruit visualization
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f'{image_name}_fruit_{i+1}.png')
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            output_paths.append(output_path)
            
            logger.debug(f"Saved fruit {i+1} visualization to {output_path}")
        
        return output_paths
    
    def create_summary_statistics_plot(self, results_per_image: List[Dict],
                                     output_name: str = 'summary_statistics.png') -> str:
        """
        Create summary statistics visualization across all processed images.
        
        Args:
            results_per_image: List of result dictionaries for each image
            output_name: Name of output file
            
        Returns:
            Path to saved statistics plot
        """
        logger.info("Creating summary statistics plot")
        
        if not results_per_image:
            logger.warning("No results to visualize")
            return ""
        
        # Extract statistics
        image_names = [result['image_name'] for result in results_per_image]
        fruit_counts = [len(result['fruits']) for result in results_per_image]
        candidate_counts = [
            sum(len(fruit['candidates']) for fruit in result['fruits'])
            for result in results_per_image
        ]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=self.dpi)
        
        # 1. Fruit count per image
        axes[0, 0].bar(range(len(image_names)), fruit_counts)
        axes[0, 0].set_title('Number of Fruits per Image')
        axes[0, 0].set_ylabel('Fruit Count')
        axes[0, 0].set_xticks(range(len(image_names)))
        axes[0, 0].set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                                  for name in image_names], rotation=45)
        
        # 2. Candidate count per image
        axes[0, 1].bar(range(len(image_names)), candidate_counts)
        axes[0, 1].set_title('Number of Candidates per Image')
        axes[0, 1].set_ylabel('Candidate Count')
        axes[0, 1].set_xticks(range(len(image_names)))
        axes[0, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                                  for name in image_names], rotation=45)
        
        # 3. Score distribution
        all_scores = []
        for result in results_per_image:
            for fruit in result['fruits']:
                for candidate in fruit['candidates']:
                    if candidate.get('valid', True):
                        all_scores.append(candidate.get('total_score', 0.0))
        
        if all_scores:
            axes[1, 0].hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Distribution of Candidate Scores')
            axes[1, 0].set_xlabel('Score')
            axes[1, 0].set_ylabel('Count')
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid candidates', ha='center', va='center')
            axes[1, 0].set_title('Distribution of Candidate Scores')
        
        # 4. Processing statistics
        processing_times = [result.get('processing_time', 0) for result in results_per_image]
        if any(t > 0 for t in processing_times):
            axes[1, 1].bar(range(len(image_names)), processing_times)
            axes[1, 1].set_title('Processing Time per Image')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_xticks(range(len(image_names)))
            axes[1, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                                      for name in image_names], rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No timing data', ha='center', va='center')
            axes[1, 1].set_title('Processing Time per Image')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved summary statistics to {output_path}")
        
        return output_path
    
    def save_visualization_legend(self, output_name: str = 'legend.png') -> str:
        """
        Create and save a legend explaining the visualization symbols.
        
        Args:
            output_name: Name of output file
            
        Returns:
            Path to saved legend image
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        ax.text(5, 7.5, 'Fruit Label Placement Analysis - Visualization Legend', 
               fontsize=16, fontweight='bold', ha='center')
        
        # Candidate circles legend
        ax.text(1, 6.5, 'Candidate Circles:', fontsize=12, fontweight='bold')
        
        # Draw example circles
        circle_high = patches.Circle((2, 6), 0.2, color='green', fill=False, linewidth=3)
        circle_med = patches.Circle((2, 5.5), 0.2, color='yellow', fill=False, linewidth=2)
        circle_low = patches.Circle((2, 5), 0.2, color='red', fill=False, linewidth=2)
        circle_invalid = patches.Circle((2, 4.5), 0.2, color='gray', fill=False, linewidth=1)
        
        ax.add_patch(circle_high)
        ax.add_patch(circle_med)
        ax.add_patch(circle_low)
        ax.add_patch(circle_invalid)
        
        ax.text(2.8, 6, 'High Score (>0.7) - Green, thick border', fontsize=10)
        ax.text(2.8, 5.5, 'Medium Score (0.4-0.7) - Yellow, medium border', fontsize=10)
        ax.text(2.8, 5, 'Low Score (<0.4) - Red, medium border', fontsize=10)
        ax.text(2.8, 4.5, 'Invalid/Rejected - Gray, thin border', fontsize=10)
        
        # Constraint colors legend
        ax.text(1, 3.5, 'Constraint Areas:', fontsize=12, fontweight='bold')
        
        constraint_colors = [
            ('Tray Boundary', 'magenta'),
            ('Fruit Contact Areas', 'cyan'),
            ('Stem Regions', 'yellow')
        ]
        
        for i, (label, color) in enumerate(constraint_colors):
            y_pos = 3 - i * 0.3
            rect = patches.Rectangle((1.5, y_pos - 0.1), 0.3, 0.2, color=color)
            ax.add_patch(rect)
            ax.text(2, y_pos, label, fontsize=10)
        
        # Planarity heatmap legend
        ax.text(6, 3.5, 'Planarity Heatmap:', fontsize=12, fontweight='bold')
        ax.text(6, 3, 'Blue → Flat surfaces (good for labels)', fontsize=10)
        ax.text(6, 2.7, 'Red → Curved surfaces (avoid for labels)', fontsize=10)
        
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization legend to {output_path}")
        
        return output_path