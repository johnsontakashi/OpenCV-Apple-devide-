"""
Main Pipeline for Fruit Label Placement Analysis

Orchestrates the complete workflow from image loading to candidate selection
and result export. Processes RGB-D image pairs to find optimal label locations.
"""

import os
import json
import pandas as pd
import yaml
import time
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Import custom modules
from preprocess import ImagePreprocessor
from segmentation import FruitSegmenter
from pointcloud import PointCloudProcessor
from planarity import PlanarityAnalyzer
from candidate_selector import CandidateSelector
from constraints import ConstraintFilter
from visualize import ResultVisualizer


class FruitLabelPipeline:
    """Main pipeline for fruit label placement analysis."""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()
        
        logger.info("Fruit Label Pipeline initialized")
        logger.info(f"Configuration: {config_path}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(self.config['output_dir'], 'pipeline.log')
                )
            ]
        )
        
        # Set up logger for this module
        global logger
        logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.preprocessor = ImagePreprocessor(self.config['preprocessing'])
        self.segmenter = FruitSegmenter(self.config['segmentation'])
        self.pointcloud_processor = PointCloudProcessor(self.config['camera'])
        self.planarity_analyzer = PlanarityAnalyzer(self.config['planarity'])
        self.candidate_selector = CandidateSelector(self.config['candidate_selection'])
        self.constraint_filter = ConstraintFilter(self.config['constraints'])
        
        # Create output directories
        os.makedirs(self.config['output_dir'], exist_ok=True)
        viz_dir = os.path.join(self.config['output_dir'], 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        self.visualizer = ResultVisualizer(self.config['visualization'], viz_dir)
    
    def find_image_pairs(self, input_dir: str) -> List[Tuple[str, str]]:
        """
        Find matching RGB and depth image pairs in input directory.
        
        Args:
            input_dir: Directory containing RGB and depth images
            
        Returns:
            List of (color_path, depth_path) tuples
        """
        logger.info(f"Scanning for image pairs in: {input_dir}")
        
        color_extensions = self.config.get('color_extensions', ['.jpg', '.png', '.jpeg'])
        depth_extensions = self.config.get('depth_extensions', ['.png', '.tiff', '.tif'])
        
        color_files = {}
        depth_files = {}
        
        # Find all color and depth files
        for file_path in Path(input_dir).rglob('*'):
            if file_path.is_file():
                stem = file_path.stem
                suffix = file_path.suffix.lower()
                
                # Check if it's a color image
                if suffix in color_extensions and 'depth' not in stem.lower():
                    color_files[stem] = str(file_path)
                
                # Check if it's a depth image
                elif suffix in depth_extensions and 'depth' in stem.lower():
                    # Extract base name (remove 'depth' suffix)
                    base_stem = stem.replace('_depth', '').replace('-depth', '').replace('depth', '')
                    depth_files[base_stem] = str(file_path)
        
        # Match pairs
        pairs = []
        for base_name in color_files:
            if base_name in depth_files:
                pairs.append((color_files[base_name], depth_files[base_name]))
            else:
                logger.warning(f"No matching depth image for: {color_files[base_name]}")
        
        # Try alternative matching strategies
        if not pairs:
            logger.info("No pairs found with standard naming. Trying alternative matching...")
            
            # Try matching by order (assumes same number of color and depth images)
            color_paths = sorted(color_files.values())
            depth_paths = sorted(depth_files.values())
            
            if len(color_paths) == len(depth_paths):
                pairs = list(zip(color_paths, depth_paths))
                logger.info(f"Matched {len(pairs)} pairs by filename order")
        
        logger.info(f"Found {len(pairs)} RGB-D image pairs")
        
        return pairs
    
    def process_image_pair(self, color_path: str, depth_path: str) -> Dict:
        """
        Process a single RGB-D image pair through the complete pipeline.
        
        Args:
            color_path: Path to color image
            depth_path: Path to depth image
            
        Returns:
            Dictionary with processing results
        """
        image_name = Path(color_path).stem
        logger.info(f"Processing image pair: {image_name}")
        
        start_time = time.time()
        
        try:
            # 1. Preprocessing
            logger.info("Step 1: Preprocessing images")
            color_image, depth_image = self.preprocessor.preprocess_pair(color_path, depth_path)
            
            # 2. Fruit Segmentation
            logger.info("Step 2: Segmenting fruits")
            instance_labels, num_instances = self.segmenter.segment(color_image, depth_image)
            
            if num_instances == 0:
                logger.warning(f"No fruits detected in {image_name}")
                return {
                    'image_name': image_name,
                    'processing_time': time.time() - start_time,
                    'fruits': [],
                    'error': 'No fruits detected'
                }
            
            # Extract instance masks
            instance_masks = self.segmenter.get_instance_masks(instance_labels)
            instance_properties = self.segmenter.get_instance_properties(
                instance_labels, color_image, depth_image
            )
            
            # 3. Planarity Analysis
            logger.info("Step 3: Analyzing surface planarity")
            planarity_maps = []
            rms_error_maps = []
            
            for i, instance_mask in enumerate(instance_masks):
                logger.debug(f"Analyzing planarity for fruit {i+1}/{num_instances}")
                planarity_map, rms_error_map = self.planarity_analyzer.compute_planarity_map(
                    depth_image, instance_mask
                )
                planarity_maps.append(planarity_map)
                rms_error_maps.append(rms_error_map)
            
            # 4. Candidate Selection
            logger.info("Step 4: Finding label candidates")
            candidates_per_fruit = []
            
            for i, (instance_mask, planarity_map, rms_error_map) in enumerate(
                zip(instance_masks, planarity_maps, rms_error_maps)
            ):
                logger.debug(f"Finding candidates for fruit {i+1}/{num_instances}")
                candidates = self.candidate_selector.find_candidates(
                    depth_image, color_image, instance_mask,
                    planarity_map, rms_error_map
                )
                
                # Add 3D coordinates
                candidates = self.candidate_selector.add_3d_coordinates(
                    candidates, self.config['camera']
                )
                
                candidates_per_fruit.append(candidates)
            
            # 5. Constraint Filtering
            logger.info("Step 5: Applying constraints")
            filtered_candidates_per_fruit, constraint_masks = self.constraint_filter.apply_all_constraints(
                candidates_per_fruit, color_image, depth_image,
                instance_labels, instance_masks
            )
            
            # 6. Create Results
            fruits_data = []
            for i, (instance_props, planarity_map, candidates) in enumerate(
                zip(instance_properties, planarity_maps, filtered_candidates_per_fruit)
            ):
                # Get planarity statistics
                planarity_stats = self.planarity_analyzer.get_planarity_statistics(
                    planarity_map, instance_masks[i]
                )
                
                fruit_data = {
                    'fruit_id': i + 1,
                    'properties': instance_props,
                    'planarity_statistics': planarity_stats,
                    'candidates': candidates,
                    'num_candidates': len([c for c in candidates if c.get('valid', True)])
                }
                
                fruits_data.append(fruit_data)
            
            # 7. Visualization
            logger.info("Step 6: Creating visualizations")
            viz_path = self.visualizer.create_comprehensive_visualization(
                image_name, color_image, depth_image, instance_labels,
                planarity_maps, instance_masks, filtered_candidates_per_fruit,
                constraint_masks
            )
            
            # Create individual fruit visualizations
            individual_viz_paths = self.visualizer.create_individual_fruit_visualizations(
                image_name, color_image, instance_masks,
                planarity_maps, filtered_candidates_per_fruit
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'image_name': image_name,
                'processing_time': processing_time,
                'color_path': color_path,
                'depth_path': depth_path,
                'num_fruits': num_instances,
                'fruits': fruits_data,
                'visualization_path': viz_path,
                'individual_visualizations': individual_viz_paths
            }
            
            logger.info(f"Completed processing {image_name} in {processing_time:.2f}s")
            logger.info(f"Found {num_instances} fruits with {sum(f['num_candidates'] for f in fruits_data)} valid candidates")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            return {
                'image_name': image_name,
                'processing_time': time.time() - start_time,
                'fruits': [],
                'error': str(e)
            }
    
    def export_results_json(self, results: List[Dict], output_path: str):
        """Export results to JSON format."""
        logger.info(f"Exporting results to JSON: {output_path}")
        
        export_data = {
            'pipeline_config': self.config,
            'processing_summary': {
                'total_images': len(results),
                'successful_images': len([r for r in results if 'error' not in r]),
                'total_fruits': sum(r.get('num_fruits', 0) for r in results),
                'total_candidates': sum(
                    sum(f['num_candidates'] for f in r.get('fruits', []))
                    for r in results
                )
            },
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"JSON export complete: {output_path}")
    
    def export_results_csv(self, results: List[Dict], output_path: str):
        """Export results to CSV format (flattened)."""
        logger.info(f"Exporting results to CSV: {output_path}")
        
        rows = []
        
        for result in results:
            if 'error' in result:
                continue
            
            image_name = result['image_name']
            
            for fruit in result.get('fruits', []):
                fruit_id = fruit['fruit_id']
                props = fruit['properties']
                planarity_stats = fruit['planarity_statistics']
                
                for candidate in fruit.get('candidates', []):
                    if not candidate.get('valid', True):
                        continue
                    
                    row = {
                        'image_name': image_name,
                        'fruit_id': fruit_id,
                        'fruit_area': props.get('area', 0),
                        'fruit_centroid_y': props.get('centroid', [0, 0])[0],
                        'fruit_centroid_x': props.get('centroid', [0, 0])[1],
                        'fruit_mean_depth': props.get('mean_depth', 0),
                        'fruit_circularity': props.get('circularity', 0),
                        'fruit_mean_planarity': planarity_stats.get('mean_planarity', 1),
                        'fruit_flat_percentage': planarity_stats.get('flat_percentage', 0),
                        'candidate_pixel_y': candidate['center_y'],
                        'candidate_pixel_x': candidate['center_x'],
                        'candidate_depth': candidate.get('depth', 0),
                        'candidate_world_x': candidate.get('world_coordinates', [0, 0, 0])[0],
                        'candidate_world_y': candidate.get('world_coordinates', [0, 0, 0])[1],
                        'candidate_world_z': candidate.get('world_coordinates', [0, 0, 0])[2],
                        'candidate_total_score': candidate.get('total_score', 0),
                        'candidate_planarity_score': candidate.get('planarity_score', 0),
                        'candidate_depth_validity_score': candidate.get('depth_validity_score', 0),
                        'candidate_color_uniformity_score': candidate.get('color_uniformity_score', 0),
                        'candidate_edge_distance_score': candidate.get('edge_distance_score', 0),
                        'candidate_edge_distance_pixels': candidate.get('edge_distance_pixels', 0),
                        'candidate_rms_error': candidate.get('rms_error', float('inf'))
                    }
                    
                    rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"CSV export complete: {output_path} ({len(rows)} candidate rows)")
        else:
            logger.warning("No valid candidates to export to CSV")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def run_pipeline(self, input_dir: str) -> List[Dict]:
        """
        Run the complete pipeline on all image pairs in input directory.
        
        Args:
            input_dir: Directory containing RGB-D image pairs
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting pipeline execution on directory: {input_dir}")
        
        # Find image pairs
        image_pairs = self.find_image_pairs(input_dir)
        
        if not image_pairs:
            logger.error("No valid RGB-D image pairs found")
            return []
        
        # Process each pair
        results = []
        
        for i, (color_path, depth_path) in enumerate(image_pairs):
            logger.info(f"Processing pair {i+1}/{len(image_pairs)}")
            
            result = self.process_image_pair(color_path, depth_path)
            results.append(result)
        
        # Export results
        logger.info("Exporting results...")
        
        json_path = os.path.join(self.config['output_dir'], 'results.json')
        csv_path = os.path.join(self.config['output_dir'], 'results.csv')
        
        self.export_results_json(results, json_path)
        self.export_results_csv(results, csv_path)
        
        # Create summary visualizations
        if results:
            self.visualizer.create_summary_statistics_plot(results)
            self.visualizer.save_visualization_legend()
        
        # Print summary
        self._print_pipeline_summary(results)
        
        logger.info("Pipeline execution complete!")
        
        return results
    
    def _print_pipeline_summary(self, results: List[Dict]):
        """Print pipeline execution summary."""
        total_images = len(results)
        successful_images = len([r for r in results if 'error' not in r])
        failed_images = total_images - successful_images
        
        total_fruits = sum(r.get('num_fruits', 0) for r in results)
        total_candidates = sum(
            sum(f['num_candidates'] for f in r.get('fruits', []))
            for r in results
        )
        
        if successful_images > 0:
            avg_fruits_per_image = total_fruits / successful_images
            avg_candidates_per_fruit = total_candidates / total_fruits if total_fruits > 0 else 0
        else:
            avg_fruits_per_image = 0
            avg_candidates_per_fruit = 0
        
        print("\n" + "="*60)
        print("FRUIT LABEL PLACEMENT PIPELINE SUMMARY")
        print("="*60)
        print(f"Images processed: {successful_images}/{total_images}")
        if failed_images > 0:
            print(f"Failed images: {failed_images}")
        print(f"Total fruits detected: {total_fruits}")
        print(f"Total valid candidates: {total_candidates}")
        print(f"Average fruits per image: {avg_fruits_per_image:.1f}")
        print(f"Average candidates per fruit: {avg_candidates_per_fruit:.1f}")
        print(f"Output directory: {self.config['output_dir']}")
        print("="*60)


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fruit Label Placement Analysis Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing RGB-D image pairs')
    parser.add_argument('--output', type=str, 
                       help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load and optionally modify config
    pipeline = FruitLabelPipeline(args.config)
    
    if args.output:
        pipeline.config['output_dir'] = args.output
        # Recreate output directories
        os.makedirs(args.output, exist_ok=True)
        viz_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        pipeline.visualizer = ResultVisualizer(pipeline.config['visualization'], viz_dir)
    
    # Run pipeline
    results = pipeline.run_pipeline(args.input)
    
    return results


if __name__ == "__main__":
    main()