# Fruit Label Placement Analysis - Proof of Concept

A complete computer vision pipeline for automatically identifying optimal label placement locations on apples and tomatoes using synchronized RGB and depth images.

## Overview

This system processes RGB-D image pairs to:
1. Segment individual fruits (apples/tomatoes) from background
2. Analyze surface curvature and planarity  
3. Identify flat regions suitable for circular label placement
4. Apply constraints to filter unsuitable locations
5. Generate comprehensive visualizations and export results

## Features

- **Multi-modal processing**: Combines RGB and depth information
- **Instance segmentation**: Separates touching fruits automatically
- **Surface analysis**: Computes local planarity using SVD plane fitting
- **Constraint filtering**: Avoids stem regions, tray edges, and contact areas
- **Comprehensive visualization**: Multiple view modes and heatmaps
- **Flexible export**: JSON and CSV output formats
- **Configurable parameters**: Easy adjustment via YAML configuration

## Installation

### Requirements

- Python 3.8+
- Local processing only (no cloud dependencies)

### Install Dependencies

```bash
pip install opencv-python numpy scipy scikit-image scikit-learn open3d matplotlib pandas PyYAML
```

### Detailed Package List

```bash
# Core image processing
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install scipy==1.11.1

# Machine learning and analysis  
pip install scikit-image==0.21.0
pip install scikit-learn==1.3.0

# 3D processing
pip install open3d==0.17.0

# Visualization and data
pip install matplotlib==3.7.2
pip install pandas==2.0.3

# Configuration
pip install PyYAML==6.0.1
```

## Project Structure

```
project/
├── preprocess.py           # RGB-D image preprocessing
├── segmentation.py         # Fruit segmentation and instance separation
├── pointcloud.py          # 3D point cloud processing
├── planarity.py           # Surface curvature and planarity analysis
├── candidate_selector.py   # Label placement candidate detection
├── constraints.py         # Constraint filtering (stem, tray, contact areas)
├── visualize.py           # Result visualization and plotting
├── main.py                # Main pipeline orchestrator
├── config.yaml            # Configuration parameters
├── README.md              # This documentation
└── output/                # Output directory (created automatically)
    ├── visualizations/    # Generated visualizations
    ├── results.json       # Detailed results in JSON format
    ├── results.csv        # Flattened results in CSV format
    └── pipeline.log       # Processing log file
```

## Quick Start

### 1. Prepare Input Data

Organize your RGB-D image pairs in a directory structure:

```
input_data/
├── image1.jpg              # Color image
├── image1_depth.png        # Corresponding depth image
├── image2.jpg              
├── image2_depth.png
└── ...
```

**Naming Convention**: Depth images should include "depth" in the filename and have the same base name as the corresponding color image.

### 2. Configure Camera Parameters

Edit `config.yaml` to match your RGB-D camera specifications:

```yaml
camera:
  camera_fx: 525.0      # Focal length X (pixels)
  camera_fy: 525.0      # Focal length Y (pixels)
  camera_cx: 320.0      # Principal point X (pixels)
  camera_cy: 240.0      # Principal point Y (pixels)
  depth_scale: 1000.0   # Raw depth units per meter
```

### 3. Run the Pipeline

```bash
python main.py --input /path/to/input_data --config config.yaml
```

### 4. View Results

Check the `output/` directory for:
- **Visualizations**: Comprehensive analysis images
- **results.json**: Detailed structured results
- **results.csv**: Tabular data for spreadsheet analysis

## Algorithm Details

### 1. Preprocessing (`preprocess.py`)

**Input**: Aligned RGB and depth image pairs  
**Processing**:
- Depth hole filling using OpenCV inpainting
- Bilateral filtering for noise reduction
- Specular highlight reduction in HSV space
- Depth normalization to meters

**Output**: Clean RGB and depth images

### 2. Fruit Segmentation (`segmentation.py`)

**Classical Computer Vision Approach**:
- HSV color thresholding for red/orange fruits
- Depth-based foreground extraction
- Morphological operations for noise removal
- Distance transform + watershed for instance separation

**Output**: Instance-labeled segmentation mask

### 3. Point Cloud Processing (`pointcloud.py`)

**Processing**:
- Convert depth pixels to 3D coordinates using camera intrinsics
- Generate colored point clouds
- Compute surface normals using Open3D
- Statistical outlier removal

**Output**: 3D point clouds with normals for each fruit

### 4. Surface Planarity Analysis (`planarity.py`)

**Local Analysis Window**:
- Window size = `label_diameter_pixels × 1.2` 
- Adaptive window sizing based on depth

**Planarity Computation**:
1. Extract 3D points in local neighborhood
2. Fit plane using SVD (least squares)
3. Compute RMS fitting error
4. PCA eigenvalue analysis for planarity score

**Output**: Planarity maps (0 = flat, 1 = curved)

### 5. Candidate Selection (`candidate_selector.py`)

**Selection Criteria**:
- Local window contains ≥ 95% valid depth
- Planarity RMS ≤ threshold
- Low color standard deviation (uniformity)
- Sufficient distance from fruit edges
- No significant depth discontinuities

**Scoring Function**:
```
score = 0.5 × (1 - RMS_norm) 
      + 0.25 × valid_depth_fraction
      + 0.15 × (1 - color_std_norm)  
      + 0.10 × distance_from_edge_norm
```

**Output**: Top 1-5 candidates per fruit with scores

### 6. Constraint Filtering (`constraints.py`)

**Stem Region Detection**:
- HSV color detection (brown/green signatures)
- Depth spike detection for protruding stems
- Region dilation for safety margins

**Tray Boundary Detection**:
- RANSAC plane fitting on background points
- Distance-based filtering from tray edges

**Contact Area Detection**:
- Depth gradient analysis at instance boundaries
- Morphological dilation for exclusion zones

**Output**: Filtered candidates avoiding constrained regions

## Configuration Guide

### Core Parameters (`config.yaml`)

**Camera Calibration**:
```yaml
camera:
  camera_fx: 525.0      # Adjust based on your camera
  camera_fy: 525.0      # Usually equal to fx for square pixels
  camera_cx: 320.0      # Half of image width (approximately)  
  camera_cy: 240.0      # Half of image height (approximately)
```

**Label Size**:
```yaml
planarity:
  label_diameter_mm: 20.0       # Physical label size
  label_diameter_pixels: 40     # Expected size in pixels
```

**Quality Thresholds**:
```yaml
candidate_selection:
  max_rms_threshold: 0.01              # Lower = stricter planarity
  min_valid_depth_fraction: 0.95      # Higher = more coverage required
  min_edge_distance: 20                # Pixels from fruit boundary
```

**Fruit Colors (HSV)**:
```yaml
segmentation:
  hsv_lower: [0, 50, 50]        # Red/orange lower bound
  hsv_upper: [25, 255, 255]     # Red/orange upper bound
```

## Output Formats

### JSON Results (`results.json`)

Complete structured data including:
- Processing metadata and timing
- Per-fruit properties (area, centroid, planarity statistics)
- Per-candidate details (location, scores, 3D coordinates)
- Configuration parameters used

### CSV Results (`results.csv`)

Flattened tabular format with columns:
- Image and fruit identifiers
- Fruit physical properties  
- Candidate pixel and world coordinates
- Individual scoring components
- Quality metrics

### Visualizations

**Comprehensive Analysis** (`*_analysis.png`):
- Original RGB image
- Depth colormap 
- Instance segmentation overlay
- Surface planarity heatmap
- Constraint visualization  
- Final candidate locations

**Individual Fruits** (`*_fruit_N.png`):
- Cropped fruit region
- Local planarity analysis
- Candidate locations with scores

## Extending the System

### Adding New Fruit Types

1. **Update color thresholds** in `config.yaml`:
```yaml
segmentation:
  hsv_lower: [40, 40, 40]     # For green fruits
  hsv_upper: [80, 255, 255]
```

2. **Adjust morphological parameters** for different sizes:
```yaml
segmentation:
  min_fruit_area: 500         # Smaller fruits
  morph_kernel_size: 3        # Finer details
```

### Custom Constraint Rules

Extend `constraints.py` with new filtering methods:

```python
def apply_custom_constraint(self, candidates, custom_mask):
    """Add your custom constraint logic here."""
    filtered_candidates = []
    for candidate in candidates:
        # Your filtering logic
        if meets_custom_criteria(candidate):
            filtered_candidates.append(candidate)
    return filtered_candidates
```

### Real-time Processing

For live camera feeds:

1. **Modify input handling** in `main.py`:
```python
def process_camera_stream(self):
    """Process live camera stream."""
    while True:
        color_frame, depth_frame = camera.get_frames()
        result = self.process_image_pair_from_arrays(color_frame, depth_frame)
        # Display results
```

2. **Optimize for speed**:
```yaml
advanced:
  max_image_size: [640, 480]    # Reduce resolution
  enable_multiprocessing: true   # Parallel processing
```

### Integration with Robot Systems

Export candidate locations in robot coordinate systems:

```python
def export_robot_coordinates(candidates, camera_to_robot_transform):
    """Transform candidates to robot coordinate frame."""
    for candidate in candidates:
        world_coords = candidate['world_coordinates']
        robot_coords = transform_coordinates(world_coords, camera_to_robot_transform)
        candidate['robot_coordinates'] = robot_coords
```

## Troubleshooting

### Common Issues

**No fruits detected**:
- Check HSV color thresholds for your fruit types
- Verify depth image alignment and scale
- Ensure adequate lighting and contrast

**Poor candidate quality**:
- Lower `max_rms_threshold` for stricter planarity
- Increase `min_edge_distance` to avoid boundaries
- Adjust `label_diameter_pixels` to match actual size

**Processing errors**:
- Check input image formats (RGB for color, 16-bit for depth)
- Verify camera parameter accuracy
- Ensure sufficient memory for large images

### Parameter Tuning

1. **Start with default configuration**
2. **Process test images and examine visualizations**
3. **Adjust one parameter category at a time**:
   - Segmentation → Better fruit detection
   - Planarity → Stricter surface requirements  
   - Candidate selection → Quality vs. quantity tradeoff
   - Constraints → Safety margins and exclusions

### Performance Optimization

For faster processing:
- Reduce image resolution in preprocessing
- Decrease sampling density in candidate selection
- Disable individual fruit visualizations
- Use parallel processing for multiple images

## Technical Details

### Coordinate Systems

- **Image coordinates**: (row, column) with origin at top-left
- **World coordinates**: (x, y, z) in meters, camera-centered
- **Robot coordinates**: Transformed using camera-to-robot calibration

### Depth Processing

- **Units**: All depths normalized to meters
- **Invalid pixels**: Set to 0, handled gracefully throughout pipeline
- **Range**: Configurable via `depth_fg_threshold` and `depth_bg_threshold`

### Memory Usage

Typical memory requirements:
- **640×480 images**: ~50MB per image pair
- **1280×960 images**: ~200MB per image pair
- **Point clouds**: Additional ~100MB per fruit instance

## Validation and Testing

### Test Dataset Preparation

Recommended test cases:
1. **Single fruits**: Isolated apples/tomatoes
2. **Multiple fruits**: Touching and separated instances  
3. **Challenging conditions**: Shadows, reflections, stems visible
4. **Different varieties**: Various colors, sizes, and shapes

### Quality Metrics

Monitor pipeline performance using:
- **Detection rate**: Percentage of fruits successfully segmented
- **Candidate density**: Average candidates per fruit
- **Score distribution**: Range and distribution of quality scores
- **Processing time**: Per image and per fruit timing

### Validation Protocol

1. **Manual annotation**: Mark ground-truth label locations
2. **Automated comparison**: Measure distance between candidates and ground truth
3. **Quality assessment**: Evaluate planarity and constraint adherence
4. **Visual inspection**: Review generated visualizations

## Citation and References

This implementation uses standard computer vision techniques:

- **Plane fitting**: SVD-based least squares fitting
- **Watershed segmentation**: Meyer, F. (1994)
- **Bilateral filtering**: Tomasi, C. and Manduchi, R. (1998)
- **RANSAC**: Fischler, M.A. and Bolles, R.C. (1981)

## License and Usage

This code is provided as a Proof of Concept implementation. Adapt and modify as needed for your specific application requirements.

## Support

For issues, questions, or suggestions:
1. Check this documentation first
2. Review the configuration parameters
3. Examine the generated log files
4. Test with simpler input data to isolate problems

---

**Note**: This system is designed for research and development purposes. For production deployment, consider additional validation, error handling, and performance optimization based on your specific requirements.