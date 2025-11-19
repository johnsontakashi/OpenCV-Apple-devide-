#!/usr/bin/env python3
"""
Minimal test script to verify pipeline components work.
Run this to check if your environment is set up correctly.
"""

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV not found. Install with: pip3 install opencv-python")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ NumPy not found. Install with: pip3 install numpy")
        return False
        
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError:
        print("✗ PyYAML not found. Install with: pip3 install PyYAML")
        return False
        
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError:
        print("✗ Matplotlib not found. Install with: pip3 install matplotlib")
        return False
        
    try:
        import open3d as o3d
        print("✓ Open3D imported successfully")
    except ImportError:
        print("✗ Open3D not found. Install with: pip3 install open3d")
        return False
        
    return True

def test_config():
    """Test if config file can be loaded."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
        print(f"  - Output directory: {config.get('output_dir', 'Not set')}")
        print(f"  - Camera fx: {config.get('camera', {}).get('camera_fx', 'Not set')}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_modules():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    modules = [
        'preprocess', 'segmentation', 'pointcloud', 
        'planarity', 'candidate_selector', 'constraints', 
        'visualize'
    ]
    
    for module_name in modules:
        try:
            exec(f"import {module_name}")
            print(f"✓ {module_name}.py imported successfully")
        except Exception as e:
            print(f"✗ {module_name}.py failed: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("=== Pipeline Environment Test ===\n")
    
    imports_ok = test_imports()
    config_ok = test_config()
    modules_ok = test_modules()
    
    print("\n=== Test Summary ===")
    if imports_ok and config_ok and modules_ok:
        print("✓ All tests passed! Pipeline should work.")
        print("\nNext steps:")
        print("1. Create input data directory with RGB-D image pairs")
        print("2. Run: python3 main.py --input your_data --config config.yaml")
    else:
        print("✗ Some tests failed. Fix the issues above before running the pipeline.")

if __name__ == "__main__":
    main()