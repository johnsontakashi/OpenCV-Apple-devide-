#!/usr/bin/env python3
"""
CPU-only detector fallback for systems without GPU acceleration.
Uses OpenCV DNN module with pre-trained models.
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result structure."""
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    processing_time: float
    frame_id: int


class OpenCVDetector:
    """OpenCV DNN-based fruit detector for CPU inference."""
    
    def __init__(self, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.output_layers = None
        self.input_size = (416, 416)
        self._setup_model()
    
    def _setup_model(self):
        """Setup OpenCV DNN model."""
        try:
            # For demo purposes, we'll use a simple blob detection
            # In production, you would load a proper ONNX model here
            logger.info("Setting up OpenCV CPU detector (demo mode)")
            
            # Mock model setup - in real implementation, load ONNX model:
            # self.net = cv2.dnn.readNetFromONNX("fruit_detector.onnx")
            # self.output_layers = self.net.getUnconnectedOutLayersNames()
            
            self._create_simple_detector()
            self.warmup()
            
        except Exception as e:
            logger.error(f"Failed to setup OpenCV detector: {e}")
            raise
    
    def _create_simple_detector(self):
        """Create simple color-based detector for demo."""
        # This is a fallback detector using color segmentation
        # In production, replace with proper ONNX model loading
        self.detector_type = "color_based"
        logger.info("Using simple color-based detector for demonstration")
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect fruits using OpenCV."""
        start_time = time.time()
        frame_id = int(time.time() * 1000) % 100000
        
        try:
            # Use simple color-based detection for demo
            boxes, scores, classes = self._detect_by_color(image)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                boxes=boxes,
                scores=scores,
                classes=classes,
                processing_time=processing_time,
                frame_id=frame_id
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionResult(
                boxes=np.array([]),
                scores=np.array([]),
                classes=np.array([]),
                processing_time=time.time() - start_time,
                frame_id=frame_id
            )
    
    def _detect_by_color(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple color-based fruit detection."""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for fruits
        color_ranges = [
            # Red apples
            ([0, 50, 50], [10, 255, 255]),
            ([160, 50, 50], [180, 255, 255]),
            # Orange fruits
            ([5, 50, 50], [25, 255, 255]),
            # Yellow fruits
            ([15, 50, 50], [35, 255, 255])
        ]
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for i, (lower, upper) in enumerate(color_ranges):
            # Create mask for this color range
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # Filter small areas
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and shape
                aspect_ratio = w / h
                area_ratio = area / (w * h)
                
                confidence = min(0.95, area_ratio * 0.8 + 0.3)
                if aspect_ratio > 0.5 and aspect_ratio < 2.0:  # Reasonable aspect ratio
                    confidence += 0.1
                
                if confidence > self.confidence_threshold:
                    all_boxes.append([x, y, x + w, y + h])
                    all_scores.append(confidence)
                    all_classes.append(0)  # All fruits are class 0
        
        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to numpy arrays
        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)
        classes = np.array(all_classes, dtype=np.int32)
        
        # Apply NMS to remove overlapping detections
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.confidence_threshold, self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            scores = scores[indices]
            classes = classes[indices]
        
        return boxes, scores, classes
    
    def warmup(self):
        """Warm up the detector."""
        logger.info("Warming up OpenCV detector...")
        dummy_image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        for _ in range(3):
            self.detect(dummy_image)
        logger.info("OpenCV detector warmed up")


class FastCPUDetector:
    """Ultra-fast CPU detector for real-time processing."""
    
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.max_processing_time = 1.0 / target_fps
        self.detector = OpenCVDetector()
        
        # Performance optimization settings
        self.resize_factor = 0.5  # Process at half resolution for speed
        self.max_detections = 10  # Limit number of detections
        
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Fast detection with time limits."""
        start_time = time.time()
        
        # Resize image for faster processing
        h, w = image.shape[:2]
        new_h, new_w = int(h * self.resize_factor), int(w * self.resize_factor)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Run detection on resized image
        result = self.detector.detect(resized)
        
        # Scale boxes back to original size
        if len(result.boxes) > 0:
            scale_x = w / new_w
            scale_y = h / new_h
            result.boxes[:, [0, 2]] *= scale_x
            result.boxes[:, [1, 3]] *= scale_y
        
        # Limit number of detections for performance
        if len(result.boxes) > self.max_detections:
            # Keep highest scoring detections
            indices = np.argsort(result.scores)[::-1][:self.max_detections]
            result.boxes = result.boxes[indices]
            result.scores = result.scores[indices]
            result.classes = result.classes[indices]
        
        result.processing_time = time.time() - start_time
        return result


if __name__ == "__main__":
    # Test the CPU detector
    import matplotlib.pyplot as plt
    
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = FastCPUDetector(target_fps=30)
    
    # Test with random image
    test_image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    
    # Add some colored regions to simulate fruits
    # Red apple
    cv2.circle(test_image, (200, 200), 50, (200, 50, 50), -1)
    # Orange
    cv2.circle(test_image, (400, 300), 40, (255, 150, 50), -1)
    # Yellow apple  
    cv2.circle(test_image, (500, 150), 45, (255, 255, 100), -1)
    
    # Run detection
    print("Running CPU detection test...")
    result = detector.detect(test_image)
    
    print(f"Detection results:")
    print(f"  Processing time: {result.processing_time:.3f}s")
    print(f"  FPS estimate: {1.0/result.processing_time:.1f}")
    print(f"  Detections: {len(result.boxes)}")
    print(f"  Scores: {result.scores}")
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    times = []
    for i in range(20):
        start = time.time()
        result = detector.detect(test_image)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Benchmark results:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Average FPS: {fps:.1f}")
    print(f"  Min time: {np.min(times):.3f}s")
    print(f"  Max time: {np.max(times):.3f}s")
    
    if fps >= 15:
        print("✅ CPU detector meets real-time requirements (15+ FPS)")
    else:
        print("❌ CPU detector too slow for real-time processing")