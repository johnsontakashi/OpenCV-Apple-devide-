#!/usr/bin/env python3
"""
Real-time Inference Engine for Fruit Label Placement
High-performance system supporting GPU acceleration and asynchronous processing.
"""

import asyncio
import threading
import queue
import time
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from abc import ABC, abstractmethod
import json
import traceback

# GPU/Acceleration imports
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """High-speed detection result."""
    boxes: np.ndarray  # [N, 4] in xyxy format
    scores: np.ndarray  # [N,] confidence scores
    classes: np.ndarray  # [N,] class indices
    processing_time: float
    frame_id: int


@dataclass
class LabelPlacement:
    """Computed label placement."""
    fruit_id: int
    center_x: int
    center_y: int
    confidence: float
    planarity_score: float
    world_coords: Tuple[float, float, float]
    processing_time: float


@dataclass
class FrameData:
    """Complete frame processing data."""
    frame_id: int
    timestamp: float
    color_image: np.ndarray
    depth_image: Optional[np.ndarray]
    detections: Optional[DetectionResult] = None
    placements: List[LabelPlacement] = field(default_factory=list)
    total_processing_time: float = 0.0
    error: Optional[str] = None


class PerformanceMonitor:
    """Real-time performance monitoring and safeguards."""
    
    def __init__(self, max_queue_size: int = 100, warning_threshold: float = 0.8):
        self.max_queue_size = max_queue_size
        self.warning_threshold = warning_threshold
        self.fps_history = []
        self.processing_times = []
        self.frame_drops = 0
        self.total_frames = 0
        self.last_report_time = time.time()
        
    def record_frame(self, processing_time: float, dropped: bool = False):
        """Record frame processing metrics."""
        current_time = time.time()
        self.total_frames += 1
        
        if dropped:
            self.frame_drops += 1
        else:
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
        
        # Calculate FPS
        if len(self.processing_times) > 0:
            fps = 1.0 / np.mean(self.processing_times[-10:])
            self.fps_history.append(fps)
            if len(self.fps_history) > 50:
                self.fps_history = self.fps_history[-50:]
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.processing_times:
            return {"fps": 0, "avg_time": 0, "drop_rate": 0}
            
        current_fps = np.mean(self.fps_history[-10:]) if self.fps_history else 0
        avg_time = np.mean(self.processing_times[-10:]) if self.processing_times else 0
        drop_rate = self.frame_drops / max(self.total_frames, 1)
        
        return {
            "fps": current_fps,
            "avg_processing_time": avg_time,
            "drop_rate": drop_rate,
            "queue_utilization": 0,  # Will be set by queue manager
            "frames_processed": self.total_frames - self.frame_drops,
            "frames_dropped": self.frame_drops
        }
    
    def should_drop_frame(self, queue_size: int) -> bool:
        """Determine if frame should be dropped to maintain performance."""
        utilization = queue_size / self.max_queue_size
        return utilization > self.warning_threshold


class BaseDetector(ABC):
    """Abstract base class for fruit detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect fruits in image."""
        pass
    
    @abstractmethod
    def warmup(self):
        """Warm up the detector."""
        pass


class YOLOv8Detector(BaseDetector):
    """YOLOv8-Nano high-speed detector."""
    
    def __init__(self, model_path: str = None, device: str = "auto", img_size: int = 416):
        self.model_path = model_path
        self.img_size = img_size
        self.device = self._get_device(device)
        self.model = None
        self._setup_model()
        
    def _get_device(self, device: str) -> str:
        """Determine best available device."""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _setup_model(self):
        """Setup YOLOv8 model."""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
                
            # For demo purposes, create a mock detector
            # In real implementation, load actual YOLOv8 model
            logger.info(f"Setting up YOLOv8-Nano detector on {self.device}")
            
            # Mock model for demonstration
            self.model = self._create_mock_detector()
            self.warmup()
            
        except Exception as e:
            logger.error(f"Failed to setup YOLOv8 detector: {e}")
            raise
    
    def _create_mock_detector(self):
        """Create mock detector for demonstration."""
        class MockYOLO:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, x):
                # Simulate detection processing time
                time.sleep(0.005)  # 5ms processing time
                
                # Return mock detections (simulate 2-4 fruits)
                batch_size = x.shape[0] if len(x.shape) == 4 else 1
                num_detections = np.random.randint(2, 5)
                
                # Random fruit detections
                boxes = []
                scores = []
                classes = []
                
                for _ in range(num_detections):
                    # Random box coordinates (normalized)
                    cx, cy = np.random.uniform(0.2, 0.8, 2)
                    w, h = np.random.uniform(0.05, 0.15, 2)
                    
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(np.random.uniform(0.7, 0.95))
                    classes.append(0)  # Fruit class
                
                return {
                    'boxes': np.array(boxes),
                    'scores': np.array(scores),
                    'classes': np.array(classes)
                }
        
        return MockYOLO(self.device)
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect fruits in image."""
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_img = self._preprocess_image(image)
            
            # Run inference
            results = self.model(processed_img)
            
            # Post-process results
            boxes = self._denormalize_boxes(results['boxes'], image.shape[:2])
            scores = results['scores']
            classes = results['classes']
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                boxes=boxes,
                scores=scores,
                classes=classes,
                processing_time=processing_time,
                frame_id=int(time.time() * 1000) % 100000
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionResult(
                boxes=np.array([]),
                scores=np.array([]),
                classes=np.array([]),
                processing_time=time.time() - start_time,
                frame_id=int(time.time() * 1000) % 100000
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Fast image preprocessing."""
        # Resize to model input size
        resized = cv2.resize(image, (self.img_size, self.img_size))
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def _denormalize_boxes(self, boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Convert normalized boxes to pixel coordinates."""
        if len(boxes) == 0:
            return boxes
            
        h, w = img_shape
        boxes_denorm = boxes.copy()
        boxes_denorm[:, [0, 2]] *= w  # x coordinates
        boxes_denorm[:, [1, 3]] *= h  # y coordinates
        return boxes_denorm.astype(int)
    
    def warmup(self):
        """Warm up the detector."""
        logger.info("Warming up YOLOv8 detector...")
        dummy_image = np.random.uint8(np.random.rand(640, 640, 3) * 255)
        for _ in range(3):
            self.detect(dummy_image)
        logger.info("YOLOv8 detector warmed up")


class ONNXDetector(BaseDetector):
    """ONNX Runtime optimized detector."""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        self.model_path = model_path
        self.providers = providers or self._get_best_providers()
        self.session = None
        self._setup_model()
    
    def _get_best_providers(self) -> List[str]:
        """Get best available ONNX providers."""
        if not ONNX_AVAILABLE:
            return ['CPUExecutionProvider']
            
        available = ort.get_available_providers()
        preferred = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        return [p for p in preferred if p in available]
    
    def _setup_model(self):
        """Setup ONNX session."""
        try:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime not available")
                
            # For demo, create mock session
            logger.info(f"Setting up ONNX detector with providers: {self.providers}")
            self.session = self._create_mock_session()
            self.warmup()
            
        except Exception as e:
            logger.error(f"Failed to setup ONNX detector: {e}")
            raise
    
    def _create_mock_session(self):
        """Create mock ONNX session."""
        class MockSession:
            def run(self, output_names, input_feed):
                # Simulate ONNX inference
                time.sleep(0.003)  # 3ms processing time
                
                # Return mock outputs
                num_detections = np.random.randint(1, 6)
                boxes = np.random.rand(num_detections, 4)
                scores = np.random.uniform(0.6, 0.95, num_detections)
                classes = np.zeros(num_detections)
                
                return [boxes, scores, classes]
        
        return MockSession()
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect using ONNX model."""
        start_time = time.time()
        
        try:
            # Preprocess
            input_tensor = self._preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(
                ['boxes', 'scores', 'classes'],
                {'input': input_tensor}
            )
            
            boxes, scores, classes = outputs
            boxes = self._denormalize_boxes(boxes, image.shape[:2])
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                boxes=boxes,
                scores=scores,
                classes=classes,
                processing_time=processing_time,
                frame_id=int(time.time() * 1000) % 100000
            )
            
        except Exception as e:
            logger.error(f"ONNX detection failed: {e}")
            return DetectionResult(
                boxes=np.array([]),
                scores=np.array([]),
                classes=np.array([]),
                processing_time=time.time() - start_time,
                frame_id=int(time.time() * 1000) % 100000
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess for ONNX model."""
        resized = cv2.resize(image, (416, 416))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized.transpose(2, 0, 1), 0)
    
    def _denormalize_boxes(self, boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Convert boxes to pixel coordinates."""
        if len(boxes) == 0:
            return boxes
            
        h, w = img_shape
        boxes_denorm = boxes.copy()
        boxes_denorm[:, [0, 2]] *= w
        boxes_denorm[:, [1, 3]] *= h
        return boxes_denorm.astype(int)
    
    def warmup(self):
        """Warm up ONNX detector."""
        logger.info("Warming up ONNX detector...")
        dummy_image = np.random.uint8(np.random.rand(640, 640, 3) * 255)
        for _ in range(3):
            self.detect(dummy_image)
        logger.info("ONNX detector warmed up")


class FastLabelPlacement:
    """Ultra-fast label placement computation."""
    
    def __init__(self, camera_params: Dict):
        self.fx = camera_params.get('fx', 640.0)
        self.fy = camera_params.get('fy', 640.0)
        self.cx = camera_params.get('cx', 320.0)
        self.cy = camera_params.get('cy', 240.0)
        
        # Pre-computed masks for speed
        self._setup_fast_masks()
    
    def _setup_fast_masks(self):
        """Pre-compute masks for fast processing."""
        # Create circular masks for different sizes
        self.circle_masks = {}
        for radius in [10, 15, 20, 25, 30]:
            mask = np.zeros((radius*2+1, radius*2+1), dtype=np.uint8)
            cv2.circle(mask, (radius, radius), radius, 1, -1)
            self.circle_masks[radius] = mask
    
    def compute_placements(self, detections: DetectionResult, 
                          color_image: np.ndarray, 
                          depth_image: Optional[np.ndarray] = None) -> List[LabelPlacement]:
        """Compute label placements immediately after detection."""
        start_time = time.time()
        placements = []
        
        if len(detections.boxes) == 0:
            return placements
        
        try:
            for i, (box, score) in enumerate(zip(detections.boxes, detections.scores)):
                if score < 0.5:  # Skip low confidence detections
                    continue
                
                placement = self._compute_single_placement(
                    box, i, color_image, depth_image, score
                )
                
                if placement:
                    placements.append(placement)
        
        except Exception as e:
            logger.error(f"Placement computation failed: {e}")
        
        processing_time = time.time() - start_time
        
        # Update processing times
        for placement in placements:
            placement.processing_time = processing_time / max(len(placements), 1)
        
        return placements
    
    def _compute_single_placement(self, box: np.ndarray, fruit_id: int,
                                 color_image: np.ndarray, depth_image: Optional[np.ndarray],
                                 confidence: float) -> Optional[LabelPlacement]:
        """Compute placement for single fruit."""
        try:
            x1, y1, x2, y2 = box.astype(int)
            
            # Crop fruit region
            fruit_region = color_image[y1:y2, x1:x2]
            if fruit_region.size == 0:
                return None
            
            # Fast center finding
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Fast planarity estimation
            planarity_score = self._estimate_planarity_fast(
                fruit_region, depth_image[y1:y2, x1:x2] if depth_image is not None else None
            )
            
            # Convert to world coordinates
            depth_value = 0.5  # Default depth
            if depth_image is not None:
                depth_value = depth_image[center_y, center_x] / 1000.0  # Convert mm to m
            
            world_coords = self._pixel_to_world(center_x, center_y, depth_value)
            
            return LabelPlacement(
                fruit_id=fruit_id,
                center_x=center_x,
                center_y=center_y,
                confidence=confidence,
                planarity_score=planarity_score,
                world_coords=world_coords,
                processing_time=0.0  # Will be set later
            )
            
        except Exception as e:
            logger.warning(f"Failed to compute placement for fruit {fruit_id}: {e}")
            return None
    
    def _estimate_planarity_fast(self, color_region: np.ndarray, 
                                depth_region: Optional[np.ndarray]) -> float:
        """Fast planarity estimation using color variance."""
        if depth_region is not None and depth_region.size > 0:
            # Use depth variance as planarity measure
            valid_depths = depth_region[depth_region > 0]
            if len(valid_depths) > 5:
                return max(0.0, 1.0 - np.std(valid_depths) / 50.0)
        
        # Fall back to color uniformity
        if color_region.size == 0:
            return 0.5
            
        color_std = np.mean(np.std(color_region.reshape(-1, 3), axis=0))
        return max(0.0, 1.0 - color_std / 100.0)
    
    def _pixel_to_world(self, x: int, y: int, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates."""
        if depth <= 0:
            return (0.0, 0.0, 0.0)
        
        world_x = (x - self.cx) * depth / self.fx
        world_y = (y - self.cy) * depth / self.fy
        world_z = depth
        
        return (world_x, world_y, world_z)


class RealTimeInferenceEngine:
    """Main real-time inference engine with asynchronous processing."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detector = None
        self.placement_computer = None
        self.performance_monitor = PerformanceMonitor(
            max_queue_size=config.get('max_queue_size', 50),
            warning_threshold=config.get('warning_threshold', 0.8)
        )
        
        # Async processing components
        self.input_queue = asyncio.Queue(maxsize=config.get('max_queue_size', 50))
        self.output_queue = asyncio.Queue(maxsize=100)
        self.processing_active = False
        self.worker_tasks = []
        
        # Thread pools for CPU intensive work
        self.detection_executor = ThreadPoolExecutor(max_workers=2)
        self.placement_executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.frame_id_counter = 0
        self.last_metrics_time = time.time()
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup detection and placement components."""
        try:
            # Setup detector based on config
            detector_type = self.config.get('detector_type', 'yolov8')
            
            if detector_type == 'yolov8':
                try:
                    self.detector = YOLOv8Detector(
                        device=self.config.get('device', 'auto'),
                        img_size=self.config.get('img_size', 416)
                    )
                except ImportError:
                    logger.warning("PyTorch not available, falling back to CPU detector")
                    from cpu_detector import FastCPUDetector
                    self.detector = FastCPUDetector(target_fps=30)
            elif detector_type == 'onnx':
                try:
                    self.detector = ONNXDetector(
                        model_path=self.config.get('model_path', 'model.onnx')
                    )
                except ImportError:
                    logger.warning("ONNX Runtime not available, falling back to CPU detector")
                    from cpu_detector import FastCPUDetector
                    self.detector = FastCPUDetector(target_fps=30)
            elif detector_type == 'cpu':
                from cpu_detector import FastCPUDetector
                self.detector = FastCPUDetector(target_fps=30)
            else:
                logger.warning(f"Unknown detector type: {detector_type}, using CPU fallback")
                from cpu_detector import FastCPUDetector
                self.detector = FastCPUDetector(target_fps=30)
            
            # Setup label placement computer
            camera_params = self.config.get('camera', {})
            self.placement_computer = FastLabelPlacement(camera_params)
            
            logger.info(f"Inference engine initialized with {detector_type} detector")
            
        except Exception as e:
            logger.error(f"Failed to setup inference engine: {e}")
            raise
    
    async def start_processing(self):
        """Start asynchronous processing."""
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start worker tasks
        num_workers = self.config.get('num_workers', 2)
        for i in range(num_workers):
            task = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started {num_workers} processing workers")
    
    async def stop_processing(self):
        """Stop asynchronous processing."""
        if not self.processing_active:
            return
        
        self.processing_active = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        # Cleanup executors
        self.detection_executor.shutdown(wait=True)
        self.placement_executor.shutdown(wait=True)
        
        logger.info("Processing stopped")
    
    async def process_frame_async(self, color_image: np.ndarray, 
                                 depth_image: Optional[np.ndarray] = None) -> bool:
        """Submit frame for asynchronous processing."""
        frame_id = self.frame_id_counter
        self.frame_id_counter += 1
        
        frame_data = FrameData(
            frame_id=frame_id,
            timestamp=time.time(),
            color_image=color_image,
            depth_image=depth_image
        )
        
        try:
            # Check if we should drop frame to maintain performance
            queue_size = self.input_queue.qsize()
            if self.performance_monitor.should_drop_frame(queue_size):
                self.performance_monitor.record_frame(0.0, dropped=True)
                logger.debug(f"Dropped frame {frame_id} due to high load")
                return False
            
            # Non-blocking put
            self.input_queue.put_nowait(frame_data)
            return True
            
        except asyncio.QueueFull:
            self.performance_monitor.record_frame(0.0, dropped=True)
            logger.warning(f"Input queue full, dropped frame {frame_id}")
            return False
    
    async def get_results_async(self, timeout: float = 0.1) -> Optional[FrameData]:
        """Get processing results asynchronously."""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def _processing_worker(self, worker_name: str):
        """Asynchronous processing worker."""
        logger.info(f"Started processing worker: {worker_name}")
        
        while self.processing_active:
            try:
                # Get frame from input queue
                frame_data = await asyncio.wait_for(
                    self.input_queue.get(), timeout=1.0
                )
                
                # Process frame in thread pool to avoid blocking
                processed_frame = await self._process_frame_in_executor(frame_data)
                
                # Put result in output queue
                try:
                    self.output_queue.put_nowait(processed_frame)
                except asyncio.QueueFull:
                    logger.warning("Output queue full, dropping processed frame")
                
                # Record performance
                self.performance_monitor.record_frame(
                    processed_frame.total_processing_time
                )
                
            except asyncio.TimeoutError:
                continue  # No frame available, keep waiting
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue
    
    async def _process_frame_in_executor(self, frame_data: FrameData) -> FrameData:
        """Process frame in thread executor."""
        loop = asyncio.get_event_loop()
        
        try:
            start_time = time.time()
            
            # Run detection in executor
            detections = await loop.run_in_executor(
                self.detection_executor,
                self.detector.detect,
                frame_data.color_image
            )
            
            frame_data.detections = detections
            
            # Run placement computation in executor
            placements = await loop.run_in_executor(
                self.placement_executor,
                self.placement_computer.compute_placements,
                detections,
                frame_data.color_image,
                frame_data.depth_image
            )
            
            frame_data.placements = placements
            frame_data.total_processing_time = time.time() - start_time
            
        except Exception as e:
            frame_data.error = str(e)
            frame_data.total_processing_time = time.time() - start_time
            logger.error(f"Frame processing error: {e}")
        
        return frame_data
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        metrics = self.performance_monitor.get_metrics()
        metrics.update({
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "processing_active": self.processing_active,
            "detector_type": self.config.get('detector_type', 'unknown')
        })
        
        # Update queue utilization
        max_queue = self.config.get('max_queue_size', 50)
        metrics["queue_utilization"] = self.input_queue.qsize() / max_queue
        
        return metrics
    
    def create_visualization(self, frame_data: FrameData) -> np.ndarray:
        """Create visualization of results."""
        vis_image = frame_data.color_image.copy()
        
        if frame_data.error:
            cv2.putText(vis_image, f"Error: {frame_data.error}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis_image
        
        # Draw detections
        if frame_data.detections and len(frame_data.detections.boxes) > 0:
            for box, score in zip(frame_data.detections.boxes, frame_data.detections.scores):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_image, f"{score:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw label placements
        for placement in frame_data.placements:
            # Draw placement center
            cv2.circle(vis_image, (placement.center_x, placement.center_y), 
                      8, (255, 0, 0), -1)
            
            # Draw confidence and planarity
            text = f"C:{placement.confidence:.2f} P:{placement.planarity_score:.2f}"
            cv2.putText(vis_image, text, (placement.center_x-30, placement.center_y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw performance info
        metrics = self.get_performance_metrics()
        cv2.putText(vis_image, f"FPS: {metrics['fps']:.1f} | Time: {metrics['avg_processing_time']:.3f}s", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Fruits: {len(frame_data.placements)} | Queue: {metrics['input_queue_size']}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image


# Configuration presets
INFERENCE_CONFIGS = {
    "ultra_fast": {
        "detector_type": "cpu",
        "device": "cpu",
        "img_size": 320,
        "max_queue_size": 30,
        "warning_threshold": 0.7,
        "num_workers": 1,
        "camera": {"fx": 320, "fy": 320, "cx": 160, "cy": 160}
    },
    "balanced": {
        "detector_type": "yolov8", 
        "device": "cuda",
        "img_size": 416,
        "max_queue_size": 50,
        "warning_threshold": 0.8,
        "num_workers": 2,
        "camera": {"fx": 640, "fy": 640, "cx": 320, "cy": 240}
    },
    "high_accuracy": {
        "detector_type": "onnx",
        "model_path": "yolov8s.onnx",
        "device": "cuda",
        "img_size": 640,
        "max_queue_size": 20,
        "warning_threshold": 0.9,
        "num_workers": 3,
        "camera": {"fx": 640, "fy": 640, "cx": 320, "cy": 240}
    },
    "cpu_optimized": {
        "detector_type": "cpu",
        "device": "cpu",
        "img_size": 320,
        "max_queue_size": 15,
        "warning_threshold": 0.8,
        "num_workers": 2,
        "camera": {"fx": 320, "fy": 320, "cx": 160, "cy": 160}
    }
}


async def main():
    """Example usage of the inference engine."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Choose config based on available hardware
    config_name = "cpu_optimized"
    if TORCH_AVAILABLE and torch.cuda.is_available():
        config_name = "balanced"
    elif ONNX_AVAILABLE:
        config_name = "ultra_fast"
    
    config = INFERENCE_CONFIGS[config_name]
    logger.info(f"Using config: {config_name}")
    
    # Create inference engine
    engine = RealTimeInferenceEngine(config)
    
    try:
        # Start processing
        await engine.start_processing()
        
        # Simulate frame processing
        for i in range(50):
            # Create dummy frame
            color_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            depth_frame = (np.random.rand(480, 640) * 1000).astype(np.uint16)
            
            # Add some colored regions to simulate fruits
            if i % 10 == 0:  # Add fruits every 10 frames
                cv2.circle(color_frame, (200, 200), 50, (200, 50, 50), -1)
                cv2.circle(color_frame, (400, 300), 40, (255, 150, 50), -1)
            
            # Submit for processing
            submitted = await engine.process_frame_async(color_frame, depth_frame)
            if submitted:
                logger.info(f"Submitted frame {i}")
            
            # Get results
            result = await engine.get_results_async(timeout=0.1)
            if result:
                logger.info(f"Processed frame {result.frame_id}: "
                          f"{len(result.placements)} placements in "
                          f"{result.total_processing_time:.3f}s")
            
            # Print metrics every 10 frames
            if i % 10 == 0:
                metrics = engine.get_performance_metrics()
                logger.info(f"Performance: {metrics['fps']:.1f} FPS, "
                          f"{metrics['drop_rate']:.1%} drop rate")
            
            # Simulate real-time frame rate
            await asyncio.sleep(1/30)  # 30 FPS input
        
    finally:
        await engine.stop_processing()
        logger.info("Inference engine demo completed")


if __name__ == "__main__":
    asyncio.run(main())