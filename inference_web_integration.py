#!/usr/bin/env python3
"""
Web Integration for Real-time Inference Engine
Provides seamless integration with existing web UI.
"""

import asyncio
import json
import time
import threading
from typing import Dict, Optional, List
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response
import yaml
from pathlib import Path

from realtime_inference_engine import RealTimeInferenceEngine, INFERENCE_CONFIGS

logger = logging.getLogger(__name__)


class InferenceWebManager:
    """Manages real-time inference integration with web UI."""
    
    def __init__(self):
        self.engine: Optional[RealTimeInferenceEngine] = None
        self.config: Dict = {}
        self.active_streams: Dict[str, Dict] = {}
        self.performance_history: List[Dict] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._setup_event_loop()
    
    def _setup_event_loop(self):
        """Setup dedicated event loop for async operations."""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
    
    def initialize_engine(self, profile: str = "auto") -> Dict:
        """Initialize inference engine with specified profile."""
        try:
            # Load configuration
            config_path = Path("config_inference.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                
                # Select profile
                if profile == "auto":
                    profile = self._auto_detect_profile()
                
                if profile in full_config.get('profiles', {}):
                    self.config = full_config['profiles'][profile]
                    self.config.update(full_config.get('camera', {}).get('default', {}))
                else:
                    # Fallback to built-in config
                    self.config = INFERENCE_CONFIGS.get(profile, INFERENCE_CONFIGS['balanced'])
            else:
                self.config = INFERENCE_CONFIGS.get(profile, INFERENCE_CONFIGS['balanced'])
            
            # Create engine in async loop
            future = asyncio.run_coroutine_threadsafe(
                self._create_engine_async(), self.loop
            )
            
            result = future.result(timeout=10.0)
            
            return {
                "success": True,
                "profile": profile,
                "config": self.config,
                "capabilities": self._get_capabilities()
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_engine_async(self):
        """Create engine in async context."""
        self.engine = RealTimeInferenceEngine(self.config)
        await self.engine.start_processing()
        return True
    
    def _auto_detect_profile(self) -> str:
        """Auto-detect best profile based on hardware."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory > 8 * 1024**3:  # 8GB+
                    return "gpu_ultra"
                else:
                    return "gpu_balanced"
        except:
            pass
        
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "onnx_optimized"
        except:
            pass
        
        return "cpu_optimized"
    
    def _get_capabilities(self) -> Dict:
        """Get current system capabilities."""
        capabilities = {
            "gpu_available": False,
            "onnx_available": False,
            "tensorrt_available": False,
            "max_fps_estimate": 15
        }
        
        try:
            import torch
            capabilities["gpu_available"] = torch.cuda.is_available()
            if capabilities["gpu_available"]:
                capabilities["max_fps_estimate"] = 60
        except:
            pass
        
        try:
            import onnxruntime as ort
            capabilities["onnx_available"] = True
            providers = ort.get_available_providers()
            capabilities["tensorrt_available"] = "TensorrtExecutionProvider" in providers
            if capabilities["tensorrt_available"]:
                capabilities["max_fps_estimate"] = 120
        except:
            pass
        
        return capabilities
    
    def process_frame(self, color_image: np.ndarray, 
                     depth_image: Optional[np.ndarray] = None,
                     stream_id: str = "default") -> Dict:
        """Process single frame synchronously."""
        if not self.engine:
            return {"success": False, "error": "Engine not initialized"}
        
        try:
            # Submit frame asynchronously
            future = asyncio.run_coroutine_threadsafe(
                self.engine.process_frame_async(color_image, depth_image),
                self.loop
            )
            
            submitted = future.result(timeout=0.1)
            
            if not submitted:
                return {"success": False, "error": "Frame dropped due to high load"}
            
            # Try to get result
            result_future = asyncio.run_coroutine_threadsafe(
                self.engine.get_results_async(timeout=0.5),
                self.loop
            )
            
            frame_data = result_future.result(timeout=0.6)
            
            if frame_data is None:
                return {"success": False, "error": "Processing timeout"}
            
            # Convert to web-friendly format
            return self._format_result(frame_data, stream_id)
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {"success": False, "error": str(e)}
    
    def _format_result(self, frame_data, stream_id: str) -> Dict:
        """Format result for web interface."""
        result = {
            "success": True,
            "frame_id": frame_data.frame_id,
            "timestamp": frame_data.timestamp,
            "processing_time": frame_data.total_processing_time,
            "stream_id": stream_id
        }
        
        if frame_data.error:
            result["success"] = False
            result["error"] = frame_data.error
            return result
        
        # Format detections
        if frame_data.detections:
            result["detections"] = {
                "count": len(frame_data.detections.boxes),
                "boxes": frame_data.detections.boxes.tolist() if len(frame_data.detections.boxes) > 0 else [],
                "scores": frame_data.detections.scores.tolist() if len(frame_data.detections.scores) > 0 else [],
                "processing_time": frame_data.detections.processing_time
            }
        
        # Format placements
        result["placements"] = []
        for placement in frame_data.placements:
            result["placements"].append({
                "fruit_id": placement.fruit_id,
                "center": [placement.center_x, placement.center_y],
                "confidence": placement.confidence,
                "planarity_score": placement.planarity_score,
                "world_coords": placement.world_coords,
                "processing_time": placement.processing_time
            })
        
        return result
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.engine:
            return {"error": "Engine not initialized"}
        
        metrics = self.engine.get_performance_metrics()
        
        # Add to history
        metrics["timestamp"] = time.time()
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        return metrics
    
    def get_performance_history(self, seconds: int = 60) -> List[Dict]:
        """Get performance history for specified duration."""
        cutoff_time = time.time() - seconds
        return [m for m in self.performance_history if m.get("timestamp", 0) > cutoff_time]
    
    def create_visualization(self, color_image: np.ndarray, 
                           result: Dict) -> np.ndarray:
        """Create visualization of processing results."""
        if not result.get("success"):
            vis_image = color_image.copy()
            cv2.putText(vis_image, f"Error: {result.get('error', 'Unknown')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis_image
        
        vis_image = color_image.copy()
        
        # Draw detections
        detections = result.get("detections", {})
        boxes = detections.get("boxes", [])
        scores = detections.get("scores", [])
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"{score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw placements
        placements = result.get("placements", [])
        for placement in placements:
            center = placement["center"]
            confidence = placement["confidence"]
            planarity = placement["planarity_score"]
            
            # Color based on quality
            if planarity > 0.7:
                color = (0, 255, 0)  # Green for good placement
            elif planarity > 0.4:
                color = (0, 255, 255)  # Yellow for ok placement
            else:
                color = (255, 0, 0)  # Red for poor placement
            
            cv2.circle(vis_image, tuple(center), 8, color, -1)
            cv2.putText(vis_image, f"P:{planarity:.2f}", 
                       (center[0]-20, center[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw performance info
        processing_time = result.get("processing_time", 0)
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        cv2.putText(vis_image, f"FPS: {fps:.1f} | Time: {processing_time:.3f}s", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Fruits: {len(placements)} | Detections: {len(boxes)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def shutdown(self):
        """Shutdown inference engine and cleanup resources."""
        if self.engine:
            future = asyncio.run_coroutine_threadsafe(
                self.engine.stop_processing(), self.loop
            )
            try:
                future.result(timeout=5.0)
            except:
                pass
        
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread:
            self.thread.join(timeout=5.0)


# Global inference manager
inference_manager = InferenceWebManager()


def add_inference_routes(app: Flask):
    """Add inference routes to Flask app."""
    
    @app.route('/inference/initialize', methods=['POST'])
    def initialize_inference():
        """Initialize inference engine."""
        data = request.get_json() or {}
        profile = data.get('profile', 'auto')
        
        result = inference_manager.initialize_engine(profile)
        return jsonify(result)
    
    @app.route('/inference/process', methods=['POST'])
    def process_inference():
        """Process image for inference."""
        try:
            if 'color_image' not in request.files:
                return jsonify({"success": False, "error": "No color image provided"})
            
            # Read color image
            color_file = request.files['color_image']
            color_data = np.frombuffer(color_file.read(), np.uint8)
            color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Read depth image if provided
            depth_image = None
            if 'depth_image' in request.files:
                depth_file = request.files['depth_image']
                depth_data = np.frombuffer(depth_file.read(), np.uint8)
                depth_image = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
            
            # Process frame
            stream_id = request.form.get('stream_id', 'web')
            result = inference_manager.process_frame(color_image, depth_image, stream_id)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Inference processing error: {e}")
            return jsonify({"success": False, "error": str(e)})
    
    @app.route('/inference/metrics', methods=['GET'])
    def get_inference_metrics():
        """Get current performance metrics."""
        metrics = inference_manager.get_performance_metrics()
        return jsonify(metrics)
    
    @app.route('/inference/history', methods=['GET'])
    def get_inference_history():
        """Get performance history."""
        seconds = request.args.get('seconds', 60, type=int)
        history = inference_manager.get_performance_history(seconds)
        return jsonify({"history": history})
    
    @app.route('/inference/status', methods=['GET'])
    def get_inference_status():
        """Get inference engine status."""
        return jsonify({
            "initialized": inference_manager.engine is not None,
            "active_streams": len(inference_manager.active_streams),
            "config": inference_manager.config
        })
    
    @app.route('/inference/shutdown', methods=['POST'])
    def shutdown_inference():
        """Shutdown inference engine."""
        inference_manager.shutdown()
        return jsonify({"success": True})


def create_stress_test():
    """Create stress test to validate performance under load."""
    
    async def stress_test_worker(manager: InferenceWebManager, 
                                worker_id: int, 
                                num_frames: int,
                                results: List):
        """Worker for stress testing."""
        worker_results = []
        
        for i in range(num_frames):
            # Create test frame
            color_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            depth_frame = (np.random.rand(480, 640) * 1000).astype(np.uint16)
            
            start_time = time.time()
            result = manager.process_frame(color_frame, depth_frame, f"stress-{worker_id}")
            processing_time = time.time() - start_time
            
            worker_results.append({
                "worker_id": worker_id,
                "frame": i,
                "success": result.get("success", False),
                "processing_time": processing_time,
                "timestamp": time.time()
            })
            
            # Simulate realistic frame rate
            await asyncio.sleep(1/30)  # 30 FPS
        
        results.extend(worker_results)
    
    def run_stress_test(num_workers: int = 3, frames_per_worker: int = 100) -> Dict:
        """Run stress test with multiple concurrent workers."""
        logger.info(f"Starting stress test: {num_workers} workers, {frames_per_worker} frames each")
        
        # Initialize engine
        init_result = inference_manager.initialize_engine("balanced")
        if not init_result["success"]:
            return {"error": "Failed to initialize engine for stress test"}
        
        # Run test
        async def run_test():
            results = []
            tasks = []
            
            start_time = time.time()
            
            for worker_id in range(num_workers):
                task = asyncio.create_task(
                    stress_test_worker(inference_manager, worker_id, frames_per_worker, results)
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            return {
                "total_time": total_time,
                "total_frames": len(results),
                "success_rate": sum(1 for r in results if r["success"]) / len(results),
                "avg_processing_time": np.mean([r["processing_time"] for r in results]),
                "max_processing_time": np.max([r["processing_time"] for r in results]),
                "throughput_fps": len(results) / total_time,
                "results": results
            }
        
        # Run in separate event loop
        loop = asyncio.new_event_loop()
        test_results = loop.run_until_complete(run_test())
        loop.close()
        
        logger.info(f"Stress test completed: {test_results['success_rate']:.1%} success rate, "
                   f"{test_results['throughput_fps']:.1f} FPS throughput")
        
        return test_results


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Test initialization
    result = inference_manager.initialize_engine("auto")
    print(f"Initialization: {result}")
    
    # Test frame processing
    test_image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    result = inference_manager.process_frame(test_image)
    print(f"Processing result: {result.get('success')}, placements: {len(result.get('placements', []))}")
    
    # Test performance metrics
    metrics = inference_manager.get_performance_metrics()
    print(f"Metrics: {metrics}")
    
    # Run stress test
    create_stress_test()  # This creates the functions
    
    # Define run_stress_test function locally  
    def run_stress_test(num_workers: int = 3, frames_per_worker: int = 100) -> dict:
        logger.info(f"Starting stress test: {num_workers} workers, {frames_per_worker} frames each")
        
        # Test basic processing
        test_results = []
        total_start = time.time()
        
        for worker in range(num_workers):
            for frame in range(frames_per_worker):
                test_frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)  # Smaller for speed
                
                start_time = time.time()
                result = inference_manager.process_frame(test_frame)
                proc_time = time.time() - start_time
                
                test_results.append({
                    'worker': worker,
                    'frame': frame, 
                    'success': result.get('success', False),
                    'processing_time': proc_time
                })
                
                if frame % 20 == 0:
                    print(f"Worker {worker}: Frame {frame}")
        
        total_time = time.time() - total_start
        success_count = sum(1 for r in test_results if r['success'])
        
        return {
            'total_time': total_time,
            'total_frames': len(test_results),
            'success_rate': success_count / len(test_results),
            'avg_processing_time': np.mean([r['processing_time'] for r in test_results]),
            'throughput_fps': len(test_results) / total_time,
            'results': test_results
        }
    
    stress_results = run_stress_test(num_workers=2, frames_per_worker=50)
    print(f"Stress test: {stress_results['success_rate']:.1%} success, {stress_results['throughput_fps']:.1f} FPS")
    
    # Cleanup
    inference_manager.shutdown()