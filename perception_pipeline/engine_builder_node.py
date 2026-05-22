import os
import sys
import rclpy
from rclpy.node import Node
import tensorrt as trt

# TensorRT Logger matched to your existing pipeline settings
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class EngineBuilderNode(Node):
    def __init__(self):
        super().__init__('engine_builder_node')
        
        # Consistent directory paths for your T4 environment
        self.onnx_dir = "./"  # Where your .onnx files sit
        self.engine_dir = "./trt_engine_cache"
        os.makedirs(self.engine_dir, exist_ok=True)

        # List of files to process sequentially
        self.models_to_build = [
            ("stereonet_fp16_720p.onnx", "stereonet.engine"),
            ("front_seg_fp16_720p.onnx", "seg_front.engine"),
            ("rear_depth_fp16.onnx", "rear_depth.engine"),
            ("rear_seg_fp16.onnx", "rear_seg.engine"),
            ("side_depth_fp16.onnx", "side_depth.engine"),
            ("side_segmentation_unified_fp16.onnx", "side_seg.engine")
        ]

    def build_all(self):
        self.get_logger().info("🚀 Starting Standalone TensorRT Pre-Compilation Node...")
        
        for onnx_file, engine_file in self.models_to_build:
            onnx_path = os.path.join(self.onnx_dir, onnx_file)
            engine_path = os.path.join(self.engine_dir, engine_file)
            
            if os.path.exists(engine_path):
                self.get_logger().info(f"✓ {engine_file} cache already exists. Skipping.")
                continue
                
            if not os.path.exists(onnx_path):
                self.get_logger().error(f"❌ Missing source file: {onnx_path}")
                continue

            self.get_logger().info(f"⚙️ Compiling: {onnx_file} -> {engine_file}")
            
            # Executing your identical compilation pipeline mechanics
            self._build_engine(onnx_path, engine_path, engine_file.replace(".engine", ""))

        self.get_logger().info("🎉 All engines successfully generated inside the cache directory.")

    def _build_engine(self, onnx_path, engine_path, engine_name):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    sys.stderr.write(f"ONNX Parse Error: {parser.get_error(error)}\n")
                raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)  
        config.builder_optimization_level = 3  
        
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            sys.stdout.write("   → FP16 ENABLED! Tensor Cores will be used.\n")
            sys.stdout.flush()
            
        # ---------------------------------------------------------
        # THE FIX: DISABLE BUGGY JIT COMPILATION FOR THIS ENGINE
        # ---------------------------------------------------------
        if hasattr(trt.TacticSource, 'JIT_CONVOLUTIONS'):
            tactics = config.get_tactic_sources()
            tactics &= ~(1 << int(trt.TacticSource.JIT_CONVOLUTIONS))
            config.set_tactic_sources(tactics)
            sys.stdout.write("   → Disabled JIT Convolutions to bypass FP16 compiler bug.\n")
            sys.stdout.flush()
        # ---------------------------------------------------------

        sys.stdout.write("   ⏳ COMPILING FP16 ENGINE... PLEASE DO NOT CLOSE (Takes 2-5 mins)...\n")
        sys.stdout.flush()
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            sys.stdout.write("   ⚠️ FP16 failed again. Forcing safe FP32...\n")
            sys.stdout.flush()
            config.clear_flag(trt.BuilderFlag.FP16)
            serialized_engine = builder.build_serialized_network(network, config)
            
        if serialized_engine is None:
            raise RuntimeError(f"Failed to build TensorRT engine: {onnx_path}")
            
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

def main(args=None):
    rclpy.init(args=args)
    builder_node = EngineBuilderNode()
    try:
        builder_node.build_all()
    except Exception as e:
        builder_node.get_logger().error(f"Engine Compilation Interrupted: {e}")
    finally:
        builder_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()