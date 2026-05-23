import os
import sys
import time
import collections
import cv2
import numpy as np
import warnings

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
import message_filters
from cv_bridge import CvBridge

import torch
import tensorrt as trt

warnings.filterwarnings("ignore")

def find_repo_root(start_path: str) -> str:
    current = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current, 'package.xml')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.abspath(os.path.join(start_path, '..', '..'))

# ==========================================
# GPU SELECTION & TENSORRT CONFIG
# ==========================================
TRT_LIB_DIR = "/usr/local/lib/python3.12/dist-packages/tensorrt_libs"
if os.path.exists(TRT_LIB_DIR):
    os.environ["LD_LIBRARY_PATH"] = TRT_LIB_DIR + ":/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
        DEVICE = torch.device("cuda:1")
        torch.cuda.set_device(1)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        DEVICE = torch.device("cuda:0")
        torch.cuda.set_device(0)
else:
    DEVICE = torch.device("cpu")

# Pre-allocate Image Normalization tensors on GPU
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
ENGINE_CACHE_DIR = os.path.join(find_repo_root(os.path.dirname(os.path.realpath(__file__))), "trt_engine_cache")

# ==========================================
# TRT ASYNC RUNNER
# ==========================================
class AsyncTRTRunner:
    def __init__(self, engine_name):
        engine_path = os.path.join(ENGINE_CACHE_DIR, f"{engine_name}.engine")
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=DEVICE)
        self.num_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io)]
        self.outputs = {}
        self.bindings = [None] * self.num_io

    def enqueue(self, *input_tensors):
        with torch.cuda.stream(self.stream):
            input_idx = 0
            for i, name in enumerate(self.tensor_names):
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    t = input_tensors[input_idx]
                    self.context.set_input_shape(name, t.shape)
                    self.bindings[i] = t.data_ptr()
                    input_idx += 1
            
            for i, name in enumerate(self.tensor_names):
                if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                    shape = tuple(self.context.get_tensor_shape(name))
                    if name not in self.outputs or self.outputs[name].shape != shape:
                        dtype = torch.float32 if self.engine.get_tensor_dtype(name) == trt.float32 else torch.float16
                        self.outputs[name] = torch.empty(shape, dtype=dtype, device=DEVICE)
                    self.bindings[i] = self.outputs[name].data_ptr()
            
            if hasattr(self.context, 'execute_async_v3'):
                for i, name in enumerate(self.tensor_names):
                    self.context.set_tensor_address(name, self.bindings[i])
                self.context.execute_async_v3(self.stream.cuda_stream)
            elif hasattr(self.context, 'execute_async_v2'):
                self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
            else:
                self.context.execute_v2(self.bindings)

    def wait_and_get_outputs(self):
        self.stream.synchronize()
        return [self.outputs[n] for n in self.tensor_names if self.engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT]


# ==========================================
# ROS 2 INFERENCE NODE
# ==========================================
class TrtEvaluationNode(Node):
    def __init__(self):
        super().__init__('trt_evaluation_node')
        self.bridge = CvBridge()
        self.get_logger().info("Loading TensorRT Engines from cache...")
        
        self.workers = {
            'front_depth': AsyncTRTRunner('stereonet'), 'front_seg': AsyncTRTRunner('seg_front'),
            'rear_depth': AsyncTRTRunner('rear_depth'), 'rear_seg': AsyncTRTRunner('rear_seg'),
            'side_left_depth': AsyncTRTRunner('side_depth'), 'side_left_seg': AsyncTRTRunner('side_seg'),
            'side_right_depth': AsyncTRTRunner('side_depth'), 'side_right_seg': AsyncTRTRunner('side_seg'),
        }

        self.inference_publishers = {}
        for cam in ['front_left', 'rear', 'side_left', 'side_right']:
            self.inference_publishers[cam] = {
                'depth': self.create_publisher(Image, f'/inference/{cam}/depth', 10),
                'seg': self.create_publisher(Image, f'/inference/{cam}/seg', 10),
            }

        self.frame_counts = {cam: 0 for cam in ['front_left', 'rear', 'side_left', 'side_right']}
        self.last_timestamps = {cam: None for cam in ['front_left', 'rear', 'side_left', 'side_right']}
        self.fps_records = {cam: collections.deque(maxlen=30) for cam in ['front_left', 'rear', 'side_left', 'side_right']}

        self.callback_group = ReentrantCallbackGroup()
        self.sub_fl = message_filters.Subscriber(self, CompressedImage, '/carla/front_left/image/compressed', callback_group=self.callback_group)
        self.sub_fr = message_filters.Subscriber(self, CompressedImage, '/carla/front_right/image/compressed', callback_group=self.callback_group)
        self.sub_r  = message_filters.Subscriber(self, CompressedImage, '/carla/rear/image/compressed', callback_group=self.callback_group)
        self.sub_sl = message_filters.Subscriber(self, CompressedImage, '/carla/side_left/image/compressed', callback_group=self.callback_group)
        self.sub_sr = message_filters.Subscriber(self, CompressedImage, '/carla/side_right/image/compressed', callback_group=self.callback_group)

        self.heartbeat_pub = self.create_publisher(Bool, '/perception/heartbeat', 10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_fl, self.sub_fr, self.sub_r, self.sub_sl, self.sub_sr], queue_size=30, slop=0.15
        )
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info("Ready.")

    def publish_prediction(self, cam_name, depth_arr, seg_arr, header):
        # cv_bridge C++ bindings are much faster than python tobytes()
        depth_msg = self.bridge.cv2_to_imgmsg(depth_arr, encoding='32FC1')
        depth_msg.header = header
        seg_msg = self.bridge.cv2_to_imgmsg(seg_arr, encoding='8UC1')
        seg_msg.header = header
        self.inference_publishers[cam_name]['depth'].publish(depth_msg)
        self.inference_publishers[cam_name]['seg'].publish(seg_msg)

    def preprocess_to_gpu(self, msg):
        """Decompresses JPEG directly into numpy memory without raw serialization block"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        
        t = torch.from_numpy(arr).to(DEVICE, non_blocking=True).float()
        return t.permute(2, 0, 1).unsqueeze(0).div_(255.0).sub_(MEAN).div_(STD)

    def synced_callback(self, msg_fl, msg_fr, msg_r, msg_sl, msg_sr):
        t0 = time.perf_counter()
        try:
            # PRE-PROCESS
            gpu_tensors = {
                'fl': self.preprocess_to_gpu(msg_fl), 'fr': self.preprocess_to_gpu(msg_fr),
                'r':  self.preprocess_to_gpu(msg_r),  'sl': self.preprocess_to_gpu(msg_sl),
                'sr': self.preprocess_to_gpu(msg_sr),
            }
            t1 = time.perf_counter()

            # INFERENCE
            self.workers['front_depth'].enqueue(gpu_tensors['fl'], gpu_tensors['fr'])
            self.workers['front_seg'].enqueue(gpu_tensors['fl'])
            self.workers['rear_depth'].enqueue(gpu_tensors['r'])
            self.workers['rear_seg'].enqueue(gpu_tensors['r'])
            self.workers['side_left_depth'].enqueue(gpu_tensors['sl'])
            self.workers['side_left_seg'].enqueue(gpu_tensors['sl'])
            self.workers['side_right_depth'].enqueue(gpu_tensors['sr'])
            self.workers['side_right_seg'].enqueue(gpu_tensors['sr'])

            res_front_depth = self.workers['front_depth'].wait_and_get_outputs()
            res_front_seg   = self.workers['front_seg'].wait_and_get_outputs()
            res_rear_depth  = self.workers['rear_depth'].wait_and_get_outputs()
            res_rear_seg    = self.workers['rear_seg'].wait_and_get_outputs()
            res_sl_depth    = self.workers['side_left_depth'].wait_and_get_outputs()
            res_sl_seg      = self.workers['side_left_seg'].wait_and_get_outputs()
            res_sr_depth    = self.workers['side_right_depth'].wait_and_get_outputs()
            res_sr_seg      = self.workers['side_right_seg'].wait_and_get_outputs()
            t2 = time.perf_counter()

            # POST-PROCESS
            pred_d_front = torch.clamp((480.0 * (1280.0/1920.0)) / torch.clamp(res_front_depth[1].squeeze(), min=0.1, max=480.0), min=1.0, max=250.0).cpu().numpy()
            pred_s_front = torch.argmax(res_front_seg[0][0], dim=0).to(torch.uint8).cpu().numpy()

            pred_d_rear = torch.clamp(res_rear_depth[0].squeeze(), min=1.0, max=150.0).cpu().numpy()
            pred_s_rear = torch.argmax(res_rear_seg[0].squeeze(), dim=0).to(torch.uint8).cpu().numpy()

            pred_d_sl = torch.clamp(res_sl_depth[0].squeeze(), min=1.0, max=80.0).cpu().numpy()
            pred_s_sl = torch.argmax(res_sl_seg[0].squeeze(), dim=0).to(torch.uint8).cpu().numpy()
            
            pred_d_sr = torch.clamp(res_sr_depth[0].squeeze(), min=1.0, max=80.0).cpu().numpy()
            pred_s_sr = torch.argmax(res_sr_seg[0].squeeze(), dim=0).to(torch.uint8).cpu().numpy()

            self.publish_prediction('front_left', pred_d_front, pred_s_front, msg_fl.header)
            self.publish_prediction('rear', pred_d_rear, pred_s_rear, msg_r.header)
            self.publish_prediction('side_left', pred_d_sl, pred_s_sl, msg_sl.header)
            self.publish_prediction('side_right', pred_d_sr, pred_s_sr, msg_sr.header)
            self.heartbeat_pub.publish(Bool(data=True))
            t3 = time.perf_counter()

            # TIMING LOGIC
            for cam in ['front_left', 'rear', 'side_left', 'side_right']:
                self.frame_counts[cam] += 1
                if self.last_timestamps[cam]:
                    self.fps_records[cam].append(1.0 / (t3 - self.last_timestamps[cam]))
                self.last_timestamps[cam] = t3

            if self.frame_counts['front_left'] % 10 == 0:
                avg_fps = float(np.mean(self.fps_records['front_left'])) if len(self.fps_records['front_left']) > 0 else 0.0
                self.get_logger().info(f"[Inference Profiler] Pre: {(t1-t0)*1000:.1f}ms | Infer: {(t2-t1)*1000:.1f}ms | Post+Pub: {(t3-t2)*1000:.1f}ms || FPS: {avg_fps:.2f}")

        except Exception as e:
            self.get_logger().error(f"Inference callback failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TrtEvaluationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()