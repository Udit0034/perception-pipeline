import os
import sys
import time
import collections
import cv2
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
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
        sys.stdout.write("🖥️ Multi-GPU system detected. TRT Context pinned to: [cuda:1]\n")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        DEVICE = torch.device("cuda:0")
        torch.cuda.set_device(0)
        sys.stdout.write("🖥️ Single GPU system detected. TRT Context pinned to: [cuda:0]\n")
else:
    DEVICE = torch.device("cpu")

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

REPO_ROOT = find_repo_root(os.path.dirname(os.path.realpath(__file__)))
ENGINE_CACHE_DIR = os.path.join(REPO_ROOT, "trt_engine_cache")

# ==========================================
# CONSTANTS & UTILS
# ==========================================
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

VIDEO_OUTPUT_PATH = "bev_live_inference.avi"
X_MIN, X_MAX = -30.0, 50.0
Y_MIN, Y_MAX = -35.0, 35.0
RESOLUTION = 15.0
BEV_WIDTH = int((Y_MAX - Y_MIN) * RESOLUTION)
BEV_HEIGHT = int((X_MAX - X_MIN) * RESOLUTION)
MIN_HEIGHT, MAX_HEIGHT = -3.0, 1.8
ego_u = int((0 - Y_MIN) * RESOLUTION)
ego_v = int((X_MAX - 0) * RESOLUTION)

CITYSCAPES_PALETTE = np.array([
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35),
    (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
    (110, 190, 160), (170, 120, 50), (55, 90, 80), (45, 60, 150), (157, 234, 50),
    (81, 0, 81), (150, 100, 100), (230, 150, 140), (180, 165, 180), (250, 170, 160)
], dtype=np.uint8)

CAMS = {
    "front_left": {"fov": 90, "x": 1.4, "y": -0.25, "z": 1.5, "pitch": -12.0, "yaw": 0.0, "roll": 0.0},
    "front_right": {"fov": 90, "x": 1.4, "y": 0.25, "z": 1.5, "pitch": -12.0, "yaw": 0.0, "roll": 0.0},
    "rear": {"fov": 100, "x": -2.0, "y": 0.0, "z": 1.6, "pitch": -26.0, "yaw": 180.0, "roll": 0.0},
    "side_left": {"fov": 120, "x": 0.0, "y": -0.8, "z": 1.8, "pitch": -41.0, "yaw": -90.0, "roll": 0.0},
    "side_right": {"fov": 120, "x": 0.0, "y": 0.8, "z": 1.8, "pitch": -41.0, "yaw": 90.0, "roll": 0.0},
}

def get_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    p, y, r = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg), np.deg2rad(roll_deg)
    R_y = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

def project_to_bev(bev_img, depth, seg, params, current_max_depth):
    if depth is None or seg is None: return bev_img
    if depth.shape != seg.shape:
        seg = cv2.resize(seg, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    seg_clean = np.where(seg == 255, 0, seg)
    rgb = CITYSCAPES_PALETTE[np.clip(seg_clean, 0, len(CITYSCAPES_PALETTE)-1)]
    h, w = depth.shape
    fx = fy = (w / 2.0) / np.tan(np.deg2rad(params["fov"]) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    R = get_rotation_matrix(params["pitch"], params["yaw"], params["roll"])
    
    Z = depth.flatten()
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v, colors = u.flatten(), v.flatten(), rgb.reshape(-1, 3)
    
    valid = (Z > 0.5) & (Z < current_max_depth)
    u, v, Z, colors = u[valid], v[valid], Z[valid], colors[valid]
    
    X_opt = (u - cx) * Z / fx
    Y_opt = (v - cy) * Z / fy
    points_c = np.vstack((Z, X_opt, -Y_opt))
    
    points_rot = R @ points_c
    X_veh = points_rot[0] + params["x"]
    Y_veh = points_rot[1] + params["y"]
    Z_veh = points_rot[2] + params["z"]
    
    valid_h = (Z_veh > MIN_HEIGHT) & (Z_veh < MAX_HEIGHT)
    X_veh, Y_veh, colors = X_veh[valid_h], Y_veh[valid_h], colors[valid_h]
    
    u_bev = ((Y_veh - Y_MIN) * RESOLUTION).astype(np.int32)
    v_bev = ((X_MAX - X_veh) * RESOLUTION).astype(np.int32)
    
    valid_bev = (u_bev >= 0) & (u_bev < BEV_WIDTH) & (v_bev >= 0) & (v_bev < BEV_HEIGHT)
    bev_img[v_bev[valid_bev], u_bev[valid_bev]] = colors[valid_bev]
    return bev_img

# ==========================================
# TRT ASYNC RUNNER (No Building Logic)
# ==========================================
class AsyncTRTRunner:
    def __init__(self, engine_name):
        engine_path = os.path.join(ENGINE_CACHE_DIR, f"{engine_name}.engine")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Missing Engine: {engine_path}. Please run EngineBuilderNode first!")
            
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=DEVICE)
        self.num_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io)]

    def infer(self, *input_arrays):
        with torch.cuda.stream(self.stream):
            device_inputs = [torch.from_numpy(arr).to(DEVICE, non_blocking=True) for arr in input_arrays]
            
            input_idx = 0
            for name in self.tensor_names:
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, device_inputs[input_idx].shape)
                    input_idx += 1
            
            outputs, bindings = {}, [None] * self.num_io
            input_idx = 0
            for i, name in enumerate(self.tensor_names):
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    bindings[i] = device_inputs[input_idx].data_ptr()
                    input_idx += 1
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
                    dtype = torch.float32 if self.engine.get_tensor_dtype(name) == trt.float32 else torch.float16
                    out_tensor = torch.empty(shape, dtype=dtype, device=DEVICE)
                    outputs[name] = out_tensor
                    bindings[i] = out_tensor.data_ptr()
            
            if hasattr(self.context, 'execute_async_v3'):
                for i, name in enumerate(self.tensor_names):
                    self.context.set_tensor_address(name, bindings[i])
                self.context.execute_async_v3(self.stream.cuda_stream)
            elif hasattr(self.context, 'execute_async_v2'):
                self.context.execute_async_v2(bindings, self.stream.cuda_stream)
            else:
                self.context.execute_v2(bindings)
            
            self.stream.synchronize()
            return [outputs[n].cpu().numpy() for n in self.tensor_names if self.engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT]

# ==========================================
# ROS 2 EVALUATION NODE
# ==========================================
class TrtEvaluationNode(Node):
    def __init__(self):
        super().__init__('trt_evaluation_node')
        self.bridge = CvBridge()
        self.get_logger().info("Loading TensorRT Engines from cache...")
        
        # Load pre-built engines only
        self.workers = {
            'front_depth': AsyncTRTRunner('stereonet'),
            'front_seg': AsyncTRTRunner('seg_front'),
            'rear_depth': AsyncTRTRunner('rear_depth'),
            'rear_seg': AsyncTRTRunner('rear_seg'),
            'side_left_depth': AsyncTRTRunner('side_depth'),
            'side_left_seg': AsyncTRTRunner('side_seg'),
            'side_right_depth': AsyncTRTRunner('side_depth'), # Reusing side model
            'side_right_seg': AsyncTRTRunner('side_seg'),
        }
        self.get_logger().info("All 8 engines loaded successfully.")

        # Setup Inference Output Publishers
        self.inference_publishers = {}
        for cam in ['front_left', 'rear', 'side_left', 'side_right']:
            self.inference_publishers[cam] = {
                'depth': self.create_publisher(Image, f'/inference/{cam}/depth', 10),
                'seg': self.create_publisher(Image, f'/inference/{cam}/seg', 10),
            }
            self.get_logger().info(f"Created Publisher: /inference/{cam}/depth")
            self.get_logger().info(f"Created Publisher: /inference/{cam}/seg")

        # Track inference performance
        self.frame_counts = {cam: 0 for cam in ['front_left', 'rear', 'side_left', 'side_right']}
        self.last_timestamps = {cam: None for cam in ['front_left', 'rear', 'side_left', 'side_right']}
        self.fps_records = {cam: collections.deque(maxlen=30) for cam in ['front_left', 'rear', 'side_left', 'side_right']}

        # Setup Approximate Time Synchronizer for 5 Camera Topics
        self.callback_group = ReentrantCallbackGroup()
        self.sub_fl = message_filters.Subscriber(self, Image, '/carla/front_left/image', callback_group=self.callback_group)
        self.sub_fr = message_filters.Subscriber(self, Image, '/carla/front_right/image', callback_group=self.callback_group)
        self.sub_r  = message_filters.Subscriber(self, Image, '/carla/rear/image', callback_group=self.callback_group)
        self.sub_sl = message_filters.Subscriber(self, Image, '/carla/side_left/image', callback_group=self.callback_group)
        self.sub_sr = message_filters.Subscriber(self, Image, '/carla/side_right/image', callback_group=self.callback_group)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_fl, self.sub_fr, self.sub_r, self.sub_sl, self.sub_sr], 
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info("Waiting for synchronized ROS 2 Camera Data...")

    def publish_prediction(self, cam_name, depth_arr, seg_arr, header):
        try:
            depth_msg = self.bridge.cv2_to_imgmsg(depth_arr.astype(np.float32), encoding='32FC1')
            depth_msg.header = header
            seg_msg = self.bridge.cv2_to_imgmsg(seg_arr.astype(np.uint8), encoding='8UC1')
            seg_msg.header = header
            self.inference_publishers[cam_name]['depth'].publish(depth_msg)
            self.inference_publishers[cam_name]['seg'].publish(seg_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish inference outputs for {cam_name}: {e}", exc_info=True)

    def update_performance(self, cam_name, now):
        if self.last_timestamps[cam_name] is not None:
            dt = now - self.last_timestamps[cam_name]
            if dt > 0:
                self.fps_records[cam_name].append(1.0 / dt)
        self.last_timestamps[cam_name] = now
        if self.frame_counts[cam_name] > 0 and self.frame_counts[cam_name] % 10 == 0:
            avg_fps = float(np.mean(self.fps_records[cam_name])) if len(self.fps_records[cam_name]) > 0 else 0.0
            self.get_logger().info(f"[{cam_name}] Inference publishing | Avg FPS: {avg_fps:.2f}")

    def synced_callback(self, msg_fl, msg_fr, msg_r, msg_sl, msg_sr):
        try:
            wall_start = time.perf_counter()

            # 1. Convert ROS Images to CV2 RGB
            cv_imgs = {
                'front_left': cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_fl, "bgr8"), cv2.COLOR_BGR2RGB),
                'front_right': cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_fr, "bgr8"), cv2.COLOR_BGR2RGB),
                'rear': cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_r, "bgr8"), cv2.COLOR_BGR2RGB),
                'side_left': cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_sl, "bgr8"), cv2.COLOR_BGR2RGB),
                'side_right': cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_sr, "bgr8"), cv2.COLOR_BGR2RGB),
            }

            # 2. Pre-process to PyTorch Tensors (NumPy format for TRT)
            tl_np = normalize(TF.to_tensor(cv_imgs['front_left'])).unsqueeze(0).numpy().astype(np.float32)
            tr_np = normalize(TF.to_tensor(cv_imgs['front_right'])).unsqueeze(0).numpy().astype(np.float32)
            
            mono_inputs = {
                cam: normalize(TF.to_tensor(cv_imgs[cam])).unsqueeze(0).numpy().astype(np.float32)
                for cam in ['rear', 'side_left', 'side_right']
            }

            # 3. Execute 8 Models in Parallel
            results = {}
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    'front_depth': executor.submit(self.workers['front_depth'].infer, tl_np, tr_np),
                    'front_seg': executor.submit(self.workers['front_seg'].infer, tl_np),
                    'rear_depth': executor.submit(self.workers['rear_depth'].infer, mono_inputs['rear']),
                    'rear_seg': executor.submit(self.workers['rear_seg'].infer, mono_inputs['rear']),
                    'side_left_depth': executor.submit(self.workers['side_left_depth'].infer, mono_inputs['side_left']),
                    'side_left_seg': executor.submit(self.workers['side_left_seg'].infer, mono_inputs['side_left']),
                    'side_right_depth': executor.submit(self.workers['side_right_depth'].infer, mono_inputs['side_right']),
                    'side_right_seg': executor.submit(self.workers['side_right_seg'].infer, mono_inputs['side_right']),
                }
                
                future_to_name = {v: k for k, v in futures.items()}
                for future in as_completed(futures.values()):
                    name = future_to_name[future]
                    results[name] = future.result()

            self.frame_counts['front_left'] += 1
            self.frame_counts['rear'] += 1
            self.frame_counts['side_left'] += 1
            self.frame_counts['side_right'] += 1

            disp_front = np.clip(results['front_depth'][1].squeeze(), 0.1, 480.0)
            pred_d_front = np.clip((480.0 * (1280/1920.0)) / disp_front, 1.0, 250.0)
            pred_s_front = np.argmax(results['front_seg'][0][0], axis=0).astype(np.uint8)

            pred_d_rear = np.clip(results['rear_depth'][0].squeeze(), 1.0, 150.0)
            pred_s_rear = np.argmax(results['rear_seg'][0].squeeze(), axis=0).astype(np.uint8)

            pred_d_sl = np.clip(results['side_left_depth'][0].squeeze(), 1.0, 80.0)
            pred_s_sl = np.argmax(results['side_left_seg'][0].squeeze(), axis=0).astype(np.uint8)
            pred_d_sr = np.clip(results['side_right_depth'][0].squeeze(), 1.0, 80.0)
            pred_s_sr = np.argmax(results['side_right_seg'][0].squeeze(), axis=0).astype(np.uint8)

            self.publish_prediction('front_left', pred_d_front, pred_s_front, msg_fl.header)
            self.publish_prediction('rear', pred_d_rear, pred_s_rear, msg_r.header)
            self.publish_prediction('side_left', pred_d_sl, pred_s_sl, msg_sl.header)
            self.publish_prediction('side_right', pred_d_sr, pred_s_sr, msg_sr.header)

            now = time.perf_counter()
            self.update_performance('front_left', now)
            self.update_performance('rear', now)
            self.update_performance('side_left', now)
            self.update_performance('side_right', now)
        except Exception as e:
            self.get_logger().error(f"Inference callback failed: {e}", exc_info=True)

    def destroy_node(self):
        super().destroy_node()

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
        rclpy.shutdown()

if __name__ == '__main__':
    main()