import os
import json
import time
import cv2
import collections
import numpy as np
from sklearn.metrics import confusion_matrix

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge

# ==========================================
# CONFIGURATION
# ==========================================
NUM_CLASSES = 30
CAM_CAPS = {
    'front_left': 250.0,
    'rear': 150.0,
    'side_left': 80.0,
    'side_right': 80.0
}

class EvaluateNode(Node):
    def __init__(self):
        super().__init__('evaluate_node')
        
        # 1. Parameter Declarations
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        
        # 2. Tracking Dictionaries
        self.stats = {
            cam: {'rmse': [], 'd1': [], 'miou': []} 
            for cam in CAM_CAPS.keys()
        }
        self.frame_counts = {cam: 0 for cam in CAM_CAPS.keys()}
        self.last_timestamps = {cam: None for cam in CAM_CAPS.keys()}
        self.fps_records = {cam: collections.deque(maxlen=30) for cam in CAM_CAPS.keys()}
        
        # 3. Setup Synchronized Subscribers per Camera
        self.sync_handlers = {}
        for cam in CAM_CAPS.keys():
            self.sync_handlers[cam] = self.create_cam_sync(cam)
            
        self.get_logger().info(f"🧪 Evaluation Node Started. Debug mode: {self.debug}")
        self.get_logger().info("Waiting for synchronized predictions...")

    def create_cam_sync(self, cam_name):
        """Creates an ApproximateTimeSynchronizer for the appropriate topics of a single camera."""
        sub_pred_d = message_filters.Subscriber(self, Image, f'/inference/{cam_name}/depth', callback_group=self.callback_group)
        sub_pred_s = message_filters.Subscriber(self, Image, f'/inference/{cam_name}/seg', callback_group=self.callback_group)
        subscribers = [sub_pred_d, sub_pred_s]

        if self.debug:
            sub_gt_d = message_filters.Subscriber(self, Image, f'/carla/{cam_name}/depth', callback_group=self.callback_group)
            sub_gt_s = message_filters.Subscriber(self, Image, f'/carla/{cam_name}/seg', callback_group=self.callback_group)
            subscribers.extend([sub_gt_d, sub_gt_s])

        ts = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=10,
            slop=0.15
        )
        ts.registerCallback(lambda *msgs, name=cam_name: self.eval_callback(name, *msgs))
        return ts

    # ==========================================
    # DECODING HELPERS
    # ==========================================
    def decode_depth(self, msg):
        """Smart decode: handles raw CARLA BGRA or direct 32FC1 float maps."""
        if msg.encoding in ['bgra8', 'rgba8']:
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            b, g, r = bgra[:, :, 0].astype(np.float64), bgra[:, :, 1].astype(np.float64), bgra[:, :, 2].astype(np.float64)
            normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            return (normalized * 1000.0).astype(np.float32)
        else:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def decode_seg(self, msg):
        """Smart decode: handles raw CARLA BGRA (Class in Red channel) or 8UC1 map."""
        if msg.encoding in ['bgra8', 'rgba8']:
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            return bgra[:, :, 2] # CARLA Semantic Segmentation class is in the Red channel
        else:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")

    # ==========================================
    # METRICS CALCULATION
    # ==========================================
    def compute_metrics(self, pred_d, gt_d, pred_s, gt_s, max_depth):
        # Depth Metrics
        mask = (gt_d >= 1.0) & (gt_d <= max_depth) & (pred_d >= 1.0) & (pred_d <= max_depth)
        rmse, d1 = np.nan, np.nan
        
        if np.any(mask):
            p, g = pred_d[mask], gt_d[mask]
            rmse = np.sqrt(((p - g)**2).mean())
            ratio = np.maximum(p/g, g/p)
            d1 = (ratio < 1.25).mean() * 100.0
        
        # Segmentation Metrics
        miou = np.nan
        if gt_s is not None and pred_s is not None:
            valid_s = (gt_s != 255) & (gt_s < NUM_CLASSES)
            if np.any(valid_s):
                cm = confusion_matrix(gt_s[valid_s].flatten(), pred_s[valid_s].flatten(), labels=np.arange(NUM_CLASSES))
                ious = []
                for i in range(1, NUM_CLASSES): # Skip class 0 if it's unlabeled (standard Cityscapes)
                    union = np.sum(cm[i,:]) + np.sum(cm[:,i]) - cm[i,i]
                    if union > 0: ious.append(cm[i,i] / union)
                if ious: 
                    miou = np.mean(ious)

        return float(rmse), float(d1), float(miou)

    # ==========================================
    # MAIN CALLBACK
    # ==========================================
    def eval_callback(self, cam_name, msg_pred_d, msg_pred_s, msg_gt_d=None, msg_gt_s=None):
        try:
            pred_d = self.bridge.imgmsg_to_cv2(msg_pred_d, desired_encoding="32FC1")
            pred_s = self.bridge.imgmsg_to_cv2(msg_pred_s, desired_encoding="8UC1")

            now = time.perf_counter()
            if self.last_timestamps[cam_name] is not None:
                dt = now - self.last_timestamps[cam_name]
                if dt > 0:
                    self.fps_records[cam_name].append(1.0 / dt)
            self.last_timestamps[cam_name] = now
            self.frame_counts[cam_name] += 1

            if not self.debug:
                if self.frame_counts[cam_name] % 10 == 0:
                    avg_fps = float(np.mean(self.fps_records[cam_name])) if len(self.fps_records[cam_name]) > 0 else 0.0
                    self.get_logger().info(
                        f"[{cam_name.upper():<10}] Inference-only mode | Frames: {self.frame_counts[cam_name]} | Avg FPS: {avg_fps:.2f}"
                    )
                return

            gt_d = self.decode_depth(msg_gt_d)
            gt_s = self.decode_seg(msg_gt_s)

            if pred_d.shape != gt_d.shape:
                pred_d = cv2.resize(pred_d, (gt_d.shape[1], gt_d.shape[0]), interpolation=cv2.INTER_LINEAR)
            if pred_s.shape != gt_s.shape:
                pred_s = cv2.resize(pred_s, (gt_s.shape[1], gt_s.shape[0]), interpolation=cv2.INTER_NEAREST)

            max_cap = CAM_CAPS[cam_name]
            rmse, d1, miou = self.compute_metrics(pred_d, gt_d, pred_s, gt_s, max_depth=max_cap)

            if not np.isnan(rmse): self.stats[cam_name]['rmse'].append(rmse)
            if not np.isnan(d1):   self.stats[cam_name]['d1'].append(d1)
            if not np.isnan(miou): self.stats[cam_name]['miou'].append(miou)

            if self.frame_counts[cam_name] % 3 == 0:
                m_str = f"{miou:.4f}" if not np.isnan(miou) else "N/A"
                r_str = f"{rmse:.2f}" if not np.isnan(rmse) else "N/A"
                d_str = f"{d1:.2f}" if not np.isnan(d1) else "N/A"
                self.get_logger().info(f"[{cam_name.upper():<10}] Cap: {max_cap}m | mIoU: {m_str} | RMSE: {r_str}m | d1: {d_str}%")
        except Exception as e:
            self.get_logger().error(f"Evaluation callback failed for {cam_name}: {e}", exc_info=True)

    # ==========================================
    # SHUTDOWN ROUTINE (JSON DUMP & PRINT)
    # ==========================================
    def destroy_node(self):
        self.get_logger().info("\n" + "="*60)
        if self.debug:
            self.get_logger().info("🛑 RUN FINISHED: CALCULATING FINAL AVERAGE METRICS")
            self.get_logger().info("="*60)

            final_metrics = {}

            print(f"\n{'CAMERA':<15} | {'mIoU (0-1)':<12} | {'RMSE (m)':<12} | {'δ < 1.25 (%)':<12} | {'Frames'}")
            print("-" * 75)

            for cam in CAM_CAPS.keys():
                cam_stats = self.stats[cam]
                
                avg_miou = np.mean(cam_stats['miou']) if len(cam_stats['miou']) > 0 else 0.0
                avg_rmse = np.mean(cam_stats['rmse']) if len(cam_stats['rmse']) > 0 else 0.0
                avg_d1   = np.mean(cam_stats['d1']) if len(cam_stats['d1']) > 0 else 0.0
                frames_processed = self.frame_counts[cam]

                final_metrics[cam] = {
                    "mIoU": avg_miou,
                    "RMSE": avg_rmse,
                    "delta_1_25": avg_d1,
                    "max_depth_cap": CAM_CAPS[cam],
                    "frames_evaluated": frames_processed
                }

                print(f"{cam:<15} | {avg_miou:<12.4f} | {avg_rmse:<12.2f} | {avg_d1:<12.2f} | {frames_processed}")

            print("=" * 75)

            save_path = os.path.join(os.getcwd(), 'metrics.json')
            try:
                with open(save_path, 'w') as f:
                    json.dump(final_metrics, f, indent=4)
                self.get_logger().info(f"💾 Metrics successfully saved to: {save_path}")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save metrics.json: {e}")
        else:
            self.get_logger().info("🛑 Evaluation node shutdown in inference-only mode.")
            print(f"\n{'CAMERA':<15} | {'Avg FPS':<10} | {'Frames':<10}")
            print("-" * 40)
            for cam in CAM_CAPS.keys():
                avg_fps = float(np.mean(self.fps_records[cam])) if len(self.fps_records[cam]) > 0 else 0.0
                print(f"{cam:<15} | {avg_fps:<10.2f} | {self.frame_counts[cam]:<10}")
            print("=" * 40)

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EvaluateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # Standard Ctrl+C triggers the destroy_node() gracefully
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()