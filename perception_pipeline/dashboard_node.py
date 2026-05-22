import numpy as np
import cv2

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge

# ==========================================================
# CONFIGURATION
# ==========================================================
CAMERA_NAMES = ["front_left", "rear", "side_left", "side_right"]

CAMS = {
    "front_left": {"fov": 90, "w": 1280, "h": 720, "x": 1.4, "y": -0.25, "z": 1.5, "pitch": -12.0, "yaw": 0.0, "roll": 0.0},
    "rear": {"fov": 100, "w": 800, "h": 400, "x": -2.0, "y": 0.0, "z": 1.6, "pitch": -26.0, "yaw": 180.0, "roll": 0.0},
    "side_left": {"fov": 120, "w": 800, "h": 400, "x": 0.0, "y": -0.8, "z": 1.8, "pitch": -41.0, "yaw": -90.0, "roll": 0.0},
    "side_right": {"fov": 120, "w": 800, "h": 400, "x": 0.0, "y": 0.8, "z": 1.8, "pitch": -41.0, "yaw": 90.0, "roll": 0.0},
}

VC = {
    "w": 1280, "h": 720, "fov": 90,
    "x": -15.0, "y": 0.0, "z": 8.0, "pitch": -20, "yaw": 0, "roll": 0
}

CITYSCAPES_PALETTE = np.array([
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230),
    (119, 11, 32), (110, 190, 160), (170, 120, 50), (55, 90, 80), (45, 60, 150), (157, 234, 50),
    (81, 0, 81), (150, 100, 100), (230, 150, 140), (180, 165, 180), (250, 170, 160)
], dtype=np.uint8)

OCCUPANCY_MULTIPLIER = np.zeros(30, dtype=np.float32)
OCCUPANCY_MULTIPLIER[[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28]] = 1.0 

# ==========================================================
# MATH UTILS
# ==========================================================
def seg_to_rgb(seg):
    seg_clean = np.where(seg == 255, 0, seg)
    return CITYSCAPES_PALETTE[np.clip(seg_clean, 0, len(CITYSCAPES_PALETTE)-1)]

def intrinsics(w, h, fov):
    f = (w / 2) / np.tan(np.deg2rad(fov) / 2)
    return f, f, w / 2, h / 2

def rot(pitch, yaw, roll):
    p, y, r = np.deg2rad([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# ==========================================================
# TEMPORAL GRIDS
# ==========================================================
class LightweightTemporalGrid:
    def __init__(self, size_m=80.0, res=0.5, hit=0.4, decay=0.15, thresh=0.4):
        self.size_m = size_m
        self.res = res
        self.grid_size = int((size_m * 2) / res)
        self.hit = hit
        self.decay = decay
        self.thresh = thresh
        
        self.conf = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.colors = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
    def update(self, pts_vehicle, colors):
        self.conf -= self.decay
        np.clip(self.conf, 0, 1.0, out=self.conf)
        if pts_vehicle.shape[1] == 0: return
            
        X, Y, Z = pts_vehicle
        valid = (X > -self.size_m) & (X < self.size_m) & (Y > -self.size_m) & (Y < self.size_m) & (Z > -5.0) & (Z < 15.0) 
        X, Y, c = X[valid], Y[valid], colors[valid]
        if len(X) == 0: return
            
        center = self.grid_size // 2
        idx_row = np.clip(center - np.floor(X / self.res).astype(int), 0, self.grid_size - 1)
        idx_col = np.clip(center + np.floor(Y / self.res).astype(int), 0, self.grid_size - 1)
        
        self.conf[idx_row, idx_col] += self.hit
        np.clip(self.conf, 0, 1.0, out=self.conf)
        self.colors[idx_row, idx_col] = c

    def get_bev_image(self, target_size=(720, 720)):
        bev = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        mask = self.conf >= self.thresh
        bev[mask] = self.colors[mask]
        bev = cv2.resize(bev, target_size, interpolation=cv2.INTER_NEAREST)
        
        cx, cy = target_size[0] // 2, target_size[1] // 2
        car_w = int((2.0 / self.res) * (target_size[0] / self.grid_size) / 2)
        car_l = int((4.5 / self.res) * (target_size[1] / self.grid_size) / 2)
        cv2.rectangle(bev, (cx - car_w, cy - car_l), (cx + car_w, cy + car_l), (0, 255, 0), -1)
        return bev

class SemanticTemporalGrid:
    def __init__(self, size_m=80.0, res=0.5, hit=0.4, decay=0.15):
        self.size_m = size_m
        self.res = res
        self.grid_size = int((size_m * 2) / res)
        self.hit = hit
        self.decay = decay
        
        self.conf = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.classes = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
    def update(self, pts_vehicle, seg_classes):
        self.conf -= self.decay
        np.clip(self.conf, 0, 1.0, out=self.conf)
        if pts_vehicle.shape[1] == 0: return
            
        X, Y, Z = pts_vehicle
        valid = (X > -self.size_m) & (X < self.size_m) & (Y > -self.size_m) & (Y < self.size_m) & (Z > -5.0) & (Z < 15.0)
        X, Y, seg = X[valid], Y[valid], seg_classes[valid]
        if len(X) == 0: return
            
        center = self.grid_size // 2
        idx_row = np.clip(center - np.floor(X / self.res).astype(int), 0, self.grid_size - 1)
        idx_col = np.clip(center + np.floor(Y / self.res).astype(int), 0, self.grid_size - 1)
        
        self.conf[idx_row, idx_col] += self.hit
        np.clip(self.conf, 0, 1.0, out=self.conf)
        self.classes[idx_row, idx_col] = seg

    def get_occupancy_image(self, target_size=(800, 800)):
        occ_prob = self.conf * OCCUPANCY_MULTIPLIER[self.classes]
        occ_colored = cv2.applyColorMap((occ_prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        occ_colored = cv2.resize(occ_colored, target_size, interpolation=cv2.INTER_NEAREST)
        
        cx, cy = target_size[0] // 2, target_size[1] // 2
        car_w = int((2.0 / self.res) * (target_size[0] / self.grid_size) / 2)
        car_l = int((4.5 / self.res) * (target_size[1] / self.grid_size) / 2)
        cv2.rectangle(occ_colored, (cx - car_w, cy - car_l), (cx + car_w, cy + car_l), (255, 255, 0), -1)
        return occ_colored

# ==========================================================
# CHASE CAMERA RENDERER
# ==========================================================
def render_chase_from_dense(pts_vehicle, colors):
    canvas = np.zeros((VC["h"], VC["w"], 3), dtype=np.uint8)
    if pts_vehicle.shape[1] == 0: return canvas
        
    Rv = rot(VC["pitch"], VC["yaw"], VC["roll"])
    Tv = np.array([[VC["x"]], [VC["y"]], [VC["z"]]])
    pts_vc = Rv.T @ (pts_vehicle - Tv)
    X_fwd, Y_right, Z_up = pts_vc
    
    valid = X_fwd > 0.5
    X_fwd, Y_right, Z_up, colors = X_fwd[valid], Y_right[valid], Z_up[valid], colors[valid]
    if len(X_fwd) == 0: return canvas
        
    fxv, fyv, cxv, cyv = intrinsics(VC["w"], VC["h"], VC["fov"])
    u2 = (Y_right * fxv / X_fwd + cxv).astype(int)
    v2 = (-Z_up * fyv / X_fwd + cyv).astype(int)
    
    mask = (u2 >= 0) & (u2 < VC["w"]) & (v2 >= 0) & (v2 < VC["h"])
    u2_m, v2_m, colors_m, X_fwd_m = u2[mask], v2[mask], colors[mask], X_fwd[mask]
    
    if len(X_fwd_m) > 0:
        sort_indices = np.argsort(-X_fwd_m)
        canvas[v2_m[sort_indices], u2_m[sort_indices]] = colors_m[sort_indices]
    return canvas

# ==========================================================
# ROS 2 NODE
# ==========================================================
class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.bridge = CvBridge()

        # Instantiate Grids
        self.pred_grid = LightweightTemporalGrid()
        self.pred_occ = SemanticTemporalGrid()
        
        if self.debug:
            self.gt_grid = LightweightTemporalGrid()
            self.gt_occ = SemanticTemporalGrid()

        # Publishers (RViz Image Displays)
        self.pub_pred_bev = self.create_publisher(Image, '/dashboard/pred/bev', 10)
        self.pub_pred_chase = self.create_publisher(Image, '/dashboard/pred/chase', 10)
        self.pub_pred_occ = self.create_publisher(Image, '/dashboard/pred/occupancy', 10)
        
        if self.debug:
            self.pub_gt_bev = self.create_publisher(Image, '/dashboard/gt/bev', 10)
            self.pub_gt_chase = self.create_publisher(Image, '/dashboard/gt/chase', 10)
            self.pub_gt_occ = self.create_publisher(Image, '/dashboard/gt/occupancy', 10)

        self.get_logger().info(f"📊 Dashboard Live | Debug (GT) enabled: {self.debug}")
        self.callback_group = ReentrantCallbackGroup()

        # Setup Prediction Synchronizer (8 topics)
        pred_subs = []
        for cam in CAMERA_NAMES:
            pred_subs.append(message_filters.Subscriber(self, Image, f'/inference/{cam}/depth', callback_group=self.callback_group))
            pred_subs.append(message_filters.Subscriber(self, Image, f'/inference/{cam}/seg', callback_group=self.callback_group))
            
        self.ts_pred = message_filters.ApproximateTimeSynchronizer(pred_subs, queue_size=10, slop=0.2)
        self.ts_pred.registerCallback(self.pred_callback)

        # Setup GT Synchronizer if debug
        if self.debug:
            gt_subs = []
            for cam in CAMERA_NAMES:
                gt_subs.append(message_filters.Subscriber(self, Image, f'/carla/{cam}/depth', callback_group=self.callback_group))
                gt_subs.append(message_filters.Subscriber(self, Image, f'/carla/{cam}/seg', callback_group=self.callback_group))
                
            self.ts_gt = message_filters.ApproximateTimeSynchronizer(gt_subs, queue_size=10, slop=0.2)
            self.ts_gt.registerCallback(self.gt_callback)

    def extract_and_project(self, msgs, is_gt=False):
        all_pts, all_colors, all_classes = [], [], []
        
        # msgs arrives as [depth_fl, seg_fl, depth_r, seg_r, depth_sl, seg_sl, depth_sr, seg_sr]
        for i, cam_name in enumerate(CAMERA_NAMES):
            msg_depth = msgs[i*2]
            msg_seg = msgs[(i*2)+1]
            
            # Smart Decode
            if msg_depth.encoding in ['bgra8', 'rgba8']:
                bgra = self.bridge.imgmsg_to_cv2(msg_depth, "passthrough")
                b, g, r = bgra[:, :, 0].astype(np.float64), bgra[:, :, 1].astype(np.float64), bgra[:, :, 2].astype(np.float64)
                depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
                depth = (depth * 1000.0).astype(np.float32)
            else:
                depth = self.bridge.imgmsg_to_cv2(msg_depth, "32FC1")
                
            if msg_seg.encoding in ['bgra8', 'rgba8']:
                bgra = self.bridge.imgmsg_to_cv2(msg_seg, "passthrough")
                seg = bgra[:, :, 2]
            else:
                seg = self.bridge.imgmsg_to_cv2(msg_seg, "8UC1")
                
            if pred_needs_scale := (not is_gt and np.nanmax(depth) <= 1.5):
                depth = depth * 1000.0

            cam = CAMS[cam_name]
            h, w = depth.shape
            fx, fy, cx, cy = intrinsics(w, h, cam["fov"])
            
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            u, v, Z = u.flatten(), v.flatten(), depth.flatten()
            colors = seg_to_rgb(seg).reshape(-1, 3)
            seg_flat = seg.flatten()
            
            valid = np.isfinite(Z) & (Z > 0.5) & (Z < 150)
            u, v, Z, colors, seg_flat = u[valid], v[valid], Z[valid], colors[valid], seg_flat[valid]
            if len(Z) == 0: continue
            
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            pts_cam = np.vstack((Z, X, -Y))
            R_cam = rot(cam["pitch"], cam["yaw"], cam["roll"])
            pts_vehicle = (R_cam @ pts_cam)
            
            pts_vehicle[0] += cam["x"]
            pts_vehicle[1] += cam["y"]
            pts_vehicle[2] += cam["z"]
            
            all_pts.append(pts_vehicle)
            all_colors.append(colors)
            all_classes.append(seg_flat)
            
        if not all_pts:
            return np.empty((3, 0)), np.empty((0, 3)), np.empty((0,))
            
        return np.hstack(all_pts), np.vstack(all_colors), np.hstack(all_classes)

    def pred_callback(self, *msgs):
        try:
            pts, colors, classes = self.extract_and_project(msgs, is_gt=False)
            
            self.pred_grid.update(pts, colors)
            self.pred_occ.update(pts, classes)
            
            bev = self.pred_grid.get_bev_image()
            chase = render_chase_from_dense(pts, colors)
            occ = self.pred_occ.get_occupancy_image()

            cv2.putText(bev, "PRED: 80m Grid BEV", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(chase, "PRED: Chase View", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(occ, "PRED: OCCUPANCY (80m)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # The BEV and chase renderers create RGB images; convert to BGR for OpenCV bridge.
            bev = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
            chase = cv2.cvtColor(chase, cv2.COLOR_RGB2BGR)
            
            header = msgs[0].header
            self.pub_pred_bev.publish(self.bridge.cv2_to_imgmsg(bev, encoding="bgr8", header=header))
            self.pub_pred_chase.publish(self.bridge.cv2_to_imgmsg(chase, encoding="bgr8", header=header))
            self.pub_pred_occ.publish(self.bridge.cv2_to_imgmsg(occ, encoding="bgr8", header=header))
        except Exception as e:
            self.get_logger().error(f"Dashboard prediction callback failed: {e}", exc_info=True)

    def gt_callback(self, *msgs):
        try:
            pts, colors, classes = self.extract_and_project(msgs, is_gt=True)
            
            self.gt_grid.update(pts, colors)
            self.gt_occ.update(pts, classes)
            
            bev = self.gt_grid.get_bev_image()
            chase = render_chase_from_dense(pts, colors)
            occ = self.gt_occ.get_occupancy_image()
            
            cv2.putText(bev, "GT: 80m Grid BEV", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(chase, "GT: Chase View", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(occ, "GT: OCCUPANCY (80m)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            bev = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
            chase = cv2.cvtColor(chase, cv2.COLOR_RGB2BGR)

            header = msgs[0].header
            self.pub_gt_bev.publish(self.bridge.cv2_to_imgmsg(bev, encoding="bgr8", header=header))
            self.pub_gt_chase.publish(self.bridge.cv2_to_imgmsg(chase, encoding="bgr8", header=header))
            self.pub_gt_occ.publish(self.bridge.cv2_to_imgmsg(occ, encoding="bgr8", header=header))
        except Exception as e:
            self.get_logger().error(f"Dashboard GT callback failed: {e}", exc_info=True)

def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
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