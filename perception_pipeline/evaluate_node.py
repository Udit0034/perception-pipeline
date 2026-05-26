import os
import json
import time
import math
import collections
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import signal
from sklearn.metrics import confusion_matrix

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
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

# ==========================================
# MATH UTILITIES
# ==========================================
def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def angle_error(est, gt):
    """Calculate signed angle error wrapped to [-pi, pi]"""
    return wrap_angle(est - gt)

def quaternion_to_euler(w, x, y, z):
    """Convert Quaternion to Yaw, Pitch, Roll"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return wrap_angle(yaw), wrap_angle(pitch), wrap_angle(roll)


# ==========================================
# UNIFIED EVALUATION NODE
# ==========================================
class EvaluateNode(Node):
    def __init__(self):
        super().__init__('evaluate_node')
        
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        
        self.stats = {
            cam: {'rmse': [], 'd1': [], 'miou': []} 
            for cam in CAM_CAPS.keys()
        }
        self.frame_counts = {cam: 0 for cam in CAM_CAPS.keys()}
        self.last_timestamps = {cam: None for cam in CAM_CAPS.keys()}
        self.fps_records = {cam: collections.deque(maxlen=30) for cam in CAM_CAPS.keys()}
        
        self.gt_data = []
        self.ekf_data = []
        
        self.sync_handlers = {}
        for cam in CAM_CAPS.keys():
            self.sync_handlers[cam] = self.create_cam_sync(cam)
            
        self.sub_gt_odom = self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.gt_odom_callback, 10, callback_group=self.callback_group)
        self.sub_ekf_odom = self.create_subscription(Odometry, '/ekf/odometry', self.ekf_odom_callback, 10, callback_group=self.callback_group)
            
        self.get_logger().info(f"🧪 Unified Evaluation Node Started. Debug mode: {self.debug}")
        if self.debug:
            self.get_logger().info("Plots will be generated upon Ctrl+C (destroy_node).")

    # ==========================================
    # VISION CALLBACKS & SYNC
    # ==========================================
    def create_cam_sync(self, cam_name):
        sub_pred_d = message_filters.Subscriber(self, Image, f'/inference/{cam_name}/depth', callback_group=self.callback_group)
        sub_pred_s = message_filters.Subscriber(self, Image, f'/inference/{cam_name}/seg', callback_group=self.callback_group)
        subscribers = [sub_pred_d, sub_pred_s]

        if self.debug:
            sub_gt_d = message_filters.Subscriber(self, Image, f'/carla/{cam_name}/depth', callback_group=self.callback_group)
            sub_gt_s = message_filters.Subscriber(self, Image, f'/carla/{cam_name}/seg', callback_group=self.callback_group)
            subscribers.extend([sub_gt_d, sub_gt_s])

        ts = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=30, slop=0.15)
        ts.registerCallback(lambda *msgs, name=cam_name: self.eval_callback(name, *msgs))
        return ts

    def decode_depth(self, msg):
        if msg.encoding in ['bgra8', 'rgba8']:
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            b, g, r = bgra[:, :, 0].astype(np.float64), bgra[:, :, 1].astype(np.float64), bgra[:, :, 2].astype(np.float64)
            normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            return (normalized * 1000.0).astype(np.float32)
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def decode_seg(self, msg):
        if msg.encoding in ['bgra8', 'rgba8']:
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            return bgra[:, :, 2] 
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")

    def compute_metrics(self, pred_d, gt_d, pred_s, gt_s, max_depth):
        mask = (gt_d >= 1.0) & (gt_d <= max_depth) & (pred_d >= 1.0) & (pred_d <= max_depth)
        rmse, d1 = np.nan, np.nan
        
        if np.any(mask):
            p, g = pred_d[mask], gt_d[mask]
            rmse = np.sqrt(((p - g)**2).mean())
            ratio = np.maximum(p/g, g/p)
            d1 = (ratio < 1.25).mean() * 100.0
        
        miou = np.nan
        if gt_s is not None and pred_s is not None:
            valid_s = (gt_s != 255) & (gt_s < NUM_CLASSES)
            if np.any(valid_s):
                cm = confusion_matrix(gt_s[valid_s].flatten(), pred_s[valid_s].flatten(), labels=np.arange(NUM_CLASSES))
                ious = [cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]) - cm[i,i]) for i in range(1, NUM_CLASSES) if (np.sum(cm[i,:]) + np.sum(cm[:,i]) - cm[i,i]) > 0]
                if ious: miou = np.mean(ious)

        return float(rmse), float(d1), float(miou)

    def eval_callback(self, cam_name, msg_pred_d, msg_pred_s, msg_gt_d=None, msg_gt_s=None):
        try:
            pred_d = self.bridge.imgmsg_to_cv2(msg_pred_d, desired_encoding="32FC1")
            pred_s = self.bridge.imgmsg_to_cv2(msg_pred_s, desired_encoding="8UC1")

            now = time.perf_counter()
            if self.last_timestamps[cam_name] is not None:
                dt = now - self.last_timestamps[cam_name]
                if dt > 0: self.fps_records[cam_name].append(1.0 / dt)
            self.last_timestamps[cam_name] = now
            self.frame_counts[cam_name] += 1

            if not self.debug:
                if self.frame_counts[cam_name] % 10 == 0:
                    avg_fps = float(np.mean(self.fps_records[cam_name])) if len(self.fps_records[cam_name]) > 0 else 0.0
                    self.get_logger().info(f"[{cam_name.upper():<10}] Inference-only | Frames: {self.frame_counts[cam_name]} | Avg FPS: {avg_fps:.2f}")
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

        except Exception as e:
            self.get_logger().error(f"Evaluation callback failed for {cam_name}: {e}")

    # ==========================================
    # ODOMETRY DATA GATHERING (LIVE)
    # ==========================================
    def gt_odom_callback(self, msg: Odometry):
        if not self.debug: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        yaw, pitch, roll = quaternion_to_euler(q.w, q.x, q.y, q.z)
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        self.gt_data.append({
            'Timestamp': t, 'Loc_X': pos.x, 'Loc_Y': pos.y, 'Loc_Z': pos.z,
            'Yaw_Degrees': math.degrees(yaw), 
            'Pitch_Degrees': -math.degrees(pitch),  # ⬅️ FIXED: Inverted CARLA Left-Hand Pitch to match EKF
            'Roll_Degrees': -math.degrees(roll),    # ⬅️ FIXED: Inverted CARLA Left-Hand Roll to match EKF
            'GT_Velocity': speed
        })

    def ekf_odom_callback(self, msg: Odometry):
        if not self.debug: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        yaw, pitch, roll = quaternion_to_euler(q.w, q.x, q.y, q.z)
        
        cov_x = msg.pose.covariance[0] if len(msg.pose.covariance) == 36 else 0.0
        cov_y = msg.pose.covariance[7] if len(msg.pose.covariance) == 36 else 0.0

        # ⬇️ FIXED: Extract biases from the angular fields
        bias_x = msg.twist.twist.angular.x
        bias_y = msg.twist.twist.angular.y
        bias_z = msg.twist.twist.angular.z

        self.ekf_data.append({
            'Timestamp': t, 'Est_X': pos.x, 'Est_Y': pos.y, 'Est_Z': pos.z,
            'Est_Yaw': yaw, 'Est_Pitch': pitch, 'Est_Roll': roll,
            'Est_Qw': q.w, 'Est_Qx': q.x, 'Est_Qy': q.y, 'Est_Qz': q.z,
            'Est_Vx': v.x, 'Est_Vy': v.y, 'P_dp_x': cov_x, 'P_dp_y': cov_y,
            'Est_Bias_Gx': bias_x, 'Est_Bias_Gy': bias_y, 'Est_Bias_Gz': bias_z # Added!
        })

    # ==========================================
    # SHUTDOWN ROUTINE (METRICS & PLOTTING)
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
                
                final_metrics[cam] = {
                    "mIoU": avg_miou, "RMSE": avg_rmse, "delta_1_25": avg_d1,
                    "max_depth_cap": CAM_CAPS[cam], "frames_evaluated": self.frame_counts[cam]
                }
                print(f"{cam:<15} | {avg_miou:<12.4f} | {avg_rmse:<12.2f} | {avg_d1:<12.2f} | {self.frame_counts[cam]}")

            print("=" * 75)
            
            save_path = os.path.join(os.getcwd(), 'metrics.json')
            try:
                with open(save_path, 'w') as f: json.dump(final_metrics, f, indent=4)
                self.get_logger().info(f"💾 Metrics successfully saved to: {save_path}")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save metrics.json: {e}")

            if len(self.gt_data) > 0 and len(self.ekf_data) > 0:
                self.generate_ekf_plots()
            else:
                self.get_logger().warn("⚠️ Not enough Odometry data collected to generate EKF plots.")
                
        else:
            self.get_logger().info("🛑 Evaluation node shutdown in inference-only mode.")
            print(f"\n{'CAMERA':<15} | {'Avg FPS':<10} | {'Frames':<10}")
            print("-" * 40)
            for cam in CAM_CAPS.keys():
                avg_fps = float(np.mean(self.fps_records[cam])) if len(self.fps_records[cam]) > 0 else 0.0
                print(f"{cam:<15} | {avg_fps:<10.2f} | {self.frame_counts[cam]:<10}")
            print("=" * 40)

        super().destroy_node()

    def generate_ekf_plots(self):
        run_dir = os.path.join(os.getcwd(), 'eval_plots')
        plots_dir = os.path.join(run_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        self.get_logger().info(f"\n--- Generating EKF Plotting Suite in {run_dir} ---")

        sns.set_theme(style="darkgrid", palette="deep")

        # ⬇️ FIXED: Reset indices to prevent .iloc overlapping bugs
        odom_df = pd.DataFrame(self.gt_data).sort_values('Timestamp').reset_index(drop=True)
        m1_df = pd.DataFrame(self.ekf_data).sort_values('Timestamp').reset_index(drop=True)

        merged = pd.merge_asof(m1_df, odom_df, on='Timestamp', direction='nearest').dropna().reset_index(drop=True)

        if merged.empty:
            self.get_logger().error("Merge failed: Timestamps out of sync between GT and EKF.")
            return

        # ==========================================
        # ORIGIN ALIGNMENT (EKF Local -> GT Global)
        # ==========================================
        offset_x = merged['Loc_X'].iloc[0] - merged['Est_X'].iloc[0]
        offset_y = merged['Loc_Y'].iloc[0] - merged['Est_Y'].iloc[0]
        offset_z = merged['Loc_Z'].iloc[0] - merged['Est_Z'].iloc[0]

        merged['Est_X'] += offset_x
        merged['Est_Y'] += offset_y
        merged['Est_Z'] += offset_z

        m1_df['Est_X'] += offset_x
        m1_df['Est_Y'] += offset_y
        m1_df['Est_Z'] += offset_z

        # ==========================================
        # ⬇️ THE FIX: FILTER OUT THE SETTLING SHOCK 
        # ==========================================
        start_time = merged['Timestamp'].min()
        settling_time = 1.5 # Ignore the first 1.5 seconds of EKF initialization
        
        merged = merged[merged['Timestamp'] > (start_time + settling_time)].reset_index(drop=True)
        m1_df = m1_df[m1_df['Timestamp'] > (start_time + settling_time)].reset_index(drop=True)
        odom_df = odom_df[odom_df['Timestamp'] > (start_time + settling_time)].reset_index(drop=True)

        if merged.empty:
            self.get_logger().warn("⚠️ Run was too short. Need more than 1.5 seconds of data to plot.")
            return

        # Core Metrics (Proceeds as normal, but without the 25,000 m/s spike!)
        merged['error_x'] = merged['Est_X'] - merged['Loc_X']
        merged['error_y'] = merged['Est_Y'] - merged['Loc_Y']
        merged['error_z'] = merged['Est_Z'] - merged['Loc_Z']
        merged['pos_error'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2 + merged['error_z']**2)
        
        merged['yaw_error'] = merged.apply(lambda row: angle_error(row['Est_Yaw'], math.radians(row['Yaw_Degrees'])), axis=1)
        merged['pitch_error'] = merged.apply(lambda row: angle_error(row['Est_Pitch'], math.radians(row['Pitch_Degrees'])), axis=1)
        merged['roll_error'] = merged.apply(lambda row: angle_error(row['Est_Roll'], math.radians(row['Roll_Degrees'])), axis=1)
        
        merged['quat_norm'] = np.sqrt(merged['Est_Qw']**2 + merged['Est_Qx']**2 + merged['Est_Qy']**2 + merged['Est_Qz']**2)

        # Jerk Calculation
        dt = merged['Timestamp'].diff().where(lambda x: x > 0.01, 0.01) 
        velocity_series = merged['GT_Velocity'].values
        
        if len(velocity_series) > 21:  
            smoothed_vel = merged['GT_Velocity'].rolling(window=11, center=True, min_periods=1).mean()
            try:
                smoothed_vel = pd.Series(signal.savgol_filter(smoothed_vel, window_length=21, polyorder=3), index=merged.index)
            except:
                pass
        else:
            smoothed_vel = merged['GT_Velocity'].rolling(window=20, center=True, min_periods=1).mean()
        
        smoothed_vel = smoothed_vel.ffill().bfill()
        merged['Long_Accel'] = smoothed_vel.diff() / dt
        merged['Long_Jerk'] = merged['Long_Accel'].diff() / dt
        
        yaw_rate_rad = np.radians(merged['Yaw_Degrees'].diff() / dt)
        merged['Lat_Accel'] = smoothed_vel * yaw_rate_rad
        merged['Lat_Jerk'] = merged['Lat_Accel'].diff() / dt

        start_time = merged['Timestamp'].min()
        valid_data = merged[merged['Timestamp'] > (start_time + 2.0)]
        abs_jerk = valid_data['Long_Jerk'].abs().dropna()

        # ==========================================
        # PLOT 1: Trajectory Fusion & 3D Error 
        # ==========================================
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        ax1 = fig.add_subplot(gs[0])
        margin = 5.0  
        ax1.plot(odom_df['Loc_X'], odom_df['Loc_Y'], color='black', linestyle='--', label='Ground Truth', linewidth=2)
        ax1.plot(m1_df['Est_X'], m1_df['Est_Y'], color='#1f77b4', linestyle='-', label='Live EKF Trajectory', linewidth=2)
        
        # ⬇️ FIXED: Grabbing Start/End directly from the aligned 'merged' frame
        start_x, start_y = merged['Loc_X'].iloc[0], merged['Loc_Y'].iloc[0]
        end_x, end_y = merged['Loc_X'].iloc[-1], merged['Loc_Y'].iloc[-1]
        
        ax1.scatter(start_x, start_y, marker='o', color='green', s=120, zorder=5, label='Start', edgecolor='white')
        ax1.scatter(end_x, end_y, marker='X', color='red', s=120, zorder=5, label='End', edgecolor='white')

        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title('Trajectory Sensor Fusion Comparison')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend(loc='best')

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(merged['Timestamp'], merged['pos_error'], color='#1f77b4', label='Position Error', linewidth=2)
        ax2.set_title('3D Localization Error Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '1_trajectory_and_error.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 2: Altitude Tracking
        # ==========================================
        plt.figure(figsize=(12, 4))
        plt.plot(merged['Timestamp'], merged['Loc_Z'], color='black', linestyle='--', label='Ground Truth Z', linewidth=2)
        plt.plot(merged['Timestamp'], merged['Est_Z'], color='#1f77b4', linestyle='-', label='EKF Z', linewidth=2, alpha=0.8)
        plt.fill_between(merged['Timestamp'], merged['Loc_Z'], merged['Est_Z'], alpha=0.2, color='gray')
        plt.title('Altitude Tracking (Z-Axis)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '2_altitude_tracking.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 3: Orientation Analysis
        # ==========================================
        fig, axes = plt.subplots(3, 2, figsize=(16, 10))
        est_yaw, gt_yaw = np.degrees(merged['Est_Yaw']), merged['Yaw_Degrees']
        est_pitch, gt_pitch = np.degrees(merged['Est_Pitch']), merged['Pitch_Degrees']
        est_roll, gt_roll = np.degrees(merged['Est_Roll']), merged['Roll_Degrees']

        axes[0, 0].plot(merged['Timestamp'], gt_yaw, 'k--', label='GT')
        axes[0, 0].plot(merged['Timestamp'], est_yaw, color='#1f77b4', label='EKF')
        axes[0, 0].set_title('Tracking')
        axes[0, 0].set_ylabel('Yaw (°)')
        axes[0, 0].legend()

        axes[1, 0].plot(merged['Timestamp'], gt_pitch, 'k--')
        axes[1, 0].plot(merged['Timestamp'], est_pitch, color='#2ca02c')
        axes[1, 0].set_ylabel('Pitch (°)')

        axes[2, 0].plot(merged['Timestamp'], gt_roll, 'k--')
        axes[2, 0].plot(merged['Timestamp'], est_roll, color='#d62728')
        axes[2, 0].set_ylabel('Roll (°)')

        axes[0, 1].plot(merged['Timestamp'], np.degrees(merged['yaw_error']), color='#1f77b4')
        axes[0, 1].set_title('Error')
        axes[0, 1].set_ylabel('Error (°)')
        
        axes[1, 1].plot(merged['Timestamp'], np.degrees(merged['pitch_error']), color='#2ca02c')
        axes[1, 1].set_ylabel('Error (°)')
        
        axes[2, 1].plot(merged['Timestamp'], np.degrees(merged['roll_error']), color='#d62728')
        axes[2, 1].set_ylabel('Error (°)')

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '3_orientation_analysis.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 4: Gyro Bias Convergence
        # ==========================================
        has_3axis = all(col in merged.columns for col in ['Est_Bias_Gx', 'Est_Bias_Gy', 'Est_Bias_Gz'])

        if has_3axis:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for bias_col, ax, label, color in [
                (merged['Est_Bias_Gx'], axes[0], 'Gyro Bias X', '#9467bd'),
                (merged['Est_Bias_Gy'], axes[1], 'Gyro Bias Y', '#e377c2'),
                (merged['Est_Bias_Gz'], axes[2], 'Gyro Bias Z', '#17becf')]:
                ax.plot(merged['Timestamp'], bias_col, color=color, linewidth=2, alpha=0.8)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_title(label)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Bias (rad/s)')
        else:
            plt.figure(figsize=(10, 4))
            plt.plot(merged['Timestamp'], merged.get('Est_Gyro_Bias', np.zeros(len(merged))), color='#9467bd', linewidth=2, alpha=0.8)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title('Gyro Bias Estimation Convergence')
            plt.xlabel('Time (s)')
            plt.ylabel('Bias (rad/s)')

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '4_gyro_bias_convergence.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 5: Jerk Analysis
        # ==========================================
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(merged['Timestamp'], merged['Long_Jerk'], color='#ff7f0e', alpha=0.9, linewidth=2)
        ax1.axhline(y=3, color='green', linestyle='--', alpha=0.6, label='Comfort Limit')
        ax1.axhline(y=-3, color='green', linestyle='--', alpha=0.6)
        ax1.set_title('Longitudinal Jerk Over Time')
        ax1.set_ylim(-15, 15)
        ax1.legend()

        ax2 = fig.add_subplot(gs[1])
        h = ax2.hist2d(merged['Lat_Jerk'].fillna(0).clip(-10, 10), merged['Long_Jerk'].fillna(0).clip(-10, 10), bins=40, cmap='mako', range=[[-10, 10], [-10, 10]])
        fig.colorbar(h[3], ax=ax2, label='Frequency')
        ax2.set_title('2D Jerk Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '5_jerk_analysis.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 6: Spatial Error Map & Covariance
        # ==========================================
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2)
        
        ax1 = fig.add_subplot(gs[0])
        scatter = ax1.scatter(merged['Est_X'], merged['Est_Y'], c=merged['pos_error'], cmap='flare_r', s=30, zorder=3)
        ax1.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', alpha=0.4, label='Ground Truth', zorder=2)
        fig.colorbar(scatter, ax=ax1, label='3D Position Error (m)')
        ax1.set_title('EKF 2D Error Map')
        ax1.set_aspect('equal', adjustable='datalim')

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(merged['Est_X'], merged['Est_Y'], color=(237/255, 59/255, 178/255), label='Estimated Path', linewidth=2, zorder=2)
        for idx, row in merged.iloc[::30].iterrows():
            if row['P_dp_x'] > 0 and row['P_dp_y'] > 0:
                width, height = 2 * 2 * np.sqrt([row['P_dp_x'], row['P_dp_y']])
                ellipse = Ellipse((row['Est_X'], row['Est_Y']), width, height, angle=0, alpha=0.3, color="#098931", zorder=1)
                ax2.add_patch(ellipse)
        ax2.set_title('Position Covariance Ellipses (2σ)')
        ax2.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '6_spatial_error_and_covariance.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 7: Speed Tracking
        # ==========================================
        plt.figure(figsize=(12, 5))
        ekf_speed = np.sqrt(merged['Est_Vx']**2 + merged['Est_Vy']**2)
        plt.plot(merged['Timestamp'], ekf_speed, color='#1f77b4', linewidth=2, label='EKF Speed', alpha=0.9)
        plt.plot(merged['Timestamp'], merged['GT_Velocity'], color='black', linestyle='--', linewidth=2, label='GT Speed', alpha=0.7)
        plt.fill_between(merged['Timestamp'], ekf_speed, merged['GT_Velocity'], alpha=0.2, color='gray')
        plt.title('EKF Speed Tracking vs Ground Truth')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '7_speed_tracking.png'), dpi=120)
        plt.close()

        # ==========================================
        # GENERATE CORE METRICS JSON
        # ==========================================
        duration = merged['Timestamp'].max() - merged['Timestamp'].min()
        
        metrics = {
            "duration_s": float(duration),
            "samples": len(merged),
            "rmse_pos_3d": float(np.sqrt(np.mean(merged['pos_error']**2))),
            "rmse_x": float(np.sqrt(np.mean(merged['error_x']**2))),
            "rmse_y": float(np.sqrt(np.mean(merged['error_y']**2))),
            "rmse_z": float(np.sqrt(np.mean(merged['error_z']**2))),
            "max_error_3d": float(merged['pos_error'].max()),
            "mean_error_3d": float(merged['pos_error'].mean()),
            "rmse_yaw_deg": float(np.sqrt(np.mean(np.degrees(merged['yaw_error'])**2))),
            "rmse_pitch_deg": float(np.sqrt(np.mean(np.degrees(merged['pitch_error'])**2))),
            "rmse_roll_deg": float(np.sqrt(np.mean(np.degrees(merged['roll_error'])**2))),
            "avg_speed": float(merged['GT_Velocity'].mean()),
            "max_jerk": float(abs_jerk.max()) if not abs_jerk.empty else 0.0,
        }

        if has_3axis:
            metrics["final_bias_gx"] = float(merged['Est_Bias_Gx'].iloc[-1])
            metrics["final_bias_gy"] = float(merged['Est_Bias_Gy'].iloc[-1])
            metrics["final_bias_gz"] = float(merged['Est_Bias_Gz'].iloc[-1])
            metrics["final_bias_magnitude"] = float(np.sqrt(metrics["final_bias_gx"]**2 + metrics["final_bias_gy"]**2 + metrics["final_bias_gz"]**2))

        metrics_path = os.path.join(run_dir, 'ekf_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.get_logger().info(f"✅ Generated 7 Dashboard-Style plots in {run_dir}/")
        self.get_logger().info(f"✅ Evaluated {int(metrics['samples'])} samples over {metrics['duration_s']:.1f}s")
        self.get_logger().info(f"✅ Final 3D RMSE: {metrics['rmse_pos_3d']:.3f}m | Yaw RMSE: {metrics['rmse_yaw_deg']:.3f}°")

def main(args=None):
    rclpy.init(args=args)
    node = EvaluateNode()
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