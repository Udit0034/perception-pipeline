import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import math

# ==========================================
# LIGHTWEIGHT VO CLASS 
# ==========================================
class VisualOdometry:
    def __init__(self, fov=90.0, width=512, height=256):
        self.width = width
        self.height = height
        
        # Compute Camera Intrinsics
        focal = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        self.K = np.array([
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((4, 1))

        # Tracking State
        self.initialized = False
        self.prev_img = None
        self.prev_depth = None
        self.prev_kps = None
        
        # Global Pose Tracking
        self.global_x = 0.0
        self.global_y = 0.0
        self.global_z = 0.0
        self.global_roll = 0.0
        self.global_pitch = 0.0
        self.global_yaw = 0.0

        # ---------------------------------------------------------
        # ⬇️ PERFORMANCE LIMITERS: REDUCED FEATURES & TRACKING MATH
        # ---------------------------------------------------------
        # winSize 15x15 and maxLevel 2 drastically reduces optical flow CPU usage
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # maxCorners reduced from 1000 to 250. Higher quality and minDistance 
        # forces it to only pick the absolute best, spread-out features.
        self.feature_params = dict(maxCorners=250, qualityLevel=0.05, minDistance=20, blockSize=3)

    def initialize(self, rgb, depth, x, y, z, roll, pitch, yaw):
        """Initializes the VO tracker with the first frame and starting pose."""
        self.prev_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.prev_depth = depth.copy()
        
        # Extract initial limited feature set
        self.prev_kps = cv2.goodFeaturesToTrack(self.prev_img, mask=None, **self.feature_params)
        
        self.global_x, self.global_y, self.global_z = x, y, z
        self.global_roll, self.global_pitch, self.global_yaw = roll, pitch, yaw
        
        self.initialized = True
        print(f"✅ [VO] Lightweight RGB-D Odometry Initialized at ({x:.2f}, {y:.2f}, {z:.2f})")

    def process_frame(self, rgb, depth):
        """Computes odometry using the new RGB frame and Depth map."""
        if not self.initialized:
            return self.global_x, self.global_y, self.global_z, self.global_roll, self.global_pitch, self.global_yaw

        # ==========================================
        # PLACEHOLDER: Your KLT / PnP logic goes here
        # ==========================================
        
        # If your tracked features drop too low, remember to re-extract using self.feature_params
        # if len(self.prev_kps) < 50:
        #     self.prev_kps = cv2.goodFeaturesToTrack(...)

        return self.global_x, self.global_y, self.global_z, self.global_roll, self.global_pitch, self.global_yaw


# ==========================================
# ROS2 NODE FOR VISUAL ODOMETRY
# ==========================================
class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')

        self.bridge = CvBridge()
        self.vo = VisualOdometry()

        # Throttling to save CPU: 1 = process every frame, 2 = process half the frames
        self.frame_counter = 0
        self.process_every_n_frames = 2 

        # Publisher for the resulting Odometry
        self.odom_pub = self.create_publisher(Odometry, '/vo/odometry', 10)

        # Set up subscribers using message_filters to synchronize the two streams
        self.rgb_sub = message_filters.Subscriber(self, Image, '/carla/front_left/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/carla/front_left/depth')

        # ApproximateTimeSynchronizer handles varying arrival times of the two threads
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1 # Allowed time difference between frames (seconds)
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Lightweight Visual Odometry Node started...")

    def sync_callback(self, rgb_msg, depth_msg):
        """Called only when a matching RGB and Depth frame arrive together."""
        # ⬇️ THROTTLE LOGIC: Skip frames to save CPU
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return

        try:
            # 1. Convert ROS messages to OpenCV formats
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # 2. Pass to VO Logic
            if not self.vo.initialized:
                # Initialize at origin (or fetch real starting pose via TF/GNSS if needed)
                self.vo.initialize(cv_rgb, cv_depth, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
                return

            x, y, z, roll, pitch, yaw = self.vo.process_frame(cv_rgb, cv_depth)

            # 3. Publish Output
            self.publish_odometry(x, y, z, roll, pitch, yaw, rgb_msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f"Error in VO processing: {e}")

    def publish_odometry(self, x, y, z, roll, pitch, yaw, timestamp):
        """Formats and publishes the standard nav_msgs/Odometry message."""
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'camera_link'

        # Set Position
        odom_msg.pose.pose.position.x = float(x)
        odom_msg.pose.pose.position.y = float(y)
        odom_msg.pose.pose.position.z = float(z)

        # Set Orientation (Convert Euler to Quaternion)
        q = self.euler_to_quaternion(roll, pitch, yaw)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        self.odom_pub.publish(odom_msg)

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """Helper to convert Euler angles to Quaternion without external ROS2 tf dependencies."""
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()