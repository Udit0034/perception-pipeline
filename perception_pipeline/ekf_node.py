import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
import message_filters
import numpy as np
import math

# ==========================================
# MATH UTILITIES
# ==========================================
def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])

def small_angle_quaternion(delta_theta: np.ndarray) -> np.ndarray:
    d = np.asarray(delta_theta, dtype=float).reshape(3)
    theta = np.linalg.norm(d)
    if theta < 1e-10:
        return normalize_quaternion(np.array([1.0, 0.5 * d[0], 0.5 * d[1], 0.5 * d[2]]))
    axis = d / theta
    half = 0.5 * theta
    return np.array([np.cos(half), *(axis * np.sin(half))])

def euler_to_quaternion(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return normalize_quaternion(np.array([qw, qx, qy, qz]))

def quaternion_to_euler(q):
    qw, qx, qy, qz = normalize_quaternion(q)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return wrap_angle(yaw), wrap_angle(pitch), wrap_angle(roll)

def quaternion_to_yaw(q: np.ndarray) -> float:
    yaw, _, _ = quaternion_to_euler(q)
    return yaw


# ==========================================
# EKF CLASS
# ==========================================
class ESEKF12:
    IX, IY, IZ = 0, 1, 2
    IQW, IQX, IQY, IQZ = 3, 4, 5, 6
    IVX, IVY, IVZ = 7, 8, 9
    IBGX, IBGY, IBGZ = 10, 11, 12

    NOMINAL_DIM = 13
    ERROR_DIM = 12

    def __init__(self):
        self.x = np.zeros((self.NOMINAL_DIM, 1))
        self.x[self.IQW, 0] = 1.0
        self.P = np.eye(self.ERROR_DIM) * 1.0

        self.Q = np.diag([
            2.5107352169022525, 0.2990501885783244, 1e-05,
            2.083317118060237e-05, 0.004702805081859701, 0.0520821914365421,
            0.00046629944609785705, 0.00014261275497685626, 2.904437062539407e-05,
            1.0788749814621699e-08, 2.6315423599172955e-07, 0.00819159600316757,
        ])

        self.R_gnss = np.diag([0.029675973485482014, 0.28730269734687247, 1.4238959381497729])
        self.R_vel_3d = np.diag([0.04239846648067301, 1.0, 0.00029661512082430056])
        self.R_compass = np.array([[0.049387892829388665]])
        self.R_gnss_yaw = np.array([[0.049387892829388665]]) 
        self.R_dual_gnss_yaw = np.array([[0.05]])          
        self.R_pitch_roll = np.diag([0.0001, 0.20660105101293294])
        self.R_vo_pos = np.diag([1.0568245483161656, 0.2960289179277321, 0.15765852629457774])
        self.R_vo_orient = np.diag([0.1646091287778927, 0.7567409919301394, 3.1622776601683795])
        
        # New Covariances for Proxy Sensors
        self.R_altimeter = np.array([[0.09]]) # 0.3^2
        self.R_wheel_odom = np.array([[0.01]]) # 0.1^2

        self._initialized = False

    def initialize_state(self, x, y, z, vx, yaw, pitch, roll, vy=0.0, vz=0.0, bias_x=0.0, bias_y=0.0, bias_z=0.0):
        q = euler_to_quaternion(yaw, pitch, roll)
        self.x = np.array([
            [x], [y], [z],
            [q[0]], [q[1]], [q[2]], [q[3]],
            [vx], [vy], [vz],
            [bias_x], [bias_y], [bias_z],
        ])
        self._initialized = True

    def get_quaternion(self) -> np.ndarray:
        return self.x[self.IQW:self.IQZ + 1, 0].copy()

    def get_euler(self):
        return quaternion_to_euler(self.get_quaternion())

    def _inject_error_state(self, delta_x: np.ndarray):
        d = delta_x.reshape(-1)
        self.x[self.IX:self.IZ + 1, 0] += d[0:3]
        q = self.get_quaternion()
        dq = small_angle_quaternion(d[3:6])
        self.x[self.IQW:self.IQZ + 1, 0] = normalize_quaternion(quaternion_multiply(q, dq))
        self.x[self.IVX:self.IVZ + 1, 0] += d[6:9]
        self.x[self.IBGX:self.IBGZ + 1, 0] += d[9:12]

    def predict(self, dt: float, accel_x: float, accel_y: float, gyro_x: float, gyro_y: float, gyro_z: float):
        if not self._initialized or dt <= 0.0: return
        p = self.x[self.IX:self.IZ + 1, 0]
        q = self.get_quaternion()
        v = self.x[self.IVX:self.IVZ + 1, 0]
        bg = self.x[self.IBGX:self.IBGZ + 1, 0]

        omega = np.array([gyro_x, gyro_y, gyro_z]) - bg
        dq = small_angle_quaternion(omega * dt)
        q_new = normalize_quaternion(quaternion_multiply(q, dq))
        yaw = quaternion_to_yaw(q_new)
        
        a_world = np.array([
            accel_x * np.cos(yaw) - accel_y * np.sin(yaw),
            accel_x * np.sin(yaw) + accel_y * np.cos(yaw),
            0.0,
        ])

        v_new = v + a_world * dt
        p_new = p + v * dt

        self.x[self.IX:self.IZ + 1, 0] = p_new
        self.x[self.IQW:self.IQZ + 1, 0] = q_new
        self.x[self.IVX:self.IVZ + 1, 0] = v_new

        F = np.eye(self.ERROR_DIM)
        F[0:3, 6:9] = np.eye(3) * dt
        F[3:6, 9:12] = -np.eye(3) * dt
        F[6, 5] = (-accel_x * np.sin(yaw) - accel_y * np.cos(yaw)) * dt
        F[7, 5] = ( accel_x * np.cos(yaw) - accel_y * np.sin(yaw)) * dt

        q_dt = self.Q * max(dt, 1e-3)
        self.P = F @ self.P @ F.T + q_dt

    def _error_update(self, residual: np.ndarray, H: np.ndarray, R: np.ndarray, angle_rows=()): 
        y = residual.copy()
        for idx in angle_rows:
            y[idx, 0] = wrap_angle(y[idx, 0])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y
        self._inject_error_state(delta_x)

        I_KH = np.eye(self.ERROR_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # --- Updates ---
    def update_gnss_3d(self, meas_x, meas_y, meas_z):
        if not self._initialized: return
        z = np.array([[meas_x], [meas_y], [meas_z]])
        p = self.x[self.IX:self.IZ + 1, :]
        residual = z - p
        H = np.zeros((3, self.ERROR_DIM))
        H[0:3, 0:3] = np.eye(3)
        self._error_update(residual, H, self.R_gnss)

    def update_compass_yaw(self, meas_yaw):
        if not self._initialized: return
        yaw, _, _ = self.get_euler()
        residual = np.array([[wrap_angle(meas_yaw - yaw)]])
        H = np.zeros((1, self.ERROR_DIM))
        H[0, 5] = 1.0
        self._error_update(residual, H, self.R_compass, angle_rows=(0,))

    def update_gnss_yaw(self, meas_yaw):
        if not self._initialized: return
        yaw, _, _ = self.get_euler()
        residual = np.array([[wrap_angle(meas_yaw - yaw)]])
        H = np.zeros((1, self.ERROR_DIM))
        H[0, 5] = 1.0
        self._error_update(residual, H, self.R_gnss_yaw, angle_rows=(0,))

    def update_dual_gnss_yaw(self, meas_yaw):
        if not self._initialized: return
        yaw, _, _ = self.get_euler()
        residual = np.array([[wrap_angle(meas_yaw - yaw)]])
        H = np.zeros((1, self.ERROR_DIM))
        H[0, 5] = 1.0
        self._error_update(residual, H, self.R_dual_gnss_yaw, angle_rows=(0,))

    def update_pitch_roll_from_accel(self, meas_pitch, meas_roll, accel_magnitude):
        if not self._initialized: return
        if abs(accel_magnitude - 9.81) > 0.5 and accel_magnitude > 0.5: return 
        _, pitch, roll = self.get_euler()
        z = np.array([[meas_pitch], [meas_roll]])
        h = np.array([[pitch], [roll]])
        residual = z - h
        H = np.zeros((2, self.ERROR_DIM))
        H[0, 4] = 1.0 
        H[1, 3] = 1.0 
        self._error_update(residual, H, self.R_pitch_roll, angle_rows=(0, 1))

    # --- New Proxy Updates ---
    def update_altimeter(self, meas_z):
        if not self._initialized: return
        residual = np.array([[meas_z - self.x[self.IZ, 0]]])
        H = np.zeros((1, self.ERROR_DIM))
        H[0, 2] = 1.0 # 2 is index for IZ in error state (dx, dy, dz)
        self._error_update(residual, H, self.R_altimeter)

    def update_wheel_odom(self, meas_speed):
        if not self._initialized: return
        yaw = quaternion_to_yaw(self.get_quaternion())
        v_x = self.x[self.IVX, 0]
        v_y = self.x[self.IVY, 0]
        
        # Project world velocity to body forward axis
        v_forward_est = v_x * np.cos(yaw) + v_y * np.sin(yaw)
        residual = np.array([[meas_speed - v_forward_est]])
        
        H = np.zeros((1, self.ERROR_DIM))
        H[0, 6] = np.cos(yaw) # 6 is dvx in error state
        H[0, 7] = np.sin(yaw) # 7 is dvy in error state
        self._error_update(residual, H, self.R_wheel_odom)


# ==========================================
# ROS2 NODE INTEGRATION
# ==========================================
class EKFFusionNode(Node):
    def __init__(self):
        super().__init__('ekf_fusion_node')

        self.ekf = ESEKF12()

        # Timing and state tracking
        self.last_time = None
        self.gnss_origin = None  
        self.prev_gnss_x = None
        self.prev_gnss_y = None
        self.prev_gnss_time = None

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/ekf/odometry', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/carla/imu/primary', self.imu_callback, 10)
        self.vo_sub = self.create_subscription(Odometry, '/vo/odometry', self.vo_callback, 10)
        
        # Proxy Subscribers
        self.altimeter_sub = self.create_subscription(PointStamped, '/carla/altimeter', self.altimeter_callback, 10)
        self.wheel_odom_sub = self.create_subscription(Odometry, '/carla/ego_vehicle/wheel_odom', self.wheel_odom_callback, 10)

        # Dual-GNSS Synchronization
        self.gnss_front_sub = message_filters.Subscriber(self, NavSatFix, '/carla/gnss/front')
        self.gnss_rear_sub = message_filters.Subscriber(self, NavSatFix, '/carla/gnss/rear')
        
        self.gnss_sync = message_filters.ApproximateTimeSynchronizer(
            [self.gnss_front_sub, self.gnss_rear_sub], queue_size=10, slop=0.05
        )
        self.gnss_sync.registerCallback(self.dual_gnss_callback)

        self.get_logger().info("EKF Fusion Node Started. Waiting for GNSS/IMU to initialize...")

    def get_time_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def latlon_to_xy(self, lat, lon):
        EARTH_RADIUS = 6378137.0
        if self.gnss_origin is None:
            return 0.0, 0.0
        lat0, lon0 = self.gnss_origin[0], self.gnss_origin[1]
        
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        
        x = dlon * math.cos(math.radians(lat0)) * EARTH_RADIUS
        y = dlat * EARTH_RADIUS
        return x, y

    def imu_callback(self, msg: Imu):
        current_time = self.get_time_sec(msg.header.stamp)
        
        if self.last_time is None:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        
        # ⬇️ FIXED: Bound dt to prevent physics explosions from ROS 2 startup lag
        dt = max(0.01, min(dt, 0.1)) 
        self.last_time = current_time

        if not self.ekf._initialized:
            return

        accel_x = msg.linear_acceleration.x
        accel_y = msg.linear_acceleration.y
        accel_z = msg.linear_acceleration.z
        gyro_x = msg.angular_velocity.x
        gyro_y = msg.angular_velocity.y
        gyro_z = msg.angular_velocity.z

        self.ekf.predict(dt, accel_x, -accel_y, gyro_x, gyro_y, gyro_z)

        q = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        yaw, pitch, roll = quaternion_to_euler(q)

        # Apply the 90 degree (pi/2) offset to fix the frame mismatch
        compass_yaw = wrap_angle(yaw - (math.pi / 2))

        self.ekf.update_compass_yaw(compass_yaw)
        
        accel_mag = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        self.ekf.update_pitch_roll_from_accel(pitch, roll, accel_mag)

        self.publish_ekf_state(msg.header.stamp)

    def dual_gnss_callback(self, front_msg: NavSatFix, rear_msg: NavSatFix):
        current_time = self.get_time_sec(front_msg.header.stamp)

        # ⬇️ FIXED: Initialization block now correctly calculates true starting yaw
        if self.gnss_origin is None:
            # Set the origin FIRST so latlon_to_xy works!
            self.gnss_origin = (front_msg.latitude, front_msg.longitude, front_msg.altitude)
            
            # Now calculate the real relative coordinates (front will be 0,0)
            curr_front_x, curr_front_y = self.latlon_to_xy(front_msg.latitude, front_msg.longitude)
            curr_rear_x, curr_rear_y = self.latlon_to_xy(rear_msg.latitude, rear_msg.longitude)
            
            # Calculate absolute yaw from dual antenna
            dx = curr_front_x - curr_rear_x
            dy = curr_front_y - curr_rear_y
            dual_gnss_yaw = math.atan2(dy, dx)
            
            if not self.ekf._initialized:
                self.ekf.initialize_state(
                    x=0.0, y=0.0, z=0.0,
                    vx=0.0, yaw=dual_gnss_yaw, pitch=0.0, roll=0.0
                )
                self.get_logger().info(f"EKF Initialized. TRUE Init Yaw: {math.degrees(dual_gnss_yaw):.2f}°")
            
            # Initialize timing tracking to avoid 2nd frame jumps
            self.prev_gnss_time = current_time
            self.prev_gnss_x = curr_front_x
            self.prev_gnss_y = curr_front_y
            return

        # --- Subseqent Frame Processing ---
        curr_front_x, curr_front_y = self.latlon_to_xy(front_msg.latitude, front_msg.longitude)
        curr_rear_x, curr_rear_y = self.latlon_to_xy(rear_msg.latitude, rear_msg.longitude)
        
        curr_front_z = front_msg.altitude - self.gnss_origin[2]

        dx = curr_front_x - curr_rear_x
        dy = curr_front_y - curr_rear_y
        dual_gnss_yaw = math.atan2(dy, dx)

        # 1. Base 3D Position Update
        self.ekf.update_gnss_3d(curr_front_x, curr_front_y, curr_front_z)

        # 2. Single GNSS (COG) Update
        if self.prev_gnss_x is not None:
            gnss_dt = current_time - self.prev_gnss_time
            if gnss_dt > 0.0:
                vx = (curr_front_x - self.prev_gnss_x) / gnss_dt
                vy = (curr_front_y - self.prev_gnss_y) / gnss_dt
                speed = math.sqrt(vx**2 + vy**2)
                if speed > 1.0:
                    gnss_yaw = math.atan2(vy, vx)
                    self.ekf.update_gnss_yaw(gnss_yaw)

        # 3. Dual GNSS Absolute Heading Update
        self.ekf.update_dual_gnss_yaw(dual_gnss_yaw)

        self.prev_gnss_x = curr_front_x
        self.prev_gnss_y = curr_front_y
        self.prev_gnss_time = current_time

    def altimeter_callback(self, msg: PointStamped):
        if not self.ekf._initialized: return
        self.ekf.update_altimeter(msg.point.z)

    def wheel_odom_callback(self, msg: Odometry):
        if not self.ekf._initialized: return
        self.ekf.update_wheel_odom(msg.twist.twist.linear.x)

    def vo_callback(self, msg: Odometry):
        if not self.ekf._initialized:
            return

        meas_x = msg.pose.pose.position.x
        meas_y = msg.pose.pose.position.y
        meas_z = msg.pose.pose.position.z
        
        q = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])
        meas_yaw, meas_pitch, meas_roll = quaternion_to_euler(q)

        self.ekf.update_vo(meas_x, meas_y, meas_z, meas_yaw, meas_pitch, meas_roll)

    def publish_ekf_state(self, stamp):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set Position & Velocity
        odom_msg.pose.pose.position.x = float(self.ekf.x[self.ekf.IX, 0])
        odom_msg.pose.pose.position.y = float(self.ekf.x[self.ekf.IY, 0])
        odom_msg.pose.pose.position.z = float(self.ekf.x[self.ekf.IZ, 0])
        
        odom_msg.twist.twist.linear.x = float(self.ekf.x[self.ekf.IVX, 0])
        odom_msg.twist.twist.linear.y = float(self.ekf.x[self.ekf.IVY, 0])
        odom_msg.twist.twist.linear.z = float(self.ekf.x[self.ekf.IVZ, 0])

        # Set Orientation
        q = self.ekf.get_quaternion()
        odom_msg.pose.pose.orientation.w = float(q[0])
        odom_msg.pose.pose.orientation.x = float(q[1])
        odom_msg.pose.pose.orientation.y = float(q[2])
        odom_msg.pose.pose.orientation.z = float(q[3])

        # ⬇️ FIXED: Pack Gyro Biases into the unused Angular Twist fields
        odom_msg.twist.twist.angular.x = float(self.ekf.x[self.ekf.IBGX, 0])
        odom_msg.twist.twist.angular.y = float(self.ekf.x[self.ekf.IBGY, 0])
        odom_msg.twist.twist.angular.z = float(self.ekf.x[self.ekf.IBGZ, 0])

        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFFusionNode()
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