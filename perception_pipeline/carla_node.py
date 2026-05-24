import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import CompressedImage, Image, Imu, NavSatFix, PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

import carla
import numpy as np
import time
import queue
import math

# Physical Camera Configurations
CAMS = {
    "front_left": {"fov": 90, "w": 1280, "h": 720, "x": 1.4, "y": -0.25, "z": 1.5, "pitch": -12.0, "yaw": 0.0, "roll": 0.0},
    "front_right": {"fov": 90, "w": 1280, "h": 720, "x": 1.4, "y": 0.25, "z": 1.5, "pitch": -12.0, "yaw": 0.0, "roll": 0.0},
    "rear": {"fov": 100, "w": 800, "h": 400, "x": -2.0, "y": 0.0, "z": 1.6, "pitch": -26.0, "yaw": 180.0, "roll": 0.0},
    "side_left": {"fov": 120, "w": 800, "h": 400, "x": 0.0, "y": -0.8, "z": 1.8, "pitch": -41.0, "yaw": -90.0, "roll": 0.0},
    "side_right": {"fov": 120, "w": 800, "h": 400, "x": 0.0, "y": 0.8, "z": 1.8, "pitch": -41.0, "yaw": 90.0, "roll": 0.0},
}

SENSOR_BLUEPRINTS = {
    'image': {'type': 'sensor.camera.rgb', 'encoding': 'bgr8', 'topic_suffix': 'image'},
    'depth': {'type': 'sensor.camera.depth', 'encoding': 'bgra8', 'topic_suffix': 'depth'},
    'seg': {'type': 'sensor.camera.semantic_segmentation', 'encoding': 'bgra8', 'topic_suffix': 'seg'},
}

class CarlaNode(Node):
    def __init__(self):
        super().__init__('carla_node')
        
        self.declare_parameter('debug', False)
        self.declare_parameter('integration_mode', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.integration_mode = self.get_parameter('integration_mode').get_parameter_value().bool_value
        self.get_logger().info(f"CARLA debug mode: {self.debug} | integration_mode: {self.integration_mode}")

        self.bridge = CvBridge()
        self.sensor_queue = queue.Queue()
        self.sensor_callback_count = 0  # <--- ADD THIS LINE
        self.actor_list = []
        self.running = True

        # ROS2 Publishers
        self.imu_pubs = {
            'primary': self.create_publisher(Imu, '/carla/imu/primary', 10),
            'backup': self.create_publisher(Imu, '/carla/imu/backup', 10)
        }
        self.gnss_pubs = {
            'front': self.create_publisher(NavSatFix, '/carla/gnss/front', 10),
            'rear': self.create_publisher(NavSatFix, '/carla/gnss/rear', 10)
        }
        self.camera_pubs = {
            'front_left': self.create_publisher(Image, '/carla/front_left/image_raw', 10)
        }
        self.gt_odom_pub = self.create_publisher(Odometry, '/carla/ego_vehicle/odometry', 10)
        
        # Connect to CARLA
        self.connect_to_carla()

        # SINGLE clean timer to drive the simulation at exactly 20Hz (0.05s)
        self.tick_timer = self.create_timer(0.05, self.world_tick)

    def connect_to_carla(self):
        self.get_logger().info("Connecting to CARLA on 127.0.0.1:2000...")
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)
        
        self.world = self.client.get_world()
        if self.world.get_map().name.split('/')[-1] != 'Town01':
            self.world = self.client.load_world('Town01')
            
        # PROPER SYNCHRONOUS SETUP
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 5
        self.world.apply_settings(settings)
        
        # FORCE AUTOPILOT TO SYNC WITH PHYSICS
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        self.tm = traffic_manager

        self.spawn_ego_and_sensors()
        self.setup_spectator_camera()

    def setup_spectator_camera(self):
        self.spectator_offset = carla.Vector3D(x=-15.0, y=0.0, z=8.0)
        self.spectator_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)
        self.get_logger().info("Spectator camera initialized (chase view mode)")

    @staticmethod
    def euler_to_quaternion(yaw, pitch, roll):
        yaw, pitch, roll = math.radians(yaw), math.radians(pitch), math.radians(roll)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        return np.array([
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ])

    def publish_gt_odometry(self, stamp):
        if self.ego_vehicle is None or not self.ego_vehicle.is_alive:
            return

        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = transform.location.x
        odom_msg.pose.pose.position.y = transform.location.y
        odom_msg.pose.pose.position.z = transform.location.z

        q = self.euler_to_quaternion(transform.rotation.yaw, transform.rotation.pitch, transform.rotation.roll)
        odom_msg.pose.pose.orientation.w = float(q[0])
        odom_msg.pose.pose.orientation.x = float(q[1])
        odom_msg.pose.pose.orientation.y = float(q[2])
        odom_msg.pose.pose.orientation.z = float(q[3])

        odom_msg.twist.twist.linear.x = velocity.x
        odom_msg.twist.twist.linear.y = velocity.y
        odom_msg.twist.twist.linear.z = velocity.z

        self.gt_odom_pub.publish(odom_msg)

    def make_sensor_callback(self, name, sensor_type):
        def callback(data):
            self._sensor_callback(name, sensor_type, data)
        return callback

    def _sensor_callback(self, name, sensor_type, data):
        self.sensor_callback_count += 1
        self.sensor_queue.put((name, sensor_type, data))

    def update_spectator_camera(self):
        if self.ego_vehicle is None or not self.ego_vehicle.is_alive:
            return
        ego_transform = self.ego_vehicle.get_transform()
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        up = ego_transform.get_up_vector()

        offset = self.spectator_offset
        spectator_loc = carla.Location(
            x=ego_transform.location.x + forward.x * offset.x + right.x * offset.y + up.x * offset.z,
            y=ego_transform.location.y + forward.y * offset.x + right.y * offset.y + up.y * offset.z,
            z=ego_transform.location.z + forward.z * offset.x + right.z * offset.y + up.z * offset.z,
        )

        ego_yaw = ego_transform.rotation.yaw
        spectator_rot = carla.Rotation(pitch=self.spectator_rotation.pitch, yaw=ego_yaw, roll=self.spectator_rotation.roll)
        self.world.get_spectator().set_transform(carla.Transform(spectator_loc, spectator_rot))

    def spawn_ego_and_sensors(self):
        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.find('vehicle.tesla.model3')
        
        # Spawn Ego
        spawn_points = self.world.get_map().get_spawn_points()
        self.ego_vehicle = None
        for spawn_point in spawn_points[:10]:
            try:
                self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
                break
            except RuntimeError:
                continue
                
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")
            
        self.actor_list.append(self.ego_vehicle)
        self.ego_vehicle.set_autopilot(True, self.tm.get_port())
        self.get_logger().info("Ego vehicle spawned and autopilot engaged.")

        # Expected sensors: 2 IMU, 2 GNSS, and front_left camera
        sensor_types = ['image']
        self.expected_sensor_frames = 2 + 2 + 1
        self.get_logger().info(f"Expecting {self.expected_sensor_frames} CARLA sensor streams per tick.")

        # ==========================================
        # CAMERAS
        # ==========================================
        # Only front_left camera is enabled here; CARLA synchronous mode drives frames.
        for name, ext in CAMS.items():
            if name != 'front_left':
                continue
            tf = carla.Transform(
                carla.Location(x=ext['x'], y=ext['y'], z=ext['z']), 
                carla.Rotation(pitch=ext['pitch'], yaw=ext['yaw'], roll=ext['roll'])
            )
            for sensor_type in sensor_types:
                blueprint = bp_lib.find(SENSOR_BLUEPRINTS[sensor_type]['type'])
                blueprint.set_attribute('image_size_x', str(ext['w']))
                blueprint.set_attribute('image_size_y', str(ext['h']))
                blueprint.set_attribute('fov', str(ext['fov']))
                # CARLA sync mode controls timing, no explicit sensor_tick required.
                # blueprint.set_attribute('sensor_tick', '1.0')

                cam = self.world.spawn_actor(blueprint, tf, attach_to=self.ego_vehicle)
                cam.listen(self.make_sensor_callback(name, sensor_type))
                self.actor_list.append(cam)

        # ==========================================
        # IMU (Primary & Backup)
        # ==========================================
        imu_bp = bp_lib.find('sensor.other.imu')
        # NOTE: CARLA synchronous mode and world.tick() drive all sensor updates.
        # sensor_tick is not needed here and can conflict with fixed_delta_seconds.
        # imu_bp.set_attribute('sensor_tick', '1.0') # FORCED 1Hz
        center_tf = carla.Transform(carla.Location(x=0, z=0))
        
        for name in ['primary', 'backup']:
            imu = self.world.spawn_actor(imu_bp, center_tf, attach_to=self.ego_vehicle)
            imu.listen(self.make_sensor_callback(name, 'imu'))
            self.actor_list.append(imu)

        # ==========================================
        # GNSS (Front & Rear)
        # ==========================================
        gnss_bp = bp_lib.find('sensor.other.gnss')
        # NOTE: CARLA synchronous mode and world.tick() drive all sensor updates.
        # sensor_tick is not needed here and can conflict with fixed_delta_seconds.
        # gnss_bp.set_attribute('sensor_tick', '1.0') # FORCED 1Hz
        
        gnss_f = self.world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=1.0, z=1.5)), attach_to=self.ego_vehicle)
        gnss_f.listen(self.make_sensor_callback('front', 'gnss'))
        self.actor_list.append(gnss_f)

        gnss_r = self.world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=-1.0, z=1.5)), attach_to=self.ego_vehicle)
        gnss_r.listen(self.make_sensor_callback('rear', 'gnss'))
        self.actor_list.append(gnss_r)

        # ==========================================
        # LIDAR
        # ==========================================
        # LIDAR disabled:
        # lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        # lidar_bp.set_attribute('channels', '32')
        # lidar_bp.set_attribute('range', '100')
        # lidar_bp.set_attribute('rotation_frequency', '1') # Matched to 1Hz world tick
        # lidar_bp.set_attribute('points_per_second', '500000')
        # lidar_bp.set_attribute('sensor_tick', '1.0') # FORCED 1Hz

        # lidar = self.world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8)), attach_to=self.ego_vehicle)
        # lidar.listen(self.make_sensor_callback('top', 'lidar'))
        # self.actor_list.append(lidar)

        # ==========================================
        # RADARS
        # ==========================================
        # Radar sensors disabled:
        # def spawn_radar(name, x, y, z, yaw, h_fov, v_fov, rng):
        #     bp = bp_lib.find('sensor.other.radar')
        #     bp.set_attribute('horizontal_fov', str(h_fov))
        #     bp.set_attribute('vertical_fov', str(v_fov))
        #     bp.set_attribute('range', str(rng))
        #     bp.set_attribute('sensor_tick', '1.0') # FORCED 1Hz
        #     
        #     tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
        #     radar = self.world.spawn_actor(bp, tf, attach_to=self.ego_vehicle)
        #     radar.listen(self.make_sensor_callback(name, 'radar'))
        #     self.actor_list.append(radar)

        # spawn_radar('front',  2.0,  0.0, 0.5, 0,   30, 15, 100)
        # spawn_radar('rear',  -2.0,  0.0, 0.5, 180, 30, 15, 100)
        # spawn_radar('fl',     1.8, -0.8, 0.5, -45,  90, 30, 30)
        # spawn_radar('fr',     1.8,  0.8, 0.5, 45,   90, 30, 30)
        # spawn_radar('rl',    -1.8, -0.8, 0.5, -135, 90, 30, 30)
        # spawn_radar('rr',    -1.8,  0.8, 0.5, 135,  90, 30, 30)

        self.get_logger().info("All V6 sensors spawned successfully at 1Hz.")

    def world_tick(self):
        if not self.running:
            return
        
        try:
            # 1. Step the physics engine & autopilot
            self.world.tick()
            
            # 2. Update Ground Truth & Camera
            if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
                self.update_spectator_camera()
                stamp = self.get_clock().now().to_msg()
                self.publish_gt_odometry(stamp)

            # 3. Drain the sensor queue for this frame
            expected = getattr(self, 'expected_sensor_frames', 0)
            gathered = 0
            
            while gathered < expected:
                try:
                    name, sensor_type, data = self.sensor_queue.get(timeout=1.0)
                    
                    if sensor_type == 'imu':
                        msg = Imu()
                        msg.header.stamp = stamp
                        msg.header.frame_id = f"imu_{name}"
                        msg.linear_acceleration.x = data.accelerometer.x
                        msg.linear_acceleration.y = data.accelerometer.y
                        msg.linear_acceleration.z = data.accelerometer.z
                        msg.angular_velocity.x = data.gyroscope.x
                        msg.angular_velocity.y = data.gyroscope.y
                        msg.angular_velocity.z = data.gyroscope.z
                        self.imu_pubs[name].publish(msg)

                    elif sensor_type == 'gnss':
                        msg = NavSatFix()
                        msg.header.stamp = stamp
                        msg.header.frame_id = f"gnss_{name}"
                        msg.latitude = data.latitude
                        msg.longitude = data.longitude
                        msg.altitude = data.altitude
                        self.gnss_pubs[name].publish(msg)

                    elif sensor_type == 'image':
                        msg = Image()
                        msg.header.stamp = stamp
                        msg.header.frame_id = 'front_left'
                        msg.height = data.height
                        msg.width = data.width
                        msg.encoding = 'bgra8'
                        msg.is_bigendian = 0
                        msg.step = data.width * 4
                        msg.data = data.raw_data
                        self.camera_pubs[name].publish(msg)

                    gathered += 1
                except queue.Empty:
                    self.get_logger().warn(f"Sensor queue timeout! Gathered {gathered}/{expected}")
                    break

        except Exception as e:
            self.get_logger().error(f"CARLA world tick failed: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down CARLA node and cleaning up actors...")
        self.running = False
        try:
            self.control_timer.cancel()
        except Exception:
            pass
        try:
            self.sensor_timer.cancel()
        except Exception:
            pass
        for actor in list(self.actor_list):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as e:
                self.get_logger().error(f"Failed to destroy CARLA actor: {e}")
        self.actor_list.clear()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CarlaNode()
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