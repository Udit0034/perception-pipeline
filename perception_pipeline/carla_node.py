import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import carla
import numpy as np
import time
import queue

# Physical Camera Configurations from your original script
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
        
        # Parameters
        self.declare_parameter('debug', False)
        self.declare_parameter('integration_mode', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.integration_mode = self.get_parameter('integration_mode').get_parameter_value().bool_value
        self.get_logger().info(f"CARLA debug mode: {self.debug} | integration_mode: {self.integration_mode}")

        # Tools
        self.bridge = CvBridge()
        self.sensor_queue = queue.Queue()
        self.actor_list = []
        self.tick_counter = 0
        self.sensor_callback_count = 0
        
        # ROS2 Publishers for each camera
        self.publishers_dict = {}
        for cam_name in CAMS.keys():
            self.publishers_dict[cam_name] = {}
            self.publishers_dict[cam_name]['image'] = self.create_publisher(Image, f'/carla/{cam_name}/image', 10)
            self.get_logger().info(f"Created Publisher: /carla/{cam_name}/image")
            if self.debug:
                self.publishers_dict[cam_name]['depth'] = self.create_publisher(Image, f'/carla/{cam_name}/depth', 10)
                self.publishers_dict[cam_name]['seg'] = self.create_publisher(Image, f'/carla/{cam_name}/seg', 10)
                self.get_logger().info(f"Created Publisher: /carla/{cam_name}/depth")
                self.get_logger().info(f"Created Publisher: /carla/{cam_name}/seg")

        # Connect to CARLA
        self.connect_to_carla()
        
        # Timer to tick CARLA and publish frames at 1Hz or 50Hz for integration/testing
        timer_period = 1.0 if self.integration_mode else 0.02
        self.timer = self.create_timer(timer_period, self.tick_and_publish)

    def connect_to_carla(self):
        self.get_logger().info("Connecting to CARLA on 127.0.0.1:2000...")
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(60.0)
        
        self.world = self.client.get_world()
        if self.world.get_map().name.split('/')[-1] != 'Town01':
            self.world = self.client.load_world('Town01')
            
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 if self.integration_mode else 0.02
        self.world.apply_settings(settings)
        
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        self.spawn_ego_and_sensors()
        self.setup_spectator_camera()

    def setup_spectator_camera(self):
        """Set spectator camera to chase view matching dashboard_node VC config"""
        # Chase camera config from dashboard_node.py VC
        self.spectator_offset = carla.Vector3D(x=-15.0, y=0.0, z=8.0)
        self.spectator_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)
        self.get_logger().info("Spectator camera initialized (chase view mode)")

    def make_sensor_callback(self, name, sensor_type):
        def callback(data):
            self._sensor_callback(name, sensor_type, data)
        return callback

    def _sensor_callback(self, name, sensor_type, data):
        self.sensor_callback_count += 1
        if self.sensor_callback_count == 1 or self.sensor_callback_count % 50 == 0:
            self.get_logger().info(f"CARLA sensor callback #{self.sensor_callback_count}: {name}/{sensor_type}")
        self.sensor_queue.put((name, sensor_type, data))

    def update_spectator_camera(self):
        """Update spectator camera position relative to ego vehicle"""
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

        sensor_types = ['image', 'depth', 'seg'] if self.debug else ['image']
        self.expected_sensor_frames = len(CAMS) * len(sensor_types)
        self.get_logger().info(f"Expecting {self.expected_sensor_frames} CARLA camera streams.")

        # Spawn Cameras
        for name, ext in CAMS.items():
            tf = carla.Transform(
                carla.Location(x=ext['x'], y=ext['y'], z=ext['z']), 
                carla.Rotation(pitch=ext['pitch'], yaw=ext['yaw'], roll=ext['roll'])
            )
            for sensor_type in sensor_types:
                blueprint = bp_lib.find(SENSOR_BLUEPRINTS[sensor_type]['type'])
                blueprint.set_attribute('image_size_x', str(ext['w']))
                blueprint.set_attribute('image_size_y', str(ext['h']))
                blueprint.set_attribute('fov', str(ext['fov']))
                blueprint.set_attribute('sensor_tick', '1.0' if self.integration_mode else '0.033')

                cam = self.world.spawn_actor(blueprint, tf, attach_to=self.ego_vehicle)
                cam.listen(self.make_sensor_callback(name, sensor_type))
                self.actor_list.append(cam)
                self.get_logger().info(f"Spawned {sensor_type} camera for {name}.")

    def tick_and_publish(self):
        try:
            self.world.tick()
            self.update_spectator_camera()
        except Exception as e:
            self.get_logger().error(f"CARLA tick failed: {e}", exc_info=True)
            return
        
        self.tick_counter += 1
        expected = getattr(self, 'expected_sensor_frames', len(CAMS) * (3 if self.debug else 1))
        gathered = 0
        timeout_start = time.time()

        while gathered < expected and (time.time() - timeout_start) < 2.0:
            try:
                cam_name, sensor_type, data = self.sensor_queue.get(timeout=0.1)
                bgra = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)

                if sensor_type == 'image':
                    bgr = bgra[:, :, :3]
                    msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                else:
                    msg = self.bridge.cv2_to_imgmsg(bgra, encoding='bgra8')

                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = cam_name
                self.publishers_dict[cam_name][sensor_type].publish(msg)
                gathered += 1
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Failed to publish CARLA sensor frame: {e}", exc_info=True)
                continue

        if gathered == 0:
            self.get_logger().warn(
                f"CARLA tick produced no sensor frames after {time.time() - timeout_start:.2f}s; queue size={self.sensor_queue.qsize()}"
            )
        elif self.debug and gathered < expected:
            self.get_logger().warn(
                f"CARLA tick produced partial data {gathered}/{expected}; queue size={self.sensor_queue.qsize()}"
            )
        elif self.debug and self.tick_counter % 50 == 0:
            self.get_logger().info(f"CARLA tick published {gathered}/{expected} frames; queue size={self.sensor_queue.qsize()}")

    def destroy_node(self):
        self.get_logger().info("Shutting down CARLA node and cleaning up actors...")
        for actor in list(self.actor_list):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as e:
                self.get_logger().error(f"Failed to destroy CARLA actor: {e}", exc_info=True)
        self.actor_list.clear()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CarlaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()