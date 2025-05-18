import os
import time
import queue
import random
import weakref
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import carla
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET

from environment.global_route_planner import GlobalRoutePlanner
from environment.local_planner import LocalPlanner, RoadOption
from environment.controller import PIDLateralController

# Configuration constants
IM_WIDTH = 384
IM_HEIGHT = 384
SPAWN_LOCATIONS = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/environment/spawn_locations_v2.xml"
ROUTES = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/environment/routes.xml"
TRAFFIC = True
OCCLUSION = True
MOVING_OCC = False
SPAWN_DELAY = 30
MAX_TRAFFIC = 30


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator with VLM control integration."""

    def __init__(self, render_mode=None, vlm_frames=3):
        # Standard gym environment setup
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255.0, shape=(3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
        self.render_mode = render_mode

        # VLM controller integration
        self.vlm_frames_needed = vlm_frames
        self.frame_buffer = []
        self.vlm_controller = None
        self.frame_save_dir = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/vlm_outputs/frames/example4"
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."

        # Create frame save directory if it doesn't exist
        os.makedirs(self.frame_save_dir, exist_ok=True)

        # Connect to CARLA server
        print('Connecting to CARLA server...')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print('CARLA server connected!')

        # Configure synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = 0.10
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)
        random.seed(0)

        # Route setup
        self.spawn_point1, self.route1, self.spawn_point2, self.route2 = self.generate_route()
        self.traffic_blueprints = self.filter_blueprints()
        self.alternate_spawn = False
        self.spawn_counter = 0

        # Blueprint and actor initialization
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_locations = self.get_spawn_locations()

        # Vehicle setup
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.vehicle = None
        self.prev_s = 0.0
        self.veh_transform = carla.Transform(
            carla.Location(
                float(self.spawn_locations.get("2")[1].get("x")),
                float(self.spawn_locations.get("2")[1].get("y")),
                float(self.spawn_locations.get("2")[1].get("z"))
            ),
            carla.Rotation(0, 90)
        )
        self.target_velocity = None
        self.ego_target = carla.Location(
            float(self.spawn_locations.get("2")[2].get("x")),
            float(self.spawn_locations.get("2")[2].get("y")),
            float(self.spawn_locations.get("2")[2].get("z"))
        )

        # Spectator camera
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                self.veh_transform.location + carla.Location(x=5, y=30, z=50),
                carla.Rotation(pitch=-90)
            )
        )

        # Pedestrian setup
        self.ped_bp = self.blueprint_library.filter("walker")[4]
        if self.ped_bp.has_attribute('is_invincible'):
            self.ped_bp.set_attribute('is_invincible', 'False')
        self.ped = None

        # Occlusion vehicle
        if OCCLUSION:
            self.ambulance = self.blueprint_library.filter("ambulance")[0]
            self.obsticle = None
            self.obsticle_transform = carla.Transform(
                carla.Location(
                    float(self.spawn_locations.get("2")[5].get("x")),
                    float(self.spawn_locations.get("2")[5].get("y")),
                    float(self.spawn_locations.get("2")[5].get("z"))
                ),
                carla.Rotation(0, 90)
            )

        # Sensor setup
        self.image_queue = queue.Queue()
        self.front_camera = np.zeros((3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)

        # Collision sensor
        self.collision_sensor = None
        self.collision_hist = []
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Lane invasion sensor
        self.lane_sensor = None
        self.lane_hist = []
        self.lane_hist_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')

        # Camera sensor
        self.camera_sensor = None
        self.camera_trans = carla.Transform(carla.Location(x=0.7, z=1.6))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        self.camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        self.camera_bp.set_attribute('fov', '110')
        self.camera_bp.set_attribute('sensor_tick', '0.1')

        # Navigation setup
        self.controller = None
        self.grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2)
        self.route = self.get_route()
        self.local_planner = None
        self.route_ind = 0

        # Metrics and state tracking
        self.speeds = []
        self.accs = []
        self.dets = []
        self.dist = []
        self.rewards = []
        self.timestep = 0
        self.ped_count = 0
        self.successful_ep = 0
        self.stall_ep = 0
        self.collision_ep = 0
        self.lane_ep = 0
        self.stopped = False
        self.passed = False

        # Goal location
        self.goal = carla.Location(
            float(-48.64543151855469),
            float(94),
            float(1.0)
        )

    def save_current_frame(self):
        """
        Save the current camera frame for VLM processing.
        This captures the current view from the vehicle's camera and saves it
        to disk for later analysis by the VLM.

        Returns:
            bool: True if we have enough frames for VLM processing
        """
        if self.front_camera is not None:
            # Convert from CHW to HWC format for PIL
            frame = self.front_camera.transpose(1, 2, 0)
            img = Image.fromarray(frame.astype('uint8'))

            # Create a unique timestamped filename
            timestamp = int(time.time() * 1000000)
            frame_path = os.path.join(self.frame_save_dir, f"frame_{timestamp}_{self.timestep}_opencv_detection.png")

            try:
                # Save the image to disk
                img.save(frame_path)

                # Add to frame buffer
                self.frame_buffer.append(frame_path)
                if len(self.frame_buffer) > self.vlm_frames_needed:
                    # Remove oldest frame (but don't delete it from disk - useful for analysis)
                    self.frame_buffer.pop(0)

                return True
            except Exception as e:
                print(f"Error saving frame: {e}")
                return False

        # Return True if we have enough frames for VLM processing
        return len(self.frame_buffer) >= self.vlm_frames_needed

    def get_current_vehicle_state(self):
        """
        Get relevant vehicle state information for the VLM.
        This provides context about the vehicle's current situation
        that can be used by the VLM to make better decisions.
        
        Returns:
            dict: Dictionary containing vehicle state information
        """
        # Calculate vehicle speed
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5  # m/s

        # Calculate distances to key objects
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        dist_to_goal = self.get_distance_to_goal(self.vehicle, self.goal)

        # Determine pedestrian relative position
        ped_location = self.ped.get_location()
        vehicle_location = self.vehicle.get_location()
        ped_forward = (ped_location.y > vehicle_location.y)  # Is pedestrian ahead of vehicle?
        
        # Enhanced pedestrian distance categories with increased detection range
        ped_distance_category = "NONE"
        if veh2ped_dist < 12.0:  # Increased from 7.5 to improve safety
            if veh2ped_dist < 5:  # Increased from 3 to give more reaction time
                ped_distance_category = "NEAR"
            elif veh2ped_dist < 8:  # Increased from 5 to improve medium-range response
                ped_distance_category = "MEDIUM"
            else:
                ped_distance_category = "FAR"

        # Enhanced pedestrian detection with increased range
        pedestrian_detected = veh2ped_dist < 12.0 and \
                            (self.get_distance_to_goal(self.ped, self.ped_target) > 2)

        # Get reward components if available
        reward_info = {}
        if hasattr(self, 'current_reward_components'):
            reward_info = self.current_reward_components

        # Get recent reward history (last 5 steps)
        recent_rewards = []
        if hasattr(self, 'reward_history') and len(self.reward_history) > 0:
            # Get last 5 rewards or fewer if not available
            history_length = min(5, len(self.reward_history))
            recent_rewards = self.reward_history[-history_length:]
        
        # Calculate simple trends if we have enough history
        safety_trend = "N/A"
        progress_trend = "N/A"
        smoothness_trend = "N/A"
        
        if len(recent_rewards) >= 3:
            # Get the average of the newest half vs oldest half of rewards
            midpoint = len(recent_rewards) // 2
            
            # Safety trend (more negative is worse)
            older_safety = sum([r.get('safety_reward', 0) for r in recent_rewards[:midpoint]]) / midpoint
            newer_safety = sum([r.get('safety_reward', 0) for r in recent_rewards[midpoint:]]) / (len(recent_rewards) - midpoint)
            safety_trend = "Improving" if newer_safety > older_safety else "Declining"
            
            # Progress trend (more positive is better)
            older_progress = sum([r.get('progress_reward', 0) for r in recent_rewards[:midpoint]]) / midpoint
            newer_progress = sum([r.get('progress_reward', 0) for r in recent_rewards[midpoint:]]) / (len(recent_rewards) - midpoint)
            progress_trend = "Improving" if newer_progress > older_progress else "Declining"
            
            # Smoothness trend (closer to zero is better)
            older_smoothness = sum([abs(r.get('smoothness_reward', 0)) for r in recent_rewards[:midpoint]]) / midpoint
            newer_smoothness = sum([abs(r.get('smoothness_reward', 0)) for r in recent_rewards[midpoint:]]) / (len(recent_rewards) - midpoint)
            smoothness_trend = "Improving" if newer_smoothness < older_smoothness else "Declining"
        
        # Return structured state information with reward data
        return {
            # Basic vehicle state
            "speed_ms": speed,
            "speed_kmh": speed * 3.6,
            "acceleration": speed - self.prev_s,  # Simple acceleration estimate
            
            # Pedestrian information
            "pedestrian_distance": veh2ped_dist,
            "pedestrian_distance_category": ped_distance_category,
            "pedestrian_detected": pedestrian_detected,
            "pedestrian_ahead": ped_forward,
            
            # Navigation and safety
            "distance_to_goal": dist_to_goal,
            "collision_detected": len(self.collision_hist) > 0,
            
            # History and context
            "timestep": self.timestep,
            "previous_action": self.current_vlm_action,
            "previous_justification": self.current_vlm_justification,
            "occlusion_present": OCCLUSION,
            
            # Reward information
            "current_rewards": reward_info,
            "recent_rewards": recent_rewards,
            
            # Trend analysis
            "safety_trend": safety_trend,
            "progress_trend": progress_trend,
            "smoothness_trend": smoothness_trend
        }

    def filter_blueprints(self):
        # Select some models from the blueprint library
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
        blueprints = []
        for vehicle in self.world.get_blueprint_library().filter('*vehicle*'):
            if any(model in vehicle.id for model in models):
                blueprints.append(vehicle)

        return blueprints

    def generate_route(self):
        spawn_points = self.world.get_map().get_spawn_points()
        route1_ind = [130, 29, 137, 90, 96, 3, 75, 6, 8, 16]
        route2_ind = [129, 28, 79, 86, 77, 2, 125, 7, 9, 15]
        spawn_point1 = spawn_points[130]
        route1 = []
        spawn_point2 = spawn_points[129]
        route2 = []
        # Draw the spawn point locations as numbers in the map
        for i in route1_ind:
            route1.append(spawn_points[i].location)
        for i in route2_ind:
            route2.append(spawn_points[i].location)
        return [spawn_point1, route1, spawn_point2, route2]

    def get_spawn_locations(self):
        # Use .iter() instead of deprecated .getchildren()
        data = {}
        tree = ET.parse(SPAWN_LOCATIONS)
        root = tree.getroot()
        for child in root:
            data[child.attrib["name"]] = [tag.attrib for tag in child.iter()]
        return data

    def get_route(self):
        data = {}
        route = []
        tree = ET.parse(ROUTES)
        root = tree.getroot()
        # Convert xml data to dict
        for child in root:
            data[child.attrib["id"]] = [tag.attrib for tag in child.iter()]
        # Simplify route generation with a helper method
        def create_route_from_waypoints(waypoints):
            return [
                [
                    self.grp._wmap.get_waypoint(
                        carla.Location(
                            float(waypoint.get("x")),
                            float(waypoint.get("y")),
                            float(waypoint.get("z"))
                        )
                    ),
                    RoadOption(int(waypoint.get("road_option")))
                ] for waypoint in waypoints
            ]
        return create_route_from_waypoints(data.get("0" if OCCLUSION else "1")[1:])

    def randomise_location(self, xyz):
        # Use a more robust method for float conversion
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        if xyz["orientation"] == "horizontal":
            x = random.uniform(
                safe_float(xyz["xmin"]),
                safe_float(xyz["xmax"])
            )
            y = safe_float(xyz["y"])
            z = safe_float(xyz["z"])
        elif xyz["orientation"] == "vertical":
            y = random.uniform(
                safe_float(xyz["ymin"]),
                safe_float(xyz["ymax"])
            )
            x = safe_float(xyz["x"])
            z = safe_float(xyz["z"])
        return carla.Transform(carla.Location(x, y, z), carla.Rotation(0, 0))

    def spawn_traffic(self):
        # Add some error handling and logging
        try:
            n_vehicles = len(self.world.get_actors().filter('*vehicle*'))
            # Check if we can spawn more vehicles
            if self.spawn_counter == SPAWN_DELAY and n_vehicles < MAX_TRAFFIC:
                # Choose vehicle blueprint
                vehicle_bp = random.choice(self.traffic_blueprints)
                # Select spawn point
                spawn_point = self.spawn_point1 if self.alternate_spawn else self.spawn_point2
                # Attempt to spawn vehicle
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    # Configure vehicle
                    vehicle.set_autopilot(True)
                    # Traffic manager configurations
                    self.traffic_manager.update_vehicle_lights(vehicle, True)
                    self.traffic_manager.random_left_lanechange_percentage(vehicle, 0)
                    self.traffic_manager.random_right_lanechange_percentage(vehicle, 0)
                    self.traffic_manager.auto_lane_change(vehicle, False)
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100)
                    self.traffic_manager.global_percentage_speed_difference(30)
                    # Set route
                    route = self.route1 if self.alternate_spawn else self.route2
                    self.traffic_manager.set_path(vehicle, route)
                    # Toggle spawn point
                    self.alternate_spawn = not self.alternate_spawn
                # Manage spawn counter
                self.spawn_counter = max(0, self.spawn_counter - 1)
            elif self.spawn_counter > 0:
                self.spawn_counter -= 1
            else:
                self.spawn_counter = SPAWN_DELAY
        except Exception as e:
            print(f"Error in spawn_traffic: {e}")

    def reset(self, seed=None, options=None):
        """Reset the environment to begin a new episode"""
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Update timesteps and counters
        self.timestep = 0
        self.ped_count = 0
        self.stopped = False
        # Clear frame buffer for VLM
        self.frame_buffer = []
        # Reset VLM action to default
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."
        # Clean up existing sensors
        if self.camera_sensor is not None:
            if self.camera_sensor.is_listening():
                self.camera_sensor.stop()
            if self.collision_sensor is not None and self.collision_sensor.is_listening():
                self.collision_sensor.stop()
            if self.lane_sensor is not None and self.lane_sensor.is_listening():
                self.lane_sensor.stop()
        # Clear sensor objects
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera_sensor = None
        # Clear sensor arrays
        self.collision_hist = []
        self.lane_hist = []
        self.image_queue = queue.Queue()
        # Delete all actors from previous episode
        self._clear_all_actors(
            ['sensor.other.lane_invasion', 'sensor.other.collision', 'sensor.camera.rgb', 'vehicle.*', 'walker.*'])
        self.vehicle = None
        self.obsticle = None
        self.ped = None
        # Spawn ego vehicle
        self.veh_transform = carla.Transform(
            carla.Location(
                float(self.spawn_locations.get("2")[1].get("x")),
                float(self.spawn_locations.get("2")[1].get("y")),
                float(self.spawn_locations.get("2")[1].get("z"))
            ),
            carla.Rotation(0, 90)
        )
        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.veh_transform)
        self.prev_s = 0.0
        # Spawn obstacle vehicle if occlusion is enabled
        if OCCLUSION:
            self.obsticle_transform = carla.Transform(
                carla.Location(
                    float(self.spawn_locations.get("2")[5].get("x")),
                    float(self.spawn_locations.get("2")[5].get("y")),
                    float(self.spawn_locations.get("2")[5].get("z"))
                ),
                carla.Rotation(0, 90)
            )
            while self.obsticle is None:
                self.obsticle = self.world.try_spawn_actor(self.ambulance, self.obsticle_transform)
            if MOVING_OCC:
                self.obsticle.set_autopilot(True)
        # Spawn pedestrian
        self.ped_transform = self.randomise_location(self.spawn_locations["2"][4])

        while self.ped is None:
            self.ped = self.world.try_spawn_actor(self.ped_bp, self.ped_transform)

        self.ped_target = self.ped_transform.location
        self.ped_target.x = self.ped_target.x + 10

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: CarlaEnv.get_collision_data(weak_self, event))

        # Add lane invasion sensor
        self.lane_sensor = self.world.spawn_actor(self.lane_hist_bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.lane_sensor.listen(lambda event: CarlaEnv.get_lane_data(weak_self, event))

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.vehicle)
        self.camera_sensor.listen(self.image_queue.put)

        # Initial vehicle state - stationary
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Set up navigation
        self.route_ind = 0
        self.controller = PIDLateralController(self.vehicle)
        self.local_planner = LocalPlanner(self.vehicle)
        self.local_planner.set_global_plan(self.route)

        # Initialize simulation and get first camera frame
        if self.settings.synchronous_mode:
            try:
                # Run a couple of simulation steps to initialize everything
                for x in range(2):
                    self.world.tick()

                # Ensure we have at least one camera frame
                if not self.image_queue.empty():
                    self.process_img(self.image_queue.get())

                    # Save the initial frame for VLM
                    self.save_current_frame()
                else:
                    print("Warning: No initial camera frame available")
            except Exception as e:
                print(f"Failed during initialization: {e}")
        else:
            self.world.wait_for_tick()
            self.world.wait_for_tick()

        # Return initial observation and info
        info = self.get_current_vehicle_state()
        return self.front_camera, info

    @staticmethod
    def get_collision_data(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision_hist.append(event)

    @staticmethod
    def get_lane_data(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lane_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data, dtype=np.dtype("uint8"))
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        x = cv2.UMat(i3)

        self.front_camera = x.get().transpose(2, 0, 1)  # grey(torch.from_numpy(x.get().transpose(2, 0, 1)))

        del i, i2, i3, x

    def get_distance_to_goal(self, ego, target):
        current_x = ego.get_location().x
        current_y = ego.get_location().y
        distance_to_goal = np.linalg.norm(np.array([current_x, current_y]) - \
                                          np.array([target.x, target.y]))
        return distance_to_goal

    def update_route(self):
        if self.vehicle.get_location().y > (self.route[self.route_ind][0].transform.location.y - 1):
            if self.route_ind < (len(self.route) - 1):
                self.route_ind += 1

    def step(self, action=None):
        """
        Take a step in the environment using either the provided action or the VLM's recommendation.

        Args:
            action: Numeric action value between -1.0 and 1.0, or None to use VLM action

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # If action is None, use the current VLM action value
        if action is None:
            # First check if we have a vlm_controller instance with a current_action_value
            if hasattr(self, "vlm_controller") and hasattr(self.vlm_controller, "current_action_value"):
                action = self.vlm_controller.current_action_value
            # Then check if we have a direct current_action_value attribute
            elif hasattr(self, "current_action_value"):
                action = self.current_action_value
            # Otherwise use a default safe value
            else:
                action = 0.0  # Default to neutral action if no VLM value is available

        # Update route if needed
        self.update_route()

        # Spawn traffic if enabled
        if TRAFFIC:
            self.spawn_traffic()

        # Get current speed
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5  # m/s

        # Apply the action to control the vehicle
        if action < 0:
            # Braking (negative action)
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.0,
                brake=float(abs(action)),
                steer=self.controller.run_step(self.route[self.route_ind][0])
            ))
        else:
            # Acceleration/maintaining (positive action)
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=float(action),
                brake=0.0,
                steer=self.controller.run_step(self.route[self.route_ind][0])
            ))

            # Limit maximum speed
            if (speed * 3.6) > 20:  # Convert to km/h
                self.target_velocity = carla.Vector3D(0, 5, 0)
                self.vehicle.set_target_velocity(self.target_velocity)

        # Calculate distances for monitoring and pedestrian control
        dist2cross = self.get_distance_to_goal(self.vehicle, self.goal)
        dist = self.get_distance_to_goal(self.vehicle, self.ego_target)
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        ped_dist = self.get_distance_to_goal(self.ped, self.ped_target)

        # Control pedestrian movement
        # Set pedestrian bounds
        if OCCLUSION:
            ped_lb = 1
            ped_hb = 2
        else:
            ped_lb = 7
            ped_hb = 8

        if MOVING_OCC:
            # Pedestrian crosses in front of ego vehicle
            if ((self.ped.get_location().y - self.vehicle.get_location().y) < 13) and (ped_dist > 0):
                if (3 < ped_dist < 4) and self.ped_count < 40:
                    self.ped_count += 1
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0, jump=False))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4, jump=False))
            else:
                # Pedestrian crosses in front of van
                if ((self.ped.get_location().y - self.obsticle.get_location().y) < 15.5) and (ped_dist > 0):
                    if (7 < ped_dist < 8):
                        self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0, jump=False))
                    else:
                        self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4, jump=False))
        else:
            if ((self.ped.get_location().y - self.vehicle.get_location().y) < 15) and (ped_dist > 0):
                if (ped_lb < ped_dist < ped_hb) and self.ped_count < 40:
                    self.ped_count += 1
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0, jump=False))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4, jump=False))

        # Advance the simulation
        try:
            self.world.tick()
        except Exception as e:
            print(f"Failed Tick: {e}")

        # Process the latest camera image
        try:
            if not self.image_queue.empty():
                self.process_img(self.image_queue.get())
            else:
                print("Warning: Image queue is empty")
        except Exception as e:
            print(f"Failed to process image: {e}")

        # Save frame for VLM processing
        self.save_current_frame()

        # Simple pedestrian detection (using distance)
        det = 1 if veh2ped_dist < 7.5 and (ped_dist > 2) else 0

        # Check for termination conditions
        done = False
        col = 0

        if len(self.collision_hist) != 0:
            done = True
            self.collision_ep += 1
            col = -200
            print("Episode terminated: Collision detected")
        elif (dist <= 2):
            done = True
            self.successful_ep += 1
            print("Episode terminated: Goal reached successfully")
        elif self.timestep > 1000:
            done = True
            self.stall_ep += 1
            print("Episode terminated: Maximum timesteps reached")

        # Calculate reward (keep for monitoring purposes)
        c1 = -(0.2 * ((speed ** 2) / max(0.1, veh2ped_dist) + 2) + 50 * int(veh2ped_dist < 1)) * int(det)
        c2 = 0.35 * speed * int(not (det))
        c3 = -(self.prev_s - speed) ** 2
        reward = c1 + c2 + c3 + col

            # Store individual components (add these lines)
        self.current_reward_components = {
        "safety_reward": c1,
        "progress_reward": c2,
        "smoothness_reward": c3,
        "collision_penalty": col,
        "total_reward": reward}

        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        self.reward_history.append(self.current_reward_components)
    

        # Update metrics
        self.speeds.append(speed)
        self.accs.append(speed - self.prev_s)
        self.dets.append(det)
        self.dist.append(dist2cross)
        self.rewards.append(reward)

        # Update timestep and previous speed
        self.timestep += 1
        self.prev_s = speed

        # Prepare info dictionary with vehicle state
        info = self.get_current_vehicle_state()
        
        # Add reward components to info dictionary
        info.update(self.current_reward_components)

        # Add current VLM action details to info if available
        if hasattr(self, "vlm_controller"):
            info["vlm_action_text"] = self.vlm_controller.current_action_text
            info["vlm_action_value"] = self.vlm_controller.current_action_value
            info["vlm_justification"] = self.vlm_controller.current_justification

        # Return standard environment outputs
        return (self.front_camera, reward, done, done, info)

    def seed(self, seed=None):
        """
        Set the seed for this environment's random number generators.

        Args:
            seed (int, optional): The seed to use. If None, a random seed will be used.

        Returns:
            list: The seed(s) used by the environment
        """
        # Generate a numpy random generator
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Set random seeds for additional randomness sources
        random.seed(seed)
        np.random.seed(seed)

        # Set seed for traffic manager if it exists
        if hasattr(self, 'traffic_manager'):
            self.traffic_manager.set_random_device_seed(seed)

        # Return the seed as a list for compatibility
        return [seed]

    def render(self, mode):
        pass

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _clear_all_actors(self, actor_filters):
        """
        Clear specific actors from the CARLA world.

        Args:
            actor_filters (list): List of actor filter strings (e.g., 'vehicle.*', 'walker.*')
        """
        for actor_filter in actor_filters:
            try:
                # Get all actors matching the filter
                matching_actors = self.world.get_actors().filter(actor_filter)

                # First stop any walker controllers (they need to be stopped before destruction)
                if actor_filter == 'controller.ai.walker':
                    for actor in matching_actors:
                        if actor.is_alive:
                            actor.stop()

                # Then destroy all actors of this type
                for actor in matching_actors:
                    if actor.is_alive:
                        actor.destroy()

            except Exception as e:
                print(f"Error while clearing actors with filter '{actor_filter}': {e}")
