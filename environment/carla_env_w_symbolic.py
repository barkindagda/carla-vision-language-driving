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
import glob
import re
import queue
from environment.global_route_planner import GlobalRoutePlanner
from environment.local_planner import LocalPlanner, RoadOption
from environment.controller import PIDLateralController
# Register the environment with Gymnasium
from gymnasium.envs.registration import register
from Models.vlm_controller_symbolic import VLMController

# Configuration constants
IM_WIDTH = 300
IM_HEIGHT = 300
SPAWN_LOCATIONS = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/environment/spawn_locations_v2.xml"
ROUTES = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/environment/routes.xml"
TRAFFIC = True
OCCLUSION = True
MOVING_OCC = False
SPAWN_DELAY = 30
MAX_TRAFFIC = 30

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator with VLM control integration."""

    def __init__(self, render_mode=None, vlm_frames=3, use_symbolic_rewards=False, 
                use_vlm_weights=True, use_vlm_actions=False, normalize_rewards=True):
        # Your existing code
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255.0, shape=(3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
        self.render_mode = render_mode
        self.vlm_frames_needed = vlm_frames
        
        # New parameters
        self.use_symbolic_rewards = use_symbolic_rewards
        self.use_vlm_weights = use_vlm_weights
        self.use_vlm_actions = use_vlm_actions
        self.normalize_rewards = normalize_rewards

        # To define symbolic rules
        self.symbolic_rules = SymbolicRules()
        self.prev_action = 0.0  # Track previous action for calculating jerk
        self.prev_pedestrian_distance = float('inf')  # Track previous pedestrian distance
    
        self.default_weights = {
        "w1": 1,  # Safety weight (highest priority)
        "w2": 1,  # Comfort weight
        "w3": 1   # Efficiency weight
        }
            
        # Add reward normalization tracking
        self.reward_running_mean = 0
        self.reward_running_var = 1
        self.reward_count = 0
        
        # Initialize frame directory and VLM controller
        self.frame_save_dir = None
        self.frame_buffer = []
        
        # Initialize VLM controller to None - we'll set it later in the training script
        self.vlm_controller = None
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."
        self.current_vlm_action_value = 0.0

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

    def get_current_weights(self):
        """Helper method to get current weights being used"""
        if hasattr(self, 'previous_vlm_weights'):
            return self.previous_vlm_weights
        return self.default_weights

    def save_current_frame(self):
        """
        Save the current camera frame for VLM processing.
        Uses episode and timestep indices for organized storage.
        
        Returns:
            bool: True if we have enough frames for VLM processing
        """
        if self.front_camera is not None and self.frame_save_dir is not None:
            # Convert from CHW to HWC format for PIL
            frame = self.front_camera.transpose(1, 2, 0)
            img = Image.fromarray(frame.astype('uint8'))

            # Get episode index from class attribute or use a default
            episode_idx = getattr(self, 'episode_counter', 0)
            
            # Create structured filename: ep{episode}_step{timestep}.png
            frame_path = os.path.join(self.frame_save_dir, f"ep{episode_idx}_step{self.timestep}.png")

            try:
                # Save the image to disk
                img.save(frame_path)

                # Add to frame buffer
                self.frame_buffer.append(frame_path)
                if len(self.frame_buffer) > self.vlm_frames_needed:
                    # Remove oldest frame (but don't delete it from disk - useful for analysis)
                    self.frame_buffer.pop(0)

                # ADDED: Check and limit the number of saved frames (e.g., every 50 timesteps)
                if self.timestep % 50 == 0:
                    frame_files = glob.glob(os.path.join(self.frame_save_dir, "ep*_step*.png"))
                    if len(frame_files) > 1000:
                        # Sort by episode and step (oldest first)
                        frame_files.sort(key=lambda f: [int(n) for n in re.findall(r'\d+', os.path.basename(f))])
                        # Delete the oldest files
                        for old_file in frame_files[:-1000]:
                            try:
                                os.remove(old_file)
                            except Exception as e:
                                print(f"Error deleting {old_file}: {e}")

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
                # Increment episode counter
        if not hasattr(self, 'episode_counter'):
            self.episode_counter = 0
        else:
            self.episode_counter += 1
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Update timesteps and counters
        self.timestep = 0
        self.ped_count = 0
        self.stopped = False
        
        # Initialize variables for symbolic reward calculation
        self.prev_action = 0.0
        self.prev_pedestrian_distance = float('inf')
        self.prev_acceleration = 0.0
        
        # Clear frame buffer for VLM
        self.frame_buffer = []
        
        # Reset VLM action to default
        self.current_vlm_action_value = 0.0  # Initialize default VLM action value
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."

        if self.vlm_controller is not None and hasattr(self.vlm_controller,'reset_episode_state'):
            if hasattr(self.vlm_controller,'verbose') and self.vlm_controller.verbose:
                print(f"[CarlaEnv] Episode {self.episode_counter}: Calling vlm_controller reset episode")
            self.vlm_controller.reset_episode_state()
        
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

        # Reset reward history
        self.reward_history = []

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
        vehicle_state = self.get_current_vehicle_state()
        # Determine which action to use
        if self.use_vlm_actions:
            # Only get new actions at update frequency
            should_update_action = self.timestep % self.vlm_controller.update_frequency == 0
            
            # Use VLM for action decision if it's time to update
            if should_update_action and len(self.frame_buffer) >= self.vlm_frames_needed:
                try:
                    vehicle_state = self.get_current_vehicle_state()
                    vlm_action = self.vlm_controller.get_action(vehicle_state, self.frame_buffer)
                    
                    # Convert VLM action string to numeric value
                    if vlm_action == "STOP":
                        action = -1.0  # Full brake
                    elif vlm_action == "SLOW":
                        action = -0.5  # Moderate brake
                    elif vlm_action == "MAINTAIN":
                        action = 0.0   # Maintain speed
                    elif vlm_action == "ACCELERATE":
                        action = 0.5   # Moderate acceleration
                    else:
                        # Default to maintain if unknown action
                        action = 0.0
                    
                    # Store VLM action for reference and future steps
                    self.current_vlm_action = vlm_action
                    self.current_vlm_action_value = action
                    print(f"Timestep {self.timestep}: Updated VLM action: {vlm_action} (value: {action})")
                except Exception as e:
                    print(f"Error getting VLM action: {e}")
                    # Use previous action if available, otherwise default
                    action = getattr(self, 'current_vlm_action_value', 0.0)
            else:
                # Use previous VLM action between updates
                action = getattr(self, 'current_vlm_action_value', 0.0)
                if not should_update_action:
                    print(f"Timestep {self.timestep}: Using previous VLM action: {self.current_vlm_action} (value: {action})")
        
        # If not using VLM action or VLM action retrieval failed, use provided action
        if action is None:
            print("WARNING: Action was None - this should not happen during PPO training")
            action = 0.0  # Fallback, but this should never occur during training
        
        # Convert action to scalar if it's an array
        if isinstance(action, np.ndarray):
            action = float(action.item())
        elif isinstance(action, (list, tuple)):
            action = float(action[0])

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

        # Calculate reward based on selected method
        if not self.use_symbolic_rewards:
            # Original reward calculation with weighted components
            c1 = -(0.2 * ((speed ** 2) / max(0.1, veh2ped_dist) + 2) + 50 * int(veh2ped_dist < 1)) * int(det)
            c2 = 0.35 * speed * int(not (det))
            c3 = -(self.prev_s - speed) ** 2
            col_penalty = col
            
            # Get weights from VLM or defaults
            weights = self.vlm_controller.get_reward_weights(vehicle_state, self.frame_buffer, self) if self.use_vlm_weights else self.default_weights
            
            # Calculate weighted total reward
            reward = (
                weights["w1"] * c1 +  # Safety component
                weights["w2"] * c3 +  # Comfort component
                weights["w3"] * c2    # Efficiency component
            ) + col_penalty  # Always apply collision penalty in full
            
            # Store reward components
            self.current_reward_components = {
                "safety_reward": c1,
                "comfort_reward": c3,
                "efficiency_reward": c2,
                "collision_penalty": col_penalty,
                "safety_weight": float(weights["w1"]),
                "comfort_weight": float(weights["w2"]),
                "efficiency_weight": float(weights["w3"]),
                "total_reward": reward
            }
        else:
            # Symbolic rules with VLM weights
            # Prepare state dictionary for symbolic rules
            current_state = {
                "speed_ms": speed,
                "speed_kmh": speed * 3.6,
                "acceleration": speed - self.prev_s,
                "pedestrian_distance": veh2ped_dist,
                "pedestrian_detected": det == 1,
                "pedestrian_ahead": self.ped.get_location().y > self.vehicle.get_location().y,
                "collision_detected": len(self.collision_hist) > 0,
                "action_value": action,
                "prev_acceleration": getattr(self, 'prev_acceleration', 0.0),
                "prev_pedestrian_distance": self.prev_pedestrian_distance
            }
            
            # Get symbolic rewards
            symbolic_rewards = self.calculate_symbolic_rewards(current_state, action)
            
            # Get weights from VLM or defaults
            weights = self.get_vlm_weights() if self.use_vlm_weights else self.default_weights
            
            # Calculate weighted total reward
            reward = (
                weights["w1"] * symbolic_rewards["safety_reward"] +
                weights["w2"] * symbolic_rewards["comfort_reward"] +
                weights["w3"] * symbolic_rewards["efficiency_reward"]
            )
            
            if len(self.collision_hist) > 0:
                reward += col  # Add collision penalty
            
            # Generate explanation for the current action
            explanation = self.symbolic_rules.generate_explanation(current_state, action, symbolic_rewards)
            
            # Store reward components and weights
            self.current_reward_components = {
                "safety_reward": float(symbolic_rewards["safety_reward"]),
                "comfort_reward": float(symbolic_rewards["comfort_reward"]),
                "efficiency_reward": float(symbolic_rewards["efficiency_reward"]),
                "safety_weight": float(weights["w1"]),
                "comfort_weight": float(weights["w2"]),
                "efficiency_weight": float(weights["w3"]),
                "total_reward": float(reward),
                "explanation": explanation
            }

        # Process reward (update stats and normalize if enabled)
        if self.normalize_rewards:
            reward = self.process_reward(reward)
        
        # Convert reward to scalar
        reward = float(reward)
        
        # Store current values for next iteration
        self.prev_action = action
        self.prev_pedestrian_distance = veh2ped_dist
        self.prev_acceleration = speed - self.prev_s

        # If not tracking reward history, initialize it
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        self.reward_history.append(self.current_reward_components)

        # Update metrics
        self.speeds.append(speed)
        self.accs.append(speed - self.prev_s)
        self.dets.append(det)
        self.dist.append(dist2cross)
        self.rewards.append(reward)

        # Update timestep
        self.timestep += 1
        self.prev_s = speed

        # Prepare info dictionary with vehicle state
        info = self.get_current_vehicle_state()
        
        # Add reward components to info dictionary
        info.update(self.current_reward_components)
        
        # Add original reward to info if normalized
        if self.normalize_rewards and hasattr(self, 'reward_running_mean'):
            info["original_reward"] = self.current_reward_components["total_reward"]
            info["normalized_reward"] = reward

        # Return standard environment outputs
        return self.front_camera, reward, done, done, info

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
    def calculate_symbolic_rewards(self, state, action):
        """
        Calculate rewards based on symbolic rules.
        Args:
            state: Current vehicle and environment state
            action: Current action value
        Returns:
            Dictionary containing rewards for each component
        """
        # Add previous values to state for rule evaluation
        state['prev_acceleration'] = getattr(self, 'prev_acceleration', 0.0)
        state['prev_pedestrian_distance'] = self.prev_pedestrian_distance
        
        # Calculate rewards for each symbolic rule
        safety_reward = self.symbolic_rules.safety_rule_collision_avoidance(state, action)
        comfort_reward = self.symbolic_rules.comfort_rule_minimize_jerk(state, action, self.prev_action)
        efficiency_reward = self.symbolic_rules.efficiency_rule_avoid_unnecessary_braking(state, action)
        
        # Add collision penalty if collision occurred
        if state.get('collision_detected', False):
            safety_reward = -200.0  # Critical safety failure
        
        return {
            "safety_reward": safety_reward,
            "comfort_reward": comfort_reward,
            "efficiency_reward": efficiency_reward
        }
    
    def get_vlm_weights(self):
        """
        Get reward component weights from VLM controller, respecting update_frequency.
        
        Returns:
            Dictionary with weights for each reward component
        """
        # Return default weights if VLM weights disabled
        if not self.use_vlm_weights:
            return self.default_weights
            
        # Initialize previous weights if not existing
        if not hasattr(self, 'previous_vlm_weights'):
            self.previous_vlm_weights = self.default_weights.copy()
        
        # Only update weights at the specified frequency
        # Check if current timestep is divisible by the update_frequency
        should_update = self.timestep % self.vlm_controller.update_frequency == 0
        
        # Check if we have a VLM controller with weight determination capability
        if should_update and hasattr(self, "vlm_controller") and hasattr(self.vlm_controller, "get_reward_weights"):
            try:
                # Get current vehicle state - including all reward components
                vehicle_state = self.get_current_vehicle_state()
                
                # Add reward components if available
                if hasattr(self, 'current_reward_components'):
                    vehicle_state['current_rewards'] = self.current_reward_components
                
                # Only call the VLM if we have enough frames
                if len(self.frame_buffer) >= self.vlm_frames_needed:
                    # Get weights from VLM
                    vlm_weights = self.vlm_controller.get_reward_weights(vehicle_state, self.frame_buffer)
                    
                    # Check if all required keys exist
                    if all(key in vlm_weights for key in ["safety_weight", "comfort_weight", "efficiency_weight"]):
                        weights = {
                            "w1": vlm_weights["safety_weight"],
                            "w2": vlm_weights["comfort_weight"],
                            "w3": vlm_weights["efficiency_weight"]
                        }
                        
                        # Ensure weights sum to 1.0
                        total = sum(weights.values())
                        if total > 0:
                            for key in weights:
                                weights[key] /= total
                        
                        # Store for next steps
                        self.previous_vlm_weights = weights.copy()
                                
                        print(f"Timestep {self.timestep}: Updated VLM weights: Safety={weights['w1']:.2f}, Comfort={weights['w2']:.2f}, Efficiency={weights['w3']:.2f}")
                        return weights
                    else:
                        print(f"Missing keys in VLM weights: {vlm_weights}")
                else:
                    print(f"Not enough frames for VLM weights: {len(self.frame_buffer)}/{self.vlm_frames_needed}")
            except Exception as e:
                print(f"Error getting VLM weights: {e}")
                # Continue using previous weights
        
        # For non-update steps, use previously computed weights
        if hasattr(self, 'previous_vlm_weights'):
            if not should_update:
                print(f"Timestep {self.timestep}: Using previous weights (next update at step {((self.timestep // self.vlm_controller.update_frequency) + 1) * self.vlm_controller.update_frequency})")
            return self.previous_vlm_weights
        
        # Fallback to default weights if no previous weights
        print("Using default weights")
        return self.default_weights
    
    def is_pedestrian_moving_away(self):
        """
        Determine if pedestrian is moving away from vehicle.
        Returns:
            Boolean indicating if pedestrian is moving away
        """
        if not hasattr(self, 'ped') or not hasattr(self, 'prev_pedestrian_distance'):
            return False
        
        current_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        is_moving_away = current_dist > self.prev_pedestrian_distance
        return is_moving_away
    
    def process_reward(self, reward):
        """
        Update reward statistics and normalize reward if enabled.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Original or normalized reward value
        """
        # Update running statistics
        self.reward_count += 1
        delta = reward - self.reward_running_mean
        self.reward_running_mean += delta / self.reward_count
        delta2 = reward - self.reward_running_mean
        self.reward_running_var += delta * delta2
        
        if self.reward_count > 1:
            self.reward_running_std = np.sqrt(self.reward_running_var / (self.reward_count - 1))
        else:
            self.reward_running_std = 1.0
        
        # Prevent division by very small numbers
        self.reward_running_std = max(0.1, self.reward_running_std)
        
        # Normalize if enabled and we have enough samples
        if self.normalize_rewards and self.reward_count >= 100:
            # Z-score normalization with clipping
            normalized = (reward - self.reward_running_mean) / self.reward_running_std
            return float(np.clip(normalized, -1.0, 1.0))
        
        # Return original reward if normalization is disabled or too early
        return float(reward)
    
class SymbolicRules:
    """
    Class implementing symbolic rules for autonomous driving.
    Evaluates safety, comfort, and efficiency constraints.
    """
    def __init__(self):
        # Default safety parameters
        self.d_safe = 7.5  # Safe distance to pedestrian (meters)
        self.critical_distance = 3.0  # Critical safety distance (meters)
        
        # Default comfort parameters
        self.jerk_max = 2.0  # Maximum comfortable jerk (m/s³)
        self.min_reaction_time = 1.5  # Minimum comfortable reaction time (seconds)
        
        # Default efficiency parameters
        self.target_speed = 5.0  # Target speed in m/s (~18 km/h)
        self.unnecessary_brake_threshold = 10.0  # Distance beyond which braking for pedestrians is unnecessary
    
    def safety_rule_collision_avoidance(self, state, action):
        """G(distance_pedestrian < d_safe → brake)"""
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        pedestrian_detected = state.get('pedestrian_detected', False)
        
        # If action is negative, vehicle is braking
        is_braking = action < 0
        
        # Calculate reward based on rule
        if pedestrian_detected and pedestrian_distance < self.d_safe:
            if not is_braking:
                # Rule violated: pedestrian nearby but not braking
                # Penalty scales with closeness to pedestrian and vehicle speed
                speed = state.get('speed_ms', 0.0)
                return -((self.d_safe / max(0.1, pedestrian_distance)) * (1 + speed))
            else:
                # Rule followed: appropriate braking near pedestrian
                return 0.5 * (self.d_safe / max(0.1, pedestrian_distance))
        elif pedestrian_distance < self.critical_distance:
            # Critical safety situation - severe penalty if not braking
            return -100.0 if not is_braking else 1.0
        
        # No pedestrian detected or far away
        return 0.0
    
    def comfort_rule_minimize_jerk(self, state, action, prev_action):
        """G(jerk > jerk_max → ¬brake) unless safety requires it"""
        # Calculate jerk as change in acceleration
        current_accel = state.get('acceleration', 0.0)
        prev_accel = state.get('prev_acceleration', 0.0)
        jerk = abs(current_accel - prev_accel)
        
        # Calculate comfort reward
        if jerk > self.jerk_max:
            # High jerk detected
            pedestrian_distance = state.get('pedestrian_distance', float('inf'))
            pedestrian_detected = state.get('pedestrian_detected', False)
            
            # If safety doesn't require harsh braking, penalize it
            if not (pedestrian_detected and pedestrian_distance < self.d_safe):
                return -0.5 * (jerk - self.jerk_max)
        
        # Reward smooth actions - less change between actions is better
        action_smoothness = -abs(action - prev_action) if prev_action is not None else 0
        return action_smoothness
    
    def efficiency_rule_avoid_unnecessary_braking(self, state, action):
        """G(pedestrian moving away → ¬brake)"""
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        pedestrian_moving_away = self.is_pedestrian_moving_away(state)
        is_braking = action < 0
        speed = state.get('speed_ms', 0.0)
        
        # Calculate efficiency reward
        if pedestrian_moving_away and pedestrian_distance > self.unnecessary_brake_threshold and is_braking:
            # Unnecessary braking - penalty
            return -0.3
        elif not state.get('pedestrian_detected', False) and speed < self.target_speed:
            # Reward progress when it's safe
            return 0.35 * speed
        
        return 0.0
    
    def is_pedestrian_moving_away(self, state):
        """Determine if pedestrian is moving away from vehicle path"""
        # Using relative position, pedestrian_ahead value, and any trend info
        pedestrian_ahead = state.get('pedestrian_ahead', False)
        
        # In a real implementation, you would use more sophisticated logic
        # For now, we'll use a simple approximation based on available data
        if 'pedestrian_moving_direction' in state:
            # If we have explicit direction info
            return state['pedestrian_moving_direction'] == 'away'
        else:
            # Use other available indicators
            pedestrian_distance = state.get('pedestrian_distance', float('inf'))
            prev_pedestrian_distance = state.get('prev_pedestrian_distance', float('inf'))
            
            # If distance is increasing, pedestrian may be moving away
            return prev_pedestrian_distance < pedestrian_distance
    
    def generate_explanation(self, state, action, rewards):
        """Generate human-readable explanation for the action."""
        explanation = ""
        
        # Safety explanations (highest priority)
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        speed = state.get('speed_ms', 0.0)
        is_braking = action < 0
        
        if state.get('collision_detected', False):
            explanation = "EMERGENCY: Collision detected! Applying maximum braking."
        elif pedestrian_distance < self.critical_distance and is_braking:
            explanation = f"Emergency braking triggered because pedestrian is within critical danger zone ({pedestrian_distance:.1f}m)."
        elif pedestrian_distance < self.d_safe and not is_braking:
            explanation = f"SAFETY VIOLATION: Should be braking with pedestrian {pedestrian_distance:.1f}m away."
        elif pedestrian_distance < self.d_safe and is_braking:
            explanation = f"Braking for safety with pedestrian {pedestrian_distance:.1f}m away."
        
        # Comfort explanations (second priority)
        if not explanation:
            current_accel = state.get('acceleration', 0.0)
            prev_accel = state.get('prev_acceleration', 0.0)
            jerk = abs(current_accel - prev_accel)
            
            if jerk > self.jerk_max and is_braking:
                explanation = f"Strong braking causing discomfort (jerk: {jerk:.1f}m/s³), but necessary for safety."
            elif jerk > self.jerk_max:
                explanation = f"High jerk detected ({jerk:.1f}m/s³), should smooth acceleration profile."
        
        # Efficiency explanations (lowest priority)
        if not explanation:
            pedestrian_moving_away = self.is_pedestrian_moving_away(state)
            
            if pedestrian_moving_away and pedestrian_distance > self.unnecessary_brake_threshold and is_braking:
                explanation = f"Unnecessary braking when pedestrian is moving away at safe distance ({pedestrian_distance:.1f}m)."
            elif not state.get('pedestrian_detected', False) and is_braking:
                explanation = f"Unnecessary braking with no pedestrians detected."
            elif not state.get('pedestrian_detected', False) and speed < self.target_speed:
                explanation = f"Accelerating to reach efficient target speed ({self.target_speed * 3.6:.1f} km/h)."
            elif not state.get('pedestrian_detected', False):
                explanation = f"Maintaining efficient speed of {speed * 3.6:.1f} km/h with no pedestrians detected."
        
        # If no specific rule triggered, provide a general explanation
        if not explanation:
            if is_braking:
                explanation = "Braking for general caution."
            else:
                explanation = "Proceeding forward safely."
        
        return explanation