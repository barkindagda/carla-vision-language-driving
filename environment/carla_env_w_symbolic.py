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
from environment.global_route_planner import GlobalRoutePlanner
from environment.local_planner import LocalPlanner, RoadOption
from environment.controller import PIDLateralController
from gymnasium.envs.registration import register
from Models.vlm_controller_symbolic import VLMController

# Configuration constants
IM_WIDTH = 224
IM_HEIGHT = 224
SPAWN_LOCATIONS = "./environment/spawn_locations_v2.xml"
ROUTES = "./environment/routes.xml"
TRAFFIC = True
OCCLUSION = True
MOVING_OCC = False
SPAWN_DELAY = 30
MAX_TRAFFIC = 30

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator with VLM control integration."""
    def __init__(self, render_mode=None, vlm_frames=3, use_symbolic_rewards=True, 
                 use_vlm_weights=True, use_vlm_actions=False, normalize_rewards=True):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255.0, shape=(3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
        self.render_mode = render_mode
        self.vlm_frames_needed = vlm_frames

        self.use_symbolic_rewards = use_symbolic_rewards
        self.use_vlm_weights = use_vlm_weights
        self.use_vlm_actions = use_vlm_actions
        self.normalize_rewards = normalize_rewards

        self.symbolic_rules = SymbolicRules()
        self.prev_action = 0.0
        self.prev_pedestrian_distance = float('inf')
        self.prev_acceleration = 0.0
        self.prev_dist_to_target = None

        self.default_weights = {
            "w1": 1.0/3,  # Safety weight
            "w2": 1.0/3,  # Comfort weight
            "w3": 1.0/3   # Efficiency weight
        }

        self.reward_running_mean = 0
        self.reward_running_var = 1
        self.reward_count = 0

        self.frame_save_dir = None
        self.frame_buffer = []
        self.vlm_controller = None
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."
        self.current_vlm_action_value = 0.0

        print('Connecting to CARLA server...')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print('CARLA server connected!')

        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = 0.10
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)
        random.seed(0)

        self.spawn_point1, self.route1, self.spawn_point2, self.route2 = self.generate_route()
        self.traffic_blueprints = self.filter_blueprints()
        self.alternate_spawn = False
        self.spawn_counter = 0

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_locations = self.get_spawn_locations()

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

        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                self.veh_transform.location + carla.Location(x=5, y=30, z=50),
                carla.Rotation(pitch=-90)
            )
        )

        self.ped_bp = self.blueprint_library.filter("walker")[4]
        if self.ped_bp.has_attribute('is_invincible'):
            self.ped_bp.set_attribute('is_invincible', 'False')
        self.ped = None

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

        self.image_queue = queue.Queue()
        self.front_camera = np.zeros((3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)

        self.collision_sensor = None
        self.collision_hist = []
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        self.lane_sensor = None
        self.lane_hist = []
        self.lane_hist_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')

        self.camera_sensor = None
        self.camera_trans = carla.Transform(carla.Location(x=0.7, z=1.6))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        self.camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        self.camera_bp.set_attribute('fov', '110')
        self.camera_bp.set_attribute('sensor_tick', '0.1')

        self.controller = None
        self.grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2)
        self.route = self.get_route()
        self.local_planner = None
        self.route_ind = 0

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

        self.goal = carla.Location(
            float(-48.64543151855469),
            float(94),
            float(1.0)
        )

    def get_current_weights(self):
        """Helper method to get current weights being used"""
        if hasattr(self, 'current_vlm_weights'):
            return self.current_vlm_weights
        return self.default_weights

    def save_current_frame(self):
        """
        Save the current camera frame for VLM processing.
        Returns:
            bool: True if we have enough frames for VLM processing
        """
        if self.front_camera is not None and self.frame_save_dir is not None:
            frame = self.front_camera.transpose(1, 2, 0)
            img = Image.fromarray(frame.astype('uint8'))
            episode_idx = getattr(self, 'episode_counter', 0)
            frame_path = os.path.join(self.frame_save_dir, f"ep{episode_idx}_step{self.timestep}.png")

            try:
                img.save(frame_path)
                self.frame_buffer.append(frame_path)
                if len(self.frame_buffer) > 10:
                    self.frame_buffer.pop(0)

                if self.timestep % 50 == 0:
                    frame_files = glob.glob(os.path.join(self.frame_save_dir, "ep*_step*.png"))
                    if len(frame_files) > 1000:
                        frame_files.sort(key=lambda f: [int(n) for n in re.findall(r'\d+', os.path.basename(f))])
                        for old_file in frame_files[:-1000]:
                            try:
                                os.remove(old_file)
                            except Exception as e:
                                print(f"Error deleting {old_file}: {e}")

                return len(self.frame_buffer) >= self.vlm_frames_needed
            except Exception as e:
                print(f"Error saving frame: {e}")
                return False
        return len(self.frame_buffer) >= self.vlm_frames_needed

    def get_vlm_selected_frames(self):
        """
        Select first, middle, and last frames from the last 10 timesteps for VLM input.
        """
        if len(self.frame_buffer) < self.vlm_frames_needed:
            return self.frame_buffer[-self.vlm_frames_needed:] if len(self.frame_buffer) >= self.vlm_frames_needed else self.frame_buffer
        return [self.frame_buffer[0], self.frame_buffer[4], self.frame_buffer[9]]

    def get_current_vehicle_state(self):
        """
        Get vehicle state information for the VLM.
        """
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        dist_to_goal = self.get_distance_to_goal(self.vehicle, self.goal)
        ped_location = self.ped.get_location()
        vehicle_location = self.vehicle.get_location()
        ped_forward = ped_location.y > vehicle_location.y

        ped_distance_category = "NONE"
        if veh2ped_dist < 12.0:
            ped_distance_category = "NEAR" if veh2ped_dist < 5 else "MEDIUM" if veh2ped_dist < 8 else "FAR"

        pedestrian_detected = veh2ped_dist < 12.0 and self.get_distance_to_goal(self.ped, self.ped_target) > 2
        reward_info = getattr(self, 'current_reward_components', {})

        recent_rewards = []
        if hasattr(self, 'reward_history') and len(self.reward_history) > 0:
            history_length = min(5, len(self.reward_history))
            recent_rewards = self.reward_history[-history_length:]

        safety_trend = efficiency_trend = comfort_trend = "N/A"
        if len(recent_rewards) >= 3:
            midpoint = len(recent_rewards) // 2
            older_safety = sum(r.get('safety_reward', 0) for r in recent_rewards[:midpoint]) / midpoint
            newer_safety = sum(r.get('safety_reward', 0) for r in recent_rewards[midpoint:]) / (len(recent_rewards) - midpoint)
            safety_trend = "Improving" if newer_safety > older_safety else "Declining"
            older_efficiency = sum(r.get('efficiency_reward', 0) for r in recent_rewards[:midpoint]) / midpoint
            newer_efficiency = sum(r.get('efficiency_reward', 0) for r in recent_rewards[midpoint:]) / (len(recent_rewards) - midpoint)
            efficiency_trend = "Improving" if newer_efficiency > older_efficiency else "Declining"
            older_comfort = sum(abs(r.get('comfort_reward', 0)) for r in recent_rewards[:midpoint]) / midpoint
            newer_comfort = sum(abs(r.get('comfort_reward', 0)) for r in recent_rewards[midpoint:]) / (len(recent_rewards) - midpoint)
            comfort_trend = "Improving" if newer_comfort < older_comfort else "Declining"

        return {
            "speed_ms": speed,
            "speed_kmh": speed * 3.6,
            "acceleration": speed - self.prev_s,
            "pedestrian_distance": veh2ped_dist,
            "pedestrian_distance_category": ped_distance_category,
            "pedestrian_detected": pedestrian_detected,
            "pedestrian_ahead": ped_forward,
            "distance_to_goal": dist_to_goal,
            "collision_detected": len(self.collision_hist) > 0,
            "timestep": self.timestep,
            "previous_action": self.current_vlm_action,
            "previous_justification": self.current_vlm_justification,
            "occlusion_present": OCCLUSION,
            "current_rewards": reward_info,
            "recent_rewards": recent_rewards,
            "safety_trend": safety_trend,
            "efficiency_trend": efficiency_trend,
            "comfort_trend": comfort_trend
        }

    def filter_blueprints(self):
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
        return [vehicle for vehicle in self.world.get_blueprint_library().filter('*vehicle*') if any(model in vehicle.id for model in models)]

    def generate_route(self):
        spawn_points = self.world.get_map().get_spawn_points()
        route1_ind = [130, 29, 137, 90, 96, 3, 75, 6, 8, 16]
        route2_ind = [129, 28, 79, 86, 77, 2, 125, 7, 9, 15]
        spawn_point1 = spawn_points[130]
        route1 = [spawn_points[i].location for i in route1_ind]
        spawn_point2 = spawn_points[129]
        route2 = [spawn_points[i].location for i in route2_ind]
        return [spawn_point1, route1, spawn_point2, route2]

    def get_spawn_locations(self):
        data = {}
        tree = ET.parse(SPAWN_LOCATIONS)
        root = tree.getroot()
        for child in root:
            data[child.attrib["name"]] = [tag.attrib for tag in child.iter()]
        return data

    def get_route(self):
        data = {}
        tree = ET.parse(ROUTES)
        root = tree.getroot()
        for child in root:
            data[child.attrib["id"]] = [tag.attrib for tag in child.iter()]
        def create_route_from_waypoints(waypoints):
            return [[self.grp._wmap.get_waypoint(carla.Location(float(waypoint.get("x")), float(waypoint.get("y")), float(waypoint.get("z")))), RoadOption(int(waypoint.get("road_option")))] for waypoint in waypoints]
        return create_route_from_waypoints(data.get("0" if OCCLUSION else "1")[1:])

    def randomise_location(self, xyz):
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        if xyz["orientation"] == "horizontal":
            x = random.uniform(safe_float(xyz["xmin"]), safe_float(xyz["xmax"]))
            y = safe_float(xyz["y"])
            z = safe_float(xyz["z"])
        elif xyz["orientation"] == "vertical":
            y = random.uniform(safe_float(xyz["ymin"]), safe_float(xyz["ymax"]))
            x = safe_float(xyz["x"])
            z = safe_float(xyz["z"])
        return carla.Transform(carla.Location(x, y, z), carla.Rotation(0, 0))

    def spawn_traffic(self):
        try:
            n_vehicles = len(self.world.get_actors().filter('*vehicle*'))
            if self.spawn_counter == SPAWN_DELAY and n_vehicles < MAX_TRAFFIC:
                vehicle_bp = random.choice(self.traffic_blueprints)
                spawn_point = self.spawn_point1 if self.alternate_spawn else self.spawn_point2
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    vehicle.set_autopilot(True)
                    self.traffic_manager.update_vehicle_lights(vehicle, True)
                    self.traffic_manager.random_left_lanechange_percentage(vehicle, 0)
                    self.traffic_manager.random_right_lanechange_percentage(vehicle, 0)
                    self.traffic_manager.auto_lane_change(vehicle, False)
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100)
                    self.traffic_manager.global_percentage_speed_difference(30)
                    route = self.route1 if self.alternate_spawn else self.route2
                    self.traffic_manager.set_path(vehicle, route)
                    self.alternate_spawn = not self.alternate_spawn
                self.spawn_counter = max(0, self.spawn_counter - 1)
            elif self.spawn_counter > 0:
                self.spawn_counter -= 1
            else:
                self.spawn_counter = SPAWN_DELAY
        except Exception as e:
            print(f"Error in spawn_traffic: {e}")

    def reset(self, seed=None, options=None):
        """Reset the environment to begin a new episode"""
        self.episode_counter = getattr(self, 'episode_counter', 0) + 1
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.timestep = 0
        self.ped_count = 0
        self.stopped = False
        self.prev_action = 0.0
        self.prev_pedestrian_distance = float('inf')
        self.prev_acceleration = 0.0
        self.prev_dist_to_target = None
        self.frame_buffer = []
        self.current_vlm_action_value = 0.0
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."

        if self.vlm_controller is not None and hasattr(self.vlm_controller, 'reset_episode_state'):
            if getattr(self.vlm_controller, 'verbose', False):
                print(f"[CarlaEnv] Episode {self.episode_counter}: Calling vlm_controller reset episode")
            self.vlm_controller.reset_episode_state()

        if self.camera_sensor is not None:
            if self.camera_sensor.is_listening():
                self.camera_sensor.stop()
            if self.collision_sensor is not None and self.collision_sensor.is_listening():
                self.collision_sensor.stop()
            if self.lane_sensor is not None and self.lane_sensor.is_listening():
                self.lane_sensor.stop()

        self.collision_sensor = None
        self.lane_sensor = None
        self.camera_sensor = None
        self.collision_hist = []
        self.lane_hist = []
        self.image_queue = queue.Queue()

        self._clear_all_actors(['sensor.other.lane_invasion', 'sensor.other.collision', 'sensor.camera.rgb', 'vehicle.*', 'walker.*'])
        self.vehicle = None
        self.obsticle = None
        self.ped = None

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

        self.ped_transform = self.randomise_location(self.spawn_locations["2"][4])
        while self.ped is None:
            self.ped = self.world.try_spawn_actor(self.ped_bp, self.ped_transform)

        self.ped_target = self.ped_transform.location
        self.ped_target.x = self.ped_target.x + 10

        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: CarlaEnv.get_collision_data(weak_self, event))

        self.lane_sensor = self.world.spawn_actor(self.lane_hist_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda event: CarlaEnv.get_lane_data(weak_self, event))

        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.vehicle)
        self.camera_sensor.listen(self.image_queue.put)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        self.route_ind = 0
        self.controller = PIDLateralController(self.vehicle)
        self.local_planner = LocalPlanner(self.vehicle)
        self.local_planner.set_global_plan(self.route)

        if self.settings.synchronous_mode:
            try:
                for _ in range(2):
                    self.world.tick()
                if not self.image_queue.empty():
                    self.process_img(self.image_queue.get())
                    self.save_current_frame()
                else:
                    print("Warning: No initial camera frame available")
            except Exception as e:
                print(f"Failed during initialization: {e}")
        else:
            self.world.wait_for_tick()
            self.world.wait_for_tick()

        self.reward_history = []
        return self.front_camera, self.get_current_vehicle_state()

    @staticmethod
    def get_collision_data(weak_self, event):
        self = weak_self()
        if self:
            self.collision_hist.append(event)

    @staticmethod
    def get_lane_data(weak_self, event):
        self = weak_self()
        if self:
            self.lane_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data, dtype=np.uint8)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        x = cv2.UMat(i3)
        self.front_camera = x.get().transpose(2, 0, 1)
        del i, i2, i3, x

    def get_distance_to_goal(self, ego, target):
        current_x = ego.get_location().x
        current_y = ego.get_location().y
        return np.linalg.norm(np.array([current_x, current_y]) - np.array([target.x, target.y]))

    def update_route(self):
        if self.vehicle.get_location().y > (self.route[self.route_ind][0].transform.location.y - 1) and self.route_ind < (len(self.route) - 1):
            self.route_ind += 1

    def step(self, action=None):
        """
        Take a step in the environment.
        """
        vehicle_state = self.get_current_vehicle_state()
        if self.use_vlm_actions and self.timestep % self.vlm_controller.update_frequency == 0 and len(self.frame_buffer) >= self.vlm_frames_needed:
            try:
                vlm_action = self.vlm_controller.get_action(vehicle_state, self.frame_buffer)
                action_map = {"STOP": -1.0, "SLOW": -0.5, "MAINTAIN": 0.0, "ACCELERATE": 0.5}
                action = action_map.get(vlm_action, 0.0)
                self.current_vlm_action = vlm_action
                self.current_vlm_action_value = action
                if getattr(self.vlm_controller, 'verbose', False):
                    print(f"Timestep {self.timestep}: Updated VLM action: {vlm_action} (value: {action})")
            except Exception as e:
                print(f"Error getting VLM action: {e}")
                action = self.current_vlm_action_value
        else:
            action = self.current_vlm_action_value
            if getattr(self.vlm_controller, 'verbose', False) and not (self.timestep % self.vlm_controller.update_frequency == 0):
                print(f"Timestep {self.timestep}: Using previous VLM action: {self.current_vlm_action} (value: {action})")

        if action is None:
            print("WARNING: Action was None - using default 0.0")
            action = 0.0

        if isinstance(action, (np.ndarray, list, tuple)):
            action = float(action[0] if isinstance(action, (list, tuple)) else action.item())

        self.update_route()
        if TRAFFIC:
            self.spawn_traffic()

        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5

        if action < 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=float(abs(action)), steer=self.controller.run_step(self.route[self.route_ind][0])))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=float(action), brake=0.0, steer=self.controller.run_step(self.route[self.route_ind][0])))
            if speed * 3.6 > 20:
                self.target_velocity = carla.Vector3D(0, 5, 0)
                self.vehicle.set_target_velocity(self.target_velocity)

        dist2cross = self.get_distance_to_goal(self.vehicle, self.goal)
        dist = self.get_distance_to_goal(self.vehicle, self.ego_target)
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        ped_dist = self.get_distance_to_goal(self.ped, self.ped_target)

        ped_lb, ped_hb = (1, 2) if OCCLUSION else (7, 8)
        if MOVING_OCC:
            if (self.ped.get_location().y - self.vehicle.get_location().y) < 13 and ped_dist > 0:
                if 3 < ped_dist < 4 and self.ped_count < 40:
                    self.ped_count += 1
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0, jump=False))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4, jump=False))
            elif (self.ped.get_location().y - self.obsticle.get_location().y) < 15.5 and ped_dist > 0:
                if 7 < ped_dist < 8:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0, jump=False))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4, jump=False))
        else:
            if (self.ped.get_location().y - self.vehicle.get_location().y) < 15 and ped_dist > 0:
                if ped_lb < ped_dist < ped_hb and self.ped_count < 40:
                    self.ped_count += 1
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0, jump=False))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4, jump=False))

        try:
            self.world.tick()
        except Exception as e:
            print(f"Failed Tick: {e}")

        try:
            if not self.image_queue.empty():
                self.process_img(self.image_queue.get())
            else:
                print("Warning: Image queue is empty")
        except Exception as e:
            print(f"Failed to process image: {e}")

        self.save_current_frame()
        det = 1 if veh2ped_dist < 7.5 and ped_dist > 2 else 0

        done = False
        col = 0
        if len(self.collision_hist) > 0:
            done = True
            self.collision_ep += 1
            col = -100
            print("Episode terminated: Collision detected")
        elif dist <= 2:
            done = True
            self.successful_ep += 1
            print("Episode terminated: Goal reached successfully")
        elif self.timestep > 1000:
            done = True
            self.stall_ep += 1
            print("Episode terminated: Maximum timesteps reached")

        current_state = {
            "speed_ms": speed,
            "speed_kmh": speed * 3.6,
            "acceleration": speed - self.prev_s,
            "pedestrian_distance": veh2ped_dist,
            "pedestrian_detected": det == 1,
            "pedestrian_ahead": self.ped.get_location().y > self.vehicle.get_location().y,
            "collision_detected": len(self.collision_hist) > 0,
            "action_value": action,
            "prev_acceleration": self.prev_acceleration,
            "prev_pedestrian_distance": self.prev_pedestrian_distance,
            "distance_to_goal": dist2cross,
            "lane_invasion": len(self.lane_hist) > 0
        }

        symbolic_rewards = self.calculate_symbolic_rewards(current_state, action)
        weights = self.get_vlm_weights() if self.use_vlm_weights else self.default_weights

        reward = (
            weights["w1"] * symbolic_rewards["safety_reward"] +
            weights["w2"] * symbolic_rewards["comfort_reward"] +
            weights["w3"] * symbolic_rewards["efficiency_reward"]
        )
        if len(self.collision_hist) > 0:
            reward += col

        explanation = self.symbolic_rules.generate_explanation(current_state, action, symbolic_rewards, self)

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

        if self.normalize_rewards:
            reward = self.process_reward(reward)

        reward = float(reward)

        self.prev_action = action
        self.prev_pedestrian_distance = veh2ped_dist
        self.prev_acceleration = speed - self.prev_s
        self.prev_dist_to_target = dist2cross

        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        self.reward_history.append(self.current_reward_components)

        self.speeds.append(speed)
        self.accs.append(speed - self.prev_s)
        self.dets.append(det)
        self.dist.append(dist2cross)
        self.rewards.append(reward)

        self.timestep += 1
        self.prev_s = speed

        info = self.get_current_vehicle_state()
        info.update(self.current_reward_components)
        if self.normalize_rewards and hasattr(self, 'reward_running_mean'):
            info["original_reward"] = self.current_reward_components["total_reward"]
            info["normalized_reward"] = reward

        return self.front_camera, reward, done, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self, 'traffic_manager'):
            self.traffic_manager.set_random_device_seed(seed)
        return [seed]

    def render(self, mode):
        pass

    def _set_synchronous_mode(self, synchronous=True):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _clear_all_actors(self, actor_filters):
        for actor_filter in actor_filters:
            try:
                matching_actors = self.world.get_actors().filter(actor_filter)
                if actor_filter == 'controller.ai.walker':
                    for actor in matching_actors:
                        if actor.is_alive:
                            actor.stop()
                for actor in matching_actors:
                    if actor.is_alive:
                        actor.destroy()
            except Exception as e:
                print(f"Error while clearing actors with filter '{actor_filter}': {e}")

    def calculate_symbolic_rewards(self, state, action):
        """
        Calculate rewards based on symbolic rules.
        """
        state['prev_acceleration'] = self.prev_acceleration
        state['prev_pedestrian_distance'] = self.prev_pedestrian_distance
        state['lane_invasion'] = len(self.lane_hist) > 0

        safety_reward = self.symbolic_rules.safety_rule_collision_avoidance(state, action)
        comfort_reward = self.symbolic_rules.comfort_rule_minimize_jerk(state, action, self.prev_action)
        efficiency_reward = self.symbolic_rules.efficiency_rule_avoid_unnecessary_braking(state, action, self)

        if state.get('collision_detected', False):
            safety_reward = -200.0

        return {
            "safety_reward": safety_reward,
            "comfort_reward": comfort_reward,
            "efficiency_reward": efficiency_reward
        }

    def get_vlm_weights(self):
        """
        Get reward component weights from VLM controller, normalized to sum to 1.
        """
        if not self.use_vlm_weights:
            return self.default_weights

        if self.timestep % self.vlm_controller.update_frequency == 0 and hasattr(self.vlm_controller, "get_reward_weights"):
            try:
                selected_frames = self.get_vlm_selected_frames()
                if len(selected_frames) >= self.vlm_frames_needed:
                    vehicle_state = self.get_current_vehicle_state()
                    if hasattr(self, 'current_reward_components'):
                        vehicle_state['current_rewards'] = self.current_reward_components
                    vlm_output = self.vlm_controller.get_reward_weights(vehicle_state, selected_frames)
                    if vlm_output and all(k in vlm_output for k in ["safety_weight", "comfort_weight", "efficiency_weight"]):
                        weights = {
                            "w1": max(0.0, float(vlm_output["safety_weight"])),
                            "w2": max(0.0, float(vlm_output["comfort_weight"])),
                            "w3": max(0.0, float(vlm_output["efficiency_weight"]))
                        }
                        total = sum(weights.values())
                        self.current_vlm_weights = {key: value / total if total > 0 else 1.0/3 for key, value in weights.items()}
                        if getattr(self.vlm_controller, 'verbose', False):
                            print(f"Timestep {self.timestep}: VLM weights: {self.current_vlm_weights}")
                    else:
                        print(f"VLM output missing keys: {vlm_output}. Using previous weights.")
                else:
                    print(f"Not enough frames ({len(selected_frames)}/{self.vlm_frames_needed}). Using previous weights.")
            except Exception as e:
                print(f"Error getting VLM weights: {e}. Using previous weights.")
        return getattr(self, 'current_vlm_weights', self.default_weights)

    def is_pedestrian_moving_away(self):
        """
        Determine if pedestrian is moving away from vehicle.
        """
        if not hasattr(self, 'ped') or not hasattr(self, 'prev_pedestrian_distance'):
            return False
        current_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        return current_dist > self.prev_pedestrian_distance

    def process_reward(self, reward):
        """
        Update reward statistics and normalize reward if enabled.
        """
        self.reward_count += 1
        delta = reward - self.reward_running_mean
        self.reward_running_mean += delta / self.reward_count
        delta2 = reward - self.reward_running_mean
        self.reward_running_var += delta * delta2

        if self.reward_count > 1:
            self.reward_running_std = max(0.1, np.sqrt(self.reward_running_var / (self.reward_count - 1)))
        else:
            self.reward_running_std = 1.0

        if self.normalize_rewards and self.reward_count >= 100:
            normalized = (reward - self.reward_running_mean) / self.reward_running_std
            return float(np.clip(normalized, -5.0, 5.0))
        return float(reward)

class SymbolicRules:
    """
    Class implementing symbolic rules for autonomous driving.
    """
    def __init__(self):
        self.d_safe = 7.5
        self.critical_distance = 3.0
        self.jerk_max = 2.0
        self.min_reaction_time = 1.5
        self.target_speed = 5.0
        self.unnecessary_brake_threshold = 10.0
        self.lane_invasion_penalty = -10.0

    def safety_rule_collision_avoidance(self, state, action):
        """G(distance_pedestrian < d_safe → brake)"""
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        pedestrian_detected = state.get('pedestrian_detected', False)
        speed = state.get('speed_ms', 0.0)
        is_braking = action < 0
        adaptive_d_safe = min(self.d_safe + speed * 0.5, 12.0)

        if pedestrian_detected and pedestrian_distance < adaptive_d_safe:
            return -((adaptive_d_safe / max(0.1, pedestrian_distance)) * (1 + speed)) if not is_braking else 0.5 * (adaptive_d_safe / max(0.1, pedestrian_distance))
        elif pedestrian_distance < self.critical_distance:
            return -100.0 if not is_braking else 1.0
        return 0.0

    def comfort_rule_minimize_jerk(self, state, action, prev_action):
        """G(jerk > jerk_max → ¬brake) unless safety requires it"""
        current_accel = state.get('acceleration', 0.0)
        prev_accel = state.get('prev_acceleration', 0.0)
        jerk = abs(current_accel - prev_accel)
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        pedestrian_detected = state.get('pedestrian_detected', False)

        if jerk > self.jerk_max and not (pedestrian_detected and pedestrian_distance < self.d_safe):
            return -0.5 * (jerk - self.jerk_max)
        return -abs(action - prev_action) if prev_action is not None else 0.0

    def efficiency_rule_avoid_unnecessary_braking(self, state, action, env):
        """G(pedestrian moving away → ¬brake) and reward progress"""
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        pedestrian_moving_away = env.is_pedestrian_moving_away()
        is_braking = action < 0
        speed = state.get('speed_ms', 0.0)
        distance_to_goal = state.get('distance_to_goal', float('inf'))
        lane_invasion = state.get('lane_invasion', False)

        prev_dist = env.prev_dist_to_target if env.prev_dist_to_target is not None else distance_to_goal
        progress = max(0, prev_dist - distance_to_goal)

        reward = 0.0
        if pedestrian_moving_away and pedestrian_distance > self.unnecessary_brake_threshold and is_braking:
            reward -= 0.5
        elif not state.get('pedestrian_detected', False):
            reward += 1.0 * speed + 2.0 * progress
        if lane_invasion:
            reward += self.lane_invasion_penalty
        return reward

    def generate_explanation(self, state, action, rewards, env):
        """Generate human-readable explanation for the action."""
        explanation = ""
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        speed = state.get('speed_ms', 0.0)
        is_braking = action < 0
        lane_invasion = state.get('lane_invasion', False)

        if state.get('collision_detected', False):
            explanation = "EMERGENCY: Collision detected! Applying maximum braking."
        elif pedestrian_distance < self.critical_distance and is_braking:
            explanation = f"Emergency braking: pedestrian at {pedestrian_distance:.1f}m."
        elif pedestrian_distance < self.d_safe and not is_braking:
            explanation = f"SAFETY VIOLATION: Should brake with pedestrian at {pedestrian_distance:.1f}m."
        elif pedestrian_distance < self.d_safe and is_braking:
            explanation = f"Braking for safety: pedestrian at {pedestrian_distance:.1f}m."
        else:
            jerk = abs(state.get('acceleration', 0.0) - state.get('prev_acceleration', 0.0))
            if jerk > self.jerk_max and is_braking:
                explanation = f"Strong braking (jerk: {jerk:.1f}m/s³), necessary for safety."
            elif jerk > self.jerk_max:
                explanation = f"High jerk ({jerk:.1f}m/s³), should smooth acceleration."
            else:
                pedestrian_moving_away = env.is_pedestrian_moving_away()
                if pedestrian_moving_away and pedestrian_distance > self.unnecessary_brake_threshold and is_braking:
                    explanation = f"Unnecessary braking: pedestrian moving away at {pedestrian_distance:.1f}m."
                elif not state.get('pedestrian_detected', False) and is_braking:
                    explanation = "Unnecessary braking: no pedestrians detected."
                elif not state.get('pedestrian_detected', False):
                    explanation = f"Maintaining efficient speed of {speed * 3.6:.1f} km/h."
                else:
                    explanation = "Braking for caution." if is_braking else "Proceeding safely."

        if lane_invasion:
            explanation = f"LANE VIOLATION: Lane invasion detected. {explanation}"
        return explanation