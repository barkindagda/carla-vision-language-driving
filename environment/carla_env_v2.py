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
SPAWN_LOCATIONS = "/home/server01/BARKIN/carla-vision-language-driving/environment/spawn_locations_v2.xml"
ROUTES = "/home/server01/BARKIN/carla-vision-language-driving/environment/routes.xml"
TRAFFIC = True
OCCLUSION = True
MOVING_OCC = False
SPAWN_DELAY = 30
MAX_TRAFFIC = 30

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator with VLM control integration."""

    def __init__(self, render_mode=None, vlm_frames=3, scenario=1):
        self.scenario = str(scenario)  # Ensure scenario is string for dictionary lookup
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255.0, shape=(3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
        self.render_mode = render_mode
        self.vlm_frames_needed = vlm_frames
        self.frame_buffer = []
        self.vlm_controller = None
        self.frame_save_dir = "/home/server01/BARKIN/carla-vision-language-driving/vlm_outputs/frames/example4"
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."
        os.makedirs(self.frame_save_dir, exist_ok=True)

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

        # Validate scenario and spawn point
        if self.scenario not in self.spawn_locations:
            raise ValueError(f"Scenario '{self.scenario}' not found in {SPAWN_LOCATIONS}")
        scenario_data = self.spawn_locations[self.scenario]
        ego_vehicle_right = None
        for item in scenario_data:
            if item.get('tag') == 'ego_vehicle_right':
                ego_vehicle_right = item
                break
        if ego_vehicle_right is None and self.scenario == "2":
            raise ValueError(f"<ego_vehicle_right> not found in scenario '{self.scenario}'")

        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.vehicle = None
        self.prev_s = 0.0
        if self.scenario == "2":
            # Use right-lane spawn for scenario 2
            self.veh_transform = carla.Transform(
                carla.Location(
                    float(ego_vehicle_right.get("x")),
                    float(ego_vehicle_right.get("y")),
                    float(ego_vehicle_right.get("z"))
                ),
                carla.Rotation(
                    float(ego_vehicle_right.get("pitch", 0)),
                    float(ego_vehicle_right.get("yaw", 359)),
                    float(ego_vehicle_right.get("roll", 0))
                )
            )
        else:
            # Use default ego_vehicle for other scenarios
            ego_vehicle = next(item for item in scenario_data if item.get('tag') == 'ego_vehicle')
            self.veh_transform = carla.Transform(
                carla.Location(
                    float(ego_vehicle.get("x")),
                    float(ego_vehicle.get("y")),
                    float(ego_vehicle.get("z"))
                ),
                carla.Rotation(
                    float(ego_vehicle.get("pitch", 0)),
                    float(ego_vehicle.get("yaw", 359)),
                    float(ego_vehicle.get("roll", 0))
                )
            )

        self.target_velocity = None
        goal_point = next(item for item in scenario_data if item.get('tag') == 'goal_point')
        self.ego_target = carla.Location(
            float(goal_point.get("x")),
            float(goal_point.get("y")),
            float(goal_point.get("z"))
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
            obsticle_spawn = next((item for item in scenario_data if item.get('tag') == 'obsticle_spawn'), None)
            if obsticle_spawn:
                self.obsticle_transform = carla.Transform(
                    carla.Location(
                        float(obsticle_spawn.get("x")),
                        float(obsticle_spawn.get("y")),
                        float(obsticle_spawn.get("z"))
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
        self.camera_bp.set_attribute('fov', '108')
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
        self.current_lane_id = None
        self.last_lane_change_timestep = 0

        self.goal = carla.Location(
            float(goal_point.get("x")),
            float(goal_point.get("y")),
            float(goal_point.get("z"))
        )

    def get_spawn_locations(self):
        """Parse spawn_locations_v2.xml and return a dictionary of scenario spawn points."""
        data = {}
        try:
            tree = ET.parse(SPAWN_LOCATIONS)
            root = tree.getroot()
            for child in root:
                scenario_name = child.attrib["name"]
                # Collect only child tags, excluding the scenario tag itself
                spawn_points = [
                    {**tag.attrib, 'tag': tag.tag} for tag in child if tag.tag != 'scenario'
                ]
                data[scenario_name] = spawn_points
            return data
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse {SPAWN_LOCATIONS}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"XML file not found: {SPAWN_LOCATIONS}")

    # Rest of the methods (unchanged from previous version)
    def save_current_frame(self):
        """Save the current camera frame for VLM processing."""
        if self.front_camera is not None:
            frame = self.front_camera.transpose(1, 2, 0)
            img = Image.fromarray(frame.astype('uint8'))
            timestamp = int(time.time() * 1000000)
            frame_path = os.path.join(self.frame_save_dir, f"frame_{timestamp}_{self.timestep}_opencv_detection.png")
            try:
                img.save(frame_path)
                self.frame_buffer.append(frame_path)
                if len(self.frame_buffer) > self.vlm_frames_needed:
                    self.frame_buffer.pop(0)
                return True
            except Exception as e:
                print(f"Error saving frame: {e}")
                return False
        return len(self.frame_buffer) >= self.vlm_frames_needed

    def get_current_vehicle_state(self):
        """Get relevant vehicle state information for the VLM."""
        velocity = self.vehicle.get_velocity()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        dist_to_goal = self.get_distance_to_goal(self.vehicle, self.goal)
        ped_location = self.ped.get_location()
        ped_forward = ped_location.y > vehicle_location.y
        ped_distance_category = "NONE"
        if veh2ped_dist < 12.0:
            if veh2ped_dist < 5:
                ped_distance_category = "NEAR"
            elif veh2ped_dist < 8:
                ped_distance_category = "MEDIUM"
            else:
                ped_distance_category = "FAR"
        pedestrian_detected = veh2ped_dist < 12.0 and self.get_distance_to_goal(self.ped, self.ped_target) > 2

        waypoint = self.world.get_map().get_waypoint(vehicle_location)
        self.current_lane_id = waypoint.lane_id

        veh2obs_dist = self.get_distance_to_goal(self.vehicle, self.obsticle.get_location()) if OCCLUSION else float('inf')

        nearby_vehicles = [actor for actor in self.world.get_actors().filter('*vehicle*') 
                        if actor.id != self.vehicle.id and 
                        self.get_distance_to_goal(self.vehicle, actor.get_location()) < 15.0]

        reward_info = getattr(self, 'current_reward_components', {})
        recent_rewards = []
        if hasattr(self, 'reward_history') and len(self.reward_history) > 0:
            history_length = min(5, len(self.reward_history))
            recent_rewards = self.reward_history[-history_length:]
        
        safety_trend = progress_trend = smoothness_trend = lane_change_trend = "N/A"
        if len(recent_rewards) >= 3:
            midpoint = len(recent_rewards) // 2
            older_safety = sum([r.get('safety_reward', 0) for r in recent_rewards[:midpoint]]) / midpoint
            newer_safety = sum([r.get('safety_reward', 0) for r in recent_rewards[midpoint:]]) / (len(recent_rewards) - midpoint)
            safety_trend = "Improving" if newer_safety > older_safety else "Declining"
            older_progress = sum([r.get('progress_reward', 0) for r in recent_rewards[:midpoint]]) / midpoint
            newer_progress = sum([r.get('progress_reward', 0) for r in recent_rewards[midpoint:]]) / (len(recent_rewards) - midpoint)
            progress_trend = "Improving" if newer_progress > older_progress else "Declining"
            older_smoothness = sum([abs(r.get('smoothness_reward', 0)) for r in recent_rewards[:midpoint]]) / midpoint
            newer_smoothness = sum([abs(r.get('smoothness_reward', 0)) for r in recent_rewards[midpoint:]]) / (len(recent_rewards) - midpoint)
            smoothness_trend = "Improving" if newer_smoothness < older_smoothness else "Declining"
            older_lane_change = sum([r.get('lane_change_reward', 0) for r in recent_rewards[:midpoint]]) / midpoint
            newer_lane_change = sum([r.get('lane_change_reward', 0) for r in recent_rewards[midpoint:]]) / (len(recent_rewards) - midpoint)
            lane_change_trend = "Improving" if newer_lane_change > older_lane_change else "Declining"

        # Debug logging to verify values
        print(f"Vehicle State: X={vehicle_location.x:.2f}, Y={vehicle_location.y:.2f}, Z={vehicle_location.z:.2f}, Yaw={vehicle_rotation.yaw:.1f}")

        return {
            "speed_ms": speed,
            "speed_kmh": speed * 3.6,
            "acceleration": speed - self.prev_s,
            "pedestrian_distance": veh2ped_dist,
            "pedestrian_distance_category": ped_distance_category,
            "pedestrian_detected": pedestrian_detected,
            "pedestrian_ahead": ped_forward,
            "distance_to_goal": dist_to_goal,
            "distance_to_obstacle": veh2obs_dist,
            "collision_detected": len(self.collision_hist) > 0,
            "lane_invasion_detected": len(self.lane_hist) > 0,
            "current_lane_id": self.current_lane_id,
            "nearby_vehicles_count": len(nearby_vehicles),
            "timestep": self.timestep,
            "previous_action": self.current_vlm_action,
            "previous_justification": self.current_vlm_justification,
            "occlusion_present": OCCLUSION,
            "current_rewards": reward_info,
            "recent_rewards": recent_rewards,
            "safety_trend": safety_trend,
            "progress_trend": progress_trend,
            "smoothness_trend": smoothness_trend,
            "lane_change_trend": lane_change_trend,
            "vehicle_location_x": vehicle_location.x,
            "vehicle_location_y": vehicle_location.y,  # Added
            "vehicle_location_z": vehicle_location.z,  # Added
            "yaw": vehicle_rotation.yaw  # Added
        }

    def filter_blueprints(self):
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
        route1 = [spawn_points[i].location for i in route1_ind]
        spawn_point2 = spawn_points[129]
        route2 = [spawn_points[i].location for i in route2_ind]
        return [spawn_point1, route1, spawn_point2, route2]

    def get_route(self):
        data = {}
        tree = ET.parse(ROUTES)
        root = tree.getroot()
        for child in root:
            data[child.attrib["id"]] = [tag.attrib for tag in child.iter()]
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
        """Reset the environment to begin a new episode."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.timestep = 0
        self.ped_count = 0
        self.stopped = False
        self.frame_buffer = []
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."
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

        # Spawn ego vehicle
        scenario_data = self.spawn_locations[self.scenario]
        if self.scenario == "2":
            ego_vehicle_right = next(item for item in scenario_data if item.get('tag') == 'ego_vehicle_right')
            self.veh_transform = carla.Transform(
                carla.Location(
                    float(ego_vehicle_right.get("x")),
                    float(ego_vehicle_right.get("y")),
                    float(ego_vehicle_right.get("z"))
                ),
                carla.Rotation(
                    float(ego_vehicle_right.get("pitch", 0)),
                    float(ego_vehicle_right.get("yaw", 359)),
                    float(ego_vehicle_right.get("roll", 0))
                )
            )
        else:
            ego_vehicle = next(item for item in scenario_data if item.get('tag') == 'ego_vehicle')
            self.veh_transform = carla.Transform(
                carla.Location(
                    float(ego_vehicle.get("x")),
                    float(ego_vehicle.get("y")),
                    float(ego_vehicle.get("z"))
                ),
                carla.Rotation(
                    float(ego_vehicle.get("pitch", 0)),
                    float(ego_vehicle.get("yaw", 359)),
                    float(ego_vehicle.get("roll", 0))
                )
            )
        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.veh_transform)
        self.prev_s = 0.0

        # Spawn obstacle vehicle
        if OCCLUSION:
            obsticle_spawn = next((item for item in scenario_data if item.get('tag') == 'obsticle_spawn'), None)
            if obsticle_spawn:
                self.obsticle_transform = carla.Transform(
                    carla.Location(
                        float(obsticle_spawn.get("x")),
                        float(obsticle_spawn.get("y")),
                        float(obsticle_spawn.get("z"))
                    ),
                    carla.Rotation(0, 90)
                )
                while self.obsticle is None:
                    self.obsticle = self.world.try_spawn_actor(self.ambulance, self.obsticle_transform)
                if MOVING_OCC:
                    self.obsticle.set_autopilot(True)

        # Spawn pedestrian
        pedestrian = next(item for item in scenario_data if item.get('tag') == 'pedestrian')
        self.ped_transform = self.randomise_location(pedestrian)
        while self.ped is None:
            self.ped = self.world.try_spawn_actor(self.ped_bp, self.ped_transform)
        self.ped_target = self.ped_transform.location
        self.ped_target.x = self.ped_target.x + 10

        # Sensors
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
        self.current_lane_id = self.world.get_map().get_waypoint(self.vehicle.get_location()).lane_id
        self.last_lane_change_timestep = 0
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
        self.last_lane_change_timestep = self.timestep

    def process_img(self, image):
        i = np.array(image.raw_data, dtype=np.dtype("uint8"))
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        x = cv2.UMat(i3)
        self.front_camera = x.get().transpose(2, 0, 1)
        del i, i2, i3, x

    def get_distance_to_goal(self, ego, target):
        current_x = ego.get_location().x
        current_y = ego.get_location().y
        distance_to_goal = np.linalg.norm(np.array([current_x, current_y]) - np.array([target.x, target.y]))
        return distance_to_goal

    def update_route(self):
        if self.vehicle.get_location().y > (self.route[self.route_ind][0].transform.location.y - 1):
            if self.route_ind < (len(self.route) - 1):
                self.route_ind += 1

    def step(self, action=None):
        """Take a step in the environment using either the provided action or the VLM's recommendation."""
        if action is None:
            if hasattr(self, "vlm_controller") and hasattr(self.vlm_controller, "current_action_value"):
                action = self.vlm_controller.current_action_value
            elif hasattr(self, "current_action_value"):
                action = self.current_action_value
            else:
                action = [0.0, 0.0]

        action = np.array(action, dtype=np.float32)
        if action.shape != (2,):
            action = np.array([action[0], 0.0]) if action.shape == (1,) else action[:2]

        self.update_route()
        if TRAFFIC:
            self.spawn_traffic()

        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        throttle_action, steer_action = action

        if throttle_action < 0:
            control = carla.VehicleControl(
                throttle=0.0,
                brake=float(abs(throttle_action)),
                steer=float(steer_action)
            )
        else:
            control = carla.VehicleControl(
                throttle=float(throttle_action),
                brake=0.0,
                steer=float(steer_action)
            )
        self.vehicle.apply_control(control)

        if (speed * 3.6) > 20:
            self.target_velocity = carla.Vector3D(0, 5, 0)
            self.vehicle.set_target_velocity(self.target_velocity)

        dist2cross = self.get_distance_to_goal(self.vehicle, self.goal)
        dist = self.get_distance_to_goal(self.vehicle, self.ego_target)
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location())
        ped_dist = self.get_distance_to_goal(self.ped, self.ped_target)
        veh2obs_dist = self.get_distance_to_goal(self.vehicle, self.obsticle.get_location()) if OCCLUSION else float('inf')

        if OCCLUSION:
            ped_lb, ped_hb = 1, 2
        else:
            ped_lb, ped_hb = 7, 8
        if MOVING_OCC:
            if (self.ped.get_location().y - self.vehicle.get_location().y) < 13 and ped_dist > 0:
                if 3 < ped_dist < 4 and self.ped_count < 40:
                    self.ped_count += 1
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0, jump=False))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4, jump=False))
            else:
                if (self.ped.get_location().y - self.obsticle.get_location().y) < 15.5 and ped_dist > 0:
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

        current_lane_id = self.world.get_map().get_waypoint(self.vehicle.get_location()).lane_id
        lane_change_occurred = len(self.lane_hist) > 0 and self.lane_hist[-1].timestamp >= self.timestep * self.settings.fixed_delta_seconds
        right_lane_x = float(self.spawn_locations[self.scenario][-1].get("x"))  # ego_vehicle_right
        left_lane_x = float(self.spawn_locations[self.scenario][0].get("x"))    # ego_vehicle
        current_x = self.vehicle.get_location().x
        is_in_right_lane = abs(current_x - right_lane_x) < 1.0
        is_in_left_lane = abs(current_x - left_lane_x) < 1.0

        lane_change_reward = 0
        nearby_vehicles = [actor for actor in self.world.get_actors().filter('*vehicle*') 
                         if actor.id != self.vehicle.id and 
                         self.get_distance_to_goal(self.vehicle, actor.get_location()) < 15.0]
        if lane_change_occurred:
            if is_in_right_lane and veh2obs_dist < 20.0:
                lane_change_reward = 10
            elif len(nearby_vehicles) > 0 or veh2ped_dist < 10.0:
                lane_change_reward = -20
            else:
                lane_change_reward = -5
        elif is_in_right_lane and veh2obs_dist < 20.0:
            lane_change_reward = -10

        done = False
        col = 0
        if len(self.collision_hist) != 0:
            done = True
            self.collision_ep += 1
            col = -200
            print("Episode terminated: Collision detected")
        elif dist <= 2:
            done = True
            self.successful_ep += 1
            print("Episode terminated: Goal reached successfully")
        elif self.timestep > 2000:
            done = True
            self.stall_ep += 1
            print("Episode terminated: Maximum timesteps reached")

        c1 = -(0.2 * ((speed ** 2) / max(0.1, veh2ped_dist) + 2) + 50 * int(veh2ped_dist < 1)) * int(det)
        c2 = 0.35 * speed * int(not det)
        c3 = -(self.prev_s - speed) ** 2
        c4 = lane_change_reward
        reward = c1 + c2 + c3 + c4 + col

        self.current_reward_components = {
            "safety_reward": c1,
            "progress_reward": c2,
            "smoothness_reward": c3,
            "lane_change_reward": c4,
            "collision_penalty": col,
            "total_reward": reward
        }
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
        self.current_lane_id = current_lane_id

        info = self.get_current_vehicle_state()
        info.update(self.current_reward_components)
        if hasattr(self, "vlm_controller"):
            info["vlm_action_text"] = self.vlm_controller.current_action_text
            info["vlm_action_value"] = self.vlm_controller.current_action_value
            info["vlm_justification"] = self.vlm_controller.current_justification

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