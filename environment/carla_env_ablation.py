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
from transformers import AutoProcessor, AutoModelForCausalLM
import re

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


class VLMController:
    def __init__(self, model_name="VideoLLaMA3-2B"):
        print(f"Loading VLM: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        print("VLM loaded.")

        # Runtime state
        self.current_action_value = 0.0
        self.current_action_text = "MAINTAIN"
        self.current_justification = "Starting the journey safely."
        self.response_time = 0.0

    @torch.no_grad()
    def query(self, frame_paths, state_dict):
        start_time = time.time()

        prompt = (
            f"Pedestrian distance: {state_dict['pedestrian_distance']:.1f}m, "
            f"speed: {state_dict['speed_kmh']:.1f} km/h, "
            f"goal: {state_dict['distance_to_goal']:.1f}m. "
            f"Output a single number in [-1,1] (negative=brake). "
            f"Also explain in one sentence."
        )

        try:
            pil_frames = [Image.open(p).convert("RGB") for p in frame_paths]
            inputs = self.processor(images=pil_frames, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generated = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            )

            self.response_time = time.time() - start_time
            text_out = self.processor.decode(generated.sequences[0], skip_special_tokens=True)

            # Extract action
            action_match = re.search(r"[-+]?\d*\.?\d+", text_out)
            action_value = 0.0
            if action_match:
                action_value = float(action_match.group())
                action_value = np.clip(action_value, -1.0, 1.0)

            # Extract justification
            justification = text_out.split(action_match.group())[-1].strip() if action_match else text_out
            justification = justification.split('\n')[0]
            if not justification:
                justification = "No justification provided."

            # Update state
            self.current_action_value = action_value
            self.current_action_text = "BRAKE" if action_value < -0.3 else "ACCELERATE" if action_value > 0.3 else "MAINTAIN"
            self.current_justification = justification

            return action_value

        except Exception as e:
            print(f"VLM query failed: {e}")
            self.response_time = time.time() - start_time
            return 0.0


class CarlaEnv(gym.Env):
    """CARLA environment with VLM zero-shot driving, weather control, and full logging."""
    
    def __init__(self, render_mode=None, vlm_frames=3, scenario=1, weather=None):
        self.scenario = scenario
        self.weather = weather
        self.vlm_frames_needed = vlm_frames
        self.frame_buffer = []
        self.frame_save_dir = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/vlm_outputs/frames/example4"
        os.makedirs(self.frame_save_dir, exist_ok=True)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255.0, shape=(3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
        self.render_mode = render_mode

        # Connect to CARLA
        print('Connecting to CARLA server...')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print('CARLA server connected!')

        # Synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = 0.10
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # Traffic manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)
        random.seed(0)

        # Route & spawn
        self.spawn_point1, self.route1, self.spawn_point2, self.route2 = self.generate_route()
        self.traffic_blueprints = self.filter_blueprints()
        self.alternate_spawn = False
        self.spawn_counter = 0

        # Blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_locations = self.get_spawn_locations()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        # Actors
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
        self.ego_target = carla.Location(
            float(self.spawn_locations.get("2")[2].get("x")),
            float(self.spawn_locations.get("2")[2].get("y")),
            float(self.spawn_locations.get("2")[2].get("z"))
        )

        # Pedestrian
        self.ped_bp = self.blueprint_library.filter("walker")[4]
        if self.ped_bp.has_attribute('is_invincible'):
            self.ped_bp.set_attribute('is_invincible', 'False')
        self.ped = None

        # Occlusion
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

        # Sensors
        self.image_queue = queue.Queue()
        self.front_camera = np.zeros((3, IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
        self.collision_sensor = None
        self.collision_hist = []
        self.collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.lane_sensor = None
        self.lane_hist = []
        self.lane_hist_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.camera_sensor = None
        self.camera_trans = carla.Transform(carla.Location(x=0.7, z=1.6))
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        self.camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        self.camera_bp.set_attribute('fov', '110')
        self.camera_bp.set_attribute('sensor_tick', '0.1')

        # Navigation
        self.controller = None
        self.grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2)
        self.route = self.get_route()
        self.local_planner = None
        self.route_ind = 0

        # Metrics
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

        self.goal = carla.Location(-48.64543151855469, 94, 1.0)

        # VLM
        self.vlm_controller = None

    def set_weather(self, weather_preset=None):
        if weather_preset is None:
            presets = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.HardRainNoon]
            weather_preset = random.choice(presets)
        elif isinstance(weather_preset, str):
            weather_preset = getattr(carla.WeatherParameters, weather_preset)
        self.world.set_weather(weather_preset)
        print(f"Weather set to: {weather_preset}")

    def save_current_frame(self):
        if self.front_camera is not None:
            frame = self.front_camera.transpose(1, 2, 0)
            img = Image.fromarray(frame.astype('uint8'))
            timestamp = int(time.time() * 1000000)
            frame_path = os.path.join(self.frame_save_dir, f"frame_{timestamp}_{self.timestep}.png")
            img.save(frame_path)
            self.frame_buffer.append(frame_path)
            if len(self.frame_buffer) > self.vlm_frames_needed:
                self.frame_buffer.pop(0)
            return True
        return False

    def get_current_vehicle_state(self):
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location()) if self.ped else 999
        dist_to_goal = self.get_distance_to_goal(self.vehicle, self.goal)
        ped_forward = (self.ped.get_location().y > self.vehicle.get_location().y) if self.ped else False
        ped_distance_category = "NONE"
        if veh2ped_dist < 12.0:
            if veh2ped_dist < 5: ped_distance_category = "NEAR"
            elif veh2ped_dist < 8: ped_distance_category = "MEDIUM"
            else: ped_distance_category = "FAR"
        pedestrian_detected = veh2ped_dist < 12.0 and (self.get_distance_to_goal(self.ped, self.ped_target) > 2) if self.ped else False

        reward_info = getattr(self, 'current_reward_components', {})

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
            "previous_action": getattr(self.vlm_controller, 'current_action_text', "N/A"),
            "previous_justification": getattr(self.vlm_controller, 'current_justification', "N/A"),
            "occlusion_present": OCCLUSION,
            "current_rewards": reward_info,
        }

    def filter_blueprints(self):
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
        return [v for v in self.blueprint_library.filter('*vehicle*') if any(m in v.id for m in models)]

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
        for child in tree.getroot():
            data[child.attrib["name"]] = [tag.attrib for tag in child.iter()]
        return data

    def get_route(self):
        tree = ET.parse(ROUTES)
        data = {child.attrib["id"]: [tag.attrib for tag in child.iter()] for child in tree.getroot()}
        def create_route(waypoints):
            return [(self.grp._wmap.get_waypoint(carla.Location(float(wp.get("x")), float(wp.get("y")), float(wp.get("z")))), RoadOption(int(wp.get("road_option")))) for wp in waypoints]
        return create_route(data.get("0" if OCCLUSION else "1")[1:])

    def randomise_location(self, xyz):
        def safe_float(v, d=0.0):
            try: return float(v)
            except: return d
        if xyz["orientation"] == "horizontal":
            x = random.uniform(safe_float(xyz["xmin"]), safe_float(xyz["xmax"]))
            y = safe_float(xyz["y"])
            z = safe_float(xyz["z"])
        else:
            y = random.uniform(safe_float(xyz["ymin"]), safe_float(xyz["ymax"]))
            x = safe_float(xyz["x"])
            z = safe_float(xyz["z"])
        return carla.Transform(carla.Location(x, y, z), carla.Rotation(0, 0))

    def spawn_traffic(self):
        try:
            n_vehicles = len(self.world.get_actors().filter('*vehicle*'))
            if self.spawn_counter == SPAWN_DELAY and n_vehicles < MAX_TRAFFIC:
                bp = random.choice(self.traffic_blueprints)
                spawn_point = self.spawn_point1 if self.alternate_spawn else self.spawn_point2
                vehicle = self.world.try_spawn_actor(bp, spawn_point)
                if vehicle:
                    vehicle.set_autopilot(True)
                    self.traffic_manager.update_vehicle_lights(vehicle, True)
                    self.traffic_manager.auto_lane_change(vehicle, False)
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100)
                    route = self.route1 if self.alternate_spawn else self.route2
                    self.traffic_manager.set_path(vehicle, route)
                    self.alternate_spawn = not self.alternate_spawn
                    self.spawn_counter = max(0, self.spawn_counter - 1)
                else:
                    self.spawn_counter = SPAWN_DELAY
            elif self.spawn_counter > 0:
                self.spawn_counter -= 1
            else:
                self.spawn_counter = SPAWN_DELAY
        except Exception as e:
            print(f"Traffic spawn error: {e}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Weather
        weather_opt = options.get('weather') if options else None
        weather_opt = weather_opt or self.weather
        self.set_weather(weather_opt)

        self.timestep = 0
        self.ped_count = 0
        self.stopped = False
        self.frame_buffer = []
        self.current_vlm_action = "MAINTAIN"
        self.current_vlm_justification = "Starting the journey safely."

        # Clean sensors
        for sensor in [self.camera_sensor, self.collision_sensor, self.lane_sensor]:
            if sensor and sensor.is_listening():
                sensor.stop()
        self.collision_sensor = self.lane_sensor = self.camera_sensor = None
        self.collision_hist = []
        self.lane_hist = []
        self.image_queue = queue.Queue()

        # Clear actors
        self._clear_all_actors(['sensor.*', 'vehicle.*', 'walker.*', 'controller.ai.walker'])
        self.vehicle = self.obsticle = self.ped = None

        # Spawn ego
        while not self.vehicle:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.veh_transform)
        self.prev_s = 0.0

        # Occlusion
        if OCCLUSION:
            while not self.obsticle:
                self.obsticle = self.world.try_spawn_actor(self.ambulance, self.obsticle_transform)
            if MOVING_OCC:
                self.obsticle.set_autopilot(True)

        # Pedestrian
        self.ped_transform = self.randomise_location(self.spawn_locations["2"][4])
        while not self.ped:
            self.ped = self.world.try_spawn_actor(self.ped_bp, self.ped_transform)
        self.ped_target = self.ped_transform.location
        self.ped_target.x += 10

        # Sensors
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda e: CarlaEnv.get_collision_data(weak_self, e))
        self.lane_sensor = self.world.spawn_actor(self.lane_hist_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda e: CarlaEnv.get_lane_data(weak_self, e))
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.vehicle)
        self.camera_sensor.listen(self.image_queue.put)

        # Navigation
        self.route_ind = 0
        self.controller = PIDLateralController(self.vehicle)
        self.local_planner = LocalPlanner(self.vehicle)
        self.local_planner.set_global_plan(self.route)

        # Initialize VLM
        self.vlm_controller = VLMController()

        # Init simulation
        for _ in range(2):
            self.world.tick()
        if not self.image_queue.empty():
            self.process_img(self.image_queue.get())
        self.save_current_frame()

        info = self.get_current_vehicle_state()
        return self.front_camera, info

    @staticmethod
    def get_collision_data(weak_self, event):
        self = weak_self()
        if self: self.collision_hist.append(event)

    @staticmethod
    def get_lane_data(weak_self, event):
        self = weak_self()
        if self: self.lane_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data, dtype=np.dtype("uint8"))
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
        self.front_camera = cv2.UMat(i2).get().transpose(2, 0, 1)

    def get_distance_to_goal(self, ego, target):
        ex, ey = ego.get_location().x, ego.get_location().y
        return np.linalg.norm(np.array([ex, ey]) - np.array([target.x, target.y]))

    def update_route(self):
        if self.vehicle.get_location().y > (self.route[self.route_ind][0].transform.location.y - 1):
            if self.route_ind < len(self.route) - 1:
                self.route_ind += 1

    def step(self, action=None):
        # VLM Query
        vlm_action = None
        vlm_response_time = 0.0
        vlm_justification = "N/A"

        if self.save_current_frame() and len(self.frame_buffer) >= self.vlm_frames_needed:
            frames = self.frame_buffer[-self.vlm_frames_needed:]
            state = self.get_current_vehicle_state()
            vlm_action = self.vlm_controller.query(frames, state)
            vlm_response_time = self.vlm_controller.response_time
            vlm_justification = self.vlm_controller.current_justification

        if action is None:
            action = vlm_action if vlm_action is not None else 0.0

        self.update_route()
        if TRAFFIC:
            self.spawn_traffic()

        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        acceleration = speed - self.prev_s

        # Apply action
        if action < 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=abs(action), steer=self.controller.run_step(self.route[self.route_ind][0])))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=action, brake=0.0, steer=self.controller.run_step(self.route[self.route_ind][0])))

        if speed * 3.6 > 20:
            self.vehicle.set_target_velocity(carla.Vector3D(0, 5, 0))

        # Pedestrian logic
        dist2cross = self.get_distance_to_goal(self.vehicle, self.goal)
        dist = self.get_distance_to_goal(self.vehicle, self.ego_target)
        veh2ped_dist = self.get_distance_to_goal(self.vehicle, self.ped.get_location()) if self.ped else 999
        ped_dist = self.get_distance_to_goal(self.ped, self.ped_target) if self.ped else 999

        ped_lb, ped_hb = (1, 2) if OCCLUSION else (7, 8)
        if MOVING_OCC:
            if (self.ped.get_location().y - self.vehicle.get_location().y) < 13 and ped_dist > 0:
                if 3 < ped_dist < 4 and self.ped_count < 40:
                    self.ped_count += 1
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4))
        else:
            if (self.ped.get_location().y - (self.obsticle.get_location().y if OCCLUSION else self.vehicle.get_location().y)) < 15.5 and ped_dist > 0:
                if ped_lb < ped_dist < ped_hb and self.ped_count < 40:
                    self.ped_count += 1
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), speed=0))
                else:
                    self.ped.apply_control(carla.WalkerControl(carla.Vector3D(1, 0, 0), speed=4))

        self.world.tick()
        if not self.image_queue.empty():
            self.process_img(self.image_queue.get())
        self.save_current_frame()

        det = 1 if veh2ped_dist < 7.5 and ped_dist > 2 else 0
        done = col = 0
        if self.collision_hist:
            done = True
            self.collision_ep += 1
            col = -200
        elif dist <= 2:
            done = True
            self.successful_ep += 1
        elif self.timestep > 900:
            done = True
            self.stall_ep += 1

        c1 = -(0.2 * ((speed ** 2) / max(0.1, veh2ped_dist) + 2) + 50 * int(veh2ped_dist < 1)) * det
        c2 = 0.35 * speed * (1 - det)
        c3 = -(self.prev_s - speed) ** 2
        reward = c1 + c2 + c3 + col

        self.current_reward_components = {
            "safety_reward": c1, "progress_reward": c2, "smoothness_reward": c3,
            "collision_penalty": col, "total_reward": reward
        }
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        self.reward_history.append(self.current_reward_components)

        self.speeds.append(speed)
        self.accs.append(acceleration)
        self.dets.append(det)
        self.dist.append(dist2cross)
        self.rewards.append(reward)
        self.timestep += 1
        self.prev_s = speed

        info = self.get_current_vehicle_state()
        info.update(self.current_reward_components)
        info.update({
            "acceleration": acceleration,
            "vlm_response_time_ms": vlm_response_time * 1000,
            "vlm_justification": vlm_justification,
            "vlm_action_value": vlm_action if vlm_action is not None else action,
            "vlm_action_text": self.vlm_controller.current_action_text,
        })

        return self.front_camera, reward, done, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self, 'traffic_manager'):
            self.traffic_manager.set_random_device_seed(seed)
        return [seed]

    def render(self, mode): pass

    def _clear_all_actors(self, filters):
        for f in filters:
            for actor in self.world.get_actors().filter(f):
                if f == 'controller.ai.walker' and actor.is_alive:
                    actor.stop()
                if actor.is_alive:
                    actor.destroy()