from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork, BSRoad
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from typing import Dict, Text, Tuple, List  # , Self
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.objects import RF_BS, THz_BS

from ..sinr import *
from ..Shared import *
import pandas as pd

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20, #20 20
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

class HighwayEnvBS(HighwayEnvFast):

    def __init__(self, config: dict = None, render_mode: str = None) -> None:
        super().__init__(config)
        self.render_mode = render_mode

    @classmethod
    def default_config(cls) -> dict:
        conf = super().default_config()
        conf.update({
            "obstacle_count": 20,
            "action": {
                "type": "DiscreteDualObjectMetaAction",
            },
            "termination_agg_fn": 'any',
            "ho_reward": -5,
            "normalize_reward": True,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicleWithTelecom",
            "lanes_count": 3,
            "road_start": 0,
            "road_length": 10000,
            "observation": {
                "type": "KinematicsTele",
                "features": ["presence", "x", "y", "vx", "vy", 'gbs_cnt', 'haps_cnt'],
                'vehicles_count': 5,
            },
            "max_detection_distance": 1000,
            "controlled_vehicles": 1,
        })
        return conf

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.road.update()

    def _create_road(self) -> None:
        network = RoadNetwork.straight_road_network(
            self.config["lanes_count"],
            self.config['road_start'],
            self.config['road_length'],
            speed_limit=30
        )
        self.road = BSRoad(
            self.config["lanes_count"],
            self.config['road_start'],
            self.config['road_length'],
            network=network,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )
        for _ in range(self.config['obstacle_count']):
            obstacle_lane = np.random.choice([0, 8])
            obstacle_dist = np.random.randint(300, 10000)
            self.road.objects.append(Obstacle(self.road, [obstacle_dist, obstacle_lane]))

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        self.controlled_vehicles = []
        vehicle_dist = 0.0
        id = 0
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(self.road, speed=25, lane_id=self.config["initial_lane_id"], spacing=self.config["ego_spacing"])
            vehicle = self.action_type.vehicle_class(
                id, self.road, vehicle.position, vehicle.heading, vehicle.speed, max_dd=self.config["max_detection_distance"])
            id += 1
            if self.config['controlled_vehicles']:
                lanes = [4 * lane for lane in range(self.config["lanes_count"])]
                vehicle_lane = np.random.choice(lanes)
                vehicle_dist += 25
                vehicle.position = np.array([vehicle_dist, vehicle_lane])
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            else:
                self.controlled_vehicles.append(vehicle)
            for _ in range(others):
                vehicle = Vehicle.create_random(self.road, spacing=1/self.config["vehicles_density"])
                vehicle = other_vehicles_type(
                    id, self.road, vehicle.position, vehicle.heading, vehicle.speed, max_dd=self.config["max_detection_distance"])
                id += 1
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info['other_vehicle_collision'] = sum(vehicle.crashed for vehicle in self.road.vehicles if vehicle not in self.controlled_vehicles)
        info['agents_ho_prob'] = tuple(self.get_ho(action, vehicle)["ho_prob"] for vehicle in self.controlled_vehicles)
        # Store GBS and HAPS datarate separately for each agent
        perf_table = self.road.get_performance_table()
        info['agents_gbs_rate'] = tuple(
            perf_table['gbs'][vehicle.id, vehicle.target_current_bs] if vehicle.target_current_bs is not None and self.road.kind_of_bs(vehicle.target_current_bs) == 'gbs' else 0.0
            for vehicle in self.controlled_vehicles
        )
        info['agents_haps_rate'] = tuple(
            # perf_table['haps'][vehicle.id, 0] if vehicle.target_current_bs is not None and self.road.kind_of_bs(vehicle.target_current_bs) == 'haps' else 0.0
            perf_table['haps'][vehicle.id, 0] for vehicle in self.controlled_vehicles
        )
        info['agents_rewards'] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info['agents_collided'] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info['distance_travelled'] = tuple(vehicle.position[0] for vehicle in self.controlled_vehicles)
        info['agents_survived'] = self._is_truncated()
        return info

    def _agent_is_terminal(self, vehicle) -> bool:
        return vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _is_terminated(self) -> bool:
        agent_terminal = [self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles]
        agg_fn = {'any': any, 'all': all}[self.config['termination_agg_fn']]
        return agg_fn(agent_terminal)

    def _simulate(self, action) -> None:
        super()._simulate(action)
        self.road.update()

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents"""
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        # Weighted sum, can be customized
        reward = rewards["tran_reward"] + rewards["tele_reward"]
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) else vehicle.lane_index[2]
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        vid = vehicle.id

        # Communication datarate (handover-aware)
        result_rf = 0
        if vehicle.target_current_bs is not None:
            bs_kind = self.road.kind_of_bs(vehicle.target_current_bs)
            perf_table = self.road.get_performance_table()
            if bs_kind == 'gbs':
                result_rf = perf_table['gbs'][vid, vehicle.target_current_bs]
            elif bs_kind == 'haps':
                result_rf = perf_table['haps'][vid, 0]
        result_rf = utils.lmap(result_rf, [0, 4e8], [0, 2])

        # Transportation reward (normalized)
        tran_reward = (
            self.config.get("collision_reward", 0) * float(vehicle.crashed) +
            self.config.get("right_lane_reward", 0) * (lane / max(len(neighbours) - 1, 1)) +
            self.config.get("high_speed_reward", 0) * np.clip(scaled_speed, 0, 1)
        )
        tran_reward = utils.lmap(
            tran_reward,
            [self.config["collision_reward"], self.config["high_speed_reward"] + self.config["right_lane_reward"]],
            [0, 1]
        )
        tran_reward *= float(vehicle.on_road)

        return {
            "tran_reward": float(tran_reward),
            "tele_reward": float(result_rf)
        }

    def get_ho(self, action: int, vehicle: Vehicle) -> dict:
        ho_density = vehicle.target_ho / vehicle.position[0] if vehicle.position[0] != 0 else 0
        ho_prob = vehicle.target_ho / max((self.steps), 1)
        return {
            "ho_density": float(ho_density),
            "ho_prob": float(ho_prob),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self.viewer.get_image()
        elif self.render_mode == "human":
            self.viewer.show()

