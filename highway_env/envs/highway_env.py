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
            "reward_speed_range": [15, 25],#[20, 30]
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
            "vehicles_count": 20,
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

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        # self.shared_state = SharedState()

    @classmethod
    def default_config(cls) -> dict:
        conf = super().default_config()
        conf.update({
            "obstacle_count": 20,
            # https://github.com/eleurent/highway-env/issues/35#issuecomment-1206427869
            # https://github.com/eleurent/highway-env/pull/352/files
            "action": {
                "type": "DiscreteDualObjectMetaAction",
            },
            "termination_agg_fn": 'any',
            'rf_bs_count': 5,  #20
            'thz_bs_count': 20,  #100
            'rf_bs_max_connections': 10,  # 最大连接数量
            'thz_bs_max_connections': 5,
            "tele_reward": 4.5 / (10 ** 6.5),#3e-6,
            "tele_reward_threshold": 4.5 * (10 ** 6.5),#3e-6,
            # "dr_reward": 0.2,
            "ho_reward": -5,
            "normalize_reward": True,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicleWithTelecom",
            "lanes_count": 3, #4
            "road_start": 0,
            "road_length": 10000,
            "observation": {
                    "type": "KinematicsTele",
                    "features": ["presence", "x", "y", "vx", "vy", 'rf_cnt', 'thz_cnt'],
                'vehicles_count': 5,
            },
            "max_detection_distance": 1000,  # 观测距离
        })
        return conf

    def _reset(self) -> None:
        # super()._reset()
        # self.shared_state = SharedState()
        self._create_road()
        self._create_vehicles()
        self.road.update()
        # self._create_bs_assignment_table()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        network = RoadNetwork.straight_road_network(self.config["lanes_count"],
                                                    self.config['road_start'],
                                                    self.config['road_length'],
                                                    speed_limit=30)
        # RF bss的创建和管理移到了BSRoad中
        self.road = BSRoad(self.config['rf_bs_count'],
                           self.config['thz_bs_count'],
                           self.config['rf_bs_max_connections'],
                           self.config['thz_bs_max_connections'],
                           self.config["lanes_count"],
                           self.config['road_start'],
                           self.config['road_length'],
                           network=network,
                           np_random=self.np_random,
                           record_history=self.config["show_trajectories"])
        # print('lanes count', self.config["lanes_count"])#debug
        # Adding obstacles at random places on the lanes
        for _ in range(self.config['obstacle_count']):
            # lanes = [4 * lane for lane in range(self.config["lanes_count"])]
            # why 0 or 8
            obstacle_lane = np.random.choice([0, 8])  #random generate lane number (integer) obstacle_lane = np.random.choice(lanes)
            # obstacle_lane = 0
            obstacle_dist = np.random.randint(300, 10000)
            self.road.objects.append(Obstacle(self.road, [obstacle_dist, obstacle_lane]))


    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        vehicle_dist = 0.0
        id = 0  # 每个vehicle的id, 需要与self.road.vehicles的添加顺序一致
        # lanes = [4 * lane for lane in range(self.config["lanes_count"])]
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(self.road, speed=25, lane_id=self.config["initial_lane_id"], spacing=self.config["ego_spacing"])
            vehicle = self.action_type.vehicle_class(
                id, self.road, vehicle.position, vehicle.heading, vehicle.speed, max_dd = self.config["max_detection_distance"])
            id += 1
            if self.config['controlled_vehicles']:
                # vehicle_lane = np.random.choice(lanes)
                # To make sure the agents doesn't collide on the start itself because of the random obstacles.
                # vehicle.position = np.array([vehicle_dist, vehicle_lane])
                # vehicle_dist += 25
                lanes = [4 * lane for lane in range(self.config["lanes_count"])]
                vehicle_lane = np.random.choice(lanes)
                vehicle_dist += 25
                vehicle.position = np.array([vehicle_dist, vehicle_lane])
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
                # self.shared_state.vehicles.append(vehicle)  # shared state append
            else:
                self.controlled_vehicles.append(vehicle)
            for _ in range(others):
                vehicle = Vehicle.create_random(self.road, spacing=1/self.config["vehicles_density"])
                vehicle = other_vehicles_type(
                    id, self.road, vehicle.position, vehicle.heading, vehicle.speed, max_dd = self.config["max_detection_distance"])
                id += 1
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                # self.shared_state.vehicles.append(vehicle)  # shared state append

    # No longer used
    def _create_assignment_rf_matrix(self):
        '''
        distance matrice between AVs and RF BSs.
        '''
        bss = self.shared_state.rf_bss
        vehicles = self.shared_state.vehicles
        # 用一个numpy表示就行了, 顺序是不会乱的
        return np.zeros((len(vehicles), len(bss)))


    # No longer used
    def _create_assignment_thz_matrix(self):
        '''
        distance matrice between AVs and RF BSs.
        '''
        bss = self.shared_state.thz_bss
        vehicles = self.shared_state.vehicles
        return np.zeros((len(vehicles), len(bss)))


    # not used
    def _get_bs_assignment_table(self) -> np.ndarray:
        return self.shared_state.bs_assignment_table

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info['other_vehicle_collision'] = \
            sum(vehicle.crashed for vehicle in self.road.vehicles if vehicle not in self.controlled_vehicles)

        # info['agents_te_rewards'] = tuple(self._agent_rewards(action, vehicle)['tele_reward'] for vehicle in self.controlled_vehicles)
        info['agents_ho_prob'] = tuple(self.get_ho(action, vehicle)["ho_prob"] for vehicle in self.controlled_vehicles)

        info['agents_tran_all_rewards'] = tuple(self.get_seperate_reward(action, vehicle)["tran_reward"] for vehicle in self.controlled_vehicles)
        info['agents_tele_all_rewards'] = tuple(self._agent_rewards(action, vehicle)["tele_reward"] for vehicle in self.controlled_vehicles)

        info['agents_rewards'] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        # info['agents_tr_rewards'] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) to be implemented
        info['agents_collided'] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info['distance_travelled'] = tuple(vehicle.position[0] for vehicle in self.controlled_vehicles)
        info['agents_survived'] = self._is_truncated()
        return info

    # To check if a single agent has collided
    def _agent_is_terminal(self, vehicle) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road)

    # To terminate when the duration limit has reached.
    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

    # To terminate training based on any or all agent has collided.
    def _is_terminated(self) -> bool:
        # https://github.com/eleurent/highway-env/issues/35#issuecomment-1206427869
        agent_terminal = [self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles]
        agg_fn = {'any': any, 'all': all}[self.config['termination_agg_fn']]
        return agg_fn(agent_terminal)

    def _simulate(self, action) -> None:
        super()._simulate(action)
        # 更新距离参数
        self.road.update()

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents"""
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)
        # return sum_total_reward #,sum_tr_reward,sum_te_reward

    # TODO: 什么时候会调用这个函数? MORL   very important
    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""

        # agents_rewards = [self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles]
        # return {
        #     name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
        #     for name in agents_rewards[0].keys()
        # }

        agents_rewards = [self.get_seperate_reward(action, vehicle) for vehicle in self.controlled_vehicles]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }


    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""

        tran_reward = self.get_seperate_reward(action, vehicle)["tran_reward"]
        tele_reward = self._agent_rewards(action, vehicle)["tele_reward"]

        
        reward = tran_reward + tele_reward
        # print('reward *=', reward)
        return reward  #,reward_tr,reward_te

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        vid = vehicle.id

        result_rf = 0
        if vehicle.target_current_bs is not None:
            result_rf = self.road.get_performance_table()[vid, vehicle.target_current_bs]
            # print('self step is ========================',self.steps,type(self.steps))
            # print('ho',float(vehicle.target_ho))
            # print('steps',steps)
            # print('result_rf',result_rf)
            if self.steps > 2: # 3
                result_rf *=  1 - (vehicle.target_ho/(self.steps))
            
            result_rf = utils.lmap(result_rf,[0, self.config["tele_reward_threshold"]],[0, 2])#1e8
            # result_rf = "{:.2f}".format(result_rf)
            # print('final result_rf',result_rf)
        
        # reward_ho = vehicle.target_ho / vehicle.position[0]  # assume this is MyMDPVehicle

        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road),
            "tele_reward": float(result_rf) # / 10e7
            # "ho_reward": float(reward_ho) #* 10
            # "thz_reward": float(max_rate_thz)
            # "rf_reward": float(1/rf_sinr_specific_vehicle)
        }
    
    def get_seperate_reward(self, action: int, vehicle: Vehicle) -> float:
        tranKeys = ["collision_reward","right_lane_reward","high_speed_reward","on_road_reward"]
        # teleKeys = ["tele_reward"] #,"ho_reward"
        rewards = self._agent_rewards(action, vehicle)

        filterByKey = lambda keys: {x: rewards[x] for x in keys}
        tranData = filterByKey(tranKeys)
        # teleData = filterByKey(teleKeys)

        tran_reward = sum(self.config.get(name, 0) * reward for name, reward in tranData.items())
        # tele_reward = sum(self.config.get(name, 0) * reward for name, reward in teleData.items()) 
        tran_reward = utils.lmap(tran_reward,
                    [self.config["collision_reward"], self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                    [0, 1])
        tran_reward *= rewards['on_road_reward']

        tele_reward = self._agent_rewards(action, vehicle)["tele_reward"]
        return {
            "tran_reward": float(tran_reward),
            "tele_reward": float(tele_reward),
        }
    
    def get_ho(self, action: int, vehicle: Vehicle) -> float:
        ho_density = vehicle.target_ho / vehicle.position[0]  # assume this is MyMDPVehicle
        ho_prob = vehicle.target_ho/(self.steps+1e-5) # avoid divide 0
        return {
            "ho_density": float(ho_density),
            "ho_prob": float(ho_prob),
        }

    # No longer used
    def _get_distance_rf_matrix(self) -> np.ndarray:
        '''
        distance matrice between AVs and RF BSs.
        计算vehicles 与bss的距离, return: [len(vehicles), len(bss)]
        '''

        vehicles = self.shared_state.vehicles
        bss = self.shared_state.rf_bss

        vehicles_pose = np.array([v.position for v in vehicles])
        bss_pose = np.array([b.position for b in bss])

        distance_matrix = np.sqrt(((vehicles_pose[:, None, :] - bss_pose)**2).sum(axis=-1))
        return distance_matrix


    # No longer used
    def _get_distance_thz_matrix(self):
        '''
        distance matrice between AVs and Thz BSs.
        '''
        bss = self.shared_state.thz_bss
        vehicles = self.shared_state.vehicles

        vehicles_pose = np.array([v.position for v in vehicles])
        bss_pose = np.array([b.position for b in bss])

        distance_matrix = np.sqrt(((vehicles_pose[:, None, :] - bss_pose)**2).sum(axis=-1))
        return distance_matrix

    # No longer used
    def recursive_select_max_bs(self, result):
        # 从result中选择 最大且可行 的值
        i = 0
        # print("result rf\n",result_rf)
        bs_vacant_list = self.get_vacant_bs_list()  # 剩余连接数量
        bs_max_name = result.idxmax()
        length = result.size
        while (i < length):
            if self.check_connect_with_bs(bs_vacant_list, bs_max_name):
                bs_max_name = result.idxmax()
                max_rate = np.max(result)
                break
            result.drop(bs_max_name)  # drop the maximum one due to limitation of capacity
            bs_max_name = result.idxmax()
            i = i + 1
        return bs_max_name, max_rate

    # No longer used
    def get_performance_assignment_tables(self):
        # return the mobility aware throughput table and base station assignment table
        return self.shared_state.bs_performance_table, self.shared_state.bs_assignment_table

    # No longer used
    def get_concurrent_user(self):
        # Based on the base station assignment table, we calculate how many concurrent AVs connect with the specific base stations.
        '''
        INPUT
            bs1	bs2	bs3	bs4	bs5	bs6	bs7	bs8	bs9	bs10
        AV1	0	1	0	1	1	1	1	0	0	1
        AV2	1	1	1	1	0	0	0	0	1	0
        AV3	0	0	1	1	1	0	0	0	0	0
        AV4	1	0	0	1	0	0	1	1	1	0
        AV5	1	0	1	1	1	1	1	1	1	0
        AV6	1	0	1	1	0	0	1	0	1	0
        AV7	0	0	0	0	1	0	1	0	0	1
        AV8	1	0	1	0	0	1	1	0	1	1
        AV9	0	0	1	1	1	1	0	1	0	1
        RETURN
        bs1	bs2	bs3	bs4	bs5	bs6	bs7	bs8	bs9	bs10
        5	2	6	7	5	4	6	3	5	4
        '''
        # print(self.bs_assignment_table.sum())
        return self.shared_state.bs_assignment_table.sum()

    # No longer used
    def get_config(self, attr):
        return self.config[attr]

    # No longer used
    def get_current_user(self):
        return self.get_concurrent_user(self)

    # No longer used
    def get_vacant_bs_list(self):  # previously get_vacant_rf_bs_list extend to the thz
        # Based on the concurrent user, we generate how many vacants on each base stations
        '''
            bs1	bs2	bs3	bs4	bs5	bs6	bs7	bs8	bs9	bs10
currentuser	7	4	6	4	3	2	5	5	3	4
tupple	    5	5	5	5	5	5	5	5	5	5
result	    -2	1	-1	1	2	3	0	0	2	1
        '''
        # n_rf = self.config['rf_bs_count']
        # n_thz = self.config['thz_bs_count']
        n_rf = len(self.shared_state.rf_bss)
        n_thz = len(self.shared_state.thz_bss)
        tupples_rf = np.ones(n_rf) * 10
        tupples_thz = np.ones(n_thz) * 5
        tupples = np.concatenate((tupples_rf, tupples_thz), axis=None)
        result = tupples
        current_users = self.get_concurrent_user()
        try:
            result = np.subtract(tupples, current_users)
            result1 = pd.Series(result, index=current_users.index)
        except:
            print("current_users and tupples length is unequal, current users and tupples length are", len(current_users), len(tupples))
        return result1

    # No longer used
    def check_connect_with_bs(self, vacant_list, bs):
        # print("vacant_list\n",vacant_list)
        # print("bs\n",bs)
        num_vacant = int(vacant_list.loc[bs])

        return (num_vacant > 0)
