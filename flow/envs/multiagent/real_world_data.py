"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from copy import deepcopy
from flow.core import rewards
from flow.envs.real_world_data import RealWorldPOEnv
from flow.envs.multiagent import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1


class MultiRealWorldPOEnv(RealWorldPOEnv, MultiEnv):
    """Multiagent shared model version of TrafficLightGridPOEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def __init__(self, env_params, sim_params, network, simulator='traci_rd'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        if len(self.network.vehicles.selected_intersection) > 0:
            self.select_vehicles()
            ## re-do kernel vehicle initialize after selection of vehicles
            # initial the vehicles kernel using the VehicleParams object
            self.k.vehicle.initialize(deepcopy(self.network.vehicles))
            # store the initial vehicle ids
            self.initial_ids = deepcopy(self.network.vehicles.ids)

    @property
    def observation_space(self):
        """
        Distance and speed to the coming intersection, 
        the in-coming flow volume of each intersection path. 
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(1, ),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1, ),
            dtype=np.float32)

    def select_vehicles(self):
        from flow.core.params import VehicleParams
        from flow.controllers import RealDataRouter,RoutingLaneChanger, LCIDMController
        infected_list = []
        for junction_id in self.network.vehicles.selected_intersection:
            infected_list.extend(self.k.network.get_incedge(junction_id))
        routing = deepcopy(self.network.vehicles.vehicle_routing)
        for idx in range(len(routing)-1, -1, -1):
            match = set(routing[idx]['route']) & set(infected_list)
            if len(match) == 0:
                del routing[idx]
        
        current_vehicle_num = len(routing)
        while len(routing)>(1-self.network.vehicles.sparsing_traffic)*current_vehicle_num:
            rand_idx = np.random.randint(len(routing))
            del routing[rand_idx]

        new_vehicles = VehicleParams()
        new_vehicles.vehicle_routing = deepcopy(routing)
        for veh in new_vehicles.vehicle_routing:
            new_vehicles.add(
                veh_id=veh['id'],
                acceleration_controller=(LCIDMController, {}),
                lane_change_controller=(RoutingLaneChanger, {}),
                routing_controller=(RealDataRouter, {}),
                depart=veh['depart'],
                num_vehicles=1)
        self.network.vehicles = deepcopy(new_vehicles)

    def get_state(self):
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            observation = np.array([0])
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        """
        for veh_id in self.k.vehicle.get_rl_ids():
            self.k.vehicle.apply_acceleration(veh_id, rl_actions[veh_id])

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}

        # print(str(rl_actions))
        if self.env_params.evaluate:
            rew = -rewards.min_delay_unscaled(self)
        else:
            rew = -rewards.min_delay_unscaled(self) \
                  + rewards.penalize_standstill(self, gain=0.2)

        # each agent receives reward normalized by number of lights
        rew /= len(self.k.vehicle.get_rl_ids())+ 0.0001

        rews = {}
        for rl_id_num, rl_id in enumerate(self.k.vehicle.get_rl_ids()):
            rews[rl_id] = rew
        return rews

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        for veh_ids in self.observed_ids:
            for veh_id in veh_ids:
                self.k.vehicle.set_observed(veh_id)
