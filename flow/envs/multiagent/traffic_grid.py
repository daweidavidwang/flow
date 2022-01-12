"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.traffic_grid import TrafficGridPOEnv
from flow.envs.multiagent import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1


class MultiTrafficGridPOEnv(TrafficGridPOEnv, MultiEnv):
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

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)

    @property
    def observation_space(self):
        """
        Distance and speed to the coming intersection, 
        the in-coming flow volume of each intersection path. 
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(2 + 3 * 4 * self.num_observed +
                   2 * self.num_local_edges, 
                   ),
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

    def get_state(self):
        """Observations for each traffic light agent.

        :return: dictionary which contains agent-wise observations as follows:
        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity, distance to
        intersection, edge number.
        - For edges in the network, gives the density and average velocity.
        - For the self.num_local_lights number of nearest lights (itself
        included), gives the traffic light information, including the last
        change time, light direction (i.e. phase), and a currently_yellow flag.
        """
        # Normalization factors
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        # TODO(cathywu) refactor TrafficLightGridPOEnv with convenience
        # methods for observations, but remember to flatten for single-agent

        # Observed vehicle information
        # get intersection information and corresponding vehicle's info
        speeds = []
        dist_to_intersec = []
        edge_number = []

        node_mapping = self.network.node_mapping
        for rl_id in self.k.vehicle.get_rl_ids():
            all_observed_ids = []
            edge_id = self.k.vehicle.get_edge(rl_id)
            center_id = -1
            obs_rotation = 0
            for c_id in range(len(node_mapping)):
                if edge_id in node_mapping[c_id][1] or edge_id[1:8] == node_mapping[c_id][0]:
                    for ei in range(len(node_mapping[c_id][1])):
                        if edge_id == node_mapping[c_id][1][ei]:
                            obs_rotation = ei
                    center_id = c_id
                    break
            local_speeds = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            if center_id < 0:
                local_speeds = [0,0,0,0,0,0,0,0]
                local_dists_to_intersec = [0,0,0,0,0,0,0,0]
                local_edge_numbers = [0,0,0,0,0,0,0,0]
            else:
                for idx in range(len(node_mapping[center_id][1])):
                    edge = node_mapping[center_id][1][(idx+obs_rotation)%len(node_mapping[center_id][1])]
                    observed_ids = \
                        self.get_closest_to_intersection(edge, self.num_observed)
                    all_observed_ids.append(observed_ids)

                    # check which edges we have so we can always pad in the right
                    # positions
                    local_speeds.extend(
                        [self.k.vehicle.get_speed(veh_id) / max_speed for veh_id in
                        observed_ids])
                    local_dists_to_intersec.extend([(self.k.network.edge_length(
                        self.k.vehicle.get_edge(
                            veh_id)) - self.k.vehicle.get_position(
                        veh_id)) / max_dist for veh_id in observed_ids])
                    local_edge_numbers.extend([self._convert_edge(
                        self.k.vehicle.get_edge(veh_id)) / (
                        self.k.network.network.num_edges - 1) for veh_id in
                                            observed_ids])
                    if len(observed_ids) < self.num_observed:
                        diff = self.num_observed - len(observed_ids)
                        local_speeds.extend([1] * diff)
                        local_dists_to_intersec.extend([1] * diff)
                        local_edge_numbers.extend([0] * diff)
            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)

        # Edge information
        density = []
        velocity_avg = []
        for edge in self.k.network.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                # TODO(cathywu) Why is there a 5 here?
                density += [5 * len(ids) / self.k.network.edge_length(edge)]
                velocity_avg += [np.mean(
                    [self.k.vehicle.get_speed(veh_id) for veh_id in
                     ids]) / max_speed]
            else:
                density += [0]
                velocity_avg += [0]
        density = np.array(density)
        velocity_avg = np.array(velocity_avg)

        # combination of obs
        obs = {}
        for rl_id_num, rl_id in enumerate(self.k.vehicle.get_rl_ids()):
            edge_id = self.k.vehicle.get_edge(rl_id)
            center_id = -1
            obs_rotation = 0
            for c_id in range(len(node_mapping)):
                if edge_id in node_mapping[c_id][1] or edge_id[1:8] == node_mapping[c_id][0]:
                    for ei in range(len(node_mapping[c_id][1])):
                        if edge_id == node_mapping[c_id][1][ei]:
                            obs_rotation = ei
                    center_id = c_id
                    break
            if center_id>-1:
                local_edges = [node_mapping[center_id][1][(idx+obs_rotation)%len(node_mapping[center_id][1])] for idx in range(len(node_mapping[center_id][1]))]
                # local_edges = node_mapping[center_id][1]
                # print("local egese!!!!!!!!!"+str(local_edges))
                local_edge_numbers = [self.k.network.get_edge_list().index(e)
                                        for e in local_edges]
                # print("local edge number !!!!!!!!"+str(local_edge_numbers))
                # print("velocity avg"+str(velocity_avg[local_edge_numbers])+str())
                observation = np.clip(np.array(np.concatenate(
                [[self.k.vehicle.get_speed(rl_id) / max_speed],
                [(self.k.network.edge_length(
                        self.k.vehicle.get_edge(
                            rl_id)) - self.k.vehicle.get_position(
                        rl_id)) / max_dist],
                speeds[rl_id_num], dist_to_intersec[rl_id_num],
                    edge_number[rl_id_num], density[local_edge_numbers],
                    velocity_avg[local_edge_numbers]
                    ])), 0, 1)
                obs.update({rl_id: observation})
            else:
                ## the vehicle already leaving the map
                observation = np.clip(np.array(np.concatenate(
                [[self.k.vehicle.get_speed(rl_id) / max_speed],
                [(self.k.network.edge_length(
                        self.k.vehicle.get_edge(
                            rl_id)) - self.k.vehicle.get_position(
                        rl_id)) / max_dist],
                speeds[rl_id_num], dist_to_intersec[rl_id_num],
                    edge_number[rl_id_num], [0,0,0,0],
                    [0,0,0,0]
                    ])), 0, 1)
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
        rew /= len(self.k.vehicle.get_rl_ids())

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
