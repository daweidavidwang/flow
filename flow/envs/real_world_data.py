"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}


class RealWorldEnv(Env):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum time a light must be constant before
      it switches (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL,
      options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.

    Attributes
    ----------
    grid_array : dict
        Array containing information on the traffic light grid, such as the
        length of roads, row_num, col_num, number of initial cars
    rows : int
        Number of rows in this traffic light grid network
    cols : int
        Number of columns in this traffic light grid network
    num_traffic_lights : int
        Number of intersection in this traffic light grid network
    tl_type : str
        Type of traffic lights, either 'actuated' or 'static'
    steps : int
        Horizon of this experiment, see EnvParams.horion
    obs_var_labels : dict
        Referenced in the visualizer. Tells the visualizer which
        metrics to track
    node_mapping : dict
        Dictionary mapping intersections / nodes (nomenclature is used
        interchangeably here) to the edges that are leading to said
        intersection / node
    last_change : np array [num_traffic_lights]x1 np array
        Multi-dimensional array keeping track, in timesteps, of how much time
        has passed since the last change to yellow for each traffic light
    direction : np array [num_traffic_lights]x1 np array
        Multi-dimensional array keeping track of which direction in traffic
        light is flowing. 0 indicates flow from top to bottom, and
        1 indicates flow from left to right
    currently_yellow : np array [num_traffic_lights]x1 np array
        Multi-dimensional array keeping track of whether or not each traffic
        light is currently yellow. 1 if yellow, 0 if not
    min_switch_time : np array [num_traffic_lights]x1 np array
        The minimum time in timesteps that a light can be yellow. Serves
        as a lower bound
    discrete : bool
        Indicates whether or not the action space is discrete. See below for
        more information:
        https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))


        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        self.discrete = env_params.additional_params.get("discrete", False)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """Distance and speed to the coming intersection,
         the in-coming flow volume of each intersection path. """
        return Box(low=0, high=1, shape=(1 , ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # compute the normalizers
        pass


    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass


    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return - rewards.min_delay_unscaled(self) \
            - rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)

    # ===============================
    # ============ UTILS ============
    # ===============================

    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.

        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)

        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def _convert_edge(self, edges):
        """Convert the string edge to a number.

        Start at the bottom left vertical edge and going right and then up, so
        the bottom left vertical edge is zero, the right edge beside it  is 1.

        The numbers are assigned along the lowest column, then the lowest row,
        then the second lowest column, etc. Left goes before right, top goes
        before bottom.

        The values are zero indexed.

        Parameters
        ----------
        edges : list of str or str
            name of the edge(s)

        Returns
        -------
        list of int or int
            a number uniquely identifying each edge
        """
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Act as utility function for convert_edge."""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("center")[1][0])
                base = ((self.cols + 1) * self.rows * 2) \
                    + ((self.rows + 1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def _get_relative_node(self, agent_id, direction):
        """Yield node number of traffic light agent in a given direction.

        For example, the nodes in a traffic light grid with 2 rows and 3
        columns are indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

        See flow.networks.traffic_light_grid for more information.

        Example of function usage:
        - Seeking the "top" direction to ":center0" would return 3.
        - Seeking the "bottom" direction to ":center0" would return -1.

        Parameters
        ----------
        agent_id : str
            agent id of the form ":center#"
        direction : str
            top, bottom, left, right

        Returns
        -------
        int
            node number
        """
        ID_IDX = 1
        agent_id_num = int(agent_id.split("center")[ID_IDX])
        if direction == "top":
            node = agent_id_num + self.cols
            if node >= self.cols * self.rows:
                node = -1
        elif direction == "bottom":
            node = agent_id_num - self.cols
            if node < 0:
                node = -1
        elif direction == "left":
            if agent_id_num % self.cols == 0:
                node = -1
            else:
                node = agent_id_num - 1
        elif direction == "right":
            if agent_id_num % self.cols == self.cols - 1:
                node = -1
            else:
                node = agent_id_num + 1
        else:
            raise NotImplementedError

        return node

    def additional_command(self):
        """See parent class.

        Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge.
        """
        for veh_id in self.k.vehicle.get_ids():
            self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        """Reroute vehicle associated with veh_id.

        Checks if an edge is the final edge. If it is return the route it
        should start off at.
        """
        edge = self.k.vehicle.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are
        # going to remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            type_id = self.k.vehicle.get_type(veh_id)
            lane_index = self.k.vehicle.get_lane(veh_id)
            # remove the vehicle
            self.k.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            self.k.vehicle.add(
                veh_id=veh_id,
                edge=route_id,
                type_id=str(type_id),
                lane=str(lane_index),
                pos="0",
                speed="max")

    def get_closest_to_intersection(self, edges, num_closest, padding=False):
        """Return the IDs of the vehicles that are closest to an intersection.

        For each edge in edges, return the IDs (veh_id) of the num_closest
        vehicles in edge that are closest to an intersection (the intersection
        they are heading towards).

        This function performs no check on whether or not edges are going
        towards an intersection or not, it just gets the vehicles that are
        closest to the end of their edges.

        If there are less than num_closest vehicles on an edge, the function
        performs padding by adding empty strings "" instead of vehicle ids if
        the padding parameter is set to True.

        Parameters
        ----------
        edges : str | str list
            ID of an edge or list of edge IDs.
        num_closest : int (> 0)
            Number of vehicles to consider on each edge.
        padding : bool (default False)
            If there are less than num_closest vehicles on an edge, perform
            padding by adding empty strings "" instead of vehicle ids if the
            padding parameter is set to True (note: leaving padding to False
            while passing a list of several edges as parameter can lead to
            information loss since you will not know which edge, if any,
            contains less than num_closest vehicles).

        Usage
        -----
        For example, consider the following network, composed of 4 edges
        whose ids are "edge0", "edge1", "edge2" and "edge3", the numbers
        being vehicles all headed towards intersection x. The ID of the vehicle
        with number n is "veh{n}" (edge "veh0", "veh1"...).

                            edge1
                            |   |
                            | 7 |
                            | 8 |
               -------------|   |-------------
        edge0    1 2 3 4 5 6  x                 edge2
               -------------|   |-------------
                            | 9 |
                            | 10|
                            | 11|
                            edge3

        And consider the following example calls on the previous network:

        >>> get_closest_to_intersection("edge0", 4)
        ["veh6", "veh5", "veh4", "veh3"]

        >>> get_closest_to_intersection("edge0", 8)
        ["veh6", "veh5", "veh4", "veh3", "veh2", "veh1"]

        >>> get_closest_to_intersection("edge0", 8, padding=True)
        ["veh6", "veh5", "veh4", "veh3", "veh2", "veh1", "", ""]

        >>> get_closest_to_intersection(["edge0", "edge1", "edge2", "edge3"],
                                         3, padding=True)
        ["veh6", "veh5", "veh4", "veh8", "veh7", "", "", "", "", "veh9",
         "veh10", "veh11"]

        Returns
        -------
        str list
            If n is the number of edges given as parameters, then the returned
            list contains n * num_closest vehicle IDs.

        Raises
        ------
        ValueError
            if num_closest <= 0
        """
        if num_closest <= 0:
            raise ValueError("Function get_closest_to_intersection called with"
                             "parameter num_closest={}, but num_closest should"
                             "be positive".format(num_closest))

        if isinstance(edges, list):
            ids = [self.get_closest_to_intersection(edge, num_closest)
                   for edge in edges]
            # flatten the list and return it
            return [veh_id for sublist in ids for veh_id in sublist]

        # get the ids of all the vehicles on the edge 'edges' ordered by
        # increasing distance to end of edge (intersection)
        veh_ids_ordered = sorted(self.k.vehicle.get_ids_by_edge(edges),
                                 key=self.get_distance_to_intersection)

        # return the ids of the num_closest vehicles closest to the
        # intersection, potentially with ""-padding.
        pad_lst = [""] * (num_closest - len(veh_ids_ordered))
        return veh_ids_ordered[:num_closest] + (pad_lst if padding else [])


class RealWorldPOEnv(RealWorldEnv):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.
    * num_observed: number of vehicles nearest each intersection that is
      observed in the state space; defaults to 2

    States
        An observation is the number of observed vehicles in each intersection
        closest to the traffic lights, a number uniquely identifying which
        edge the vehicle is on, and the speed of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the delay of each vehicle minus a penalty for switching
        traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.

    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_PO_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 2)

        # used during visualization
        self.observed_ids = []

    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, edge information, and traffic light
        state.
        """
        tl_box = Box(
            low=0.,
            high=3,
            shape=(1,),
            dtype=np.float32)
        return tl_box

    def get_state(self):
        return np.array([0])

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        [self.k.vehicle.set_observed(veh_id) for veh_id in self.observed_ids]

