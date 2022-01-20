"""Contains the traffic light grid scenario class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.core.params import NetParams
from collections import defaultdict
import numpy as np
import _thread

ADDITIONAL_NET_PARAMS = {
    # dictionary of traffic light grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": 3,
        # number of vertical columns of edges
        "col_num": 2,
        # length of inner edges in the traffic light grid network
        "inner_length": None,
        # length of edges where vehicles enter the network
        "short_length": None,
        # length of edges where vehicles exit the network
        "long_length": None,
        # number of cars starting at the edges heading to the top
        "cars_top": 20,
        # number of cars starting at the edges heading to the bottom
        "cars_bot": 20,
        # number of cars starting at the edges heading to the left
        "cars_left": 20,
        # number of cars starting at the edges heading to the right
        "cars_right": 20,
    },
    # number of lanes in the horizontal edges
    "horizontal_lanes": 1,
    # number of lanes in the vertical edges
    "vertical_lanes": 1,
    # speed limit for all edges, may be represented as a float value, or a
    # dictionary with separate values for vertical and horizontal lanes
    "speed_limit": {
        "horizontal": 35,
        "vertical": 35
    }
}


class RealWorldNetwork(Network):
    """Traffic Light Grid network class.

    The traffic light grid network consists of m vertical lanes and n
    horizontal lanes, with a total of nxm intersections where the vertical
    and horizontal edges meet.

    Requires from net_params:

    * **grid_array** : dictionary of grid array data, with the following keys

      * **row_num** : number of horizontal rows of edges
      * **col_num** : number of vertical columns of edges
      * **inner_length** : length of inner edges in traffic light grid network
      * **short_length** : length of edges that vehicles start on
      * **long_length** : length of final edge in route
      * **cars_top** : number of cars starting at the edges heading to the top
      * **cars_bot** : number of cars starting at the edges heading to the
        bottom
      * **cars_left** : number of cars starting at the edges heading to the
        left
      * **cars_right** : number of cars starting at the edges heading to the
        right

    * **horizontal_lanes** : number of lanes in the horizontal edges
    * **vertical_lanes** : number of lanes in the vertical edges
    * **speed_limit** : speed limit for all edges. This may be represented as a
      float value, or a dictionary with separate values for vertical and
      horizontal lanes.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import TrafficLightGridNetwork
    >>>
    >>> network = TrafficLightGridNetwork(
    >>>     name='grid',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'grid_array': {
    >>>                 'row_num': 3,
    >>>                 'col_num': 2,
    >>>                 'inner_length': 500,
    >>>                 'short_length': 500,
    >>>                 'long_length': 500,
    >>>                 'cars_top': 20,
    >>>                 'cars_bot': 20,
    >>>                 'cars_left': 20,
    >>>                 'cars_right': 20,
    >>>             },
    >>>             'horizontal_lanes': 1,
    >>>             'vertical_lanes': 1,
    >>>             'speed_limit': {
    >>>                 'vertical': 35,
    >>>                 'horizontal': 35
    >>>             }
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize an n*m traffic light grid network."""

        net_params = NetParams(template='/headless/ray_results/flow/real_data/CSeditClean_1.net_TLRemoved.xml')

        # name of the network (DO NOT CHANGE)
        self.name = "BobLoblawsLawBlog"

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_routes(self, net_params):
        """See parent class."""
        routes = defaultdict(list)

        # build row routes (vehicles go from left to right and vice versa)
        for veh in self.vehicles.vehicle_routing:
            routes[str(veh['route'][0])] = veh['route']

        return routes

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):

        startpositions = [["229357869#2","0"],["36978229#3","0"]]
        startlanes = ["random","random"]


        return startpositions, startlanes

    def compute_best_lane(self, kernel_net, num_cpu):
        """ to add the best lane candidiates in vehicles (flow.core.params.VehicleParams)"""
        def _seg_compute(ind_a, ind_len, kernel_net):
            for idx in range(ind_a, ind_a + ind_len):
                routes = self.vehicles.vehicle_routing[idx]['route']
                candidate_lane = []
                candidate_lane.append([i for i in range(kernel_net.num_lanes(routes[len(routes)-1]))])
                for edge_idx in range(len(routes)-2, -1, -1):
                    can = []
                    for last_lane_id in candidate_lane[-1]:
                        res = kernel_net.prev_edge(routes[edge_idx+1], last_lane_id)
                        for edge, lane in res: 
                            if edge[0] == ":":
                                res.extend(kernel_net.prev_edge(edge, lane))
                        # print(str(res))
                        # print(routes[edge_idx])
                        for edge, lane in res:
                            if edge == routes[edge_idx]:
                                can.extend([lane])
                    can.sort()
                    candidate_lane.append(list(set(can)))
                candidate_lane.reverse()
                # print(str(routes))
                # print(str(candidate_lane))
                self.vehicles.vehicle_routing[idx]['can_lane'] = candidate_lane
            _thread.exit()

        for k in range(num_cpu):
            if k < num_cpu - 1:
                ind_len = int(len(self.vehicles.vehicle_routing) / num_cpu)
                ind_a = k * ind_len
            else:
                ind_len = len(self.vehicles.vehicle_routing) - int(len(self.vehicles.vehicle_routing) / num_cpu) * (num_cpu - 1)
                ind_a = len(self.vehicles.vehicle_routing) - ind_len
            _thread.start_new_thread(_seg_compute,
                                     (ind_a, ind_len, kernel_net))


        # single thread version
        # for idx in range(len(self.vehicles.vehicle_routing)):
        #     routes = self.vehicles.vehicle_routing[idx]['route']
        #     candidate_lane = []
        #     candidate_lane.append([i for i in range(kernel_net.num_lanes(routes[len(routes)-1]))])
        #     for edge_idx in range(len(routes)-2, -1, -1):
        #         can = []
        #         for last_lane_id in candidate_lane[-1]:
        #             res = kernel_net.prev_edge(routes[edge_idx+1], last_lane_id)
        #             for edge, lane in res: 
        #                 if edge[0] == ":":
        #                     res.extend(kernel_net.prev_edge(edge, lane))
        #             # print(str(res))
        #             # print(routes[edge_idx])
        #             for edge, lane in res:
        #                 if edge == routes[edge_idx]:
        #                     can.extend([lane])
        #         can.sort()
        #         candidate_lane.append(list(set(can)))
        #     candidate_lane.reverse()
        #     # print(str(routes))
        #     # print(str(candidate_lane))
        #     self.vehicles.vehicle_routing[idx]['can_lane'] = candidate_lane