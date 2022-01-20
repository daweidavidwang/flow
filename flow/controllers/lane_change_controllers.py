"""Contains a list of custom lane change controllers."""

from random import betavariate
from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController


class SimLaneChangeController(BaseLaneChangeController):
    """A controller used to enforce sumo lane-change dynamics on a vehicle.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return None


class StaticLaneChanger(BaseLaneChangeController):
    """A lane-changing model used to keep a vehicle in the same lane.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return 0


class RoutingLaneChanger(BaseLaneChangeController):
    """A lane-changing model used to keep a vehicle in the same lane.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        def compute_can_list(env):
            route_contr = env.k.vehicle.get_routing_controller(
                    self.veh_id)
            routing_result = route_contr.choose_route(env)
            current_lane = env.k.vehicle.get_lane(self.veh_id)
            current_edge = env.k.vehicle.get_edge(self.veh_id)

            if routing_result is None or current_edge != routing_result[0]:
                return []
            if len(routing_result)>2:
                ll2 = [i for i in range(env.k.network.num_lanes(routing_result[2]))]
                ll1 = []
                for last_lane_id in ll2:
                    res = env.k.network.prev_edge(routing_result[2], last_lane_id)
                    for edge, lane in res: 
                        if edge[0] == ":":
                            res.extend(env.k.network.prev_edge(edge, lane))
                    # print(str(res))
                    # print(routes[edge_idx])
                    for edge, lane in res:
                        if edge == routing_result[1]:
                            ll1.extend([lane])
                ll1.sort()
                ll1 = list(set(ll1))
                ll0 = []
                for last_lane_id in ll1:
                    res = env.k.network.prev_edge(routing_result[1], last_lane_id)
                    for edge, lane in res: 
                        if edge[0] == ":":
                            res.extend(env.k.network.prev_edge(edge, lane))
                    # print(str(res))
                    # print(routes[edge_idx])
                    for edge, lane in res:
                        if edge == routing_result[0]:
                            ll0.extend([lane])
                ll0.sort()
                ll0 = list(set(ll0))
                return ll0
            else:
                return []

        current_lane = env.k.vehicle.get_lane(self.veh_id)
        current_edge = env.k.vehicle.get_edge(self.veh_id)
        route_contr = env.k.vehicle.get_routing_controller(
                self.veh_id)
        routing_result = route_contr.choose_route(env)

        # two step look can list.
        # can_lane_list = compute_can_list(env)

        # for idx in range(len(env.network.vehicles.vehicle_routing)):
        #     if env.network.vehicles.vehicle_routing[idx]['id'] == self.veh_id:
        #         for ridx in range(len(env.network.vehicles.vehicle_routing[idx]['route'])):
        #             if env.network.vehicles.vehicle_routing[idx]['route'][ridx] == current_edge:
        #                 can_lane_list =  env.network.vehicles.vehicle_routing[idx]['can_lane'][ridx]
        #                 break                        
        #         break
        ## if can lane list exist, we first use it to make decision
        # if len(can_lane_list)>0:
        #     if current_lane in can_lane_list:
        #         return 0
        #     else:
        #         closest = 1000
        #         for lid in can_lane_list:
        #             if abs(closest)> abs(lid-current_lane):
        #                 closest = lid-current_lane
        #         if closest>0:
        #             return 1
        #         else:
        #             return -1
                
        # one step look version

        #resolve needed lane idx
        if routing_result is None or current_edge != routing_result[0]:
            return 0
        else:
            current_conn = env.k.network.next_edge(current_edge, current_lane)
            for edge, lane in current_conn:
                if edge == routing_result[1]:
                    # no lane changing needed
                    return 0
                    # one more look!
                if edge[0] == ":":
                    #if the connection connect to a "via", we further query it
                    via_conn = env.k.network.next_edge(edge, lane)
                    if len(via_conn)>0:
                        for vedge, vlane in via_conn:
                            if vedge == routing_result[1]:
                                return 0

            l_idx_shifting = 1
            while len(env.k.network.next_edge(current_edge, current_lane-l_idx_shifting))>0 or \
                len(env.k.network.next_edge(current_edge, current_lane+l_idx_shifting))>0:

                ## right look first
                conn = env.k.network.next_edge(current_edge, current_lane+l_idx_shifting)
                for edge, lane in conn:
                    if edge == routing_result[1]:
                    ## lane change 1 needed 
                        return 1
                    if edge[0] == ":":
                        #if the connection connect to a "via", we further query it
                        via_conn = env.k.network.next_edge(edge, lane)
                        if len(via_conn)>0:
                            for vedge, vlane in via_conn:
                                if vedge == routing_result[1]:
                                ## lane change 1 needed 
                                    return 1
                ## left look
                conn = env.k.network.next_edge(current_edge, current_lane-l_idx_shifting)
                for edge, lane in conn:
                    if edge == routing_result[1]:
                    ## lane change -1 needed 
                        return -1
                    if edge[0] == ":":
                        #if the connection connect to a "via", we further query it
                        via_conn = env.k.network.next_edge(edge, lane)
                        if len(via_conn)>0:
                            for vedge, vlane in via_conn:
                                if vedge == routing_result[1]:
                                ## lane change -1 needed 
                                    return -1
                l_idx_shifting += 1
        
        return 0