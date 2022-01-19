"""Contains a list of custom lane change controllers."""

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
        route_contr = env.k.vehicle.get_routing_controller(
                        self.veh_id)
        current_lane = env.k.vehicle.get_lane(self.veh_id)
        current_edge = env.k.vehicle.get_edge(self.veh_id)
        routing_result = route_contr.choose_route(env)


        #resolve needed lane idx
        if routing_result is None or current_edge != routing_result[0]:
            return 0
        else:
            current_conn = env.k.network.next_edge(current_edge, current_lane)
            # print(str(routing_result))
            # print(str(current_conn))
            for edge, lane in current_conn:
                if edge == routing_result[1]:
                    # no lane changing needed
                    return 0
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
                # print(str(conn))
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