"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import MultiTrafficGridPOEnv
from flow.networks import TrafficGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from flow.controllers import RLController, IDMController, ContinuousRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

# Experiment parameters
N_ROLLOUTS = 50  # number of rollouts per training iteration
N_CPUS = 15  # number of parallel workers

# Environment parameters
HORIZON = 2000  # time horizon of a single rollout
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
N_ROWS = 2  # number of row of bidirectional lanes
N_COLUMNS = 2  # number of columns of bidirectional lanes
NUM_AUTOMATED = 2
AUTO_PLATOON = 1
HUMAN_PLATOON = 1
DENSITY = 2
# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
num_vehicles = 3
num_human = num_vehicles - NUM_AUTOMATED


# Add one automated vehicle.
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    # car_following_params=SumoCarFollowingParams(
    #     min_gap=2.5,
    #     max_speed=V_ENTER,
    #     decel=7.5,  # avoid collisions at emergency stops
    #     speed_mode="right_of_way",
    # ),
    routing_controller=(GridRouter, {}),
    num_vehicles=NUM_AUTOMATED)

# Add a fraction of the remaining human vehicles.
vehicles_to_add = num_human
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=num_human)


# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    for _ in range(DENSITY):
        for _ in range(HUMAN_PLATOON):
            inflow.add(
                veh_type="human",
                edge=edge,
                vehs_per_hour=EDGE_INFLOW,
                departLane="free",
                departSpeed=V_ENTER)
        for _ in range(AUTO_PLATOON):
            inflow.add(
                veh_type="rl",
                edge=edge,
                vehs_per_hour=EDGE_INFLOW,
                departLane="free",
                departSpeed=V_ENTER)

flow_params = dict(
    # name of the experiment
    exp_tag="grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

    # name of the flow environment the experiment is running on
    env_name=MultiTrafficGridPOEnv,

    # name of the network class the experiment is running on
    network=TrafficGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 50,
            "switch_time": 3,
            "num_observed": 2,
            "discrete": False,
            "tl_type": "actuated",
            "num_local_edges": 4,
            "num_local_lights": 4,
            "max_accel": 1,
            "max_decel": 1,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params={
            "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization
    # or reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']
