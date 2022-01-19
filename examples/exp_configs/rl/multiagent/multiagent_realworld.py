"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import MultiRealWorldPOEnv
from flow.networks import RealWorldNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from flow.controllers import RLController, IDMController, ContinuousRouter, RealDataRouter,RoutingLaneChanger, LCIDMController
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

# Experiment parameters
N_ROLLOUTS = 50  # number of rollouts per training iteration
N_CPUS = 1  # number of parallel workers

# Environment parameters
HORIZON = 20000  # time horizon of a single rollout
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 5, 5, 2, 2

EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
NUM_AUTOMATED = 1
AUTO_PLATOON = 1
HUMAN_PLATOON = 1

vehicles = VehicleParams()
num_vehicles = 2
num_human = num_vehicles - NUM_AUTOMATED

vehicles.load_from_xml('/headless/ray_results/flow/real_data/newroute1220_start0.xml')

for veh in vehicles.vehicle_routing:
    vehicles.add(
        veh_id=veh['id'],
        acceleration_controller=(LCIDMController, {}),
        lane_change_controller=(RoutingLaneChanger, {}),
        routing_controller=(RealDataRouter, {}),
        depart=veh['depart'],
        num_vehicles=1)


flow_params = dict(
    # name of the experiment
    exp_tag="realworld_multiagent",

    # name of the flow environment the experiment is running on
    env_name=MultiRealWorldPOEnv,

    # name of the network class the experiment is running on
    network=RealWorldNetwork,

    # simulator that is used by the experiment
    simulator='traci_rd',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=True,
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
        additional_params={
            "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
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
        shuffle=False,
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
