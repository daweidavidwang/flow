import time
from collections import OrderedDict
import pickle
import numpy as np
import gym
import os
from flow.utils.registry import make_create_env
from multiagent_ring_env import flow_params
from imitating_controller import ImitatingController
from imitating_network import ImitatingNetwork
from flow.controllers.car_following_models import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import SumoCarFollowingParams
import tensorflow as tf
from utils import *

class Trainer(object):
    """
    Class to initialize and run training for imitation learning (with DAgger)
    """

    def __init__(self, params):

        # param setup
        self.params = params
        self.sess = create_tf_session()

        # environment setup
        create_env, _ = make_create_env(flow_params)
        self.env = create_env()
        init_state = self.env.reset()

        # vehicle setup
        self.multiagent = params['multiagent']

        # TODO: remove print
        print("MULTI: ", self.multiagent)

        if self.multiagent:
            self.vehicle_ids = list(init_state.keys())
        else:
            print("IDS: ", self.env.k.vehicle.get_ids())
            assert self.params['vehicle_id'] in self.env.k.vehicle.get_ids()
            self.vehicle_ids = [self.params['vehicle_id']]

        # neural net setup
        obs_dim = self.env.observation_space.shape[0]
        action_dim = (1,)[0]
        self.params['action_dim'] = action_dim
        self.params['obs_dim'] = obs_dim

        self.action_network = ImitatingNetwork(self.sess, self.params['action_dim'], self.params['obs_dim'], self.params['num_layers'], self.params['size'], self.params['learning_rate'], self.params['replay_buffer_size'], inject_noise=self.params['inject_noise'], noise_variance=self.params['noise_variance'])

        tf.global_variables_initializer().run(session=self.sess)

        # controllers setup
        car_following_params = SumoCarFollowingParams()
        self.expert_controllers = []
        self.controllers = []
        for vehicle_id in self.vehicle_ids:
            self.expert_controllers.append(FollowerStopper(vehicle_id, car_following_params=car_following_params))
            self.controllers.append(ImitatingController(vehicle_id, self.action_network, self.multiagent, car_following_params=car_following_params))


    def run_training_loop(self, n_iter):
        """
        Trains imitator for n_iter iterations

        Args:
            param n_iter:  number of iterations to execute training
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # collect trajectories, to be used for training
            if itr == 0:
                # first iteration is standard behavioral cloning
                training_returns = self.collect_training_trajectories(itr, self.params['init_batch_size'])
            else:
                training_returns = self.collect_training_trajectories(itr, self.params['batch_size'])

            paths, envsteps_this_batch = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.action_network.add_to_replay_buffer(paths)

            # train controller (using sampled data from replay buffer)
            loss = self.train_controller()

    def collect_training_trajectories(self, itr, batch_size):
        """
        Collect (state, action, reward, next_state, terminal) tuples for training

        Args:
            itr: iteration of training during which functino is called
            batch_size: number of tuples to collect
        Returns:
            paths: list of trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
        """

        if itr == 0:
            collect_controllers = self.expert_controllers
        else:
            collect_controllers = self.controllers

        print("\nCollecting data to be used for training...")
        trajectories, envsteps_this_batch = sample_trajectories(self.env, self.vehicle_ids, collect_controllers, self.expert_controllers, batch_size, self.params['ep_len'], self.multiagent)

        return trajectories, envsteps_this_batch

    def train_controller(self):
        """
            Trains controller using data sampled from replay buffer
        """

        print('Training controller using sampled data from replay buffer')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, expert_ac_batch, re_batch, next_ob_batch, terminal_batch = self.action_network.sample_data(self.params['train_batch_size'])
            self.action_network.train(ob_batch, expert_ac_batch)

    def evaluate_controller(self, num_trajs = 10):
        """
        Evaluates a trained controller on similarity with expert with respect to action taken and total reward per rollout

        Args:
            num_trajs: number of trajectories to evaluate performance on
        """

        print("\n\n********** Evaluation ************ \n")

        trajectories = sample_n_trajectories(self.env, self.vehicle_ids, self.controllers, self.expert_controllers, num_trajs, self.params['ep_len'], self.multiagent)

        average_imitator_reward = 0
        total_imitator_steps = 0
        average_imitator_reward_per_rollout = 0

        action_errors = np.array([])
        average_action_expert = 0
        average_action_imitator = 0

        # compare actions taken in each step of trajectories
        for traj in trajectories:
            imitator_actions = traj['actions']
            expert_actions = traj['expert_actions']

            average_action_expert += np.sum(expert_actions)
            average_action_imitator += np.sum(imitator_actions)

            action_error = np.linalg.norm(imitator_actions - expert_actions) / len(imitator_actions)
            action_errors = np.append(action_errors, action_error)

            average_imitator_reward += np.sum(traj['rewards'])
            total_imitator_steps += len(traj['rewards'])
            average_imitator_reward_per_rollout += np.sum(traj['rewards'])

        average_imitator_reward = average_imitator_reward / total_imitator_steps
        average_imitator_reward_per_rollout = average_imitator_reward_per_rollout / len(trajectories)

        average_action_expert = average_action_expert / total_imitator_steps
        average_action_imitator = average_action_imitator / total_imitator_steps


        expert_trajectories = sample_n_trajectories(self.env, self.vehicle_ids, self.expert_controllers, self.expert_controllers, num_trajs, self.params['ep_len'], self.multiagent)

        average_expert_reward = 0
        total_expert_steps = 0
        average_expert_reward_per_rollout = 0

        # compare reward accumulated in trajectories collected via expert vs. via imitator
        for traj in expert_trajectories:
            average_expert_reward += np.sum(traj['rewards'])
            total_expert_steps += len(traj['rewards'])
            average_expert_reward_per_rollout += np.sum(traj['rewards'])

        average_expert_reward_per_rollout = average_expert_reward_per_rollout / len(expert_trajectories)
        average_expert_reward = average_expert_reward / total_expert_steps

        print("\nAVERAGE REWARD PER STEP EXPERT: ", average_expert_reward)
        print("AVERAGE REWARD PER STEP IMITATOR: ", average_imitator_reward)
        print("AVERAGE REWARD PER STEP DIFFERENCE: ", np.abs(average_expert_reward - average_imitator_reward), "\n")

        print("AVERAGE REWARD PER ROLLOUT EXPERT: ", average_expert_reward_per_rollout)
        print("AVERAGE REWARD PER ROLLOUT IMITATOR: ", average_imitator_reward_per_rollout)
        print("AVERAGE REWARD PER ROLLOUT DIFFERENCE: ", np.abs(average_expert_reward_per_rollout - average_imitator_reward_per_rollout), "\n")

        print("MEAN ACTION ERROR: ", np.mean(action_errors), "\n")

    def save_controller_network(self):
        print("Saving tensorflow model to: ", self.params['save_path'])
        self.action_network.save_network(self.params['save_path'])