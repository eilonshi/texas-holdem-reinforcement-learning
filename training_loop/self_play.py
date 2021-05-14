import logging
import gym
import numpy as np
import pandas as pd

from agents.player import Player
from agents.agent_torch_rl import DQNPolicyModel
from agents.agent_torch_rl import DQNHandClassifierModel
from agents.agent_torch_rl import DQNTorchPlayer
from gym_env.env import HoldemTable
from tools.montecarlo_by_net import MonteCarloTreeBuilder


class SelfPlay:
    """Orchestration of playing against itself"""

    def __init__(self, render: bool, num_epochs: int, num_episodes: int, use_cpp_montecarlo: bool, funds_plot: bool,
                 num_players: int, stack: int):
        """Initialize"""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.num_players = num_players
        self.num_epochs = num_epochs
        self.num_episodes = num_episodes
        self.stack = stack
        self.log = logging.getLogger(__name__)

        self.env = None
        self.torch_dqn_policy_model = None
        self.torch_hand_classifier_model = None
        self.mc_builder = MonteCarloTreeBuilder(env=self, policy_model=self.torch_dqn_policy_model,
                                                hand_classifier_model=self.torch_hand_classifier_model)
        self.cur_agent = None
        self.last_agent = None

    def create_env(self):
        self.env = HoldemTable(initial_stacks=self.stack, funds_plot=self.funds_plot, render=self.render,
                               use_cpp_montecarlo=self.use_cpp_montecarlo)

        np.random.seed(123)
        self.env.seed(123)

        # add dqn players as the number of players that was chosen
        number_of_players = self.num_players
        if number_of_players < 2:
            raise ValueError('Number of players is smaller than 2')

        self.torch_dqn_policy_model = DQNPolicyModel()
        self.torch_hand_classifier_model = DQNHandClassifierModel()

        for _ in range(number_of_players):
            dqn_torch_player = DQNTorchPlayer(env=self.env,
                                              stack_size=self.stack,
                                              policy_model=self.torch_dqn_policy_model,
                                              hand_classifier_model=self.torch_hand_classifier_model)
            self.env.add_player(dqn_torch_player)

        self.env.reset()

    def dqn_train_torch_rl(self):
        """Implementation of torch rl deep q learning train loop."""

        # TODO: save the architectures

        self.dqn_train_game_loop(epochs=self.num_epochs, episodes=self.num_episodes)

    def dqn_play_torch_rl(self):
        pass

    def dqn_train_keras_rl(self, model_name):
        """Implementation of keras-rl deep q learning."""
        from agents.agent_consider_equity import EquityPlayer as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import RandomPlayer as RandomPlayer
        env_name = 'neuron_poker-v0'
        env = gym.make(env_name, initial_stacks=self.stack, funds_plot=self.funds_plot, render=self.render,
                       use_cpp_montecarlo=self.use_cpp_montecarlo)

        np.random.seed(123)
        env.seed(123)
        env.add_player(EquityPlayer(name='equity/50/70', min_call_equity=.5, min_bet_equity=.7))
        env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        env.add_player(RandomPlayer())
        env.add_player(RandomPlayer())
        env.add_player(RandomPlayer())
        env.add_player(Player(name='keras-rl', stack_size=self.stack))  # shell is used for callback to keras rl

        env.reset()

        dqn = DQNPlayer()
        dqn.initiate_agent(env)
        dqn.train(env_name=model_name)

    def dqn_play_keras_rl(self, model_name):
        """Create 6 players, one of them a trained DQN"""
        from agents.agent_consider_equity import EquityPlayer as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import RandomPlayer as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(Player(name='keras-rl', stack_size=self.stack))

        self.env.reset()

        dqn = DQNPlayer(load_model=model_name, env=self.env)
        dqn.play(nb_episodes=self.num_episodes, render=self.render)

    def dqn_train_custom_q1(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import EquityPlayer as EquityPlayer
        from agents.agent_custom_q1 import Player as Custom_Q1
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        # self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        # self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        # self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        # self.env.add_player(RandomPlayer())
        # self.env.add_player(RandomPlayer())
        # self.env.add_player(RandomPlayer())
        self.env.add_player(Custom_Q1(name='Deep_Q1'))

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def dqn_train_game_loop(self, epochs, episodes):

        # TODO: use tensorboard
        # TODO: look at keras-rl.core to see how they implemented the training loop

        for epoch in range(epochs):
            self.create_env()

            for episode in range(episodes):
                self.env.game_loop()

                # fit the models to the collected data
                # TODO: uncomment
                # self.torch_dqn_policy_model.fit()
                # self.torch_hand_classifier_model.fit()

                # TODO: if self.env.done: break

            # evaluate the latest agent in a game vs the prev agent and keep the net that won in the evaluation
            if self.last_agent is not None:
                self.keep_best_agent(last_agent=self.last_agent, cur_agent=self.cur_agent)

        # TODO: save the weights
        # After training is done, we save the final weights.

    def keep_best_agent(self, last_agent, cur_agent):
        best_agent = self.env.evaluate_2_players(last_agent, cur_agent)
        self.cur_agent = best_agent
