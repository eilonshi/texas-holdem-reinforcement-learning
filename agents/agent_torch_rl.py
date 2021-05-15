"""Player based on a trained neural network, implemented in torch lybrary"""
import logging
import torch
from agents.player import Player
from gym_env.env import HoldemTable
from tools.montecarlo_by_net import MonteCarloTreeBuilder

log = logging.getLogger(__name__)


class DQNHandClassifierModel(torch.nn.Module):
    def __init__(self):
        # TODO
        super().__init__()

    def forward(self, x):
        # TODO
        pass


class DQNPolicyModel(torch.nn.Module):
    def __init__(self):
        # TODO
        super().__init__()

    def build_net(self):
        # TODO
        pass

    def load_net(self, env_name):
        """Load a model"""
        pass
        # # Load the architecture
        # with open('dqn_{}_json.json'.format(env_name), 'r') as architecture_json:
        #     dqn_json = json.load(architecture_json)
        #
        # self.model = model_from_json(dqn_json)
        # self.model.load_weights('dqn_{}_weights.h5'.format(env_name))

    def update_memory(self):
        # TODO
        pass

    def fit(self):
        # TODO
        pass


class DQNTorchPlayer(Player):
    """Mandatory class with the player methods"""

    def __init__(self, env: HoldemTable, stack_size: int, policy_model, hand_classifier_model, name='DQNTorch'):
        """Initialization of an agent"""
        super().__init__(stack_size, name)

        self.env = env
        self.policy_model = self.build_model(policy_model, DQNPolicyModel)
        self.hand_classifier_model = self.build_model(hand_classifier_model, DQNHandClassifierModel)

        self.montecarlo_tree_builder = MonteCarloTreeBuilder(env=self.env,
                                                             policy_model=self.policy_model,
                                                             hand_classifier_model=self.hand_classifier_model)

    def act(self, state, legal_actions, info=None):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        policy = self.montecarlo_tree_builder.get_policy()
        action = self.choose_action_by_policy(policy=policy)
        return action

    def choose_action_by_policy(self, policy):
        # TODO: choose the best action
        return self.act_randomly()

    def act_randomly(self):
        """Custom policy for random decisions for warm up."""
        log.info("Random action")
        action = self.env.action_space.sample()
        return action

    def log_state_for_training(self, state, reward, done, info):
        # log reward
        logging.log(level=logging.INFO, msg=f'Player: {self.name}\n reward: {reward}\n')

    def load_models(self, load_models):
        # TODO
        pass

    def train(self):
        # TODO
        pass

    @staticmethod
    def build_model(model, model_class):
        if model is None:
            model = model_class()
        return model
