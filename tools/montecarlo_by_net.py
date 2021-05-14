class MonteCarloTreeBuilder:
    def __init__(self, env, policy_model, hand_classifier_model):
        self.env = env
        self.policy_model = policy_model
        self.hand_classifier_model = hand_classifier_model
        # TODO
        pass

    def build_tree(self, state):
        player_hands = self.predict_hands_of_opponents(state)
        # TODO
        pass

    def get_policy(self):
        # TODO
        pass

    def predict_hands_of_opponents(self, observation):
        player_hands = []
        for player in self.env.players:
            relative_observation = self.env.get_relative_observation(player, observation)
            player_hands.append(self.hand_classifier_model.forward(relative_observation))
        return player_hands
