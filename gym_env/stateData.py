from consts import NUM_ROUNDS
from gym_env.enums import Action, Stage


class PlayerData:
    """Player specific information"""

    def __init__(self, num_players):
        """data"""
        self.current_player_position = [False] * num_players  # ix[0] = dealer
        self.player_cards_encoding = None
        self.legal_moves = [0 for _ in Action]
        self.big_blinds = 0
        self.equity_to_river_alive = 0
        # TODO: update to_call and contribution
        self.to_call = 0
        self.contribution = 0


class CommunityData:
    """Data available to everybody"""

    def __init__(self, num_players):
        """data"""
        self.active_players = [False] * num_players  # one hot encoded, 0 = dealer
        self.stage = [False] * len(Stage)  # one hot: preflop, flop, turn, river
        self.community_pot = None
        self.current_round_pot = None


class ActionData:
    """The data of single action in a stage"""

    def __init__(self, player_seat, action, bet):
        self.player_seat = player_seat
        self.action = action
        self.bet = bet


class StageData:
    """Preflop, flop, turn and river"""

    def __init__(self, num_players):
        """data"""
        self.players_stack = [0] * num_players  # ix[0] = dealer
        self.contribution = [0] * num_players  # ix[0] = dealer
        self.community_pot_at_action = [0] * num_players  # ix[0] = dealer
        self.stage_actions = []

    def add_action_data(self, player_seat: int, action: Action, bet: int = 0):
        action_data = ActionData(player_seat, action, bet)
        self.stage_actions.append(action_data)
