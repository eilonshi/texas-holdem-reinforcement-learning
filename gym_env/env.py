import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete

from consts import ILLEGAL_MOVE_REWARD, DEFAULT_STACK, DEFAULT_SMALL_BLIND, DEFAULT_BIG_BLIND, \
    NUM_RUNS_MONTE_CARLO_STRENGTH_SIMULATION, CARDS_IN_FLOP, CARDS_IN_TURN, CARDS_IN_RIVER
from gym_env.enums import Action, Stage
from gym_env.stateData import CommunityData, PlayerData, StageData
from tools.hand_evaluator import get_winner
from gym_env.rendering import PygletWindow, WHITE, RED, GREEN, BLUE
from agents.player import Player, PlayerCycle
from gym_env.deck import Deck, Hand

log = logging.getLogger(__name__)

winner_in_episodes = []


# TODO: split this class
class HoldemTable(Env, Hand):
    """Poker game environment"""

    def __init__(self, initial_stacks: int = DEFAULT_STACK, small_blind: int = DEFAULT_SMALL_BLIND,
                 big_blind: int = DEFAULT_BIG_BLIND, render: bool = False, funds_plot: bool = True,
                 max_raising_rounds: int = 2, use_cpp_montecarlo: bool = False):
        """
        The table needs to be initialized once at the beginning

        Args:
            initial_stacks (real): initial stacks per player
            small_blind (real)
            big_blind (real)
            render (bool): render table after each move in graphical format
            funds_plot (bool): show plot of funds history at end of each episode
            max_raising_rounds (int): max raises per round per player

        """
        super().__init__()

        if use_cpp_montecarlo:
            import cppimport
            calculator = cppimport.imp("tools.montecarlo_cpp.pymontecarlo")
            get_equity = calculator.montecarlo
        else:
            from tools.montecarlo_python import get_equity

        self.get_equity = get_equity
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.num_of_players = 0
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.render_switch = render
        self.players: List[Player] = []
        self.dealer_pos = None
        self.player_status = {}  # one hot encoded
        self.current_player = None
        self.player_cycle = None  # cycle iterator
        self.stage = None
        self.viewer = None
        self.player_max_win = None  # used for side pots
        self.min_call = None
        self.community_data = None
        self.player_data = None
        self.stage_data = None
        self.deck = None
        self.action = None
        self.winner_ix = None
        self.initial_stacks = initial_stacks
        self.acting_agent = None
        self.funds_plot = funds_plot
        self.max_round_raising = max_raising_rounds

        # pots
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = {}  # individual player pots

        self.observation = None
        self.reward = None
        self.info = {}
        self.done = False
        self.funds_history = None
        self.current_state = None
        self.legal_moves = None
        self.illegal_move_reward = ILLEGAL_MOVE_REWARD
        self.action_space = Discrete(len(Action) - 2)
        self.first_action_for_hand = None

    def reset(self):
        """Reset after game over."""
        self.observation = None
        self.reward = None
        self.info = {}
        self.done = False
        self.funds_history = pd.DataFrame()
        self.first_action_for_hand = [True] * self.num_of_players

        for player in self.players:
            player.stack = self.initial_stacks

        self.dealer_pos = 0
        self.player_cycle = PlayerCycle(self.players, dealer_idx=-1, max_steps_after_raiser=self.num_of_players - 1,
                                        max_steps_after_big_blind=self.num_of_players)
        self._start_new_hand()

        return self.current_state

    def step(self, action: Action):
        """
        Next player makes a move and a new environment is observed.

        Args:
            action: Used for testing only. Needs to be of Action type

        """
        # loop over step function, calling the agent's action method until either the env is done
        self.reward = 0
        self.acting_agent = self.player_cycle.idx

        while action not in self.legal_moves:
            action = self._illegal_move(action)

        self._process_decision(action)
        self._next_player()

        if self.stage in [Stage.END_HIDDEN, Stage.SHOWDOWN]:
            self._end_hand()
            self._start_new_hand()

        self._get_environment()

        if self.first_action_for_hand[self.acting_agent] or self.done:
            self.first_action_for_hand[self.acting_agent] = False
            self._calculate_reward()

        return self.current_state, self.reward, self.done, self.info

    def game_loop(self):
        while not self.done:
            self._get_environment()
            action = self.current_player.act(state=self.observation, legal_actions=self.legal_moves, info=self.info)
            self.step(action)

            if self.current_player.is_trainable:
                self.current_player.log_state_for_training(self.current_state, self.reward, self.done, self.info)

    def add_player(self, player: Player):
        """Add a player to the table. Has to happen at the very beginning"""
        player.seat = self.num_of_players  # assign next seat number to player
        player.stack = self.initial_stacks
        self.num_of_players += 1
        self.players.append(player)
        self.player_status[player] = True
        self.player_pots[player] = 0

    def render(self, mode='human'):
        """Render the current state"""
        screen_width = 600
        screen_height = 400
        table_radius = 200
        face_radius = 10

        if self.viewer is None:
            self.viewer = PygletWindow(screen_width + 50, screen_height + 50)
        self.viewer.reset()
        self.viewer.circle(screen_width / 2, screen_height / 2, table_radius, color=BLUE,
                           thickness=0)

        for i in range(self.num_of_players):
            degrees = i * (360 / self.num_of_players)
            radian = (degrees * (np.pi / 180))
            x = (face_radius + table_radius) * np.cos(radian) + screen_width / 2
            y = (face_radius + table_radius) * np.sin(radian) + screen_height / 2
            if self.player_cycle.alive[i]:
                color = GREEN
            else:
                color = RED
            self.viewer.circle(x, y, face_radius, color=color, thickness=2)

            try:
                if i == self.current_player.seat:
                    self.viewer.rectangle(x - 60, y, 150, -50, (255, 0, 0, 10))
            except AttributeError:
                pass
            self.viewer.text(f"{self.players[i].name}", x - 60, y - 15,
                             font_size=10,
                             color=WHITE)
            self.viewer.text(f"Player {self.players[i].seat}: {self.players[i].cards}", x - 60, y,
                             font_size=10,
                             color=WHITE)
            equity_alive = int(round(float(self.players[i].equity_alive) * self.initial_stacks))

            self.viewer.text(f"${self.players[i].stack} (EQ: {equity_alive}%)", x - 60, y + 15, font_size=10,
                             color=WHITE)
            try:
                self.viewer.text(self.players[i].last_action_in_stage, x - 60, y + 30, font_size=10, color=WHITE)
            except IndexError:
                pass
            x_inner = (-face_radius + table_radius - 60) * np.cos(radian) + screen_width / 2
            y_inner = (-face_radius + table_radius - 60) * np.sin(radian) + screen_height / 2
            self.viewer.text(f"${self.player_pots[self.players[i]]}", x_inner, y_inner, font_size=10, color=WHITE)
            self.viewer.text(f"{self.cards}", screen_width / 2 - 40, screen_height / 2, font_size=10,
                             color=WHITE)
            self.viewer.text(f"${self.community_pot}", screen_width / 2 - 15, screen_height / 2 + 30, font_size=10,
                             color=WHITE)
            self.viewer.text(f"${self.current_round_pot}", screen_width / 2 - 15, screen_height / 2 + 50,
                             font_size=10,
                             color=WHITE)

            x_button = (-face_radius + table_radius - 20) * np.cos(radian) + screen_width / 2
            y_button = (-face_radius + table_radius - 20) * np.sin(radian) + screen_height / 2
            try:
                if i == self.player_cycle.dealer_idx:
                    self.viewer.circle(x_button, y_button, 5, color=BLUE, thickness=2)
            except AttributeError:
                pass

        self.viewer.update()

    def _illegal_move(self, action):
        log.warning(f"{action} is an Illegal move, try again. Currently allowed: {self.legal_moves}")
        self.reward = self.illegal_move_reward

        if self.current_player.is_trainable:
            self.current_player.log_state_for_training(self.current_state, self.reward, self.done, self.info)

        action = self.current_player.act(state=self.observation, legal_actions=self.legal_moves, info=self.info)

        return action

    def _agent_is_autoplay(self, idx=None):
        if not idx:
            return hasattr(self.current_player.agent_obj, 'autoplay')
        return hasattr(self.players[idx].agent_obj, 'autoplay')

    def _get_environment(self):
        """Observe the environment"""
        if not self.done:
            self._get_legal_moves()

        if self.current_player is None:  # game over
            self.current_player = self.players[self.winner_ix]

        self.observation = None
        self.reward = 0
        self.info = {}

        self.community_data = CommunityData(self.num_of_players)
        self.community_data.community_pot = self.community_pot / self.big_blind
        self.community_data.current_round_pot = self.current_round_pot / self.big_blind
        self.community_data.stage[self.stage.value] = 1
        self.community_data.legal_moves = [action in self.legal_moves for action in Action]
        # TODO: add active players to community data

        self.player_data = PlayerData(self.num_of_players)
        self.player_data.stack = [player.stack / self.big_blind for player in self.players]
        self.player_data.current_player_position[self.current_player.seat] = 1
        self.player_data.equity_to_river_alive = self.get_equity(set(self.current_player.cards_str_list()),
                                                                 set(self.cards_str_list()),
                                                                 sum(self.player_cycle.alive),
                                                                 NUM_RUNS_MONTE_CARLO_STRENGTH_SIMULATION)

        # update state
        player_cards = self.current_player.get_cards_encodings()
        community_cards = self.get_cards_encodings()
        player_stack = self.current_player.stack
        stage = self.stage.value
        stage_data = self.stage_data
        player_data = self.player_data
        community_data = self.community_data

        # TODO: check if the state is invariant to player position
        self.current_state = np.concatenate([player_cards, community_cards]).flatten()

        self.observation = self.current_state
        self._get_legal_moves()

        self.info.update({'player_data': self.player_data.__dict__,
                          'community_data': self.community_data.__dict__,
                          'stage_data': [stage.__dict__ for stage in self.stage_data],
                          'legal_moves': self.legal_moves})

        self.observation_space = self.current_state.shape

        if self.render_switch:
            self.render()

    def _calculate_reward(self):
        """Preliminary implementation of reward function."""
        stack_with_pot = self.current_player.stack + self.player_pots[self.current_player]
        stack_diff_normalized = (stack_with_pot - self.current_player.last_stack) / self.big_blind
        self.reward = stack_diff_normalized
        self.current_player.last_stack = stack_with_pot
        log.info(f"Reward for {self.current_player.name} is {self.reward}")

    def _process_decision(self, action):  # pylint: disable=too-many-statements
        """Process the decisions that have been made by an agent."""
        if action not in [Action.SMALL_BLIND, Action.BIG_BLIND]:
            assert action in set(self.legal_moves), "Illegal decision"

        if action == Action.FOLD:
            self.player_cycle.deactivate_current()
            self.player_cycle.mark_folder()

        else:

            if action == Action.CALL:
                contribution = min(self.min_call - self.player_pots[self.current_player],
                                   self.current_player.stack)
                self.callers.append(self.current_player.seat)
                self.last_caller = self.current_player.seat

            # verify the player has enough in his stack
            elif action == Action.CHECK:
                contribution = 0
                self.player_cycle.mark_checker()

            elif action == Action.RAISE_3BB:
                contribution = 3 * self.big_blind - self.player_pots[self.current_player]
                self.raisers.append(self.current_player.seat)

            elif action == Action.RAISE_HALF_POT:
                contribution = (self.community_pot + self.current_round_pot) / 2
                self.raisers.append(self.current_player.seat)

            elif action == Action.RAISE_POT:
                contribution = (self.community_pot + self.current_round_pot)
                self.raisers.append(self.current_player.seat)

            elif action == Action.RAISE_2POT:
                contribution = (self.community_pot + self.current_round_pot) * 2
                self.raisers.append(self.current_player.seat)

            elif action == Action.ALL_IN:
                contribution = self.current_player.stack
                self.raisers.append(self.current_player.seat)

            elif action == Action.SMALL_BLIND:
                contribution = np.minimum(self.small_blind, self.current_player.stack)

            elif action == Action.BIG_BLIND:
                contribution = np.minimum(self.big_blind, self.current_player.stack)
                self.player_cycle.mark_bb()
            else:
                raise RuntimeError("Illegal action.")

            if contribution > self.min_call:
                self.player_cycle.mark_raiser()
                self.min_call = contribution

            self.current_player.stack -= contribution
            self.player_pots[self.current_player] += contribution
            self.current_round_pot += contribution
            self.last_player_pot = self.player_pots[self.current_player]

            if self.current_player.stack == 0 and contribution > 0:
                self.player_cycle.mark_out_of_cash_but_contributed()

            self.current_player.actions.append(action)
            self.current_player.last_action_in_stage = action.name
            self.current_player.temp_stack.append(self.current_player.stack)

            self.player_max_win[self.current_player.seat] += contribution  # side pot

            pos = self.player_cycle.idx
            self.stage_data[self.stage.value].add_action_data(pos, action, bet=contribution / self.big_blind)
            self.stage_data[self.stage.value].players_stack[pos] = self.current_player.stack / self.big_blind
            self.stage_data[self.stage.value].contribution[pos] += contribution / self.big_blind
            self.stage_data[self.stage.value].community_pot_at_action[pos] = self.community_pot / self.big_blind

        self.player_cycle.update_alive()

        log.info(
            f"Seat {self.current_player.seat} ({self.current_player.name}): {action} - "
            f"Remaining stack: {self.current_player.stack}, "
            f"Round pot: {self.current_round_pot}, Community pot: {self.community_pot}, "
            f"player pot: {self.player_pots[self.current_player]}")

    def _start_new_hand(self):
        """Deal new cards to players and reset table states."""
        self._save_funds_history()

        if self._check_game_over():
            return

        log.info("")
        log.info("++++++++++++++++++")
        log.info("Starting new hand.")
        log.info("++++++++++++++++++")
        self.clear_cards()
        self.deck = Deck()
        self.stage = Stage.PREFLOP

        self.stage_data = [StageData(num_players=self.num_of_players)]

        # pots
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = {player: 0 for player in self.players}
        self.player_max_win = [0] * self.num_of_players
        self.last_player_pot = 0
        self.played_in_round = 0
        self.first_action_for_hand = [True] * self.num_of_players

        for player in self.players:
            player.clear_cards()

        self._next_dealer()

        self._distribute_cards()
        self._initiate_round()

    def _save_funds_history(self):
        """Keep track of player funds history"""
        funds_dict = {i: player.stack for i, player in enumerate(self.players)}
        self.funds_history = pd.concat([self.funds_history, pd.DataFrame(funds_dict, index=[0])])

    def _check_game_over(self):
        """Check if only one player has money left"""
        player_alive = []
        self.player_cycle.new_hand_reset()

        for idx, player in enumerate(self.players):
            if player.stack > 0:
                player_alive.append(True)
            else:
                self.player_status[player] = False
                self.player_cycle.deactivate_player(idx)

        remaining_players = sum(player_alive)
        if remaining_players < 2:
            self._game_over()
            return True
        return False

    def _game_over(self):
        """End of an episode."""
        log.info("Game over.")
        self.done = True
        player_names = [f"{i} - {player.name}" for i, player in enumerate(self.players)]
        self.funds_history.columns = player_names
        if self.funds_plot:
            self.funds_history.reset_index(drop=True).plot()
        log.info(self.funds_history)
        plt.show()

        winner_in_episodes.append(self.winner_ix)
        league_table = pd.Series(winner_in_episodes).value_counts()
        best_player = league_table.index[0]
        log.info(league_table)
        log.info(f"Best Player: {best_player}")

    def _initiate_round(self):
        """A new round (flop, turn, river) is initiated"""
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.min_call = 0
        for player in self.players:
            player.last_action_in_stage = ''
        self.player_cycle.new_round_reset()

        if self.stage == Stage.PREFLOP:
            log.info("")
            log.info("===Round: Stage: PREFLOP")
            # max steps total will be adjusted again at bb
            self.player_cycle.max_steps_total = self.num_of_players * self.max_round_raising + 2

            self._next_player()
            self._process_decision(Action.SMALL_BLIND)
            self._next_player()
            self._process_decision(Action.BIG_BLIND)
            self._next_player()

        elif self.stage in [Stage.FLOP, Stage.TURN, Stage.RIVER]:
            self.player_cycle.max_steps_total = self.num_of_players * self.max_round_raising

            self._next_player()

        elif self.stage == Stage.SHOWDOWN:
            log.info("Showdown")

        else:
            raise RuntimeError()

    def _end_round(self):
        """End of preflop, flop, turn or river"""
        self._close_round()
        if self.stage == Stage.PREFLOP:
            self.stage = Stage.FLOP
            self._distribute_cards_to_table(amount_of_cards=CARDS_IN_FLOP)

        elif self.stage == Stage.FLOP:
            self.stage = Stage.TURN
            self._distribute_cards_to_table(amount_of_cards=CARDS_IN_TURN)

        elif self.stage == Stage.TURN:
            self.stage = Stage.RIVER
            self._distribute_cards_to_table(amount_of_cards=CARDS_IN_RIVER)

        elif self.stage == Stage.RIVER:
            self.stage = Stage.SHOWDOWN

        self.stage_data.append(StageData(num_players=self.num_of_players))

        log.info("--------------------------------")
        log.info(f"===ROUND: {self.stage} ===")
        self._clean_up_pots()

    def _clean_up_pots(self):
        self.community_pot += self.current_round_pot
        self.current_round_pot = 0
        self.player_pots = {player: 0 for player in self.players}

    def _end_hand(self):
        self._clean_up_pots()
        self.winner_ix = self._get_winner()
        self._award_winner(self.winner_ix)

    def _get_winner(self):
        """Determine which player has won the hand"""
        potential_winners = self.player_cycle.get_potential_winners()

        potential_winner_idx = [i for i, potential_winner in enumerate(potential_winners) if potential_winner]
        if sum(potential_winners) == 1:
            winner_ix = [i for i, active in enumerate(potential_winners) if active][0]
            winning_card_type = 'Only remaining player in round'

        else:
            assert self.stage == Stage.SHOWDOWN
            remaining_player_winner_ix, winning_card_type = get_winner([player.cards_str_list()
                                                                        for ix, player in enumerate(self.players) if
                                                                        potential_winners[ix]],
                                                                       self.cards_str_list())
            winner_ix = potential_winner_idx[remaining_player_winner_ix]
        log.info(f"Player {winner_ix} won: {winning_card_type}")
        return winner_ix

    def _award_winner(self, winner_ix):
        """Hand the pot to the winner and handle side pots"""
        max_win_per_player_for_winner = self.player_max_win[winner_ix]
        total_winnings = sum(np.minimum(max_win_per_player_for_winner, self.player_max_win))
        remains = np.maximum(0, np.array(self.player_max_win) - max_win_per_player_for_winner)  # to be returned

        self.players[winner_ix].stack += total_winnings
        self.winner_ix = winner_ix
        if total_winnings < sum(self.player_max_win):
            log.info("Returning side pots")
            for i, player in enumerate(self.players):
                player.stack += remains[i]

    def _next_dealer(self):
        self.dealer_pos = self.player_cycle.next_dealer().seat

    def _next_player(self):
        """Move to the next player"""
        self.current_player = self.player_cycle.next_player()
        if not self.current_player:
            if sum(self.player_cycle.alive) < 2:
                log.info("Only one player remaining in round")
                self.stage = Stage.END_HIDDEN
            else:
                log.info("End round - no current player returned")
                self._end_round()
                # todo: in some cases no new round should be initialized bc only one player is playing only it seems
                self._initiate_round()

        elif self.current_player == 'max_steps_total' or self.current_player == 'max_steps_after_raiser':
            log.debug(self.current_player)
            log.info("End of round ")
            self._end_round()
            return

    def _get_legal_moves(self):
        """Determine what moves are allowed in the current state"""
        self.legal_moves = []
        if self.player_pots[self.current_player] == max(self.player_pots.values()):
            self.legal_moves.append(Action.CHECK)
        else:
            self.legal_moves.append(Action.CALL)
            self.legal_moves.append(Action.FOLD)

        if self.current_player.stack >= 3 * self.big_blind - self.player_pots[self.current_player]:
            self.legal_moves.append(Action.RAISE_3BB)

            if self.current_player.stack >= ((self.community_pot + self.current_round_pot) / 2) >= self.min_call:
                self.legal_moves.append(Action.RAISE_HALF_POT)

            if self.current_player.stack >= (self.community_pot + self.current_round_pot) >= self.min_call:
                self.legal_moves.append(Action.RAISE_POT)

            if self.current_player.stack >= ((self.community_pot + self.current_round_pot) * 2) >= self.min_call:
                self.legal_moves.append(Action.RAISE_2POT)

            if self.current_player.stack > 0:
                self.legal_moves.append(Action.ALL_IN)

        log.debug(f"Community+current round pot pot: {self.community_pot + self.current_round_pot}")

    def _close_round(self):
        """put player_pots into community pots"""
        self.community_pot += sum(self.player_pots.values())
        self.player_pots = {player: 0 for player in self.players}
        self.played_in_round = 0

    def _distribute_cards(self):
        log.info(f"Dealer is at position {self.dealer_pos}")
        self.deck.deal_2_cards_to_all_players(self.players)

    def _distribute_cards_to_table(self, amount_of_cards):
        for _ in range(amount_of_cards):
            self.deck.deal_to(self, num_cards=1)
        log.info(f"Cards on table: {self.cards_str_list()}")

    @staticmethod
    def evaluate_2_players(player1: Player, player2: Player) -> Player:
        if player1.stack > player2.stack:
            return player1
        return player2
