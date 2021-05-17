import numpy as np
import logging
import abc

from gym_env.enums import Action
from gym_env.deck import Hand

log = logging.getLogger(__name__)


class PlayerCycle:
    """Handle the circularity of the Table."""

    def __init__(self, lst, start_idx=0, dealer_idx=0, max_steps_total=None,
                 last_raiser_step=None, max_steps_after_raiser=None, max_steps_after_big_blind=None):
        """Cycle over a list"""
        self.lst = lst
        self.start_idx = start_idx
        self.size = len(lst)
        self.max_steps_total = max_steps_total
        self.last_raiser_step = last_raiser_step
        self.max_steps_after_raiser = max_steps_after_raiser
        self.max_steps_after_big_blind = max_steps_after_big_blind
        self.last_raiser = None
        self.step_counter = 0
        self.steps_for_blind_betting = 2
        self.second_round = False
        self.idx = 0
        self.dealer_idx = dealer_idx
        self.can_still_make_moves_in_this_hand = []  # if the player can still play in this round
        self.alive = [True] * len(self.lst)  # if the player can still play in the following rounds
        self.out_of_cash_but_contributed = [False] * len(self.lst)
        self.new_hand_reset()
        self.checkers = 0
        self.folder = None

    def new_hand_reset(self):
        """Reset state if a new hand is dealt"""
        self.idx = self.start_idx
        self.can_still_make_moves_in_this_hand = [True] * len(self.lst)
        self.out_of_cash_but_contributed = [False] * len(self.lst)
        self.folder = [False] * len(self.lst)
        self.step_counter = 0

    def new_round_reset(self):
        """Reset the state for the next stage: flop, turn or river"""
        self.step_counter = 0
        self.second_round = False
        self.idx = self.dealer_idx
        self.last_raiser_step = len(self.lst)
        self.checkers = 0

    def next_player(self, step=1):
        """Switch to the next player in the round."""
        if sum(np.array(self.can_still_make_moves_in_this_hand) + np.array(self.out_of_cash_but_contributed)) < 2:
            log.debug("Only one player remaining")
            return None  # only one player remains

        self.idx += step
        self.step_counter += step
        self.idx %= len(self.lst)
        if self.step_counter > len(self.lst):
            self.second_round = True
        if self.max_steps_total and (self.step_counter >= self.max_steps_total):
            log.debug("Max steps total has been reached")
            return None

        if self.last_raiser:
            raiser_reference = self.last_raiser
            if self.max_steps_after_raiser and (self.step_counter > self.max_steps_after_raiser + raiser_reference):
                log.debug("max steps after raiser has been reached")
                return None
        elif self.max_steps_after_raiser and \
                (self.step_counter > self.max_steps_after_big_blind + self.steps_for_blind_betting):
            log.debug("max steps after raiser has been reached")
            return None

        if self.checkers == sum(self.alive):
            log.debug("All players checked")
            return None

        while True:
            if self.can_still_make_moves_in_this_hand[self.idx]:
                break

            self.idx += 1
            self.step_counter += 1
            self.idx %= len(self.lst)
            if self.max_steps_total and self.step_counter >= self.max_steps_total:
                log.debug("Max steps total has been reached after jumping some folders")
                return None

        self.update_alive()

        return self.lst[self.idx]

    def next_dealer(self):
        """Move the dealer to the next player that's still in the round."""
        self.dealer_idx += 1
        self.dealer_idx %= len(self.lst)

        while True:
            if self.can_still_make_moves_in_this_hand[self.dealer_idx]:
                break

            self.dealer_idx += 1
            self.dealer_idx %= len(self.lst)

        return self.lst[self.dealer_idx]

    def set_idx(self, idx):
        """Set the index to a specific player"""
        self.idx = idx

    def deactivate_player(self, idx):
        """Deactivate a pleyr if he has folded or is out of cash."""
        assert self.can_still_make_moves_in_this_hand[idx], "Already deactivated"
        self.can_still_make_moves_in_this_hand[idx] = False

    def deactivate_current(self):
        """Deactivate the current player if he has folded or is out of cash."""
        assert self.can_still_make_moves_in_this_hand[self.idx], "Already deactivated"
        self.can_still_make_moves_in_this_hand[self.idx] = False

    def mark_folder(self):
        """Mark a player as no longer eligible to win cash from the current hand"""
        self.folder[self.idx] = True

    def mark_raiser(self):
        """Mark a raise for the current player."""
        if self.step_counter > 2:
            self.last_raiser = self.step_counter

    def mark_checker(self):
        """Counter the number of checks in the round"""
        self.checkers += 1

    def mark_out_of_cash_but_contributed(self):
        """Mark current player as a raiser or caller, but is out of cash."""
        self.out_of_cash_but_contributed[self.idx] = True
        self.deactivate_current()

    def mark_bb(self):
        """Ensure bb can raise"""
        self.last_raiser_step = self.step_counter + len(self.lst)
        self.max_steps_total = self.step_counter + len(self.lst) * 2

    def is_raising_allowed(self):
        """Check if raising is still allowed at this position"""
        return self.step_counter <= self.last_raiser_step

    def update_alive(self):
        """Update the alive property"""
        self.alive = np.array(self.can_still_make_moves_in_this_hand) + np.array(self.out_of_cash_but_contributed)

    def get_potential_winners(self):
        """Players eligible to win the pot"""
        potential_winners = np.logical_and(np.logical_or(np.array(self.can_still_make_moves_in_this_hand),
                                                         np.array(self.out_of_cash_but_contributed)),
                                           np.logical_not(np.array(self.folder)))
        return potential_winners


class Player(Hand):
    """Player shell"""

    def __init__(self, stack_size: int, name: str):
        """Initialization of an agent"""
        super().__init__()

        self.stack = stack_size
        self.last_stack = stack_size
        self.name = name
        self.seat = None
        self.agent_obj = None
        self.equity_alive = 0
        self.actions = []
        self.temp_stack = []
        self.last_action_in_stage = ''
        self.is_trainable = False

    @abc.abstractmethod
    def act(self, state, legal_actions, info=None) -> Action:
        pass
