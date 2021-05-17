import logging
import random

import numpy as np

from consts import SUITS, RANKS, NUM_SUITS, NUM_RANKS, TOT_COMMUNITY_CARDS, NUM_CARDS_FOR_PLAYER

log = logging.getLogger(__name__)


class Card:
    """
    Card class
    """

    def __init__(self, rank: str, suit: str, face_up=False):
        self._rank = rank
        self._suit = suit
        self._encoding = self.create_encoding()

        self._face_up = face_up

    def __str__(self):
        return str(self._rank) + str(self._suit)

    def create_encoding(self):
        encoding = np.zeros([NUM_RANKS, NUM_SUITS])
        encoding[RANKS.index(self._rank), SUITS.index(self._suit)] = 1
        return encoding

    @property
    def get_encoding(self):
        return self._encoding

    def open_card(self):
        self._face_up = True

    @staticmethod
    def get_empty_card_encoding():
        encoding = np.zeros([NUM_RANKS, NUM_SUITS])
        return encoding


class Hand:
    def __init__(self):
        self.cards = []

    def cards_str(self):
        cards_str = ''
        for card in self.cards:
            cards_str += str(card)
        return cards_str

    def __str__(self):
        return self.cards_str()

    def cards_str_list(self):
        return [card.__str__() for card in self.cards]

    def open_cards(self):
        for card in self.cards:
            card.open_card()

    def add_card(self, card):
        self.cards.append(card)

    def clear_cards(self):
        self.cards = []

    def get_cards_encodings(self):
        if len(self.cards) == 0:
            return Card.get_empty_card_encoding()

        cards_encodings = np.asarray([card.get_encoding for card in self.cards])
        encoding = cards_encodings.sum(axis=0)

        return encoding


class Deck:
    """
    represents the card deck - shuffled each round
    """

    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        self.cards = []
        # create a deck with all the cards
        for rank in RANKS:
            for suit in SUITS:
                card = Card(rank, suit)
                self.cards.append(card)
        # shuffle the deck
        random.shuffle(self.cards)

    def deal_to(self, hand, num_cards=1):

        # can't deal the cards
        if len(self.cards) == 0:
            print('deck is empty')

        elif len(self.cards) < num_cards:
            print('not enough cards to deal')

        # deal the cards
        else:
            for i in range(0, num_cards):
                card = self.cards.pop()
                hand.add_card(card)

    def deal_2_cards_to_all_players(self, players):
        for player in players:
            self.deal_to(player, num_cards=2)
            log.info(f"Player {player.seat} got {player.cards} and ${player.stack}")
