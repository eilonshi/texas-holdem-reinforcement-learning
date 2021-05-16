import logging
import random

log = logging.getLogger(__name__)


class Card:
    """
    Card class
    """

    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    NUM_RANKS = 13
    SUITS = ['C', 'D', 'H', 'S']
    NUM_SUITS = 4

    def __init__(self, rank: str, suit: str, face_up=False):
        self._rank = rank
        self._suit = suit
        self._encoding = [Card.RANKS.index(self._rank), Card.SUITS.index(self._suit)]

        self._face_up = face_up

    def __str__(self):
        return str(self._rank) + str(self._suit)

    @property
    def get_index(self):
        return self._encoding

    def open_card(self):
        self._face_up = True


class Hand:
    def __init__(self, hands=None):
        self.cards = []
        if hands is not None:
            for hand in hands:
                for card in hand.cards:
                    self.cards.append(card)

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
        for rank in Card.RANKS:
            for suit in Card.SUITS:
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
