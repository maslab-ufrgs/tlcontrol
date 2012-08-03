"""Classes implementing the general q-learning algorithm.
"""

import random as rand
from random import random


class QTable(object):
    """A table of state-action qvalues, with the learning logic.
    """

    DEFAULT_QVALUE = 0

    def __init__(self, states, actions, learning_rate,
                 discount_factor, default_qvalue=None):
        """Initializes the table and checks parameter restrictions.

        In case a parameter restriction isn't met, a ValueError is raised:
          - 0 <= learning_rate <= 1
          - 0 <= discount_factor < 1
        """
        # Verify and initialize learning_rate
        if 0 <= learning_rate and learning_rate <= 1:
            self.__learning_rate = learning_rate
        else:
            raise ValueError("Invalid learning rate of %d not contained in [0,1]."
                             % learning_rate)

        # Verify and initialize discount_factor
        if 0 <= discount_factor and discount_factor < 1:
            self.__discount_factor = discount_factor
        else:
            raise ValueError("Invalid discount factor of %d not contained in [0,1)."
                             % discount_factor)

        # Initialize the table of qvalues
        if default_qvalue is None:
            default_qvalue = self.DEFAULT_QVALUE

        self.__table = dict( (s, dict((a, default_qvalue) for a in actions))
                             for s in states )
        self.__states = tuple(states)
        self.__actions = tuple(actions)

    def __getitem__(self, pair):
        return self.__table[pair[0]][pair[1]]

    def subtable(self, state):
        """Obtain the part of the table describing the given state.

        The result is a dict from actions to their q-value
        on the given state.
        """
        return self.__table[state]

    @property
    def states(self):
        """Return a tuple with all possible states."""
        return self.__states

    @property
    def actions(self):
        """Return a tuple witha all possible actions."""
        return self.__actions

    def observe(self, state, action, new_state, reward):
        """Update q-values according to the observed behavior."""
        max_future = max( self[new_state, new_action]
                          for new_action in self.__actions )
        old_val = self[state, action]

        change = reward + (self.__discount_factor * max_future) - old_val
        self.__table[state][action] = old_val + (self.__learning_rate * change)

    def act(self, state):
        """Return the recommended action for this state.

        The choice of action may include random exploration.
        """
        raise NotImplementedError("The basic QTable has no policy.")


class EpsilonGreedyQTable(QTable):
    """QTable with epsilon-greedy strategy.

    With a given probability, called curiosity, chooses a random action.
    Otherwise, does a greedy choice (action with greatest qvalue, in case
    of a tie randomly picks one of the best).

    The curiosity decays a given percentage after each random choice.
    """

    DEFAULT_CURIOSITY_DECAY = 0

    def __init__(self, states, actions, learning_rate, discount_factor, curiosity,
                 curiosity_decay=None, default_qvalue=None):
        """Initializes the table and checks for parameter restrictions.

        In case a parameter restriction isn't met, a ValueError is raised:
          - all restrictions from QTable apply
          - 0 <= curiosity_decay < 1
        """
        super(EpsilonGreedyQTable, self).__init__(states, actions, learning_rate,
                                                  discount_factor, default_qvalue)

        self.__curiosity = curiosity

        # Verify and initialize curiosity_decay
        if curiosity_decay is None:
            curiosity_decay = self.DEFAULT_CURIOSITY_DECAY

        if 0 <= curiosity_decay and curiosity_decay < 1:
            self.__curiosity_factor = 1 - curiosity_decay
        else:
            raise ValueError("Invalid curiosity decay %d not contained in [0,1)."
                             % curiosity_decay)

    def act(self, state):
        if random() < self.__curiosity:
            return self.random_choice(state)
        else:
            return self.greedy_choice(state)

    def random_choice(self, state):
        """Makes a uniformly random choice between all actions for this state."""
        self.__curiosity *= self.__curiosity_factor
        return rand.choice(self.actions)

    def greedy_choice(self, state):
        """Makes a uniformly random choice between the actions with best q-value.

        The analysed q-values are those related with actions on this state."""
        best_qval = max(self.subtable(state).values())
        best_actions = [act for (act, qval) in self.subtable(state).items()
                        if qval == best_qval]
        return rand.choice(best_actions)


class EpsilonFirstQTable(EpsilonGreedyQTable):
    """QTable with epsilon-greedy strategy limited to the first N actions."""

    def __init__(self, states, actions, learning_rate, discount_factor, curiosity,
                 exploration_period, curiosity_decay=None, default_qvalue=None):
        super(EpsilonFirstQTable, self).__init__(states, actions, learning_rate,
                                                 discount_factor, curiosity,
                                                 curiosity_decay, default_qvalue)
        if exploration_period > 0:
            self.__remaining_exploration = exploration_period
        else:
            raise ValueError("Invalid non-positive exploration period %d."
                             % exploration_period)

    def act(self, state):
        if self.__remaining_exploration > 0:
            self.__remaining_exploration -= 1
            return super(EpsilonFirstQTable, self).act(state)
        else:
            return self.greedy_choice(state)
