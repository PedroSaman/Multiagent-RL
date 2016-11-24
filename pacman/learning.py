#!/usr/bin/env python
#  -*- coding: utf-8 -*-

"""Collection of reinforcement learning algorithms."""

from __future__ import division
import random

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class LearningAlgorithm(object):
    """Base Class for a Learning Algorithm subclass.

    Define methods learn and act that must be overwrited in subclasses.
    """

    def learn(self, state, action, reward):
        """Learn from experience.

        Learn by applying the reward received when transitioning from the
        current state to the new one by executing an action.

        The subclasses must overwrite learn.

        Args:
            state: Agent state after executing the action.
            action: Executed action.
            reward: Reward received after executing the action.
        Raises:
            NotImplementedError: Subclass does not implement learn.
        """
        raise (NotImplementedError,
               '%s does not implement "learn" method' % str(type(self)))

    def act(self, state):
        """Select an action for the given state.

        By exploiting learned model, the algorithm selects the best action to
        be executed by the agent.

        The subclasses must overwrite learn.

        Args:
            state: Agent state to select an action.
        Raises:
            Subclass does not implement act.
        """
        raise (NotImplementedError,
               '%s does not implement "act" method' % str(type(self)))


class QLearning(LearningAlgorithm):
    """Q-learning algorithm implementation.

    Q-learning is a model free reinforcement learning algorithm that tries and
    learning state values and chooses actions that maximize the expected
    discounted reward for the current state.

    Attributes:
        previous_state: State in which the algorithm currently is.
        q_values: Storage for (state, action) pair estimated values.
        learning_rate: Value in [0, 1] interval that determines how much of the
            new information overrides the previous value. Deterministic
            scenarios may have optimal results with learning rate of 1,
            which means the new information completely replaces the old one.
        discount_factor: Value in [0, 1) interval that determines the
            importance of future rewards. 0 makes the agent myopic and greedy,
            trying to achieve higher rewards in the next step. Closer to 1
            makes the agent maximize long-term rewards. Although values of 1
            and higher are possible, it may make the expected discounted reward
            infinite or divergent.
    """

    def __init__(self, initial_state=0, learning_rate=1, discount_factor=1,
                 actions=None):
        """Constructor for the QLearning class.

        Extend the LearningAlgorithm constructor.

        Args:
            initial_state: State where the algorithm begins.
            learning_rate: A value in a [0, 1] interval.
            discount_factor: A value in [0, 1) interval.
            actions: The agent actions, if parameter not specified initialize
                an empty list.
        """
        super(QLearning, self).__init__()
        self.previous_state = initial_state
        self.q_values = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        if actions:
            self.actions = actions
        else:
            self.actions = []

    def __str__(self):
        """Define the behavior when str() is called.

        Returns:
            Q-values string representation.
        """
        results = []
        results.append('Q-values\n')
        for state in self.q_values:
            results.append(str(state))
            for action in self.q_values[state]:
                results.append(str(self.q_values[state][action]))
                results.append('\t')
            results.append('\n')
        return ''.join(results)

    def update_state(self, state):
        """Update Q Learning current state.

        Args:
            state: State to which the learning algorithm is going.
        """
        self.previous_state = state

    def initialize_unknown_state(self, state):
        """Initialize Q-values for states that were not previously seen.

        Args:
            state: Environment state.
        """
        if state not in self.q_values:
            self.q_values[state] = {}
            for action_ in self.actions:
                self.q_values[state][action_] = 0.0

    def get_q_value(self, state, action):
        """Get the current estimated value for the state-action pair.

        Args:
            state: Environment state.
            action: Agent action.
        Returns:
            The current QValue.
        """
        self.initialize_unknown_state(state)
        return self.q_values[state][action]

    def set_q_value(self, state, action, value):
        """Set a new estimated value for the state-action pair.

        Args:
            state: Environment state.
            action: Agent action.
            value: New estimated value.
        """
        self.q_values[state][action] = value

    def _get_max_action_from_list(self, state, action_list):
        """Get the action with maximum estimated value.

        Get the action with maximum estimated value from the given list of
        actions.

        Args:
            state: Environment state.
            action_list: Actions to be evaluated.
        Returns:
            The action with maximum estimated value.
        """
        actions = filter(lambda a: a in action_list, self.q_values[state])
        values = [self.q_values[state][action] for action in actions]
        max_value = max(values)
        max_actions = [action for action in actions
                       if self.q_values[state][action] == max_value]

        return random.choice(max_actions)

    def get_max_action(self, state):
        """Get the action with maximum estimated value.

        Args:
            state: Environment state.
        Returns:
            Action with maximum estimated value.
        """
        self.initialize_unknown_state(state)
        return self._get_max_action_from_list(state, self.actions)

    def get_max_q_value(self, state):
        """Get the QValue for a max_action.

        This QValue will be max.

        Args:
            state: A given agent state.
        Returns:
            The max QValue.
        """
        max_action = self.get_max_action(state)
        return self.q_values[state][max_action]

    def learn(self, state, action, reward):
        """Learn by updating the (state, action) reward.

        Learn by applying the reward received when transitioning from the
        current state to the new one by executing an action.

        Args:
            state: Agent state after executing the action.
            action: Executed action.
            reward: Reward received after executing the action.
        """
        old_value = self.get_q_value(self.previous_state, action)
        next_expected_value = self.get_max_q_value(state)
        new_value = (old_value + self.learning_rate * (reward +
                                                       self.discount_factor *
                                                       next_expected_value -
                                                       old_value))
        self.set_q_value(self.previous_state, action, new_value)
        self.update_state(state)

    def act(self, state, legal_actions):
        """Select the best legal action for the given state.

        Args:
            state: Agent state to select an action.
            legal_actions: Actions allowed in the current state.
        Returns:
            The best action of a list of legal actions.
        """
        return self._get_max_action_from_list(state, legal_actions)


class QLearningWithApproximation(LearningAlgorithm):
    """Q-learning algorithm implementation with function aproximation.

    Attributes:
        actions: A list of action to action-state pair. It might be a list of
            behaviors, if using behaviors-states pairs.
        features: A list of features.
        exploration_rate: The rate the agent will explore.
        weights: The weights of an feature.
        previous_state: State in which the algorithm currently is.
        learning_rate: Value in [0, 1] interval that determines how much of the
            new information overrides the previous value. Deterministic
            scenarios may have optimal results with learning rate of 1,
            which means the new information completely replaces the old one.
        discount_factor: Value in [0, 1) interval that determines the
            importance of future rewards. 0 makes the agent myopic and greedy,
            trying to achieve higher rewards in the next step. Closer to 1
            makes the agent maximize long-term rewards. Although values of 1
            and higher are possible, it may make the expected discounted reward
            infinite or divergent.
    """

    def __init__(self, actions=None, features=None, learning_rate=1,
                 discount_factor=1, exploration_rate=0):
        """Constructor for QLearningWithApproximation class.

        Extends LearningAlgorithm constructor.

        Args:
            actions: A list of actions or behaviors, defaut is None.
            features: A list of features, default is None.
            learning_rate: A value in [0, 1] interval for the learning, default
                is 1.
            discount_factor: A value in [0, 1) interval that determines the
                importance of future rewards, default is 1.
            exploration_rate: The rate the agent will choose a explore action,
                default is 0.
        """
        super(QLearningWithApproximation, self).__init__()
        self.actions = actions
        self.features = features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.previous_state = None
        self.exploration_rate = exploration_rate
        self.weights = {}
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights values for each behavior."""
        for action in self.actions:

            self.weights[str(action)] = [random.random()
                                         for _ in range(len(self.features))]

    def get_weights(self):
        """Get the weights on the class attribute.

        Returns:
            self.weights: The weights.
        """
        return self.weights

    def set_weights(self, weights):
        """Set weights on the class attribute.

        Args:
            weights: New weights.
        """
        self.weights = weights

    def get_q_value(self, state, action):
        """Get QValues of a given (state, action) pair.

        Args:
            state: Environment state.
            action: An action for the agent state.
        Retunrs:
            The QValue.
        """
        q_value = 0

        for weight, feature in zip(self.weights[str(action)], self.features):
            q_value += weight * feature(state, action)

        return q_value

    def _get_max_action_from_list(self, state, action_list):
        """Get the action with maximum estimated value.

        Get the action with maximum estimated value from the given list of
        actions.

        Args:
            state: Environment state.
            action_list: Actions to be evaluated.
        Returns:
            A random choice on the max_actions list.
        """
        actions = filter(lambda a: a in action_list, self.actions)

        values = [self.get_q_value(state, action) for action in actions]
        max_value = max(values)
        max_actions = [action for action in actions
                       if self.get_q_value(state, action) == max_value]

        return random.choice(max_actions)

    def get_max_action(self, state):
        """Get the max action.

        Args:
            state: Environment state.
        Retuns:
            Max action.
        """
        return self._get_max_action_from_list(state, self.actions)

    def get_max_q_value(self, state):
        """Get the max QValue.

        Args:
            state: Environment state.
        Returns:
            Max QValue.
        """
        action = self.get_max_action(state)
        return self.get_q_value(state, action)

    def _update_weights(self, action, delta):
        """Update the weights.

        Args:
            action: Action for the weights update.
            delta: Reward + discount_factor * max_q_value - old qvalue.
        """
        self.weights[str(action)] = [weight + self.learning_rate * delta *
                                     feature(self.previous_state, action)
                                     for weight, feature in
                                     zip(self.weights[str(action)],
                                         self.features)]

    def learn(self, state, action, reward):
        """Update the weights and set the previous state.

        Args:
            state: The passed state.
            action: The action taken.
            reward: The reward received.
        """
        if self.previous_state:
            delta = (reward + self.discount_factor *
                     self.get_max_q_value(state) -
                     self.get_q_value(self.previous_state, action))

            self._update_weights(action, delta)

        self.previous_state = state

    def _explore(self):
        """Explore action.

        Returns:
            A random choice of self.actions
        """
        return random.choice(self.actions)

    def _exploit(self, state):
        """Exploit acion.

        Returns:
            Max action from list of pair state-action.
        """
        return self._get_max_action_from_list(state, self.actions)

    def _exploitComm(self, state, actual_behavior):
        """Exploit communication action.

        Returns:
            Max action from list of pair state-action.
        """
        actions = []
        actions.append(actual_behavior)
        return self._get_max_action_from_list(state, actions)

    def act(self, state, actual_behavior, communicationHappened):
        """Choose if the agent will explore or exploit and act.

        Args:
            state: The actual state for exploitation.
        Returns:
            Choice for the explore or exploit.
        """
        p = random.random()

        if communicationHappened:
            # print("ExploitCommunication!")
            return self._exploitComm(state, actual_behavior)
        elif p < self.exploration_rate:
            return self._explore()
        else:
            return self._exploit(state)
