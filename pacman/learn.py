#!/usr/bin/env python
#  -*- coding: utf-8 -*-

"""Implement Controllers, Adapters and Agents to create a working QAgent.

Test the agent in a WindyWater environment.
"""

import random

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class ProblemController(object):
    """Controls the execution of episodes.

    Controls the execution of episodes in a given problem adapter and with an
    agent.

    Attributes:
        num_episodes: The number of episodes the ProblemController will control
        the execution.
        problem_adapter: A problem adapter object for the ProblemAdapter class.
        agent: A agent object for the Agent Class.
    """

    def __init__(self, num_episodes, problem_adapter, agent):
        """The constructor method for the ProblemController.

        Args:
            num_episodes: The number of episodes.
            problem_adapter: A ProblemAdapter object.
            agents: A Agent object.
        """
        self.num_episodes = num_episodes
        self.problem_adapter = problem_adapter
        self.agent = agent

    def run(self):
        """Call execute_episodes.

        Call execute_episodes to get the average rewared and steps, and print
        it on the screen.
        """
        avg_reward, avg_steps = self.execute_episodes()
        print 'Average reward:', avg_reward
        print 'Average steps:', avg_steps

    def execute_episodes(self):
        """Execute the episodes.

        Call execute_episode num_episodes times.

        Returns:
            avg_reward: Average Rewards.
            avg_steps: Average Steps.
        """
        episodes_rewards = []
        episodes_steps = []

        for _ in range(self.num_episodes):
            cumulative_reward, steps = self.execute_episode
            (self.problem_adapter, self.agent)

            episodes_rewards.append(cumulative_reward)
            episodes_steps.append(steps)

        avg_reward = sum(episodes_rewards) / self.num_episodes
        avg_steps = sum(episodes_steps) / self.num_episodes

        return avg_reward, avg_steps

    def execute_episode(self, problem_adapter, agent):
        """Execute a single episode.

        Args:
            problem_adapter: The class problem_adapter.
            agent: The class agent.
        Returns:
            cumulative_reward: The cumulative reward for the episode.
            steps: The steps made by the agent in the episode.
        """
        cumulative_reward = 0
        steps = 0
        state = self.problem_adapter.initial_state
        self.problem_adapter.prepate_new_episode()

        while not self.problem_adapter.is_episode_finished():
            action = agent.act(state)
            state = self.problem_adapter.calculate_state(action)
            reward = self.problem_adapter.calculate_reward(state)
            agent.learn(action, state, reward)

            cumulative_reward += reward
            steps += 1

        return cumulative_reward, steps


class Agent(object):
    """Agent capable of learning and exploring.

    All action, state and reward variables must be numerical values. Besides,
    action and state must be an uniquely identified integer.

    Attributes:
        learning_element: An object of the QLearner Class.
        exploration_element: An object of the EGreedyExplorer Class.
    """

    def __init__(self):
        """Constructor method for the Agent Class.

        Set class atributes to None.
        """
        self.learning_element = None
        self.exploration_element = None

    def learn(self, action, state, reward):
        """Execute the learning algorithm.

        Args:
            action: An action value for the agent.
            state:  A state value for the agent.
            reward: Reward received after executing the action.
        """
        self.learning_element.learn(state, action, reward)

    def act(self, state):
        """Select an action to be executed.

        Selects an action to be executed by consulting both the learning and
        exploration algorithms.

        Args:
            state: A state value for the agent.
        Returns:
            selected_action: The selected action from the suggested action.
        """
        suggested_action = self.learning_element.act(state)
        selected_action = self.exploration_element.select_action
        (suggested_action)

        return selected_action


class ProblemAdapter(object):
    """Adapter for a specific learning problem.

    Problem adapter stores specific information about the problem where the
    agent is running.

    Attributes:
        initial_state: The initial state, default is 0.
        num_actions: The number of actions, default is 1.
        num_states: The number of states, default is 1.
    """

    def __init__(self, initial_state=0, num_actions=1, num_states=1):
        """Constructor method for ProblemAdapter class.

        Args:
            initial_state: The initial state.
            num_actions: The number of actions.
            num_states: The number of states.
        """
        self.initial_state = initial_state
        self.num_actions = num_actions
        self.num_states = num_states

    def prepare_new_episode(self):
        """Preparation for a new episode to be executed.

        The subclass must overwrite this function.

        Raises:
            NotImplementedError.
        """
        raise NotImplementedError

    def calculate_state(self, action):
        """Calculate the new agent state for a given action.

        The subclass must overwrite this function.

        Args:
            action: A given action.
        Raises:
            NotImplementedError.
        """
        raise NotImplementedError

    def calculate_reward(self, state):
        """Calculate the reward for the given state.

        The subclass must overwrite this function.
        Args:
            state: A given state.
        Raises:
            NotImplementedError.
        """
        raise NotImplementedError

    def is_episode_finished(self):
        """Check whether the current episode has finished.

        The subclass must overwrite this function.

        Raises:
            NotImplementedError.
        """
        raise NotImplementedError


class Learner(object):
    """Learning algorithm interface.

    A learning algorithm can select the best-suited action for any state and
    adapt it's belief according to reward information.
    """

    def learn(self, state, action, reward):
        """Learn state-action value by incorporating the reward information.

        The subclass must overwrite this function.

        Args:
            state: A state value.
            action: An action value.
            reward: Reward received after executing the action.
        Raises:
            NotImplementedError.
        """
        raise NotImplementedError

    def act(self, state):
        """Select an action for the given state.

        The subclass must overwrite this function.

        Args:
            state: A state value
        Raises:
            NotImplementedError.
        """
        raise NotImplementedError


class Explorer(object):
    """Exploration algorithm interface.

    An exploration algorithm is used by an agent to select actions other than
    the optimal one, potentially increasing the rewards in the long run.
    """

    def select_action(self, suggested_action):
        """Select an action given the one suggested by the learning algorithm.

        This method needs to be overwrited on a subclass.

        Args:
            suggested_action: The action suggested by the learning algorithm.
        Raises:
            NotImplementedError
        """
        raise NotImplementedError


class QLearner(Learner):
    """Q-learning algorithm implementation.

    Q-learning is a model free reinforcement learning algorithm that tries and
    learning state values and chooses actions that maximize the expected
    discounted reward for the current state.

    Attributes:
        current_state: State in which the algorithm currently is.
        q_values: Matrix that stores the value for a (state, action) pair.
        learning_rate: Value in [0, 1] interval that determines how much of the
        new information overrides the previous value. Deterministic scenarios
        may have optimal results with learning rate of 1, which means the new
        information completely replaces the old one.
        discount_factor: Value in [0, 1) interval that determines the
        importance of future rewards. 0 makes the agent myopic and greedy,
        trying to achieve higher rewards in the next step. Closer to 1 makes
        the agent maximize long-term rewards. Although values of 1 and higher
        are possible, it may make the expected discounted reward infinite or
        divergent.
    """

    def __init__(self, initial_state=0, num_states=0, num_actions=0,
                 learning_rate=1, discount_factor=1):
        """Constructor.

        Args:
            initial_state: State where the algorithm begins.
            num_states: Number of states to be represented.
            num_actions: Number of actions to be represented.
        """
        self.current_state = initial_state
        self.q_values = QValues(num_states=num_states, num_actions=num_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def update_state(self, new_state):
        """Update Q Learning current state.

        Args:
            new_state: State to which the learning algorithm is going.
        """
        self.current_state = new_state

    def learn(self, state, action, reward):
        """Learn by updating the (state, action) reward.

        Learn by applying the reward received when transitioning from the
        current state to the new one by executing an action.

        Parameters:
            state: Agent state after executing the action.
            action: Executed action.
            reward: Reward received after executing the action.
        """
        old_value = self.q_values.get(self.current_state, action)
        next_expected_value = self.q_values.get_max_value(state)
        new_value = old_value + self.learning_rate * (reward +
                                                      self.discount_factor *
                                                      next_expected_value -
                                                      old_value)

        self.q_values.set(self.current_state, action, new_value)

        self.update_state(state)

    def act(self, state):
        """Select an action for the given state.

        Args:
            state: Agent state to select an action.
        Returns:
            The max action for a certain state on the q_values map.
        """
        return self.q_values.get_max_action(state)

    def __str__(self):
        """Define behavior for when str() is called.

        Returns:
            String of q_values.
        """
        return ('Q-learning\n' + str(self.q_values))


class QValues(object):
    """Container for Q values.

    Attributes:
        num_states: Number of states that will be stored.
        num_actions: Number of actions that will be stored.
        q_values: Container for Q values.
    """

    def __init__(self, num_states=0, num_actions=0):
        """Constructor for the QValues class.

        Args:
            num_states: The number of states, default is 0.
            num_actions: The number of actions, default is 0.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_values = [[0 for _ in xrange(num_actions)]
                         for _ in xrange(num_states)]

    def get(self, state, action):
        """Get stored Q value for a (state, action) pair.

        Args:
            state: State index.
            action: Action index.
        Returns:
            The QValue for a pair state, action.
        """
        return self.q_values[state][action]

    def set(self, state, action, q_value):
        """Set Q value for a (state, action) pair.

        Args:
            state: State index.
            action: Action index.
            q_value: Q value to be stored.
        """
        self.q_values[state][action] = q_value

    def get_max_value(self, state):
        """Get the maximum Q value possible for the given state.

        Args:
            state: State from which to find the maximum Q value possible.
        Returns:
            The maximum QValue possible.
        """
        return max(self.q_values[state])

    def get_max_action(self, state):
        """Get a max action.

        Return the action index for which the Q value is maximum for the
        given state.

        Args:
            state: State from which to find the action.
        Returns:
            A random choice of a list of actions which the QValue is maximum.
        """
        max_value = self.get_max_value(state)
        actions = [action for action, value in enumerate(self.q_values[state])
                   if value == max_value]

        return random.choice(actions)

    def __str__(self):
        """Define the behavior for when str() is called.

        Returns:
            Print the states and its QValues.
        """
        output = ['\t%d' % action for action in range(self.num_actions)]
        output.append('\n')
        for state, values in enumerate(self.q_values):
            output.append('%d' % state)
            for value in values:
                output.append('\t%1.1f' % value)
            output.append('\n')
        return ''.join(output)


class EGreedyExplorer(Explorer):
    """E-Greedy exploration algorithm.

    Selects the suggested action or another random action with the given
    exploration_frequency.

    Attributes:
        actions: The range of the number of actions.
        exploration_frequency: The frequency the agent will explore.
    """

    def __init__(self, num_actions=1, exploration_frequency=0.0):
        """The constructor for the EGreedyExplorer class.

        Args:
            num_actions: Number of actions, default is 1.
            exploration_frequency: The exploration_frequency, default is 0.
        """
        self.actions = range(num_actions)
        self.exploration_frequency = exploration_frequency

    def select_action(self, suggested_action):
        """Select an action.

        Select an suggested action or a random action based on the exploration
        frequency.

        Args:
            suggested_action: The suggested action for the agent.
        Return:
            An action.
        """
        if random.random() < self.exploration_frequency:
            return random.choice(self.actions)
        else:
            return suggested_action


class QAgent(Agent):
    """Example agent with Q-learning and e-greedy exploration algorithms.

    Attributes:
        learning_element: An object of the QLearner Class
        exploration_element: An object of the EGreedyExplorer Class
    """

    def __init__(self, initial_state, num_states, num_actions):
        """Construtor for the QAgent class.

        Args:
            initial_state: A initial state position.
            num_states: Number of states.
            num_actions: Number of actions.
        """
        self.learning_element = QLearner(
            initial_state=initial_state,
            num_states=num_states,
            num_actions=num_actions,
            learning_rate=0.9,
            discount_factor=0.9,
        )
        self.exploration_element = EGreedyExplorer(
            num_actions=num_actions,
            exploration_frequency=0.1,
        )


class WindyWaterAdapter(ProblemAdapter):
    """Windy water example problem.

    The agent lives in the following world:
    * * * W W * * * * *
    S * * * * * * G * *
    * * * W W * * * * *
    * * * W W * * * * *
    * * * * * * * * * *
    * * * * * * * * * *
    * * * * * * * * * *

    Where:
    S: initial state
    W: water that gives penalty
    G: goal state

    Each step gives a reward of -1, going into the water rewards -100 and
    reaching the goal state rewards 100.

    Attributes:
        initial_coordinates: The initial coordinates of the agent, [1, 0].
        actions: The possible actions of an agent.
        [[0, 1], [-1, 0], [0, -1], [1, 0]]
        rows: The number of rows in the map, default is 7.
        cols: The number of cols in the map, defalut is 10.
        goal_coordinates: The coordinates of the goal, [1, 7].
        water_coordinates: The coordinates of the water tiles.
        [[0, 3], [0, 4], [2, 3], [2, 4], [3, 3], [3, 4]]
        wind_frequency: ...
    """

    def __init__(self, wind_frequency=0):
        """Constructor for WindyWaterAdapter class.

        Extent the ProblemAdapter class.

        Args:
            wind_frequency: ... , default is 0.
        """
        self.initial_coordinates = [1, 0]
        self.actions = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        self.rows = 7
        self.cols = 10
        self.goal_coordinates = [1, 7]
        self.water_coordinates = [[0, 3], [0, 4], [2, 3],
                                  [2, 4], [3, 3], [3, 4]]
        self.wind_frequency = wind_frequency

        super(WindyWaterAdapter, self).__init__(
            initial_state=self.coordinates_to_state(self.initial_coordinates),
            num_actions=len(self.actions),
            num_states=self.rows * self.cols,
        )

    def prepate_new_episode(self):
        """Prepate the new episode."""
        self.agent_coordinates = self.initial_coordinates

    def calculate_state(self, action):
        """Calculate the State.

        Calculate the state using wind direction.

        Args:
            action: An agent action.
        Returns:
            state: The calculated state.
        """
        # wind
        if random.random() < self.wind_frequency:
            wind_direction = random.randrange(0, self.num_actions)
            wind_action = [self.actions[wind_direction][0],
                           self.actions[wind_direction][1]]
        else:
            wind_action = [0, 0]

        # state generation
        self.agent_coordinates = [
            min(max(self.agent_coordinates[0] + self.actions[action][0] +
                    wind_action[0], 0), self.rows - 1),
            min(max(self.agent_coordinates[1] + self.actions[action][1] +
                    wind_action[1], 0), self.cols - 1),
        ]
        state = self.coordinates_to_state(self.agent_coordinates)

        return state

    def calculate_reward(self, state):
        """Calculate the reward.

        Calculate the reward, based on the agent coordinates.

        Args:
            state: An agent state.
        Returns:
            reward: The reward for the agent coordinate.
        """
        if (self.agent_coordinates in self.water_coordinates):
            reward = -100
        elif (self.agent_coordinates == self.goal_coordinates):
            reward = 100
        else:
            reward = -1

        return reward

    def is_episode_finished(self):
        """Check if it is the final episode.

        Returns:
            bool: True if is finished, false otherwise.
        """
        return (self.agent_coordinates == self.goal_coordinates)

    def coordinates_to_state(self, coordinates):
        """Transform a coordinate value.

        Args:
            coordinates: A coordinate.
        Returns:
            The state value for the given coordinate.
        """
        return coordinates[0] * self.cols + coordinates[1]

    def print_map(self):
        """Print the game map."""
        print
        for i in xrange(self.rows):
            for j in xrange(self.cols):
                if [i, j] == self.agent_coordinates:
                    print "A",
                elif [i, j] == self.initial_coordinates:
                    print "S",
                elif [i, j] == self.goal_coordinates:
                    print "G",
                elif [i, j] in self.water_coordinates:
                    print "W",
                else:
                    print "*",
            print
