#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Unit Tests for classes and functions in the learn.py file."""

import unittest
import learn

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class TestLearn(unittest.TestCase):
    """Unit tests for the Learner class."""

    def test_learn_method_raises_not_implemented_error(self):
        """Test the learn function.

        For a given state, action and reward, it should return
        NotImplementedError.
        """
        state = None
        action = None
        reward = None
        l = learn.Learner()

        with self.assertRaises(NotImplementedError):
            l.learn(state, action, reward)

    def test_select_method_raises_not_implemented_error(self):
        """Test the act function.

        For a given state, it should return NotImplementedError.
        """
        state = None
        l = learn.Learner()

        with self.assertRaises(NotImplementedError):
            l.act(state)


class TestQValues(unittest.TestCase):
    """Unit tests for the QValues class."""

    def test_default_num_states(self):
        """Test the default number of states.

        Should be 0.
        """
        qv = learn.QValues()

        self.assertEqual(qv.num_states, 0)

    def test_default_num_actions(self):
        """Test the default number of actions.

        Should be 0.
        """
        qv = learn.QValues()

        self.assertEqual(qv.num_actions, 0)

    def test_initial_num_states(self):
        """Test the initial number os states.

        Should be 1.
        """
        qv = learn.QValues(num_states=1)

        self.assertEqual(qv.num_states, 1)

    def test_initial_num_actions(self):
        """Test initial number of actions.

        Should be 1.
        """
        qv = learn.QValues(num_actions=1)

        self.assertEqual(qv.num_actions, 1)

    def test_get_default_q_value(self):
        """Test the get function.

        For a given state, and action should expect a 0.
        """
        state = 0
        action = 0
        expected_q_value = 0
        qv = learn.QValues(num_states=1, num_actions=1)

        q_value = qv.get(state, action)

        self.assertEqual(q_value, expected_q_value)

    def test_set_q_value(self):
        """Test the set function.

        For a given state, action and QValue, set the qvalue, and get the
        QValue set. It should be 10.
        """
        state = 0
        action = 0
        q_value = 10
        qv = learn.QValues(num_states=1, num_actions=1)

        qv.set(state, action, q_value)
        set_q_value = qv.get(state, action)

        self.assertEqual(set_q_value, q_value)

    def test_get_best_q_value(self):
        """Test the get_max_value function."""
        q_values = [[1, 2], [4, 3]]
        qv = learn.QValues(num_states=2, num_actions=2)

        for state, actions in enumerate(q_values):
            for action, value in enumerate(actions):
                qv.set(state, action, value)

        best_q_values = [qv.get_max_value(state) for state in [0, 1]]

        self.assertEqual(best_q_values, [2, 4])

    def test_get_max_action_index(self):
        """Test the get_max_action function."""
        q_values = [[1, 2], [4, 3]]
        qv = learn.QValues(num_states=2, num_actions=2)

        for state, actions in enumerate(q_values):
            for action, value in enumerate(actions):
                qv.set(state, action, value)

        actions = [qv.get_max_action(state) for state in [0, 1]]

        self.assertEqual(actions, [1, 0])


class TestQLearn(unittest.TestCase):
    """Unit tests for the QLearner class."""

    def test_default_current_state(self):
        """Test the default current_state.

        It should be 0.
        """
        expected_current_state = 0

        ql = learn.QLearner()

        self.assertEqual(ql.current_state, expected_current_state)

    def test_default_discount_factor(self):
        """Test the default discount factor.

        It should be 1.
        """
        expected_discount_factor = 1

        ql = learn.QLearner()

        self.assertEqual(ql.discount_factor, expected_discount_factor)

    def test_default_learning_rate(self):
        """Test the default learning rate.

        It should be 1.
        """
        expected_learning_rate = 1

        ql = learn.QLearner()

        self.assertEqual(ql.learning_rate, expected_learning_rate)

    def test_initial_state(self):
        """Test the initial state.

        It should be 2.
        """
        expected_current_state = 2

        ql = learn.QLearner(initial_state=2)

        self.assertEqual(ql.current_state, expected_current_state)

    def test_initial_discount_factor(self):
        """Test the discount factor.

        It should be 0.8.
        """
        expected_discount_factor = 0.8

        ql = learn.QLearner(discount_factor=0.8)

        self.assertEqual(ql.discount_factor, expected_discount_factor)

    def test_initial_learning_rate(self):
        """Test the initial learning rate.

        It should be 0.8.
        """
        expected_learning_rate = 0.8

        ql = learn.QLearner(learning_rate=0.8)

        self.assertEqual(ql.learning_rate, expected_learning_rate)

    def test_change_state_after_learning(self):
        """Test state change after the learning.

        Test the learn function, for state changing.
        """
        state = 1
        action = 0
        reward = 0
        ql = learn.QLearner(num_states=3, num_actions=2)

        ql.learn(state, action, reward)

        self.assertEqual(ql.current_state, state)

    def test_learn_with_1_state_action_q_value(self):
        """Test learn with one state-action QValue.

        Test the learn function for reward changing.
        """
        state = 0
        action = 0
        reward = 10
        ql = learn.QLearner(num_states=1, num_actions=1)

        ql.learn(state, action, reward)
        q_value = ql.q_values.get(state, action)

        self.assertEqual(q_value, 10)

    def test_learn_with_several_state_action_q_values(self):
        """Test learn with several state-action QValues."""
        current_state = 2
        next_state = 4
        action = 3
        reward = 10
        ql = learn.QLearner(initial_state=current_state, num_states=5,
                            num_actions=4)

        ql.learn(next_state, action, reward)
        q_value = ql.q_values.get(current_state, action)

        self.assertEqual(q_value, 10)

    def test_learn_two_rewards(self):
        """Test the QValue after two rewards."""
        state = 0
        action = 0
        rewards = [5, 10]
        expected_q_value = 15
        ql = learn.QLearner(num_states=1, num_actions=1)

        for reward in rewards:
            ql.learn(state, action, reward)
        q_value = ql.q_values.get(state, action)

        self.assertEqual(q_value, expected_q_value)

    def test_learn_with_discount_factor(self):
        """Test Learn with the discount factor.

        Check the QValue when learning with discount_factor of 0.5.
        """
        state = 0
        action = 0
        expected_q_value = 15
        rewards = [10, 10]
        ql = learn.QLearner(num_states=1, num_actions=1, discount_factor=0.5)

        for reward in rewards:
            ql.learn(state, action, reward)
        q_value = ql.q_values.get(state, action)

        self.assertEqual(q_value, expected_q_value)

    def test_learn_with_learning_rate(self):
        """Test learn with learning rate of 0.5.

        Check the QValue when learning with learning rate of 0.5.
        """
        state = 0
        action = 0
        expected_q_value = 10
        rewards = [10, 10]
        ql = learn.QLearner(num_states=1, num_actions=1, learning_rate=0.5)

        for reward in rewards:
            ql.learn(state, action, reward)
        q_value = ql.q_values.get(state, action)

        self.assertEqual(q_value, expected_q_value)

    def test_select_action_after_one_learning_step(self):
        """Test select action, after one learning step."""
        state, action, reward = 1, 1, 10
        ql = learn.QLearner(num_states=2, num_actions=2, learning_rate=0.5)

        ql.learn(state, action, reward)
        selected_action = ql.act(0)

        self.assertEqual(action, selected_action)

    def test_select_action_after_several_learning_steps(self):
        """Test select action after several learning steps."""
        ql = learn.QLearner(num_states=2, num_actions=2, learning_rate=0.5,
                            discount_factor=0.5)
        steps = [(1, 1, 10),
                 (0, 0, 100),
                 (1, 1, 5)]
        excepted_actions = [(0, 1),
                            (1, 0)]

        for state, action, reward in steps:
            ql.learn(state, action, reward)

        for state, excepted_action in excepted_actions:
            action = ql.act(state)
            self.assertEqual(action, excepted_action)


if __name__ == '__main__':
    unittest.main()
