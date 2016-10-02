#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""Define features used by behavior learning agents."""

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class Feature(object):
    """Superclass for other features."""

    def __call__(self, state, action):
        """Basic class __call__ method.

        This is only called when a subclass feature does not implement call.

        Args:
            state: The current game state.
            action: The action being made.
        Raises:
            NotImplementedError: A feature must implement __call__.
        """
        raise NotImplementedError('Feature must implement __call__')


class EnemyDistanceFeature(Feature):
    """Calculate the distance to the enemy."""

    def __init__(self, enemy_id):
        """Constructor for the EnemyDistanceFeature Class.

        Args:
            enemy_id: The identifier of the enemy.
        Attributes:
            enemy_id: The identifier of the enemy.
        """
        self.enemy_id = enemy_id

    def __call__(self, state, action):
        """Method executed when EnemyDistanceFeature Class is called.

        Args:
            state: Agent state.
            action: Action done for the agent state.
        Returns:
            distance (float): The distance between the agent and the enemy.
        """
        my_position = state.get_position()
        enemy_position = state.get_agent_position(self.enemy_id)
        distance = state.calculate_distance(my_position, enemy_position)

        if distance == 0.0:
            distance = 1.0

        distance = (1.0/distance)
        return distance


class FoodDistanceFeature(Feature):
    """Calculate the distance between the agent and the food."""

    def __call__(self, state, action):
        """Method executed when FoodDistanceFeature Class is called.

        Args:
            state: Agent state.
            action: Action done for the agent state.
        Returns:
            distance (float): The distance between the agent and the food.
        """
        distance = state.get_food_distance()

        if distance == 0.0:
            distance = 1.0

        distance = (1.0 / distance)
        return distance


class FragileAgentFeature(Feature):
    """Get the Fragile Agente for an agent identifier.

    A fragile agent is the when a pacman eat a pill and might be on danger.
    """

    def __init__(self, agent_id):
        """Constructor method for the FragileAgentFeature Class.

        Args:
            agent_id: The identifier of the agent.
        Attributes:
            agent_id: The identifier of the agent.
        """
        self.agent_id = agent_id

    def __call__(self, state, action):
        """Method executed when FragileAgentFeature Class is called.

        Args:
            state: Agent state.
            action: Action done for the agent state.
        Returns:
            Fragile Agente for the agent id.
        """
        return state.get_fragile_agent(self.agent_id)
