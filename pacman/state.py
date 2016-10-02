#!/usr/bin/python
# -*- coding: utf-8 -*-
"""The definition of the Map and tha GameStates."""

import math

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class Map(object):
    """Probabilistic map.

    Every cell contains a value in the interval [0, 1] indicating a
    probability. The entire map sums up to 1.

    Attributes:
        width: The map width.
        height: The map height.
        action_to_pos: Attribute the action to the matrix value.
        _walls: The positions of the walls.
        cells: Generate the map matrix cells.
    """

    paths = None

    def __init__(self, width, height, walls=[]):
        """Constructor for the Map class.

        Set the map width and height, map the actions to the matrix
        differences, set the walls, generate the cells and call normalize.

        Args:
            width: The width of the map.
            height: The height of the map.
            walls: A list of walls positions in the map.
        """
        self.width = width
        self.height = height
        self.action_to_pos = {
            'North': (1, 0),
            'South': (-1, 0),
            'East': (0, 1),
            'West': (0, -1),
            'Stop': (0, 0),
        }
        self._walls = walls
        self.cells = self.generate_cells()
        self.normalize()

    @property
    def walls(self):
        """Get the walls list.

        Returns:
            _walls: Walls list.
        """
        return self._walls

    @walls.setter
    def walls(self, walls):
        """Set the walls.

        Set Walls, calculate all paths.

        Args:
            walls: The walls positions.
        """
        self._walls = walls

        if Map.paths is None:
            self._calculate_all_paths()

    def __getitem__(self, i):
        """Get item in a cell.

        Args:
            i: The cell index.
        Returns:
            cells[i]: The content of the cell.
        """
        return self.cells[i]

    def __setitem__(self, i, item):
        """Set item in a cell.

        Args:
            i: The cell index.
            item: The content of the cell.
        """
        self.cells[i] = item

    def __iter__(self):
        """For each cell yeld a generator.

        Yelds:
            A generator for value.
        """
        for value in self.cells:
            yield value

    def __len__(self):
        """Return the number of cells in the Map.

        Returns:
            num_cells: The number of cells
        """
        return self.num_cells

    def __str__(self):
        """Implement behavior for when str is called.

        Returns:
            Print the map
        """
        string = []

        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if self._is_wall((y, x)):
                    string.append('.....')
                else:
                    string.append('%.3f' % self[y][x])
                string.append(' ')
            string.append('\n')

        return ''.join([str(line) for line in string])

    def _is_inbound(self, pos):
        """Check if a position is inside or outside the map.

        Args:
            pos: A position.
        Returns:
            A boolean value for when the pos is inside or outside the map.
        """
        return (0 <= pos[0] < self.height and 0 <= pos[1] < self.width)

    def _is_wall(self, pos):
        """Check if a position is a Wall.

        Args:
            pos: A position.
        Returns:
            A boolean value for when the pos is a wall or not.
        """
        return (pos in self._walls)

    def _is_valid_position(self, pos):
        """Check if a position is valid.

        Check if it is inside the map and if it is not a wall.

        Args:
            pos: A position
        Returns:
            A boolean value for when the position is valid or not.
        """
        return (self._is_inbound(pos) and not self._is_wall(pos))

    def max(self):
        """Get the max probability.

        Returns:
            The maximum probability.
        """
        max_prob = float('-inf')

        for y in range(self.height):
            max_row = max(self[y])
            if max_row > max_prob:
                max_prob = max_row

        return max_prob

    def normalize(self):
        """Normalize the multiplcation of the probabilities.

        Normalize the multiplcation of the probabilities back into a
        probability.
        """
        prob_sum = 0.0

        for x in range(self.width):
            for y in range(self.height):
                prob_sum += self[y][x]

        for x in range(self.width):
            for y in range(self.height):
                if self._is_wall((y, x)):
                    self[y][x] = 0.0
                elif prob_sum > 0:
                    prob = self[y][x] / prob_sum
                    self[y][x] = prob
                else:
                    self[y][x] = 1.0 / ((self.width * self.height) -
                                        len(self.walls))

    def generate_cells(self):
        """Generate Cells.

        Returns:
            The cells generated.
        """
        cells = [[0 for _ in range(self.width)]
                 for _ in range(self.height)]
        return cells

    def get_maximum_position(self):
        """Get the position with maximum probability.

        Returns:
            max_position: The position with max probability.
        """
        max_position = (0, 0)
        max_prob = 0.0

        for x in range(self.width):
            for y in range(self.height):
                if self[y][x] > max_prob:
                    max_prob = self[y][x]
                    max_position = (y, x)

        return max_position

    def observe(self, pos, measurement_prob_dist_fn, *params):
        """Calculate the probability of a position.

        Call normalize after the calculation.

        Args:
            pos: The position to calculate.
            measurement_prob_dist_fn: The distribution function.
            *params: Variable lenght arguments list.
        """
        for x in range(self.width):
            for y in range(self.height):
                old_probability = self[y][x]
                new_probability = measurement_prob_dist_fn(
                    (y, x), pos, *params) * old_probability

                self[y][x] = new_probability

        self.normalize()

    def predict(self, action, action_prob_dist_fn, *params):
        """Predict a position based on a given action.

        Args:
            action: A action made by the agent.
            action_prob_dist_fn: The distribution function.
            *params: Variable lenght arguments list.
        """
        cells = self.generate_cells()

        for x in range(self.width):
            for y in range(self.height):
                old_probability = self[y][x]

                for possible_action in self.action_to_pos:
                    next_y = y + self.action_to_pos[possible_action][0]
                    next_x = x + self.action_to_pos[possible_action][1]

                    if self._is_valid_position((next_y, next_x)):
                        action_probability = action_prob_dist_fn(
                            action, possible_action, *params)
                        new_probability = action_probability * old_probability
                        cells[next_y][next_x] += new_probability

        self.cells = cells
        self.normalize()

    def _generate_next_pos(self, pos):
        """Generate next position.

        Args:
            pos: The actual position.
        Returns:
            A dict of the next position candidate and its action.
        """
        next_pos = {}

        for action, delta in self.action_to_pos.items():
            candidate_pos = (pos[0] + delta[0], pos[1] + delta[1])

            if self._is_valid_position(candidate_pos):
                next_pos[candidate_pos] = action

        return next_pos

    def _calculate_paths(self, pos, max_distance=None):
        """Calculate the path.

        Args:
            pos: The current position
            max_distance: The maximum distance, default is None.
        """
        pos_to_path = {}
        current_pos = [pos]
        analyzed_pos = []

        while current_pos:
            p = current_pos.pop(0)
            analyzed_pos.append(p)

            if p in pos_to_path:
                path = pos_to_path[p]
            else:
                path = []

            next_pos = []
            for next_p, action in self._generate_next_pos(p).items():
                if next_p not in analyzed_pos:
                    next_pos.append(next_p)

                    if not max_distance or len(path) + 1 <= max_distance:
                        pos_to_path[next_p] = path + [action]
            current_pos.extend(next_pos)

        return pos_to_path

    def _calculate_all_paths(self, max_distance=None):
        paths = {}

        for y in range(self.height):
            for x in range(self.width):
                pos = (y, x)
                if self._is_valid_position(pos):
                    paths[pos] = self._calculate_paths(
                        pos, max_distance=max_distance)

        Map.paths = paths

    def calculate_distance(self, pos1, pos2):
        """Calculate the distance between two positions.

        Args:
            pos1: A valid position.
            pos2: A valid position.
        Returns:
            The calculated distance.
        """
        if Map.paths is None:
            self._calculate_all_paths()

        if self._is_valid_position(pos1) and self._is_valid_position(pos2):
            if pos1 == pos2:
                return 0
            else:
                return len(Map.paths[pos1][pos2])
        else:
            return float('inf')


def deterministic_distribution(action1, action2):
    """Calculate the deterministic distribution between two actions.

    Args:
        action1: A action.
        action2: A action.
    Returns:
        A value of 1 or 0.
    """
    if action1 == action2:
        return 1.0
    else:
        return 0.0


def semi_deterministic_distribution(action1, action2):
    """Calculate the semi deterministic distribution between two actions.

    Args:
        action1: A action.
        action2: A action.
    Returns:
        A value of 0.99 or 0.01.
    """
    if action1 == action2:
        return 0.99
    else:
        return 0.01


def gaussian_distribution(pos1, pos2, sd):
    """Calculate the gaussian distribution between two positions.

    Args:
        pos1: A position.
        pos2: A position.
        sd: The standard deviation of the distribution.
    Returns:
        A value of 0.99 or 0.01.
    """
    diff_y = pos2[0] - pos1[0]
    diff_x = pos2[1] - pos1[1]
    return math.exp(-(diff_x**2 + diff_y**2) / (2 * sd**2))


class GameState(object):
    """Game State Class.

    Attributes:
        width: A given width.
        height: A given height.
        walls: Dict of walls positions.
        agent_id: The identifier of the agent for the game state.
        ally_ids: The identifier of the agent_id allies.
        enemy_ids: The identifier of the agent_id enemies.
        agent_maps: The object of Map class for the agent.
        fragile_agents: A dict of fragile agents.
        eater: A boolean value whether the agent is eater.
        iteration: The number of the iteration.
        food_map: The object of Map class for the food.
        sd: The standard deviation.
    """

    def __init__(self, width, height, walls, agent_id=None, ally_ids=[],
                 enemy_ids=[], eater=True, iteration=0):
        """Constructor for GameState class.

        Args:
            width: A given width.
            height: A given height.
            walls: Dict of walls positions.
            agent_id: The identifier of the agent for the game state.
            ally_ids: The identifier of the agent_id allies.
            enemy_ids: The identifier of the agent_id enemies.
            eater: A boolean value whether the agent is eater.
            iteration: The number of the iteration.
        """
        self.width = width
        self.height = height
        self.walls = walls

        self.agent_id = agent_id
        self.ally_ids = ally_ids
        self.enemy_ids = enemy_ids

        self.agent_maps = {}
        for id_ in [self.agent_id] + self.ally_ids + self.enemy_ids:
            self.agent_maps[id_] = Map(width, height, walls)

        self.fragile_agents = {}
        for id_ in [self.agent_id] + self.ally_ids + self.enemy_ids:
            self.fragile_agents[id_] = 0.5

        self.eater = eater
        self.iteration = iteration
        self.food_map = None
        self.sd = 0.5

    def __str__(self):
        """Define the behavior for when str is called.

        Returns:
            The key and the value in the itens of agent maps.
        """
        string = []

        for key, value in self.agent_maps.items():
            string.append(str(key))
            string.append(str(value))

        return '\n'.join(string)

    def set_food_positions(self, food_positions):
        """Set the positions of the foods.

        Create a food_map.

        Args:
            food_positions: The positions of the foods.
        """
        if self.food_map is None:
            self.food_map = Map(self.width, self.height, self.walls)

            for x in range(self.width):
                for y in range(self.height):
                    if (y, x) in food_positions:
                        self.food_map[y][x] = 1.0
                    else:
                        self.food_map[y][x] = 0.0

    def set_walls(self, walls):
        """Set the walls in the agent map.

        Args:
            walls: The position of the walls.
        """
        for agent in self.agent_maps:
            if self.agent_maps[agent].walls == []:
                self.agent_maps[agent].walls = walls
                self.agent_maps[agent].normalize()

    def _is_this_agent(self, agent_id):
        """Check if it is agent.

        Args:
            agent_id: The identifier to check if is a agent.
        Retunrs:
            A boolean value that validates if the agent_id is in fact a agent.
        """
        return (agent_id == self.agent_id)

    def _is_ally_agent(self, agent_id):
        """Check if an agent is an ally.

        Args:
            agent_id: The identifier to check its alliance.
        Returns:
            A boolean value that validates if the agent is a ally or not.
        """
        return (agent_id in self.ally_ids)

    def _is_enemy_agent(self, agent_id):
        """Check if an agent is an enemy.

        Args:
            agent_id: The identifier to check its alliance.
        Returns:
            A boolean value that validates if the agent is an enemy or not.
        """
        return (agent_id in self.enemy_ids)

    def _is_eater_agent(self, agent_id):
        """Check if an agent is an eater.

        Args:
            agent_id: The identifier of the agent to chech if is an eater.
        Returns:
            A boolean value that validates if the agent is an eater or not.
        """
        return ((self.eater and (self._is_this_agent(agent_id) or
                                 self._is_ally_agent(agent_id))) or
                (not self.eater and self._is_enemy_agent(agent_id)))

    def observe_agent(self, agent_id, pos):
        """Call observe for the respective position with gaussian distribution.

        Args:
            agent_id: The identifier of the agent.
            pos: The position in the map.
        """
        self.agent_maps[agent_id].observe(pos, gaussian_distribution, self.sd)

    def observe_fragile_agent(self, agent_id, status):
        """Set fragile_agents for the agent_id status.

        Args:
            agent_id: The identifier of the agent.
            status: The status of the fragile agent.
        """
        self.fragile_agents[agent_id] = status

    def get_agent_position(self, agent_id):
        """Get the agent position with maximum probability.

        Args:
            agent_id: The identifier of the agent to get the position.
        Returns:
            The maximum position.
        """
        return self.agent_maps[agent_id].get_maximum_position()

    def get_position(self):
        """Get the agent position.

        Returns:
            The agent position for the respective identifier.
        """
        return self.get_agent_position(self.agent_id)

    def get_ally_positions(self):
        """Get the position of the allies.

        Returns:
            A list of allies positions.
        """
        return [self.get_agent_position(id_) for id_ in self.ally_ids]

    def get_enemy_positions(self):
        """Get the enemies positions.

        Returns:
            A list of enemies positions.
        """
        return [self.get_agent_position(id_) for id_ in self.enemy_ids]

    def get_map(self):
        """Get the map.

        Returns:
            The map of the agent_id.
        """
        return self.agent_maps[self.agent_id]

    def get_fragile_agent(self, agent_id):
        """Get the fragile agents.

        Args:
            agent_id: The identifier of the agent.
        Returns:
            The fragile agent of an agent id.
        """
        return self.fragile_agents[agent_id]

    def predict_agent(self, agent_id, action):
        """Call the predict function for the agent id.

        And predict the food position for each eater.

        Args:
            agent_id: The identifier of the agent.
            action: The respective action.
        """
        self.agent_maps[agent_id].predict(action,
                                          semi_deterministic_distribution)

        # Either the agent and its allies eat or its enemies
        if self._is_eater_agent(agent_id):
            self._predict_food_positions(agent_id)

    def _predict_food_positions(self, agent_id):
        """Predict the food positions for an agent.

        Args:
            agent_id: The identifier of an agent.
        """
        for x in range(self.width):
            for y in range(self.height):
                self.food_map[y][x] = self.food_map[y][x] * (
                    1 - self.agent_maps[agent_id][y][x])

    def calculate_distance(self, point1, point2):
        """Calculate distance between two points.

        Args:
            point1: The first point.
            point2: The second point.
        Returns:
            The distance.
        """
        return self.agent_maps[self.agent_id].calculate_distance(
            point1, point2)

    def get_food_distance(self):
        """Get the food distance.

        Get the minimum distance for the closest food.
        """
        position = self.get_agent_position(self.agent_id)
        food_prob_threshold = self.food_map.max() / 2.0
        min_dist = float('inf')

        for x in range(self.width):
            for y in range(self.height):
                if self.food_map[y][x] > food_prob_threshold:
                    dist = self.calculate_distance(position, (y, x))

                    if dist < min_dist:
                        min_dist = dist

        return min_dist

    def get_distance_to_agent(self, agent_id):
        """Get distance to an agent.

        Args:
            agent_id: The identifier of the other agent.
        Returns:
            The distance between two agents positions
        """
        my_position = self.get_agent_position(self.agent_id)
        agent_position = self.get_agent_position(agent_id)
        return self.calculate_distance(my_position, agent_position)

    def get_closest_ally(self, state):
        """Get the identifier of the closest ally.

        Args:
            state: A given state.
        Returns:
            The identifier of the closest ally.
        """
        distance = float('inf')
        closest_ally = self.ally_ids[0]

        for ally_id in self.ally_ids:
            ally_distance = self.get_distance_to_agent(ally_id)

            if ally_distance < distance:
                distance = ally_distance
                closest_ally = ally_id

        return closest_ally

    def get_closest_enemy(self, state):
        """Get the identifier of the closest enemy.

        Args:
            A given state.
        Returns:
            The identifier of the closest enemy.
        """
        distance = float('inf')
        closest_enemy = self.enemy_ids[0]

        for enemy_id in self.enemy_ids:
            enemy_distance = self.get_distance_to_agent(enemy_id)

            if enemy_distance < distance:
                distance = enemy_distance
                closest_enemy = enemy_id

        return closest_enemy


if __name__ == '__main__':
    # X X _ _ _
    # . X o _ _
    # _ X _ _ _
    # _ X _ _ _
    # _ _ _ _ _

    walls = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
    game_map = Map(10, 5, walls)
    action_to_pos = {
        'North': (1, 0),
        'South': (-1, 0),
        'East': (0, 1),
        'West': (0, -1),
        'Stop': (0, 0),
    }

    initial_pos = (1, 0)
    final_pos = (1, 2)

    print game_map
    positions = [(y, x) for y in range(game_map.height)
                 for x in range(game_map.width)]
    for pos1 in positions:
        for pos2 in positions:
            print pos1, '->', pos2, game_map.calculate_distance(pos1, pos2)

    print game_map.calculate_distance((1, 0), (1, 0))
