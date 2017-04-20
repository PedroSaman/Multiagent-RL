#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routes messages between server and agents."""

from __future__ import division
from math import log1p

import cliparser
import communication as comm
from state import GameState, Map


__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


def log(msg):
    """Log on the screen the controller message.

    Args:
        msg: The message to be logged.
    """
    print '[Controller] {}'.format(msg)


class Controller(object):
    """Menage the messages client/server and server/client for the agents.

    Attributes:
        agents: A dictionary of agents.
        agent_classes: A dictionary of agents classes.
        agent_teams: A dictionary of agents teams.
        game_states: A dictionary of game states.
        game_number: A dictionary of game numbers.
        server: A ZMQMessengerBase.
    """

    def __init__(self, server):
        """Constructor for the Controller Class.

        Set all the attributes to empty dictionaries, exept server there is set
        to the server parameter. Log 'Ready'.

        Args:
            server: A ZMQMessengerBase.
        Raises:
            ValueError: Invalid server.
        """
        if not isinstance(server, comm.ZMQMessengerBase):
            raise ValueError('Invalid server')

        self.agents = {}
        self.agent_classes = {}
        self.agent_teams = {}
        self.game_states = {}
        self.game_number = {}
        self.server = server
        self.ghostId = []
        self.probability_map = []
        self.realPositions = 0

        self.numInstances = 0
        self.instanceError = 0

        self.mse = 0.0
        self.numInstancesArray = []
        self.instanceErrorsArray = []

        self.learnTripples = []

        log('Ready')

    def __choose_action__(self, state):
        """Choose action.

        Update agent state and choose an action.

        Args:
            state: The agent state.
        Returns:
            agent_action: The action choosen.
        """
        # Update agent state.
        for id_, pos in state.agent_positions.items():
            self.game_states[state.agent_id].observe_agent(id_, pos)

        for id_, status in state.fragile_agents.items():
            self.game_states[state.agent_id].observe_fragile_agent(id_, status)

        # Choose action
        agent_state = self.game_states[state.agent_id]
        choose_action = self.agents[state.agent_id].choose_action
        agent_action = choose_action(agent_state, state.executed_action,
                                     state.reward, state.legal_actions,
                                     state.test_mode)

        for id_ in self.game_states:
            agent_state.predict_agent(id_, agent_action)

        return agent_action

    def __get_allies__(self, agent_id):
        """Get all alies of an agent.

        Args:
            agent_id: The identifier of the agent.
        Returns:
            A list of all the allies of the agent related to the agent_id.
        """
        return [id_ for id_ in self.agent_teams
                if self.agent_teams[id_] == self.agent_teams[agent_id] and
                id_ != agent_id]

    def __get_enemies__(self, agent_id):
        """Get all the enemies of an agent.

        Args:
            agent_id: The identifier of the agent.
        Returns:
            A list of all the enemies of the agent related to the agent_id.
        """
        return [id_ for id_ in self.agent_teams
                if self.agent_teams[id_] != self.agent_teams[agent_id] and
                id_ != agent_id]

    def __initialize_agent__(self, msg):
        """Initialize an agent.

        Set the agent id, it's allies and enemies, the respective game number
        to 0 and it's agents. Log the initalized agent, set a reply_msg as a
        simple acknowledgment message and send it to the server.

        Args:
            msg: A message of comm.REQUEST_INIT_MSG type.
        """
        agent_id = msg.agent_id
        ally_ids = self.__get_allies__(agent_id)
        enemy_ids = self.__get_enemies__(agent_id)

        if agent_id in self.agents:
            del self.agents[agent_id]

        self.game_number[agent_id] = 0
        self.agents[agent_id] = self.agent_classes[agent_id](agent_id,
                                                             ally_ids,
                                                             enemy_ids)
        log('Initialized {} #{}'.format(self.agent_teams[agent_id], agent_id))

        reply_msg = comm.AckMessage()
        self.server.send(reply_msg)

    def __register_agent__(self, msg):
        """Register an agent.

        Set the agent classes and team, log the registered agent, set a
        reply_msg as a simple acknowledgment message and send it to the server.

        Args:
            msg: A message of comm.REQUEST_REGISTER_MSG
        """
        self.agent_classes[msg.agent_id] = msg.agent_class
        self.agent_teams[msg.agent_id] = msg.agent_team

        log('Registered {} #{} ({})'.format(msg.agent_team, msg.agent_id,
                                            msg.agent_class.__name__))

        reply_msg = comm.AckMessage()
        self.server.send(reply_msg)

    def __request_behavior_count__(self, agent_id):
        """Request Behavior Count.

        Set the behavior count, create a reply_msg as a behavior count message
        and send it to the server. Reset the behavior count.

        Args:
            agent_id: The identifier of an agent.
        """
        count = self.agents[agent_id].behavior_count
        reply_msg = comm.BehaviorCountMessage(count)
        self.server.send(reply_msg)

        self.agents[agent_id].reset_behavior_count()

    def __reset_mse_count__(self):
        """Reset MSE Count."""
        self.numInstances = 0
        self.instanceError = 0
        self.mse = 0

        self.numInstancesArray = []
        self.instanceErrorsArray = []

    def __request_mse_count__(self, msg):
        """Request Behavior Count.

        Set the behavior count, create a reply_msg as a behavior count message
        and send it to the server. Reset the behavior count.

        Args:
            agent_id: The identifier of an agent.
        """
        mse = self.instanceError/self.numInstances
        reply_msg = comm.MSECountMessage(mse)
        self.server.send(reply_msg)

        self.__reset_mse_count__()

    def __send_agent_action__(self, msg):
        """Send the action of the agent.

        Atribute the return value of __choose_action__ for the msg parameter,
        on agent_action and send it to the server as a comm.ActionMessage.

        Args:
            msg: A message of comm.STATE_MSG type.
        Returns:
            agent_action: The action sent.
        """
        game_state = self.game_states[msg.agent_id]
        game_state.set_walls(msg.wall_positions)
        game_state.set_food_positions(msg.food_positions)

        self.realPositions = msg.realPosition
        agent_action = self.__choose_action__(msg)
        reply_msg = comm.ActionMessage(agent_id=msg.agent_id,
                                       action=agent_action)
        self.server.send(reply_msg)

        return agent_action

    def __send_policy_request__(self, msg):
        """Send policy request.

        Create a reply_msg as a policy message and send it to the server.

        Args:
            msg: A message of type comm.REQUEST_POLICY_MSG.
        """
        policy = self.agents[msg.agent_id].get_policy()
        reply_message = comm.PolicyMessage(agent_id=msg.agent_id,
                                           policy=policy)
        self.server.send(reply_message)

    def __set_agent_policy__(self, msg):
        """Set an agent policy.

        Set the policy for the msg agent id, and sand to the server a simple
        acknowledgment message.

        Args:
            msg: A message of type comm.POLICY_MSG.
        """
        self.agents[msg.agent_id].set_policy(msg.policy)
        self.server.send(comm.AckMessage())

    def __start_game_for_agent__(self, msg):
        """Start Game for an Agent.

        Call __get_allies__ and __get_enemies__, initialize a Game State for
        a agent_id from message. Send a acknowledgment message to the server.
        Log the Start Game for agent number message.

        Args:
            msg: A message of type comm.REQUEST_GAME_START_MSG.
        """
        ally_ids = self.__get_allies__(msg.agent_id)
        enemy_ids = self.__get_enemies__(msg.agent_id)

        eater = (self.agent_teams[msg.agent_id] == 'pacman')

        if(self.agent_teams[msg.agent_id] != 'pacman'):
            self.numInstancesArray.append(0)
            self.instanceErrorsArray.append(0)

        if msg.agent_id in self.game_states:
            del self.game_states[msg.agent_id]

        iteration = self.game_number[msg.agent_id]
        self.game_states[msg.agent_id] = GameState(width=msg.map_width,
                                                   height=msg.map_height,
                                                   walls=[],
                                                   agent_id=msg.agent_id,
                                                   ally_ids=ally_ids,
                                                   enemy_ids=enemy_ids,
                                                   eater=eater,
                                                   iteration=iteration)

        reply_msg = comm.AckMessage()
        self.server.send(reply_msg)
        log('Start game for {} #{}'.format(self.agent_teams[msg.agent_id],
                                           msg.agent_id))

    def __request_learn__(self, msg):
        """Request the goal.

        Args:
            msg:  A message of type GOAL_MSG
        """
        ident = msg.agent_id
        pb = self.agents[ident].previous_behavior
        reward = msg.reward
        state = self.game_states[ident]

        reply_msg = comm.SharedLearnMessage(agent_id=ident,
                                            previous_behavior=pb,
                                            reward=reward, state=state)

        self.server.send(reply_msg)

    def __request_probability_map__(self, msg):
        """Request the probability maps."""
        ident = msg.agent_id
        probability_map = self.game_states[ident].agent_maps[0]

        reply_msg = comm.ProbabilityMapMessage(agent_id=ident,
                                               probability_map=probability_map)
        self.server.send(reply_msg)

    def __set_agent_pm_mse__(self, msg):
        """Set the probability map back to the agents."""
        self.ghostId.append(msg.agent_id)
        self.probability_map.append(msg.pm)
        maxValue = 0
        maxValueX = 0
        maxValueY = 0
        aux_sum = 0.000000000000

        pacman = self.__get_enemies__(msg.agent_id)
        # print("Mapa recebido do agente {}".format(msg.agent_id))
        # print(msg.pm)
        ident = msg.agent_id

        if len(self.probability_map) == len(self.__get_allies__(ident))+1:
            width = self.probability_map[0].width
            height = self.probability_map[0].height
            walls = self.probability_map[0].walls
            sumOfValues = 0.0
            newPM = Map(width, height, walls)

            # Populate new matrix
            for x in range(height):
                for y in range(width):
                    for probMap in self.probability_map:
                        if not probMap._is_wall((x, y)):
                            # print probMap[x][y]
                            newPM[x][y] = log1p(newPM[x][y]) + log1p(probMap[x][y])
                    if not newPM._is_wall((x, y)):
                        # print newPM[x][y]
                        sumOfValues = sumOfValues + newPM[x][y]
            # Normalize it
            for x in range(height):
                for y in range(width):
                    if not newPM._is_wall((x, y)):
                        newPM[x][y] = newPM[x][y]/sumOfValues
                        aux_sum = aux_sum + newPM[x][y]
                        if newPM[x][y] > maxValue:
                            maxValue = newPM[x][y]
                            # print maxValue
                            maxValueX = x
                            maxValueY = y

            # print aux_sum
            # assert aux_sum == 1.0
            # print(">>>>>>>>>>>>>>>>>>>>>>>>")
            # print("Novo mapa de probabilidade: ")
            # print(newPM)

            # Get the max value

            self.numInstances += 1
            pacman_pos = self.realPositions
            # print("\nPacman Position: {}".format(pacman_pos))
            # print("Pacman Estimate Position: {}".format((maxValueX,
            #                                              maxValueY)))
            # print("Previous instanceError: {}".format(self.instanceError))

            self.instanceError += (abs(maxValueX - pacman_pos[0]) + abs(maxValueY - pacman_pos[1]))

            # print("New Instance Error: {}".format(abs(maxValueX - pacman_pos[0]) +
            #     abs(maxValueY - pacman_pos[1])))

            # print ">>>>>>>>>>>>>>>>>>>>>>>>"
            for agent in self.ghostId:
                self.game_states[agent].agent_maps[pacman[0]] = newPM
                # print("Mapa de probabilidade do agente {}".format(agent))
                # print self.game_states[agent].agent_maps[pacman[0]]


            self.ghostId = []
            self.probability_map = []
            self.server.send(comm.AckMessage())
        else:
            self.server.send(comm.AckMessage())

    def __set_agent_pm__(self, msg):
        """Set the probability map back to the agents."""
        self.ghostId.append(msg.agent_id)
        self.probability_map.append(msg.pm)

        pacman = self.__get_enemies__(msg.agent_id)
        # print("Mapa recebido do agente {}".format(msg.agent_id))
        # print(msg.pm)
        ident = msg.agent_id

        if len(self.probability_map) == len(self.__get_allies__(ident))+1:
            width = self.probability_map[0].width
            height = self.probability_map[0].height
            walls = self.probability_map[0].walls
            sumOfValues = 0.0
            newPM = Map(width, height, walls)

            # Populate new matrix
            for x in range(height):
                for y in range(width):
                    for probMap in self.probability_map:
                        if not probMap._is_wall((x, y)):
                            newPM[x][y] = log1p(newPM[x][y]) + log1p(probMap[x][y])

                    if not newPM._is_wall((x, y)):
                        # print newPM[x][y]
                        sumOfValues = sumOfValues + newPM[x][y]

            # Normalize it
            for x in range(height):
                for y in range(width):
                    if not newPM._is_wall((x, y)):
                        newPM[x][y] = newPM[x][y]/sumOfValues

            # print("Novo mapa de probabilidade: ")
            # print(newPM)

            for agent in self.ghostId:
                self.game_states[agent].agent_maps[pacman[0]] = newPM
                # print("Mapa de probabilidade do agente {}".format(agent))
                # print self.game_states[agent].agent_maps[pacman[0]]

            self.ghostId = []
            self.probability_map = []
            self.server.send(comm.AckMessage())
        else:
            self.server.send(comm.AckMessage())

    def __share_learn__(self, msg):
        """Set the agent new goal.

        Args:
            msg: A message of type GOAL_MSG
        """
        if (self.game_states[msg.agent_id].iteration % 5) == 0:
            self.ghostId.append(msg.agent_id)
            ps = self.agents[msg.agent_id].learning.previous_state
            self.learnTripples.append((msg.agent_id, ps, msg.state,
                                       msg.previous_behavior,
                                       msg.reward))

            num = len(self.__get_allies__(msg.agent_id))
            if len(self.learnTripples) == num:
                for agent in self.ghostId:
                    for tripple in self.learnTripples:
                        if agent != tripple[0]:
                            ps = tripple[1]
                            state = tripple[2]
                            pb = tripple[3]
                            reward = tripple[4]
                            # print "\nAgent receiving info: {}".format(agent)
                            # print "Information Recived from agent: {}"
                            # .format(tripple[0])
                            # print "Previous State: {}"
                            # .format(tripple[1].get_position())
                            # print "State: {}"
                            # .format(tripple[2].get_position())
                            # print "Behavior: {}".format(tripple[3])
                            # print "Reward: {}".format(tripple[4])
                            self.agents[agent].learning.learnFromOther(state,
                                                                       ps, pb,
                                                                       reward)

                self.ghostId = []
                self.learnTripples = []

                self.server.send(comm.AckMessage())
            else:
                self.server.send(comm.AckMessage())
        else:
            self.server.send(comm.AckMessage())

    def __set_mse__(self, msg):
        """..."""
        pacman = self.__get_enemies__(msg.agent_id)
        pacman_pos = self.realPositions
        # print pacman_pos
        pMap = self.game_states[msg.agent_id].agent_maps[pacman[0]]
        # print pMap

        maxValue = 0
        maxValueX = 0
        maxValueY = 0

        width = pMap.width
        height = pMap.height

        for x in range(height):
            for y in range(width):
                if not pMap._is_wall((x, y)):
                    if pMap[x][y] > maxValue:
                        maxValue = pMap[x][y]
                        maxValueX = x
                        maxValueY = y

        coord = (maxValueX, maxValueY)
        distance = (abs(maxValueX - pacman_pos[0]) + abs(maxValueY - pacman_pos[1]))

        self.numInstancesArray[msg.agent_id-1] += 1
        self.instanceErrorsArray[msg.agent_id-1] += distance
        # print("Agente: {}".format(msg.agent_id-1))
        # print("\nNumero instancia: {}"
        #       .format(self.numInstancesArray[msg.agent_id-1]))
        # print("Posicao pacman: {}".format(pacman_pos))
        # print("Posicao estimada: {}".format(coord))
        # print("Erro instancia: {}".format(distance))
        # print("Erro: {}".format(self.instanceErrorsArray[msg.agent_id-1]))

        self.server.send(comm.AckMessage())

    def __request_mse__(self, msg):
        """..."""
        # print("\nRealizado request")
        agent_id = msg.agent_id
        print("Agent id: {}".format(agent_id-1))
        error = self.instanceErrorsArray[agent_id-1]
        instances = self.numInstancesArray[agent_id-1]
        mse = error/instances
        print("MSE do jogo: {}".format(mse))
        if(mse > self.mse):
            self.mse = mse

        if(agent_id == len(self.agents)-1):
            print("Maior MSE: {}".format(self.mse))
            print(">>>>>>>>>>>>>>>>>>>")
            reply_msg = comm.MSECountMessage(self.mse)
            self.server.send(reply_msg)
            self.__reset_mse_count__()
        else:
            reply_msg = comm.MSECountMessage(0)
            self.server.send(reply_msg)


    def __process__(self, msg):
        """Process the message type.

        Execute correct function for the respective message type.

        Args:
            msg: A message to be processed.
        """
        if msg.type == comm.STATE_MSG:
            self.last_action = self.__send_agent_action__(msg)
        elif msg.type == comm.REQUEST_INIT_MSG:
            self.__initialize_agent__(msg)
        elif msg.type == comm.REQUEST_GAME_START_MSG:
            self.__start_game_for_agent__(msg)
            self.game_number[msg.agent_id] += 1
        elif msg.type == comm.REQUEST_REGISTER_MSG:
            self.__register_agent__(msg)
        elif msg.type == comm.REQUEST_BEHAVIOR_COUNT_MSG:
            self.__request_behavior_count__(msg.agent_id)
        elif msg.type == comm.REQUEST_POLICY_MSG:
            self.__send_policy_request__(msg)
        elif msg.type == comm.POLICY_MSG:
            self.__set_agent_policy__(msg)
        elif msg.type == comm.PROBABILITY_MAP_MSG:
            self.__set_agent_pm__(msg)
        elif msg.type == comm.REQUEST_PM_MSG:
            self.__request_probability_map__(msg)
        elif msg.type == comm.REQUEST_LEARN_MSG:
            self.__request_learn__(msg)
        elif msg.type == comm.SHARE_LEARN_MSG:
            self.__share_learn__(msg)
        elif msg.type == comm.REQUEST_MSE_COUNT_MSG:
            self.__request_mse_count__(msg)
        elif msg.type == comm.PROBABILITY_MAP_MSE_MSG:
            self.__set_agent_pm_mse__(msg)
        elif msg.type == comm.REQUEST_MSE_MSG:
            self.__request_mse__(msg)
        elif msg.type == comm.MSE_MSG:
            self.__set_mse__(msg)

    def run(self):
        """Run the Controller.

        Log 'Now running', set last_action to 'Stop'. While True, request a
        message from the server and process it.
        """
        log('Now running')

        self.last_action = 'Stop'

        while True:
            msg = self.server.receive()
            self.__process__(msg)

if __name__ == '__main__':
    try:
        controller = cliparser.get_Controller()
        controller.run()
    except KeyboardInterrupt:
        print '\n\nInterrupted execution\n'
