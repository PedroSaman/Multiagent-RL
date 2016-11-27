#!/usr/bin/env python
#  -*- coding: utf-8 -*-

"""Code for communication between controller and simulator.

Attributes:
    DEFAULT_TCP_PORT: The server port, 5555.
    DEFAULT_CLIENT_ADDRESS: The client address, 'localhost'.
    ACK_MSG = 'Acknowledgment'.
    ACTION_MSG = 'Action'.
    BEHAVIOR_COUNT_MSG = 'BehaviorCount'.
    POLICY_MSG = 'Policy'.
    REQUEST_REGISTER_MSG = 'RequestRegister'.
    REQUEST_BEHAVIOR_COUNT_MSG = 'RequestBehaviorCount'.
    REQUEST_GAME_START_MSG = 'RequestGameStart'.
    REQUEST_INIT_MSG = 'RequestInitialization'.
    REQUEST_POLICY_MSG = 'RequestPolicy'.
    STATE_MSG = 'State'.
"""

import pickle
import zmq

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"

# Default settings.
DEFAULT_CLIENT_ADDRESS = 'localhost'
DEFAULT_TCP_PORT = 5555


###############################################################################
#                                Messengers                                   #
###############################################################################
class ZMQMessengerBase(object):
    """Base class for simple communicating messages through zmq.

    Attributes:
        socket: The socket for the communication.
    """

    def __init__(self, context, socket_type):
        """Constructor for ZMQMessengerBase class.

        Args:
            context: The class constrouctor of ZMQ
            socket_type: The type of the commmunication socket.
        """
        self.socket = context.socket(socket_type)

    def receive(self):
        """Request a message and returns it.

        Returns:
            The requested message.
        """
        return pickle.loads(self.socket.recv())

    def send(self, msg):
        """Send the given message.

        Args:
            msg: The given message.
        """
        self.socket.send(pickle.dumps(msg))


class ZMQServer(ZMQMessengerBase):
    """Inter-process communication server."""

    def __init__(self, context, binding):
        """Constructor for the ZMQServer class.

        Extends the ZMQMessengerBase class.

        Args:
            context: The class constrouctor of ZMQ
            binding: The TCP binding to the server.
        """
        super(ZMQServer, self).__init__(context, socket_type=zmq.REP)
        self.socket.bind(binding)
        # http://zguide.zeromq.org/page:all#advanced-request-reply
        # The REP socket reads and saves all identity frames up to and
        # including the empty delimiter, then passes the following frame or
        # frames to the caller. REP sockets are synchronous and talk to one
        # peer at a time. If you connect a REP socket to multiple peers,
        # requests are read from peers in fair fashion, and replies are always
        # sent to the same peer that made the last request.


class ZMQClient(ZMQMessengerBase):
    """Inter-process communication server."""

    def __init__(self, context, connection):
        """Constructor for the ZMQClient class.

        Args:
            context: The class constrouctor of ZMQ
            connection: The TCP connection client server.
        """
        super(ZMQClient, self).__init__(context, socket_type=zmq.REQ)
        self.socket.connect(connection)
        # The REQ socket sends, to the network, an empty delimiter frame in
        # front of the message data. REQ sockets are synchronous. REQ sockets
        # always send one request and then wait for one reply. REQ sockets talk
        # to one peer at a time. If you connect a REQ socket to multiple peers,
        # requests are distributed to and replies expected from each peer one
        # turn at a time.


class TCPServer(ZMQServer):
    """Inter-process communication server."""

    def __init__(self, address=DEFAULT_CLIENT_ADDRESS, port=DEFAULT_TCP_PORT):
        """Constructor for the TCPServer class.

        Extends the ZMQServer base class constructor.

        Args:
            address: The address of the client, DEFAULT_CLIENT_ADDRESS.
            port: The port of the server, DEFAULT_TCP_PORT
        """
        binding = 'tcp://*:{}'.format(port)
        super(TCPServer, self).__init__(zmq.Context(), binding)


class TCPClient(ZMQClient):
    """Inter-process communication client."""

    def __init__(self, address=DEFAULT_CLIENT_ADDRESS, port=DEFAULT_TCP_PORT):
        """Constructor for the TCPClient class.

        Extends the ZMQClient base class constructor.

        Args:
            address: The address of the client, DEFAULT_CLIENT_ADDRESS.
            port: The port of the server, DEFAULT_TCP_PORT
        """
        connection = 'tcp://{}:{}'.format(address, port)
        super(TCPClient, self).__init__(zmq.Context(), connection)


###############################################################################
#                                  Messages                                   #
###############################################################################

# Message types
ACK_MSG = 'Acknowledgment'
ACTION_MSG = 'Action'
BEHAVIOR_COUNT_MSG = 'BehaviorCount'
MSE_COUNT_MSG = 'MSECount'
POLICY_MSG = 'Policy'
PROBABILITY_MAP_MSG = 'ProbabilityMap'
PROBABILITY_MAP_MSE_MSG = 'ProbabilityMapMSE'
REQUEST_REGISTER_MSG = 'RequestRegister'
REQUEST_BEHAVIOR_COUNT_MSG = 'RequestBehaviorCount'
REQUEST_GAME_START_MSG = 'RequestGameStart'
REQUEST_MSE_COUNT_MSG = 'RequestMSECount'
REQUEST_INIT_MSG = 'RequestInitialization'
REQUEST_PM_MSG = 'RequestProbabilityMap'
REQUEST_POLICY_MSG = 'RequestPolicy'
REQUEST_GOAL_MSG = 'RequestGoal'
STATE_MSG = 'State'
GOAL_MSG = 'Goal'


class BaseMessage(object):
    """Base class for Pac-Man messages.

    Attributes:
        __type: The message type.
    """

    def __init__(self, msg_type=None):
        """Constructor for BaseMessage.

        Set the __type to the msg_type.

        Args:
            msg_type: The message type.
        """
        self.__type = msg_type

    @property
    def type(self):
        """Get the message type.

        Returns:
            __type: The message type.
        """
        return self.__type


class AckMessage(BaseMessage):
    """Simple acknowledgment message."""

    def __init__(self):
        """Extend the BaseMessage constructor."""
        super(AckMessage, self).__init__(msg_type=ACK_MSG)


class ActionMessage(BaseMessage):
    """Carries the information of an agent's action.

    Attributes:
        agent_id: The identifier of an agent.
        action: The respective action.
    """

    def __init__(self, agent_id=None, action=None):
        """Constructor for ActionMessage class.

        Extends BaseMessage constructer

        Args:
            agent_id: The identifier of an agent.
            action: The respective action.
        """
        super(ActionMessage, self).__init__(msg_type=ACTION_MSG)

        self.agent_id = agent_id
        self.action = action


class BehaviorCountMessage(BaseMessage):
    """Carries the requested behavior count.

    Attributes:
        count: The behavior count.
    """

    def __init__(self, count=None):
        """Constructor for BehaviorCountMessage.

        Extends BaseMessage.

        Args:
            count: The behavior count.
        """
        super(BehaviorCountMessage, self).__init__(msg_type=BEHAVIOR_COUNT_MSG)

        self.count = count


class MSECountMessage(BaseMessage):
    """Carries the requested mean square error count.

    Attributes:
        count: The behavior count.
    """

    def __init__(self, mse=None):
        """Constructor for BehaviorCountMessage.

        Extends BaseMessage.

        Args:
            count: The behavior count.
        """
        super(MSECountMessage, self).__init__(msg_type=MSE_COUNT_MSG)

        self.mse = mse



class PolicyMessage(BaseMessage):
    """Carries the requested policy.

    Attributes:
        agent_id: The identifer on an agent.
        policy: The agent's policy.
    """

    def __init__(self, agent_id=None, policy=None):
        """Constructor for PolicyMessage class.

        Extends BaseMessage Constructor.

        Args:
            agent_id: The identifer of an agent.
            policy: The agent's policy.
        """
        super(PolicyMessage, self).__init__(msg_type=POLICY_MSG)

        self.agent_id = agent_id
        self.policy = policy


class ProbabilityMapMessage(BaseMessage):
    """Base Message for the probability map.

    Attributes:
        agent_id: The identifier of the agent.
        pm: The probability map.
    """

    def __init__(self, agent_id=None, probability_map=None):
        """Constructor for the ProbabilityMapMessage class.

        Args:
            agent_id: The identifier of the agent.
            pm: The probability map.
        """
        super(ProbabilityMapMessage,
              self).__init__(msg_type=PROBABILITY_MAP_MSG)

        self.agent_id = agent_id
        self.pm = probability_map

class ProbabilityMapMSEMessage(BaseMessage):
    """Base Message for the probability map.

    Attributes:
        agent_id: The identifier of the agent.
        pm: The probability map.
    """

    def __init__(self, agent_id=None, probability_map=None):
        """Constructor for the ProbabilityMapMessage class.

        Args:
            agent_id: The identifier of the agent.
            pm: The probability map.
        """
        super(ProbabilityMapMSEMessage,
              self).__init__(msg_type=PROBABILITY_MAP_MSE_MSG)
              
        self.agent_id = agent_id
        self.pm = probability_map
        
class GoalMessage(BaseMessage):
    """Base Message for the Behavior communication.

    Attributes:
        agent_id: The identifier of the agent.
        goal: A int value that determine wheater the agent is Seeking, Pursuing
            or Fleeing.
    """

    def __init__(self, agent_id=None, goal=None):
        """Constructor for the GoalMessage class.

        Args:
            agent_id: The identifier of the agent.
            goal: A int value that determine wheater the agent is Seeking,
                Pursuing or Fleeing.
        """
        super(GoalMessage,
              self).__init__(msg_type=GOAL_MSG)

        self.agent_id = agent_id
        self.goal = goal


class RequestMessage(BaseMessage):
    """Requests some information."""

    def __init__(self, msg_type):
        """Constructor for RequestMessage class.

        Extends BaseMessage.

        Args:
            msg_type: The type of the message.
        """
        super(RequestMessage, self).__init__(msg_type=msg_type)


class RequestInitializationMessage(RequestMessage):
    """Requests that the identified agent be REQUEST_INITIALIZED.

    Attributes:
        agent_id: The identifer of the agent to be REQUEST_INITIALIZED.
    """

    def __init__(self, agent_id=None):
        """The constructor of RequestInitializationMessage.

        Extends RequestMessage.

        Args:
            agent_id: The identifer of an agent.
        """
        super(RequestInitializationMessage,
              self).__init__(msg_type=REQUEST_INIT_MSG)

        self.agent_id = agent_id


class RequestBehaviorCountMessage(RequestMessage):
    """Requests the identified agent's RequestMessage count information.

    Attributes:
        agent_id: The identifer of an agent.
    """

    def __init__(self, agent_id=None):
        """The constructor of RequestBehaviorCountMessage.

        Extends RequestMessage.

        Args:
            agent_id: The identifer of an agent.
        """
        super(RequestBehaviorCountMessage,
              self).__init__(msg_type=REQUEST_BEHAVIOR_COUNT_MSG)

        self.agent_id = agent_id

class RequestMSECountMessage(RequestMessage):
    """Requests the identified agent's RequestMessage count information.

    Attributes:
        agent_id: The identifer of an agent.
    """

    def __init__(self):
        """The constructor of RequestBehaviorCountMessage.

        Extends RequestMessage.

        Args:
            agent_id: The identifer of an agent.
        """
        super(RequestMSECountMessage,
              self).__init__(msg_type=REQUEST_MSE_COUNT_MSG)

class RequestGameStartMessage(RequestMessage):
    """Requests that a game be started for the identified agent.

    Attributes:
        agent_id: The identifier of an agent.
        map_width: The map width.
        map_height: The map height.
    """

    def __init__(self, agent_id=None, map_width=None, map_height=None):
        """The constructor of RequestGameStartMessage.

        Extends RequestMessage.

        Args:
            agent_id: The identifer of an agent.
            map_width: The map width.
            map_height: The map height.
        """
        super(RequestGameStartMessage,
              self).__init__(msg_type=REQUEST_GAME_START_MSG)

        self.agent_id = agent_id
        self.map_width = map_width
        self.map_height = map_height


class RequestRegisterMessage(RequestMessage):
    """Requests that the identified agent be registered.

    Requests that the identified (and associated information)
    agent be registered.

    Attributes:
        agent_id: The identifer of an agent.
        agent_team: The agent team.
        agent_class: The agent class.
    """

    def __init__(self, agent_id=None, agent_team=None, agent_class=None):
        """The constructor of RequestRegisterMessage.

        Extends RequestMessage.

        Args:
            agent_id: The identifer of an agent, default is None.
            agent_team: The agent team, default is None.
            agent_class: The agent class, default is None.
        """
        super(RequestRegisterMessage,
              self).__init__(msg_type=REQUEST_REGISTER_MSG)

        self.agent_id = agent_id
        self.agent_team = agent_team
        self.agent_class = agent_class


class RequestPolicyMessage(RequestMessage):
    """Requests the identified agent's policy.

    Attributes:
        agent_id: The identifer of an agent.
    """

    def __init__(self, agent_id=None):
        """Constructor for RequestPolicyMessage class.

        Extends RequestMessage.

        Args:
            agent_id: The identifier of an agent.
        """
        super(RequestPolicyMessage, self).__init__(msg_type=REQUEST_POLICY_MSG)

        self.agent_id = agent_id


class RequestProbabilityMapMessage(RequestMessage):
    """Request the probability map.

    Attributes:
        agent_id: The identifier of an agent
    """

    def __init__(self, agent_id=None):
        """Constructor for the RequestProbabilityMapMessage.

        Args:
            agent_id: The identifier of an agent.
        """
        super(RequestProbabilityMapMessage,
              self).__init__(msg_type=REQUEST_PM_MSG)

        self.agent_id = agent_id


class RequestGoalMessage(RequestMessage):
    """Base Request Message for the Behavior communication.

    Attributes:
        agent_id: The identifier of the agent.
        goal: A int value that determine wheater the agent is Seeking, Pursuing
            or Fleeing.
    """

    def __init__(self, agent_id=None):
        """Constructor for the GoalMessage class.

        Args:
            agent_id: The identifier of the agent.
            goal: A int value that determine wheater the agent is Seeking,
                Pursuing or Fleeing.
        """
        super(RequestGoalMessage,
              self).__init__(msg_type=REQUEST_GOAL_MSG)

        self.agent_id = agent_id


class StateMessage(BaseMessage):
    """Carries the information of a game state.

    Attributes:
        agent_id: The identifier of the agent.
        agent_positions: The positions of the agents.
        food_positions: The positions of the foods.
        fragile_agents: Whethe the agent is fragile.
        wall_positions: The positions of the walls.
        legal_actions: A list of legal actions.
        reward: The expected reward.
        executed_action: The executed action.
        test_mode: Whether is test mode or not.
    """

    def __init__(self, agent_id=None, agent_positions=None,
                 food_positions=None, fragile_agents=None, wall_positions=None,
                 legal_actions=None, reward=None, executed_action=None,
                 test_mode=None):
        """The constructor for StateMessage Class.

        Args:
            agent_id: The identifier of the agent.
            agent_positions: The positions of the agents.
            food_positions: The positions of the foods.
            fragile_agents: Whethe the agent is fragile.
            wall_positions: The positions of the walls.
            legal_actions: A list of legal actions.
            reward: The expected reward.
            executed_action: The executed action.
            test_mode: Whether is test mode or not.
        """
        super(StateMessage, self).__init__(msg_type=STATE_MSG)

        self.agent_id = agent_id
        self.agent_positions = agent_positions
        self.food_positions = food_positions
        self.fragile_agents = fragile_agents
        self.wall_positions = wall_positions
        self.legal_actions = legal_actions
        self.reward = reward
        self.executed_action = executed_action
        self.test_mode = test_mode
