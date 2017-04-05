#!/usr/bin/env python
#  -*- coding: utf-8 -*-

"""Parses CLI arguments to provide Adapter and Controller instances."""


from argparse import ArgumentParser

from adapter import (Adapter, DEFAULT_GHOST_AGENT, DEFAULT_LAYOUT,
                     DEFAULT_NUMBER_OF_GHOSTS, DEFAULT_NUMBER_OF_LEARNING_RUNS,
                     DEFAULT_NUMBER_OF_TEST_RUNS, DEFAULT_OUTPUT_FILE,
                     DEFAULT_PACMAN_AGENT, DEFAULT_COMM, DEFAULT_MSE)
from agents import DEFAULT_NOISE
from controller import Controller
from communication import (TCPClient, TCPServer, DEFAULT_CLIENT_ADDRESS,
                           DEFAULT_TCP_PORT)

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


def get_Adapter():
    """Parse all the arguments to the CLI.

    Parses graphics, output_file, ghost_agent, learn_runs, layout, noise,
    num_ghosts, pacman_agent, policy_file, test_runs, address and port.

    Initialize client as a TCPClient and adapter as a Adapter, passing all its
    arguments.

    Returns:
        The adapter object of Adapter Class.
    """
    parser = ArgumentParser(description='Run Pac-Man adapter system.')
    parser.add_argument('-g', '--graphics', dest='graphics', default=False,
                        action='store_true',
                        help='display graphical user interface')
    parser.add_argument('-o', '--output', dest='output_file', type=str,
                        default=DEFAULT_OUTPUT_FILE,
                        help='results output file')

    group = parser.add_argument_group('Experimental Setup')
    group.add_argument('--ghost-agent', dest='ghost_agent', type=str,
                       choices=['random', 'ai'], default=DEFAULT_GHOST_AGENT,
                       help='select ghost agent')
    group.add_argument('-l', '--learn-num', dest='learn_runs', type=int,
                       default=DEFAULT_NUMBER_OF_LEARNING_RUNS,
                       help='number of games to learn from')
    group.add_argument('--layout', dest='layout', type=str,
                       default=DEFAULT_LAYOUT, choices=['classic', 'medium'],
                       help='Game layout')
    group.add_argument('--noise', dest='noise', type=int,
                       default=DEFAULT_NOISE,
                       help='introduce noise in position measurements')
    group.add_argument('--num-ghosts', dest='num_ghosts',
                       type=int, choices=xrange(1, 5),
                       default=DEFAULT_NUMBER_OF_GHOSTS,
                       help='number of ghosts in game')
    group.add_argument('--pacman-agent', dest='pacman_agent', type=str,
                       choices=['random', 'random2', 'ai', 'eater', 'bfs',
                                'nimble'],
                       default=DEFAULT_PACMAN_AGENT,
                       help='select Pac-Man agent')
    group.add_argument('--policy-file', dest='policy_file',
                       type=lambda s: unicode(s, 'utf8'),
                       help='load and save Pac-Man policy from the given file')
    group.add_argument('-t', '--test-num', dest='test_runs', type=int,
                       default=DEFAULT_NUMBER_OF_TEST_RUNS,
                       help='number of games to test learned policy')
    group.add_argument('--mse', dest='mse', type=int,
                       default=DEFAULT_MSE,
                       choices=[0, 1],
                       help='Enable/Disable MSE calculation')

    group = parser.add_argument_group('Communication')
    group.add_argument('--comm', dest='comm', type=str,
                       choices=['none', 'pm', 'sharedLearn', 'both', 'mse'],
                       default=DEFAULT_COMM,
                       help='Type of communication the agents will do')
    group.add_argument('--addr', dest='address', type=str,
                       default=DEFAULT_CLIENT_ADDRESS,
                       help='Client address to connect to adapter (TCP '
                            'connection)')
    group.add_argument('--port', dest='port', type=int,
                       default=DEFAULT_TCP_PORT,
                       help='Port to connect to controller (TCP connection)')

    args, unknown = parser.parse_known_args()

    client = TCPClient(args.address, args.port)

    # print(args)

    adapter = Adapter(pacman_agent=args.pacman_agent,
                      ghost_agent=args.ghost_agent,
                      num_ghosts=args.num_ghosts,
                      noise=args.noise,
                      policy_file=args.policy_file,
                      layout=args.layout,
                      learn_runs=args.learn_runs,
                      test_runs=args.test_runs,
                      client=client,
                      output_file=args.output_file,
                      graphics=args.graphics,
                      comm=args.comm,
                      mse=args.mse)

    return adapter


def get_Controller():
    """Get the Controller.

    Parse port, instantiate server as a TCPServer.

    Returns:
        The controller of the server instantiated.
    Todo:
        Setup an option for a "memory" server (direct communication with
        # Adapter) (zmq inproc?)
    """
    parser = ArgumentParser(description='Run Pac-Man controller system.')
    parser.add_argument('--port', dest='port', type=int,
                        default=DEFAULT_TCP_PORT,
                        help='TCP port to connect to adapter')
    args, unknown = parser.parse_known_args()

    server = TCPServer(port=args.port)

    return Controller(server)
