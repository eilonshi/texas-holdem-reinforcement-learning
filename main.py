"""
neuron poker

Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_keras_train [options]
  main.py selfplay dqn_keras_play [options]
  main.py selfplay dqn_torch_train [options]
  main.py selfplay dqn_torch_play [options]
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screen_log_level=<>     log level on screen
  --epochs=<>               number of epochs to play
  --episodes=<>             number of episodes to play in epoch
  --stack=<>                starting stack for each player [default: 500]
  --players=<>              number of players in the game

"""

import logging
import argparse

from consts import DEFAULT_STACK, DEFAULT_NUM_PLAYERS, DEFAULT_NUM_EPOCHS, DEFAULT_NUM_EPISODES
from tools.helper import get_config
from tools.helper import init_logger
from training_loop.self_play import SelfPlay

parser = argparse.ArgumentParser(description='Process arguments for a Texas Holdem game.')
parser.add_argument('--log', metavar='logfile', type=str, default='default',
                    help='a logfile to use for the log messages')
parser.add_argument('--name', metavar='name', type=str, default='my_model',
                    help='a name for the model')
parser.add_argument('--log_level', metavar='screen_log_level', type=str, default='INFO',
                    help='the default logging level')
parser.add_argument('--selfplay', metavar='selfplay', type=bool, default=True,
                    help='flag of using selfplay in the game')
parser.add_argument('--render', metavar='render', type=bool, default=False,
                    help='flag of rendering the game')
parser.add_argument('--cpp', metavar='use_cpp_montecarlo', type=bool, default=True,
                    help='flag of using cpp for the montecarlo')
parser.add_argument('--plot', metavar='funds_plot', type=bool, default=True,
                    help='flag of plotting the funds at the end of the selfplay')
parser.add_argument('--players', metavar='players', type=int, default=DEFAULT_NUM_PLAYERS,
                    help='number of players in the game')
parser.add_argument('--stack', metavar='stack', type=int, default=DEFAULT_STACK,
                    help='the size of the stack of each player in the game')
parser.add_argument('--epochs', metavar='epochs', type=int, default=DEFAULT_NUM_EPOCHS,
                    help='number of epochs to run the selfplay')
parser.add_argument('--episodes', metavar='episodes', type=int, default=DEFAULT_NUM_EPISODES,
                    help='number of episodes in each epoch of the selfplay')
parser.add_argument('--train_model', metavar='train_model', type=str, default='dqn_torch_train',
                    help='the model to train in the selfplay')

if __name__ == '__main__':
    # Reading arguments
    args = parser.parse_args()
    logfile = args.log
    model_name = args.name
    screen_log_level = getattr(logging, args.log_level.upper())

    # Setting logger
    _ = get_config()
    init_logger(screenlevel=screen_log_level, filename=logfile)
    print(f"Screen log level: {screen_log_level}")
    log = logging.getLogger("")
    log.info("Initializing program")

    # Managing a selfplay
    if args.selfplay:
        render = args.render
        use_cpp_montecarlo = args.cpp
        funds_plot = args.plot
        num_players = args.players
        stack = args.stack
        num_epochs = args.epochs
        num_episodes = args.episodes

        runner = SelfPlay(render=render,
                          num_players=num_players,
                          num_epochs=num_epochs,
                          num_episodes=num_episodes,
                          use_cpp_montecarlo=use_cpp_montecarlo,
                          funds_plot=funds_plot,
                          stack=stack)

        if args.train_model == 'dqn_keras_train':
            runner.dqn_train_keras_rl(model_name)
        elif args.train_model == 'dqn_keras_play':
            runner.dqn_play_keras_rl(model_name)
        elif args.train_model == 'dqn_torch_train':
            runner.dqn_train_torch_rl()
        elif args.train_model == 'dqn_torch_play':
            runner.dqn_play_torch_rl()

    else:
        raise RuntimeError("Argument not yet implemented")
