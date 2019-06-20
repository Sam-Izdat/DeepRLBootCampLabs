import threading
import time

from pysc2 import maps
from pysc2.env import available_actions_printer
# from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from absl import app
from absl import flags

from pytorch.sc2_agents.base_rl_agent import BaseRLAgent as Agent

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("train", False, "Whether we are training or running")
flags.DEFINE_integer("screen_resolution", 28,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

# flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

# flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent", "Which agent to run")
flags.DEFINE_enum("agent_race", None, sc2_env.Race._member_names_, "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.Race._member_names_, "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.Difficulty._member_names_,
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.mark_flag_as_required("map")




point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
                           
def run_thread(map_name, visualize):
  
      # feature_screen_size=(FLAGS.screen_resolution, FLAGS.screen_resolution),
      # feature_minimap_size=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
  with sc2_env.SC2Env(
      map_name=map_name,
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.feature_screen_size,
          feature_minimap=FLAGS.feature_minimap_size,
          rgb_screen=FLAGS.rgb_screen_size,
          rgb_minimap=FLAGS.rgb_minimap_size,
          action_space=FLAGS.action_space,
          use_feature_units=FLAGS.use_feature_units),
      agent_race=FLAGS.agent_race,
      bot_race=FLAGS.bot_race,
      difficulty=FLAGS.difficulty,
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agent = Agent()
    # run_loop([agent], env, FLAGS.max_agent_steps)
    agent.train(env, FLAGS.train)
    if FLAGS.save_replay:
      env.save_replay(Agent.__name__)

def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.
  run_thread(FLAGS.map, FLAGS.render)

  if FLAGS.profile:
    print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
