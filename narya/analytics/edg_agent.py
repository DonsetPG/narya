import numpy as np

import tensorflow as tf
from gfootball.env.players.ppo2_cnn import Player
from ..utils.google_football_utils import traverse, change

SMM_WIDTH = 96
SMM_HEIGHT = 72

channel_dimensions = (SMM_WIDTH, SMM_HEIGHT)

EDG_MODEL_PATH = (
    "https://storage.googleapis.com/narya-bucket-1/models/11_vs_11_selfplay_last"
)
EDG_MODEL_NAME = "11_vs_11_selfplay_last"
EDG_MODEL_TOTAR = False


class AgentValue:
    """ Creates a agent that will compute the value of tracking data
    Arguments:
      checkpoints: Checkpoint to load the agent from. Can be set to None
      policy: Policy to use with the agent
    """

    def __init__(
        self, pretrained=True, checkpoints=None, policy="gfootball_impala_cnn"
    ):
        if pretrained:
            checkpoints = tf.keras.utils.get_file(
                EDG_MODEL_NAME, EDG_MODEL_PATH, EDG_MODEL_TOTAR,
            )
        player_config = {
            "index": 0,
            "left_players": 1,
            "right_players": 0,
            "policy": policy,
            "stacked": True,
            "checkpoint": checkpoints,
        }
        self.agent = Player(player_config, env_config={})
        self.agent.reset()

    def get_value(self, observations):
        """ Computes the value of an observation. The observations is as follows : 
        The observation is
        composed of 4 planes of size 'channel_dimensions'.
        Its size is then 'channel_dimensions'x4 (or 'channel_dimensions'x16 when
        stacked is True).
        The first plane P holds the position of players on the left
        team, P[y,x] is 255 if there is a player at position (x,y), otherwise,
        its value is 0.
        The second plane holds in the same way the position of players
        on the right team.
        The third plane holds the position of the ball.
        The last plane holds the active player.
        Arguments:
          observations: A np.array of shape (72,96,16) with the stacked observations
        Returns:
          value: The value of the observations.
        Raises:
        """
        return self.agent._policy.value(observations)[0]

    def get_edg_map(
        self, observation, observation_count, init_x, init_y, entity="player"
    ):
        """ Computes the edg map of the given observation. It considers the entity (player or ball) at the position init_x,init_y,
        and changes its position on the field to compute the edg_map.
        Arguments:
          observations: A np.array of shape (72,96,16) with the stacked observations
          observations_count: A np.array of shape (72,96,16) with t
          he stacked observations counts. Observations counts stores the
                            number of entity at a position.
          init_x, init_y: The position of the entity in the observation
          entity: The entity to move (ball or player)
        Returns:
          map_value: A (72,96) array with a edg for each coordinates.
        Raises:
        """
        map_value = np.zeros((channel_dimensions[1], channel_dimensions[0]))
        change(observation, observation_count, init_x, init_y, 0, 0, entity)
        x, y = 0, 0
        while x != channel_dimensions[0] - 1 or y != channel_dimensions[1] - 1:
            value = self.get_value(observation)
            map_value[y, x] = value
            x, y = traverse(observation, observation_count, x, y, entity)
        return map_value
