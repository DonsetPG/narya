from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os
import random
import numpy as np
import pandas as pd
import pickle

from six.moves import range

SMM_WIDTH = 96
SMM_HEIGHT = 72

channel_dimensions = (SMM_WIDTH, SMM_HEIGHT)
frame = np.zeros((channel_dimensions[1], channel_dimensions[0]))

SMM_LAYERS = ["left_team", "right_team", "ball", "active"]

# Normalized minimap coordinates
MINIMAP_NORM_X_MIN = -1.0
MINIMAP_NORM_X_MAX = 1.0
MINIMAP_NORM_Y_MIN = -1.0 / 2.25
MINIMAP_NORM_Y_MAX = 1.0 / 2.25

_MARKER_VALUE = 255

SMM_WIDTH = 96
SMM_HEIGHT = 72

def _get_values():
    return MINIMAP_NORM_Y_MIN

def mark_points(frame, frame_cnt, points):
    """Draw dots corresponding to 'points'.
    Arguments:
      frame: 2-d matrix representing one SMM channel ([y, x])
      points: a list of (x, y) coordinates to be marked
    Returns:
    Raises:
    """
    for p in range(len(points) // 2):
        x = int(
            (points[p * 2] - MINIMAP_NORM_X_MIN)
            / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN)
            * frame.shape[1]
        )
        y = int(
            (points[p * 2 + 1] - MINIMAP_NORM_Y_MIN)
            / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN)
            * frame.shape[0]
        )
        x = max(0, min(frame.shape[1] - 1, x))
        y = max(0, min(frame.shape[0] - 1, y))
        frame[y, x] = _MARKER_VALUE
        frame_cnt[y, x] += 1


def _left_parser(tab, i):
    """Helper function to build google observations
    """
    coord = []
    tab_ = tab[i]["left_team_positions"]
    for list_ in tab_:
        for value in list_:
            coord.append(value)
    return coord


def _right_parser(tab, i):
    """Helper function to build google observations
    """
    coord = []
    tab_ = tab[i]["right_team_positions"]
    for list_ in tab_:
        for value in list_:
            coord.append(value)
    return coord


def _ball_parser(tab, i):
    """Helper function to build google observations
    """
    tab_ = tab[i]["ball_position"]
    return tab_[:-1]


def _active_parser(tab, i):
    """Helper function to build google observations
    """
    tab_left = tab[i]["left_team_positions"]
    tab_ball = tab[i]["ball_position"][:-1]
    min_dist = 1000000
    for indx, list_ in enumerate(tab_left):
        dist = (list_[0] - tab_ball[0]) ** 2 + (list_[1] - tab_ball[1]) ** 2
        if dist < min_dist:
            min_dist = dist
            select = indx
    return tab_left[select]


def _build_obs_stacked(tab, i):
    """ Computes an observation, readable by an agent, from _save_data output
    Arguments:
      tab: Output of _save_data : player tracking data in the right coordinates
      i: Index of the frame to create
    Returns:
      frame: Google observations of the tracking data
      frame_count:Google observations count of the tracking data
    Raises:
    """
    frame = np.zeros((channel_dimensions[1], channel_dimensions[0], 16))
    frame_count = np.zeros((channel_dimensions[1], channel_dimensions[0], 16))
    mark_points(frame[:, :, 0], frame_count[:, :, 0], _left_parser(tab, i))
    mark_points(frame[:, :, 1], frame_count[:, :, 0], _right_parser(tab, i))
    mark_points(frame[:, :, 2], frame_count[:, :, 0], _ball_parser(tab, i))
    mark_points(frame[:, :, 3], frame_count[:, :, 0], _active_parser(tab, i))
    to_add = min((i + 1), len(tab) - 1)
    mark_points(frame[:, :, 4], frame_count[:, :, 0], _left_parser(tab, to_add))
    mark_points(frame[:, :, 5], frame_count[:, :, 0], _right_parser(tab, to_add))
    mark_points(frame[:, :, 6], frame_count[:, :, 0], _ball_parser(tab, to_add))
    mark_points(frame[:, :, 7], frame_count[:, :, 0], _active_parser(tab, to_add))
    to_add = min((i + 2), len(tab) - 1)
    mark_points(frame[:, :, 8], frame_count[:, :, 0], _left_parser(tab, to_add))
    mark_points(frame[:, :, 9], frame_count[:, :, 0], _right_parser(tab, to_add))
    mark_points(frame[:, :, 10], frame_count[:, :, 0], _ball_parser(tab, to_add))
    mark_points(frame[:, :, 11], frame_count[:, :, 0], _active_parser(tab, to_add))
    to_add = min((i + 3), len(tab) - 1)
    mark_points(frame[:, :, 12], frame_count[:, :, 0], _left_parser(tab, to_add))
    mark_points(frame[:, :, 13], frame_count[:, :, 0], _right_parser(tab, to_add))
    mark_points(frame[:, :, 14], frame_count[:, :, 0], _ball_parser(tab, to_add))
    mark_points(frame[:, :, 15], frame_count[:, :, 0], _active_parser(tab, to_add))

    return frame, frame_count


# For each play, we add to ball coordinates at each frame and for each player :
def _add_ball_coordinates(dataframe: pd.DataFrame, id_ball=-1) -> pd.DataFrame:
    """ Adds the ball coordinates at each row
    Arguments:
      dataframe: pd.DataFrame with player tracking data
    Returns:
      dataframe: pd.DataFrame with player tracking data
    Raises:
    """
    list_of_plays = list(dataframe.index.get_level_values("play").unique())

    # First, a function to compute this for one play at a time :
    def _add_coord(df):
        # Getting the balls infos:
        df_ball = df[df["id"] == id_ball]

        df_ball = df_ball[["frame", "x", "y", "z"]]
        df_ball.rename(
            columns={"x": "ball_x", "y": "ball_y", "z": "ball_z"}, inplace=True
        )

        df = df.merge(df_ball, on="frame", how="left")
        df = df.drop(columns=["Unnamed: 0"])
        return df

    for i, play in enumerate(list_of_plays):

        df = dataframe.loc[play]
        df = df.reset_index()

        if i == 0:
            new_dataframe = _add_coord(df)
            new_dataframe["play"] = play
        else:
            df = _add_coord(df)
            df["play"] = play
            new_dataframe = pd.concat([new_dataframe, df])

    return new_dataframe


# We now add a boolean to mark the possession of the ball by a player
def _add_possession(dataframe: pd.DataFrame, indx_color=4) -> pd.DataFrame:
    """Add a boolean to mark which player has the ball
    Arguments:
      dataframe: pd.DataFrame with player tracking data
    Returns:
      dataframe: pd.DataFrame with player tracking data
    Raises:
    """
    dataframe["possession"] = False
    columns = list(dataframe.columns.values)
    tab = dataframe.values

    for line in tab:
        # Not ocmputing for the ball :
        if line[4] != 0:
            if (line[8] - line[11]) ** 2 + (line[9] - line[12]) ** 2 < 2:
                line[-1] = True
                # Test to display later :
                line[indx_color] = "black"

    dataframe = pd.DataFrame(tab, columns=columns)
    return dataframe


def _prepare_dataset(
    dataframe: pd.DataFrame, bgcolor_mapping, team_mapping, ball_mapping
) -> pd.DataFrame:
    dataframe["bgcolor"] = dataframe["id"]
    dataframe["bgcolor"] = dataframe["bgcolor"].replace(bgcolor_mapping)
    dataframe["team"] = dataframe["id"]
    dataframe["team"] = dataframe["team"].replace(team_mapping)
    dataframe["id"] = dataframe["id"].replace(ball_mapping)
    dataframe["player"] = dataframe["player"].replace(ball_mapping)
    dataframe["dx"] = 0
    dataframe["dy"] = 0
    dataframe["z"] = 0
    dataframe["ball_z"] = 0
    dataframe["play"] = "RM_VS_BARCA_BUT_3"
    dataframe = dataframe.drop(columns=["index", "id"])

    ordered_cols = [
        "bgcolor",
        "dx",
        "dy",
        "edgecolor",
        "player",
        "player_num",
        "team",
        "x",
        "y",
        "z",
        "ball_x",
        "ball_y",
        "ball_z",
        "play",
        "possession",
    ]

    dataframe = dataframe[ordered_cols]
    return dataframe


def _scale_mapper(x, y):
    """maps coordinates to google coordinates
    Arguments:
      x,y: coordinates
    Returns:
      new_x,new_y: Google env coordinates
    Raises:
    """
    # Takes our x,y and maps it into google's (x,y)
    new_x = (x / 100.0 - 0.5) * 2
    new_y = -(y / 100.0 - 0.5) * 2
    return (new_x, new_y)


def _save_data(df, filename):
    """Saves the tracking data into a google readable format
    Arguments:
      df:pd.DataFrame with player tracking data
      filename:a .dump filename to store the informations
    Returns:
      full_info: A list with the tracking data in a google observations format
    Raises:
    """
    df = df.reset_index()
    full_info = []
    for i in list(df["frame"].unique()):
        temp_df = df[df["frame"] == i]
        if (i + 1) in list(df["frame"].unique()):
            next_temp_df = df[df["frame"] == i + 1]
        else:
            next_temp_df = df[df["frame"] == i]

        left_df = temp_df[temp_df["team"] == "attack"]
        right_df = temp_df[temp_df["team"] == "defense"]

        next_left_df = next_temp_df[next_temp_df["team"] == "attack"]
        next_right_df = next_temp_df[next_temp_df["team"] == "defense"]

        left_team_positions = [[-1.0, 0.0]]
        right_team_positions = []
        left_team_velocity = [[0.0, 0.0]]
        right_team_velocity = []
        ball_position = []
        ball_velocity = []
        ball_team_owner = -1
        ball_player_owner = -1

        # Ball Information :

        # Left team informations :

        for indx, line in enumerate(left_df.values):
            new_x, new_y = _scale_mapper(line[8], line[9])
            next_line = next_left_df.values[indx]
            next_new_x, next_new_y = _scale_mapper(next_line[8], next_line[9])
            dx, dy = next_new_x - new_x, next_new_y - new_y

            left_team_positions.append([new_x, new_y])
            left_team_velocity.append([dx, dy])

            if line[-1] == True and ball_team_owner == -1:
                ball_team_owner = 0
                ball_player_owner = indx + 1

            if indx == 0:
                # Ball information :
                ball_x, ball_y = _scale_mapper(line[11], line[12])
                next_ball_x, next_ball_y = _scale_mapper(next_line[11], next_line[12])
                dx, dy = next_ball_x - ball_x, next_ball_y - ball_y
                ball_z = line[13]

                ball_position = [ball_x, ball_y, 0.0]
                ball_velocity = [dx, dy, 0.0]

        for indx, line in enumerate(right_df.values):
            new_x, new_y = _scale_mapper(line[8], line[9])
            next_line = next_right_df.values[indx]
            next_new_x, next_new_y = _scale_mapper(next_line[8], next_line[9])
            dx, dy = next_new_x - new_x, next_new_y - new_y

            right_team_positions.append([new_x, new_y])
            right_team_velocity.append([dx, dy])

        info = {
            "left_team_positions": left_team_positions,
            "right_team_positions": right_team_positions,
            "left_team_velocity": left_team_velocity,
            "right_team_velocity": right_team_velocity,
            "ball_position": ball_position,
            "ball_velocity": ball_velocity,
            "ball_team_owner": ball_team_owner,
            "ball_player_owner": ball_player_owner,
        }

        full_info.append(info)
    with open(filename, "wb") as fh:
        pickle.dump((full_info,), fh, pickle.HIGHEST_PROTOCOL)
    return full_info


def _reverse_points(x, y):
    """Moves the left team to the right (and the right to the left)
    Arguments:
      x,y: Coordinates to reverse
    Returns:
      x,y: Reverse coordinates
    Raises:
    """
    x = channel_dimensions[0] - 1 - x
    y = channel_dimensions[1] - 1 - y
    return (x, y)


def _change(observation, observation_count, x, y, new_x, new_y):
    """Moves an entity from x,y to new_x,new_y
    Arguments:
      observation: np.array unstack of observations
      observation_count: np.array unstack of observations count
      x,y: old coordinates
      new_x,new_y: new coordinates
    Returns:
    Raises:
    """
    observation[new_y, new_x] = _MARKER_VALUE
    observation_count[new_y, new_x] += 1
    observation_count[y, x] = max(0, observation_count[y, x] - 1)
    if observation_count[y, x] == 0:
        observation[y, x] = 0


def change(observation, observation_count, x, y, new_x, new_y, entity):
    """Moves an entity from x,y to new_x,new_y on an entire observations
    Arguments:
      observation: np.array stacked observations
      observation_count: np.array stacked observations count
      x,y: old coordinates
      new_x,new_y: new coordinates
      entity: the entity to move (player or ball)
    Returns:
    Raises:
    """
    indx_init = 0 if entity == "player" else 2
    for i in range(indx_init, 16, 4):
        unstack_observation = observation[:, :, i]
        unstack_observation_count = observation_count[:, :, i]
        _change(unstack_observation, unstack_observation_count, x, y, new_x, new_y)


def _change_random(observation, observation_count, x, y):
    """Moves an entity from x,y randomly on the field
    Arguments:
      observation: np.array unstack observations
      observation_count: np.array unstacke observations count
      x,y: old coordinates
    Returns:
    Raises:
    """
    new_x = int(random.random() * channel_dimensions[0])
    new_y = int(random.random() * channel_dimensions[1])
    _change(observation, observation_count, x, y, new_x, new_y)


def _add_noise(observation, observation_count, x, y, x_std=5, y_std=5):
    """Moves an entity from x,y randomly on the field, with a gaussian noise
    Arguments:
      observation: np.array unstack observations
      observation_count: np.array unstacke observations count
      x,y: old coordinates
      x_std,y_std : std for the noises
    Returns:
    Raises:
    """
    new_x = min(max(int(np.random.normal(x, x_std, 1)[0]), 0), channel_dimensions[0])
    new_y = min(max(int(np.random.normal(y, y_std, 1)[0]), 0), channel_dimensions[1])
    _change(observation, observation_count, x, y, new_x, new_y)


def traverse(observation, observation_count, x, y, entity):
    """Moves an entityto its next possible position. Can be used to try the entire field
    Arguments:
      observation: np.array unstack observations
      observation_count: np.array unstacke observations count
      x,y: old coordinates
      entity: ball or player
    Returns:
      new_x,new_y: the Next available position
    Raises:
    """
    new_x = x + 1
    if new_x >= channel_dimensions[0]:
        new_x = 0
        new_y = y + 1
    else:
        new_y = y
    change(observation, observation_count, x, y, new_x, new_y, entity)
    return new_x, new_y
