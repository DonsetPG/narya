from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np

def add_nan_trajectories(trajectories, max_frame):
    """Add np.nan to frame where the x,y coordinates are missing
    Arguments:
        trajectories: Dict mapping each id to a list of tuple (x,y,frame)
        max_frame: Max number of frame
    Returns:
        trajectories: Dict mapping each id to a list of tuple (x,y,frame)
    Raises:
    """
    frame_range = [i for i in range(1, max_frame)]
    full_trajectories = {}
    for ids in trajectories.keys():
        traj = trajectories[ids]
        full_trajectories[ids] = []
        cnt = 0
        for x_, y_, frame_ in traj:
            if cnt == 0:
                full_trajectories[ids].append([x_, y_, frame_])
                last_x, last_y, last_frame = x_, y_, frame_
                cnt += 1
            else:
                nb_fake_data_to_add = frame_ - last_frame
                for i in range(nb_fake_data_to_add - 1):
                    full_trajectories[ids].append([np.nan, np.nan, last_frame + i + 1])
                full_trajectories[ids].append([x_, y_, frame_])
                last_x, last_y, last_frame = x_, y_, frame_
                cnt += 1
        last_frame = frame_
        if last_frame < frame_range[-1]:
            for i in range(last_frame + 1, frame_range[-1] + 1):
                full_trajectories[ids].append([np.nan, np.nan, i])
    return full_trajectories   ## <--------- finish all the loops and return

def get_trajectory_from_id(trajectories, id_):
    """Get the trajectory of an id, and map them to lastRow data format.
    Arguments:
        trajectories: Dict mapping each id to a list of tuple (x,y,frame)
        id_: the id of the trajectory we want
    Returns:
        x,y, frames: List of x,y coordinates with frames
    Raises:
    """
    tab = trajectories[id_]
    x = []
    y = []
    frames = []
    for data_points in tab:
        x_, y_, fr = data_points[0], data_points[1], data_points[2]
        if np.isnan(x_):
            x.append(x_)
            y.append(y_)
        else:
            new_x = x_ * (100.0 / 320.0)
            new_y = y_ * (100.0 / 320.0)
            new_y = 100.0 - new_y

            x.append(new_x)
            y.append(new_y)
        frames.append(fr)

    return x, y, frames


def build_df_per_id(trajectories):
    """Build one dataframe per id
    Arguments:
        trajectories: Dict mapping each id to a list of tuple (x,y,frame)
    Returns:
        df_per_id: dict mapping each id to a dataframe with its trajectory
    Raises:
    """
    df_per_id = {}
    full_trajectories = add_nan_trajectories(trajectories, max_frame=50)
    for ids in trajectories.keys():
        x, y, frame = get_trajectory_from_id(full_trajectories,ids)
        results_ids = {"x": [], "y": [], "frame": []}
        for x_, y_, frame_ in zip(x, y, frame):
            results_ids["x"].append(x_)
            results_ids["y"].append(y_)
            results_ids["frame"].append(frame_)
        df_per_id[int(ids)] = pd.DataFrame(results_ids)
    return df_per_id


def fill_nan_trajectories(df_per_id,window_size=21):
    """Fill each trajectory, and apply a savgol filter
    Arguments:
        df_per_id: dict mapping each id to a dataframe with its trajectory
    Returns:
        df_per_id: dict mapping each id to a dataframe with its trajectory
    Raises:
    """
    for ids in df_per_id.keys():
        df_id = df_per_id[ids]
        test = df_id.set_index("frame")
        test["x"] = test["x"].interpolate(method="slinear", limit_direction="both")
        test["y"] = test["y"].interpolate(method="slinear", limit_direction="both")
        test["x"] = test["x"].interpolate(
            method="pad", limit_direction="forward", limit_area="outside"
        )
        test["y"] = test["y"].interpolate(
            method="pad", limit_direction="forward", limit_area="outside"
        )
        test["x"] = savgol_filter(test["x"], window_size, 3) if len(test["x"]) > 3 else test["x"]
        test["y"] = savgol_filter(test["y"], window_size, 3) if len(test["y"]) > 3 else test["y"]
        df_per_id[ids] = test
    return df_per_id


def get_full_results(df_per_id):
    """Build a dataframe of tracking data, in the same format as Last Row
    Arguments:
        df_per_id: dict mapping each id to a dataframe with its trajectory
    Returns:
        full_results: A dataframe with each coordinates at each frame
    Raises:
    """
    full_results = {"frame": [], "id": [], "x": [], "y": []}
    for ids in df_per_id.keys():
        df_id = df_per_id[ids].reset_index()
        for line in df_id.values:
            full_results["frame"].append(int(line[0]))
            full_results["x"].append(line[1])
            full_results["y"].append(line[2])
            full_results["id"].append(ids)
    return pd.DataFrame(full_results).set_index("frame")


def _get_max_id(traj):
    """Get the biggest id within a trajectory
    """
    return max([int(ids) for ids in traj.keys()])


def _remove_coords(traj, ids, frame):
    """Remove the x,y coordinates of an id at a given frame
    Arguments:
        traj: Dict mapping each id to a list of trajectory
        ids: the id to target
        frame: int, the frame we want to remove
    Returns:
        traj: Dict mapping each id to a list of trajectory
    Raises:
    """
    new_traj = []
    for data_points in traj[ids]:
        if data_points[2] != frame:
            new_traj.append(data_points)
    traj[ids] = new_traj
    return traj


def _remove_ids(traj, list_ids):
    """Remove ids from a trajectory
    Arguments:
        traj: Dict mapping each id to a list of trajectory
        list_ids: List of id
    Returns:
        traj: Dict mapping each id to a list of trajectory
    Raises:
    """
    for t in list_ids:
        traj.pop(t, None)
    return traj


def add_entity(traj, entity_id, entity_traj):
    """Adds a new id with a trajectory 
    Arguments:
        traj: Dict mapping each id to a list of trajectory
        entity_id: the id to add
        entity_traj: the trajectory linked to entity_id we want to add
    Returns:
        traj: Dict mapping each id to a list of trajectory
    Raises:
    """
    assert (
        entity_id not in traj.keys()
    ), "This id is already present in the trajectory, find another one"
    traj[entity_id] = entity_traj
    return traj


def add_entity_coords(traj, entity_id, entity_traj, max_frame):
    """Add some coordinates to the trajectory of a given id
    Arguments:
        traj: Dict mapping each id to a list of trajectory
        entity_id: the id to target
        entity_traj: List of (x,y,frame) to add to the trajectory of entity_id
        max_frame: int, the maximum number of frame in trajectories
    Returns:
        traj: Dict mapping each id to a list of trajectory
    Raises:
    """
    # entity_traj with shape list of (x,y,frame)
    assert entity_id in traj.keys(), "This id is not present in the trajectory"

    traj_to_add = {frame: [x, y] for x, y, frame in entity_traj}

    traj_ = {
        data_point[2]: [data_point[0], data_point[1]] for data_point in traj[entity_id]
    }

    temp_traj = []
    for frame in range(1, max_frame):
        if frame in traj_to_add.keys():
            temp_traj.append([traj_to_add[frame][0], traj_to_add[frame][1], frame])
        elif frame in traj_.keys():
            temp_traj.append([traj_[frame][0], traj_[frame][1], frame])

    traj[entity_id] = temp_traj
    return traj


def merge_id(traj, list_ids_frame):
    """Merge trajectories of different ids. 
    e.g.: (10,0,110),(12,110,300) will merge the trajectory of 10 between frame 0 and 110 to the 
        trajectory of 12 between frame 110 and 300.
    Arguments:
        traj: Dict mapping each id to a list of trajectory
        list_ids_frame: List of (id,frame_start,frame_end)
    Returns:
        traj: Dict mapping each id to a list of trajectory
    Raises:
    """
    # list_ids: sort by order of appearance
    # with the shape : list of (id,frame_start,frame_end)
    new_traj = []
    for ids, frame_start, frame_end in list_ids_frame:
        for data_points in traj[ids]:
            x, y, frame = data_points[0], data_points[1], data_points[2]
            if frame >= frame_start and frame < frame_end:
                new_traj.append([x, y, frame])
    for ids, _, _ in list_ids_frame:
        traj.pop(ids, None)
    traj[list_ids_frame[0][0]] = new_traj
    return traj


def merge_2_trajectories(traj1, traj2, id_mapper, max_frame_traj1):
    """Merge 2 dict of trajectories, if you want to merge the results of 2 tracking
    Arguments:
        traj1: Dict mapping each id to a list of trajectory
        traj2: Dict mapping each id to a list of trajectory
        id_mapper: A dict mapping each id in traj1 to id in traj2
        max_frame_traj1: Maximum number of frame in the first trajectory
    Returns:
        traj1: Dict mapping each id to a list of trajectory
    Raises:
    """
    for id_1 in id_mapper.keys():
        id_2 = id_mapper[id_1]
        traj_id1 = traj1[id_1]
        traj_id2 = traj2[id_2]
        for data_points in traj_id2:
            x, y, frame = data_points[0], data_points[1], data_points[2]
            frame += max_frame_traj1
            traj_id1.append([x, y, frame])
    for id_2 in traj2.keys():
        if id_2 not in traj1.keys():
            traj_id2 = traj2[id_2]
            new_traj = []
            for data_points in traj_id2:
                x, y, frame = data_points[0], data_points[1], data_points[2]
                frame += max_frame_traj1
                new_traj.append([x, y, frame])
            traj1[id_2] = new_traj
    return traj1
