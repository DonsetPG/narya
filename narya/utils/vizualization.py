from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import matplotlib.patheffects as path_effects
import cv2

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import arrow
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import colors

from scipy.signal import savgol_filter
from scipy.spatial import Voronoi

from shapely.geometry import Polygon

from moviepy import editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

"""

Football field vizualization function cloned from https://github.com/Friends-of-Tracking-Data-FoTD
with some minor modifications to:
    * add velocity vectors 
    * ball and possession markers

"""

X_SIZE = 105
Y_SIZE = 68

BOX_HEIGHT = (16.5 * 2 + 7.32) / Y_SIZE * 100
BOX_WIDTH = 16.5 / X_SIZE * 100

GOAL = 7.32 / Y_SIZE * 100

GOAL_AREA_HEIGHT = 5.4864 * 2 / Y_SIZE * 100 + GOAL
GOAL_AREA_WIDTH = 5.4864 / X_SIZE * 100

SCALERS = np.array([X_SIZE / 100, Y_SIZE / 100])
pitch_polygon = Polygon(((0, 0), (0, 100), (100, 100), (100, 0)))


def visualize(**images):
    """PLot images in one row.

    Arguments:
        **images: images to plot
    Returns:
        
    Raises:
        
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def draw_pitch(dpi=100, pitch_color="#a8bc95"):
    """Sets up field.
    Arguments:
      dpi: Dots per inch in the field
      pitch_color: Color of the field
    Returns:
      fig,axes: matplotlib fig and axes objects.
    Raises:
    """
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    fig.patch.set_facecolor(pitch_color)

    axes = fig.add_subplot(1, 1, 1)
    axes.set_axis_off()
    axes.set_facecolor(pitch_color)
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)

    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)

    plt.xlim([-13.32, 113.32])
    plt.ylim([-5, 105])

    fig.tight_layout(pad=3)

    draw_patches(axes)

    return fig, axes


def draw_patches(axes):
    """Draws basic field shapes on an axes
    Arguments:
      axes: matplotlib axes objects.
    Returns:
      axes: matplotlib axes objects.
    Raises:
    """
    # pitch
    axes.add_patch(plt.Rectangle((0, 0), 100, 100, edgecolor="white", facecolor="none"))

    # half-way line
    axes.add_line(plt.Line2D([50, 50], [100, 0], c="w"))

    # penalty areas
    axes.add_patch(
        plt.Rectangle(
            (100 - BOX_WIDTH, (100 - BOX_HEIGHT) / 2),
            BOX_WIDTH,
            BOX_HEIGHT,
            ec="w",
            fc="none",
        )
    )
    axes.add_patch(
        plt.Rectangle(
            (0, (100 - BOX_HEIGHT) / 2), BOX_WIDTH, BOX_HEIGHT, ec="w", fc="none"
        )
    )

    # goal areas
    axes.add_patch(
        plt.Rectangle(
            (100 - GOAL_AREA_WIDTH, (100 - GOAL_AREA_HEIGHT) / 2),
            GOAL_AREA_WIDTH,
            GOAL_AREA_HEIGHT,
            ec="w",
            fc="none",
        )
    )
    axes.add_patch(
        plt.Rectangle(
            (0, (100 - GOAL_AREA_HEIGHT) / 2),
            GOAL_AREA_WIDTH,
            GOAL_AREA_HEIGHT,
            ec="w",
            fc="none",
        )
    )

    # goals
    axes.add_patch(plt.Rectangle((100, (100 - GOAL) / 2), 1, GOAL, ec="w", fc="none"))
    axes.add_patch(plt.Rectangle((0, (100 - GOAL) / 2), -1, GOAL, ec="w", fc="none"))

    # halfway circle
    axes.add_patch(
        Ellipse(
            (50, 50),
            2 * 9.15 / X_SIZE * 100,
            2 * 9.15 / Y_SIZE * 100,
            ec="w",
            fc="none",
        )
    )

    return axes


def draw_frame(
    df,
    t,
    dpi=100,
    fps=20,
    add_vector=False,
    display_num=False,
    display_time=False,
    show_players=True,
    highlight_color=None,
    highlight_player=None,
    shadow_player=None,
    text_color="white",
    flip=False,
    **anim_args
):
    """
    Draws players from time t (in seconds) from a DataFrame df
    """
    fig, ax = draw_pitch(dpi=dpi)

    dfFrame = get_frame(df, t, fps=fps)

    if show_players:
        for pid in dfFrame.index:
            if pid == 0:
                # se for bola
                try:
                    z = dfFrame.loc[pid]["z"]
                except:
                    z = 0
                size = 1.2 + z
                lw = 0.9
                color = "black"
                edge = "white"
                zorder = 100
            else:
                # se for jogador
                size = 3
                lw = 2
                edge = dfFrame.loc[pid]["edgecolor"]

                if pid == highlight_player:
                    color = highlight_color
                else:
                    color = dfFrame.loc[pid]["bgcolor"]
                if dfFrame.loc[pid]["team"] == "attack":
                    zorder = 21
                else:
                    zorder = 20

            ax.add_artist(
                Ellipse(
                    (dfFrame.loc[pid]["x"], dfFrame.loc[pid]["y"]),
                    size / X_SIZE * 100,
                    size / Y_SIZE * 100,
                    edgecolor=edge,
                    linewidth=lw,
                    facecolor=color,
                    alpha=0.8,
                    zorder=zorder,
                )
            )

            if add_vector:

                arrow_length = (
                    math.sqrt(
                        (dfFrame.loc[pid]["dx"] ** 2 + dfFrame.loc[pid]["dy"] ** 2)
                    )
                    * 20
                )
                color_arrow = "white" if pid == 0 else color
                plt.arrow(
                    x=dfFrame.loc[pid]["x"],
                    y=dfFrame.loc[pid]["y"],
                    dx=dfFrame.loc[pid]["dx"] * 20,
                    dy=dfFrame.loc[pid]["dy"] * 20,
                    length_includes_head=True,
                    color=color_arrow,
                    edgecolor=edge,
                    head_width=1,
                    head_length=arrow_length / 4.0,
                )

            try:
                s = str(int(dfFrame.loc[pid]["player_num"]))
            except ValueError:
                s = ""
            text = plt.text(
                dfFrame.loc[pid]["x"],
                dfFrame.loc[pid]["y"],
                s,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color=text_color,
                zorder=22,
                alpha=0.8,
            )

            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground=text_color, alpha=0.8),
                    path_effects.Normal(),
                ]
            )

    return fig, ax, dfFrame


def add_voronoi_to_fig(fig, ax, dfFrame):
    """ Adds a voronoi diagram to the field, according to the players positions
    Arguments:
      fig,ax: matplotlib fig and axes objects.
      dfFrame: pd.DataFrame with player tracking data
    Returns:
      fig,ax: matplotlib fig and axes objects.
      dfFrame: pd.DataFrame with player tracking data
    Raises:
    """
    polygons = {}
    vor, dfVor = calculate_voronoi(dfFrame)
    for index, region in enumerate(vor.regions):
        if not -1 in region:
            if len(region) > 0:
                try:
                    pl = dfVor[dfVor["region"] == index]
                    polygon = Polygon(
                        [vor.vertices[i] for i in region] / SCALERS
                    ).intersection(pitch_polygon)
                    polygons[pl.index[0]] = polygon
                    color = pl["bgcolor"].values[0]
                    x, y = polygon.exterior.xy
                    plt.fill(x, y, c=color, alpha=0.30)
                except IndexError:
                    pass
                except AttributeError:
                    pass

    plt.scatter(dfVor["x"], dfVor["y"], c=dfVor["bgcolor"], alpha=0.2)

    return fig, ax, dfFrame


def calculate_voronoi(dfFrame):
    """ Computes the voronoi diagram for the players positions
    Arguments:
      dfFrame: pd.DataFrame with player tracking data
    Returns:
      vor: Voronoi dataframe (region for each coordinates)
      dfFrame: pd.DataFrame with player tracking data
    Raises:
    """
    dfTemp = dfFrame.copy().drop(0, errors="ignore")

    values = np.vstack(
        (
            dfTemp[["x", "y"]].values * SCALERS,
            [-1000, -1000],
            [+1000, +1000],
            [+1000, -1000],
            [-1000, +1000],
        )
    )

    vor = Voronoi(values)

    dfTemp["region"] = vor.point_region[:-4]

    return vor, dfTemp


def get_frame(df, t, fps=20):
    """Gets the player data from the right frame
    Arguments:
      df: pd.DataFrame with player tracking data
      t: timestamp of the play
      fps: frame per second
    Returns:
      dfFrame: pd.DataFrame with player tracking data from timestamp t
    Raises:
    """
    dfFrame = df.loc[int(t * fps)].set_index("player")
    dfFrame.player_num = dfFrame.player_num.fillna("")
    return dfFrame


def draw_frame_x(df, t, fps, voronoi=True):
    """Draw field, player and voronoi on the same image
    Arguments:
      df: pd.DataFrame with player tracking data
      t: timestamp of the play
      fps: frame per second
      voronoi : If we draw the voronoi diagram or not
    Returns:
      image: Image of the field, players, ball, and eventually voronoi diagram.
    Raises:
    """
    fig, ax, dfFrame = draw_frame(df, t=t, fps=fps, add_vector=True)
    if voronoi:
        fig, ax, dfFrame = add_voronoi_to_fig(fig, ax, dfFrame)
    image = mplfig_to_npimage(fig)
    plt.close()
    return image


def make_animation(df, fps=20, voronoi=True):
    """Makes a clip from the entire dataset
    Arguments:
      df: pd.DataFrame with player tracking data
      fps: frame per second
      voronoi : If we draw the voronoi diagram or not
    Returns:
      clip: mpy Clip object
    Raises:
    """
    # calculated variables
    length = (df.index.max() + 20) / fps
    clip = mpy.VideoClip(
        lambda x: draw_frame_x(df, t=x, fps=fps, voronoi=voronoi), duration=length - 1
    ).set_fps(fps)
    return clip


def draw_line(df_value, t, fps, smooth=True, show=True):
    """ Draw the value function overtime, and add a marker at the wanted frame
    Arguments:
      df_value: pd.Dataframe with the value of each frame
      t: timestamp of the play to mark the value function
      fps: frame per second
      smooth : If we smooth the value function or not
      show: If we show the plot (otherwise return it as an image)
    Returns:
      image, fig,ax: show or image of the plot
    Raises:
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)
    if smooth:
        df_value["Value_smooth"] = savgol_filter(df_value["value"], 21, 3)
    else:
        df_value["Value_smooth"] = df_value["value"]
    x = df_value["frame_count"].values
    y = df_value["Value_smooth"].values

    vmin = df_value["Value_smooth"].min()
    vmax = df_value["Value_smooth"].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="cool", norm=norm)
    lc.set_array(y)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.scatter(
        df_value.values[int(t * fps)][0],
        df_value.values[int(t * fps)][2],
        color="green",
        marker="o",
        linewidths=4,
    )
    if show:
        return fig, ax
    else:
        image = mplfig_to_npimage(fig)
        plt.close()
        return image


def make_animation_value(df, fps=20):
    """Makes a clip from the entire dataset of the value function
    Arguments:
      df: pd.DataFrame with player tracking data
      fps: frame per second
    Returns:
      clip: mpy Clip object
    Raises:
    """
    # calculated variables
    length = (df.index.max() + 20) / fps
    clip = mpy.VideoClip(
        lambda x: draw_line(df, t=x, fps=fps), duration=length - 1
    ).set_fps(fps)
    return clip


def add_edg_to_fig(fig, ax, edg_map, vmin=None, vmax=None):
    """Adds an edg_map to a field
    Arguments:
      fig,ax : Matplotlib object from draw_frame
      edg_map: edg_map from agent.py
      vmin,vmax : min and max value of the edg_map
    Returns:
      fig,ax: Matplotlib object from draw_frame, with the map on top
      edg_map: edg_map from agent.py
    Raises:
    """
    cmap = "bwr"
    if vmin is None:
        vmin = np.min(edg_map)
    if vmax is None:
        vmax = np.max(edg_map)
    ax = ax.imshow(
        edg_map,
        extent=(0, 100, 0, 100),
        aspect="auto",
        interpolation="mitchell",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        alpha=0.5,
    )
    cbar = plt.colorbar(ax)
    cbar.set_label("Value")
    return fig, ax, edg_map


def get_color(idx):
    idx = idx * 3
    color = ((17 * idx) % 255, (37 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.0)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.0))

    radius = max(5, int(im_w / 140.0))
    cv2.putText(
        im,
        "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = "{}".format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ", {}".format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            im,
            id_text,
            (intbox[0], intbox[1] + 30),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 255),
            thickness=text_thickness,
        )
    return im


def rgb_template_to_coord_conv_template(rgb_template):
    assert isinstance(rgb_template, np.ndarray)
    assert rgb_template.min() >= 0.0
    assert rgb_template.max() <= 1.0
    rgb_template = np.mean(rgb_template, 2)
    x_coord, y_coord = np.meshgrid(
        np.linspace(0, 1, num=rgb_template.shape[1]),
        np.linspace(0, 1, num=rgb_template.shape[0]),
    )
    coord_conv_template = np.stack((rgb_template, x_coord, y_coord), axis=2)
    return coord_conv_template


def merge_template(img, warped_template):
    valid_index = warped_template[:, :, 0] > 0.0
    overlay = (
        img[valid_index].astype("float32")
        + warped_template[valid_index].astype("float32")
    ) / 2
    new_image = np.copy(img)
    new_image[valid_index] = overlay
    return new_image
