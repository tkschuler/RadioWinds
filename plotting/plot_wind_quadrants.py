'''

Show wind quadrants used in analysis, but don't use any actual datra

Author: cem
Version: 1.0 02-Feb-2024

see also:

'''

import sys
import pandas as pd
import numpy as np
from numpy import pi
from math import radians
import matplotlib
#matplotlib.use('Qt5Agg')  # stops figure from hanging when plotted from pycharm
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from mycolorpy import colorlist as mcp
import seaborn as sns
#local libraries

import logging
logging.basicConfig(level=logging.WARNING)
# global used to stamp image with program name
subtitle = "plot_wind_quadrants"

def show_wind_rose(fig):
    '''
    Will show side-by-side plot showing wind rose quadrants in either degrees or quadrant name
    :param fig:
    :return:
    '''

    # Creates DataFrame.
    # df = pd.DataFrame(data)
    # df1 = df.iloc[0]
    # df2 = df.iloc[1]

    bar_color = mcp.gen_color(cmap="rainbow", n=8)
    #bar_color = ['yellow', 'orange', 'red', 'lime', 'green', 'lightblue', 'blue', 'purple']

    nbins = 16
    deg2rad = pi / 180.0
    # Fixing random state for reproducibility
    bins = np.arange(- (180 / nbins), 360 + (180 / nbins), 360 / nbins)
    bins_in_radians = bins * deg2rad
    increment_per_bin = (360 / nbins) * 0.5
    bin_center_pt = bins + increment_per_bin
    bin_center_pt_radians = bin_center_pt * deg2rad
    theta_angles = bin_center_pt_radians[:-1]
    x = 360 - (180 / nbins)  # anything greater needs to shifted to fit into bins
    angle_size = (2 * np.pi / nbins)
    width = angle_size - .02

    radii = np.ones(nbins) * 10
    theta = theta_angles
    bin = np.zeros(nbins)
    # http://www.chiark.greenend.org.uk/~peterb/python/polar/
    data = pd.DataFrame({'compass': ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W',
                                     'WNW', 'NW', 'NNW'],
                         'degrees': [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315,
                                     337.5]})

    # fig = plt.figure(figsize=(8, 3))
    # fig.patch.set_facecolor('lightgoldenrodyellow')
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.bar(x=data['compass'], height=data['value'], width=1)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1.patch.set_facecolor('white')
    ax1.set_title('In Degrees', fontsize=12)
    xticks = 16
    ax1.set_xticks([radians(x) for x in np.arange(0, 360, 360 / xticks)])
    ax1.set_xticklabels(data.degrees)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(325)
    ax1.set_rgrids(range(1, 10, 10), range(1, 10, 10))
    plt.ylim(0, 10)
    colors = plt.cm.hsv(theta / 2 / np.pi)
    bars1 = ax1.bar(x=theta, height=radii, width=width, edgecolor='k', linewidth=1.0, color=colors)
    fig.show()
    bars1 = ax1.bar(x=theta, height=radii, width=width, edgecolor='k', linewidth=1.0, color=colors)

    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    ax2.patch.set_facecolor('white')
    ax2.set_title('16 Quadrants', fontsize=12)
    xticks = 16
    ax2.set_xticks([radians(x) for x in np.arange(0, 360, 360 / xticks)])
    ax2.set_xticklabels(data.compass)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_rlabel_position(325)
    ax2.set_rgrids(range(1, 10, 10), range(1, 10, 10))
    ax2.set_yticklabels([])
    ax2.grid(False)
    plt.ylim(0, 10)
    bars2 = ax2.bar(x=theta, height=radii, width=width, edgecolor='k', linewidth=1.5, color=colors)

    labels = []
    bin_number = np.arange(1, 17)
    for i in range(len(bin_number)):
        level = data.degrees[i]  # bin_number[i]
        legend_label = str(level) + "$^\circ$"
        # x = ax.annotate(legend_label, xy=(0.5, 0.5 - i*.1),
        #                ha='center', va='bottom', xycoords='figure fraction',
        #                bbox=dict(boxstyle='round', fc=bar_color[i], alpha=0.5))

        labels.append(legend_label)

    # https://stackoverflow.com/questions/65993526/adding-label-to-polar-chart-in-python
    for pos, label in zip(theta, labels):
        ax1.annotate(label, xy=(pos, 6), xytext=(0, 0), textcoords="offset pixels",
                     color='black', ha='center', va='center', fontsize=12, annotation_clip=False)

    for pos, label in zip(theta, labels):
        ax2.annotate(label, xy=(pos, 6), xytext=(0, 0), textcoords="offset pixels",
                     color='black', ha='center', va='center', fontsize=7, annotation_clip=False)


    print('finished showing 16')


def show_wind_rose8(fig):
    print("only plot 8 quadrants")
    '''
    Will show side-by-side plot showing wind rose quadrants in either degrees or quadrant name
    :param fig:
    :return:
    '''

    #bar_color = sns.color_palette("pastel", 8)

    bar_color = mcp.gen_color(cmap="rainbow", n=8)

    #bar_color = ['yellow', 'orange', 'red', 'lime', 'green', 'cyan', 'blue', 'magenta']

    nbins = 8
    deg2rad = pi / 180.0
    # Fixing random state for reproducibility
    bins = np.arange(- (180 / nbins), 360 + (180 / nbins), 360 / nbins)

    increment_per_bin = (360 / nbins) * 0.5
    bin_center_pt = bins + increment_per_bin
    bin_center_pt_radians = bin_center_pt * deg2rad
    theta_angles = bin_center_pt_radians[:-1]
    x = 360 - (180 / nbins)  # anything greater needs to shifted to fit into bins
    angle_size = (2 * np.pi / nbins)
    width = angle_size - .02

    radii = np.ones(nbins) * 10
    theta = theta_angles
    bin = np.zeros(nbins)
    # http://www.chiark.greenend.org.uk/~peterb/python/polar/
    data = pd.DataFrame({'compass': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W','NW'],
                         'degrees': [0, 45, 90, 135, 180, 225, 270, 315]})

    # fig = plt.figure(figsize=(8, 3))
    # fig.patch.set_facecolor('lightgoldenrodyellow')
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.bar(x=data['compass'], height=data['value'], width=1)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1.patch.set_facecolor('white')
    ax1.set_title('In Degrees', fontsize=12)
    xticks = 8
    #ax1.set_xticks([radians(x) for x in np.arange(0, 360, 360 / xticks)])
    ax1.set_xticklabels(data.degrees)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    #ax1.set_rlabel_position(325)
    #ax1.set_rgrids(range(0, 10, 10), range(0, 10, 10))
    ax1.set_yticklabels([])
    #ax1.axis('off')
    plt.grid(False)
    plt.ylim(0, 10)
    #colors = plt.cm.hsv(theta / 2 / np.pi)
    colors = bar_color
    bars1 = ax1.bar(x=theta, height=radii, width=width, edgecolor='k', linewidth=1.0, color=colors)
    fig.show()
    bars1 = ax1.bar(x=theta, height=radii, width=width, edgecolor='k', linewidth=1.0, color=colors)

    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    ax2.patch.set_facecolor('white')
    ax2.set_title('8 Quadrants', fontsize=12)
    xticks = 8
    ax2.set_xticks([radians(x) for x in np.arange(0, 360, 360 / xticks)])
    ax2.set_xticklabels(data.compass)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_rlabel_position(325)
    ax2.set_rgrids(range(1, 10, 10), range(1, 10, 10))
    ax2.set_yticklabels([])
    ax2.grid(False)
    plt.ylim(0, 10)
    bars2 = ax2.bar(x=theta, height=radii, width=width, edgecolor='k', linewidth=1.5, color=colors)

    labels = []
    bin_number = np.arange(1, xticks+1)
    for i in range(len(bin_number)):
        level = data.degrees[i] #bin_number[i]
        legend_label = str(level) + "$^\circ$"
        # x = ax.annotate(legend_label, xy=(0.5, 0.5 - i*.1),
        #                ha='center', va='bottom', xycoords='figure fraction',
        #                bbox=dict(boxstyle='round', fc=bar_color[i], alpha=0.5))

        labels.append(legend_label)

    # https://stackoverflow.com/questions/65993526/adding-label-to-polar-chart-in-python
    for pos, label in zip(theta, labels):
        ax1.annotate(label, xy=(pos, 6), xytext=(0, 0), textcoords="offset pixels",
                     color='black', ha='center', va='center', fontsize=12, annotation_clip=False)

    for pos, label in zip(theta, labels):
        ax2.annotate(label, xy=(pos, 6), xytext=(0, 0), textcoords="offset pixels",
                     color='black', ha='center', va='center', fontsize=12, annotation_clip=False)


    print('finished showing 8')

def process_image(no_quadrants):
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    #use either 8 or 16 quadrants

    if no_quadrants == 16:
        show_wind_rose(fig)
    elif no_quadrants == 8:
        show_wind_rose8(fig)
    else:
        print("not correct number of quadrants")

    return fig

    #plt.show()




def get_colors():
    '''
    This function gives the names of common colors. It allows us to
    get a list of names and then build colormaps without knowning how
    many different colors we will need.
    @return:
    '''
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    color_names = []

    n = len(sorted_names)
    ncols = 4
    nrows = n // ncols + 1

    show_colors = False
    if show_colors:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Get height and width
        X, Y = fig.get_dpi() * fig.get_size_inches()
        h = Y / (nrows + 1)
        w = X / ncols

    for i, name in enumerate(sorted_names):
        col = i % ncols
        if col != 0:
            continue

        row = i // ncols
        if row < 2:
            continue

        if show_colors:
            y = Y - (row * h) - h

            xi_line = w * (col + 0.05)
            xf_line = w * (col + 0.25)
            xi_text = w * (col + 0.3)

            ax.text(xi_text, y, name, fontsize=(h * 0.8),
                    horizontalalignment='left',
                    verticalalignment='center')

            ax.hlines(y + h * 0.1, xi_line, xf_line,
                      color=colors[name], linewidth=(h * 0.6))

            ax.set_xlim(0, X)
            ax.set_ylim(0, Y)
            ax.set_axis_off()

            fig.subplots_adjust(left=0, right=1,
                                top=1, bottom=0,
                                hspace=0, wspace=0)
            plt.show()
        else:
            color_names.append(name)

    return color_names


def main(argv):
    no_quadrants = 8
    fig8 = process_image(no_quadrants)

    no_quadrants = 16
    fig16 = process_image(no_quadrants)

    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])

    ax1 = fig8.axes[1]
    #ax1.remove()
    ax1.figure = fig
    fig.add_axes(ax1)
    ax1.set_subplotspec(gs[0, 1])
    #fig.add_subplot(gs[0, 1], projection='polar')

    ax2 = fig16.axes[1]
    #ax2.remove()
    ax2.figure = fig
    fig.add_axes(ax2)
    ax2.set_subplotspec(gs[0, 0])

    #plt.close(fig8)
    #plt.close(fig16)

    #fig16.ax1 = fig8.ax1

    plt.show()



if __name__ == "__main__":
    main(sys.argv[1:])
