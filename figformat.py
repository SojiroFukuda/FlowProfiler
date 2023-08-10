import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.font_manager import FontProperties
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import numpy as np
import os
# FIGURE INITIAL SETTING 
style.use('default')
plt.rcParams['font.family'] ='arial' # Font
plt.rcParams["mathtext.fontset"] = "dejavuserif" # mathfont
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['legend.title_fontsize'] = 8
fontP = FontProperties()
fontP.set_size('x-small')


def get_A4size(margin_ratio=0.8, FSIZE = 7, msize=10,linewidth=0.5):
    """Return the basic dimensions of A4 paper.

    Args:
        margin_ratio (float, optional): Ratio of your figures to the A4 dimensions. Defaults to 0.8.
        FSIZE (int, optional): Font size. Defaults to 7.
        msize (int, optional): Marker size. Defaults to 10.
        linewidth (float, optional): line thickness. Defaults to 0.5.

    Returns:
        key (Dict): Dictionary of A4 paper info.
    """
    centim = 1/2.54
    key = {'width':21.0 * centim * margin_ratio,
           'half':10.5 * centim * margin_ratio,
           'twothird':15.7 * centim * margin_ratio,
           'height':29.7 * centim * margin_ratio,
           'FSIZE': FSIZE,
           'msize': msize,
           'linewidth':linewidth
          }
    return key

def listdir_nohidden(path):
    """list up the directory except for the hidden files/folders

    Args:
        path (str): folder path.

    Yields:
        str : path
    """
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def subplot2gridlist(row,col,figsize):
    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(col):
        col_list = []
        for j in range(row):
            ax = plt.subplot2grid( (row,col), (j,i), rowspan=1,colspan=1)
            col_list.append(ax)
        axes.append(col_list)
    return fig, axes

def addlabels2axes(axes,position=(-0.1, 1.05),order='C',lowercase=True,fontsize=8):
    """Labeling axes in the alphabetical way.

    Args:
        axes (list): list of axes objects
        position (tuple, optional): The relative positions of each label. Defaults to (-0.1, 1.05).
        order (str, optional): The order of figures. The otion is {‘C’, ‘F’, ‘A’, ‘K’}. Defaults to 'C'. See documentation of numpy.flatten() for the details.
        lowercase (bool, optional): Set it False if you want to use uppercase. Defaults to True.
        fontsize (int, optional): Font size of label. Defaults to 8.
    """
    import string
    for i,ax in enumerate(np.array(axes).flatten(order=order)):
        if lowercase:
            ax.text(*position, string.ascii_lowercase[i], transform=ax.transAxes, size=fontsize, weight='bold')
        else:
            ax.text(*position, string.ascii_uppercase[i], transform=ax.transAxes, size=fontsize, weight='bold')

def adjustTicks(axes,xlim,ylim,xticks=[],xlabels=[],yticks=[],ylabels=[],FSIZE=8):
    """Adjust the position and values of ticks, xlim and ylim of figures which are aligned in a grid layout.

    Args:
        axes (list): list of matplotlib.axes objects.
        xlim (list): range of x axis. e.g. [0,1]
        ylim (list): range of y axis.
        xticks (list, optional): positions of ticks you want to display. Defaults to [].
        xlabels (list, optional): labels which will be displayed at each tick of which position you specified as xticks. Defaults to [].
        yticks (list, optional): positions of ticks you want to display. Defaults to [].
        ylabels (list, optional): labels which will be displayed at each tick of which position you specified as yticks.. Defaults to [].
        FSIZE (int, optional): Font size of text. Defaults to 8.
    """
    count = 0
    num_rows = len(axes[0])
    num_cols = len(axes)
    for i in range(num_rows): 
        row = i
        for j in range(num_cols):
            col = j
            ax = axes[j][i]
            ax.tick_params(axis='x',labelsize=FSIZE); ax.tick_params(axis='y',labelsize=FSIZE)
            ax.set_xticks(ticks=xticks,labels=xlabels,size=FSIZE); ax.set_yticks(ticks=yticks,labels=ylabels,size=FSIZE)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            if col != 0 and row+1 != num_rows:
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
            elif col == 0 and row+1 != num_rows:
                plt.setp(ax.get_xticklabels(), visible=False)
            elif col != 0 and row+1 == num_rows:
                plt.setp(ax.get_yticklabels(), visible=False)
            elif col == 0 and row+1 == num_rows:
                pass

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def legends(names,flip=False):
    figure_format_df = pd.DataFrame()
    figure_format_df['exp'] = names 
    # colors = colorlist(len(names),cmap=cmap)
    # markers = ['^','s','P','*','2','8','X','$\heartsuit$','o','<','h','>','x','$\clubsuit$','D','d',"v",'p','H','$\spadesuit$']
    markers = ['^','s','P','*','2','8','X','o','<','h','>','x','$\clubsuit$','D','d',"v",'p','H','$\spadesuit$']
    colors = sns.color_palette('colorblind',len(names)+1)
    # colors = ['#F72585','#B5179E','#F72585','#560BAD','#480CA8','#3A0CA3','#3F37C9','#F1C453','#4361EE','#4895EF','#4CC9F0','#2C699A','#048BA8','#560BAD','#83E377','#EFEA5A','#F72585','#EFEA5A',"#F1C453","#F29E4C",'#7209B7','#B5179E','#0DB39E']
    mlist_temp = []
    clist_temp = []
    if len(names) > len(markers):
        floornum = len(names)//len(markers)
        modnum = len(names)%len(markers)
        for i in range(floornum):
            mlist_temp += markers
        if modnum != 0:
            mlist_temp += markers[0:modnum]
    else:
        mlist_temp = markers[0:len(names)]
    if len(names) > len(colors):
        floornum = len(names)//len(colors)
        modnum = len(names)%len(colors)
        for i in range(floornum):
            clist_temp += colors
        if modnum != 0:
            clist_temp += colors[0:modnum]
    else:
        clist_temp = colors[0:len(names)]
    if flip:
        clist_temp.reverse()
    figure_format_df['marker'] = mlist_temp
    figure_format_df['color'] = clist_temp
    marker_onlyline = [",","1","2","3","4","+","x",'$\heartsuit$','$\clubsuit$']
    return figure_format_df, marker_onlyline

def colorlist(num,cmap='hls'):
    """Return a list of descrete colors from given colormap.

    Args:
        num (int): Length of the color list.
        cmap (str, optional): Name of the colormap from which you want to generate the color pallete.. Defaults to 'hls'.

    Returns:
        clist (list): A list of colors.
    """
    c_array = sns.color_palette(cmap)
    cnum = len(c_array)
    clist = []
    for i in range(num):
        ind = i%cnum
        clist.append(c_array[ind])
    from colormap import rgb2hex
    for i,cname in enumerate(clist):
        color = np.array(cname) * 255
        color = color.astype(int).tolist()
        color = rgb2hex(*color)
        clist[i] = color
    return clist

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

