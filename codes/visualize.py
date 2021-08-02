from ImageAnalysisCodes import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

try:
    import caiman as cm
except:
    print('no caiman available')

import pandas as pd

from colour import Color
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from cmath import rect, phase
from math import radians, degrees


eva_weightings = {
    'BackwardRight': [1, 0, 0.25],
    'Backward': [1, 0, 1],
    'BackwardLeft': [0.25, 0, 1],
    'Left': [0, 0.25, 1],
    'ForwardLeft': [0, 0.75, 1],
    'Forward': [0, 1, 0],
    'ForwardRight': [0.75, 1, 0],
    'Right': [1, 0.25, 0]
}


def color_returner(val, theta, threshold=0.5):

    if theta < 0:
        theta += 360

    if val >= threshold:
        # Forward
        if theta >= 337.5 or theta <= 22.5:
            outputColor = [0, 1, 0]

        # Forward Right
        elif 22.5 < theta <= 67.5:
            outputColor = [0.75, 1, 0]

        # Right
        elif 67.5 < theta <= 112.5:
            outputColor = [1, 0.25, 0]

        # Backward Right
        elif 112.5 < theta <= 157.5:
            outputColor = [1, 0, 0.25]

        # Backward
        elif 157.5 < theta <= 202.5:
            outputColor = [1, 0, 1]

        # Backward Left
        elif 202.5 < theta <= 247.5:
            outputColor = [0.25, 0, 1]

        # Left
        elif 247.5 < theta <= 292.5:
            outputColor = [0, 0.25, 1]

        # Forward Left
        elif 292.5 < theta <= 337.5:
            outputColor = [0, 0.75, 1]

        # if somehow we make it to here just make it gray
        else:
            outputColor = [0.66, 0.66, 0.66]

    else:
        # if not above some minimum lets make it gray
        outputColor = [0.66, 0.66, 0.66]
    return outputColor


def color_span(stim_df, c1='blue', c2='red'):
    if 'stimulus' not in stim_df.columns:
        print('please map stimuli first')
        return

    stim_df = stim_df[stim_df.stimulus.notna()]
    n_colors = stim_df.stimulus.nunique()

    clrs = list(Color(c1).range_to(Color(c2), n_colors))
    clrs = [i.rgb for i in clrs]

    loc_dic = {}
    z = 0
    for i in stim_df.stimulus.unique():
        loc_dic[i] = clrs[z]
        z += 1

    stim_df.loc[:, 'color'] = stim_df.stimulus.map(loc_dic)
    return stim_df


def color_generator(good_loadings, c1=None, c2=None):
    """
    can provide custom colors to step or let it handle
    """
    if c1 and c2 is not None:
        clrs = list(Color(c1).range_to(Color(c2), good_loadings.highest_loading.nunique()))
        corr_colors = [i.rgb for i in clrs]
    else:
        corr_colors = sns.color_palette(n_colors=good_loadings.highest_loading.nunique())

    factor_choice_dic = {}
    n = 0
    for i in good_loadings.highest_loading.unique():
        factor_choice_dic[i] = n
        n += 1

    choices = []
    for nrn in good_loadings.index:
        neuron = good_loadings.loc[nrn]
        factor_choice = neuron.highest_loading
        choices.append(corr_colors[factor_choice_dic[factor_choice]])
        rvals = []

    return choices, corr_colors


def plot_cells(_path):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return

    goodLoadings = utils.threshold_by_variance(eigen)

    cell_img = np.zeros((ops['Ly'], ops['Lx']))

    cells_fnd = goodLoadings.index

    for cell_num in cells_fnd:
        ypix = stats[iscell][cell_num]['ypix']
        xpix = stats[iscell][cell_num]['xpix']
        cell_img[ypix, xpix] = 1

    masked = np.ma.masked_where(cell_img < 0.9, cell_img)

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(np.mean(image, axis=0), cmap=mpl.cm.gray)
    ax.imshow(masked, cmap=mpl.cm.gist_rainbow, interpolation=None, alpha=1)
    return


def plot_factor(_path, factors=[0,1,2], variance=0.5):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return
    goodLoadings = utils.threshold_by_variance(eigen, variance)

    factor_strings = []
    if isinstance(factors, int):
        factors = [factors]

    for i in factors:
        factor_str = "FA" + str(i)
        factor_strings.append(factor_str)

    goodLoadings = goodLoadings[goodLoadings.highest_loading.isin(factor_strings)]

    colors, factor_colors = color_generator(goodLoadings)

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in range(len(colors)):

        real_cell = goodLoadings.index[cell]
        ypix = stats[iscell][real_cell]['ypix']
        xpix = stats[iscell][real_cell]['xpix']

        for c in range(cell_img.shape[2]):
            cell_img[ypix, xpix, c] = colors[cell][c]

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(np.rot90(np.mean(image, axis=0),-1), cmap=mpl.cm.gray)
    ax.imshow(np.rot90(cell_img,-1), interpolation=None, alpha=0.7)

    c_lines = [Line2D([0], [0], color=factor_colors[i], lw=4) for i in range(len(factor_colors))]

    ax.legend(c_lines, ['factor ' + str(i) for i in range(len(factor_colors))], loc='upper right', fontsize='x-large')
    return


def plot_factor_all(_path):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return

    goodLoadings = utils.threshold_by_variance(eigen)

    colors, factor_colors = color_generator(goodLoadings)

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in range(len(colors)):

        real_cell = goodLoadings.index[cell]
        ypix = stats[iscell][real_cell]['ypix']
        xpix = stats[iscell][real_cell]['xpix']

        for c in range(cell_img.shape[2]):
            cell_img[ypix, xpix, c] = colors[cell][c]

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(np.mean(image, axis=0), cmap=mpl.cm.gray)
    ax.imshow(cell_img, interpolation=None, alpha=0.7)

    c_lines = [Line2D([0], [0], color=factor_colors[i], lw=4) for i in range(len(factor_colors))]

    ax.legend(c_lines, ['factor ' + str(i) for i in range(len(factor_colors))], loc='upper right', fontsize='x-large')
    return


def full_plot(_path, save=None, age=None, variance=0.5):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        stim_df = utils.load_stimuli(_paths['stimuli']['frame_aligned'])
    except KeyError:
        print('please align stimuli first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return

    goodLoadings = utils.threshold_by_variance(eigen, variance)

    factors, loadings, x = eigen

    cell_img = np.zeros((ops['Ly'], ops['Lx']))
    cells_fnd = goodLoadings.index

    a = []
    for val in range(goodLoadings.highest_loading.nunique()):
        a.append(len(goodLoadings[goodLoadings.highest_loading == goodLoadings.highest_loading.unique()[val]]))

    for cell_num in cells_fnd:
        ypix = stats[iscell][cell_num]['ypix']
        xpix = stats[iscell][cell_num]['xpix']
        cell_img[ypix, xpix] = 1

    masked = np.ma.masked_where(cell_img < 0.9, cell_img)
    mean_img = np.mean(image, axis=0)

    colors, factor_colors = color_generator(goodLoadings)
    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in range(len(colors)):

        real_cell = goodLoadings.index[cell]
        ypix = stats[iscell][real_cell]['ypix']
        xpix = stats[iscell][real_cell]['xpix']

        for c in range(cell_img.shape[2]):
            cell_img[ypix, xpix, c] = colors[cell][c]


    c_lines = [Line2D([0], [0], color=factor_colors[i], lw=4) for i in range(len(factor_colors))]
    c_legends = ['factor ' + str(i) for i in range(len(factor_colors))]

    fig, ax = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)

    ax[0].imshow(mean_img, cmap=mpl.cm.gray)
    ax[0].imshow(masked, cmap=mpl.cm.gist_rainbow, interpolation=None, alpha=1)

    ax[1].imshow(mean_img, cmap=mpl.cm.gray)
    ax[1].imshow(cell_img, interpolation=None, alpha=0.7)

    ax[1].legend(c_lines, c_legends, loc='upper right', fontsize='medium')
    plt.tight_layout()
    if save is not None:
        pre_sve_path = save
        ind = pre_sve_path.find('.')
        mid_string = "brain_pics" + "age_" + str(age)
        final_save_path = pre_sve_path[:ind] + mid_string + pre_sve_path[ind:]
        plt.savefig(final_save_path, dpi=300)

    plt.show()

    plt.figure(figsize=(18, 18))

    bars = plt.subplot2grid((5, 5), (0, 0), colspan=2, rowspan=3)
    bars.bar(x=np.arange(0, len(a), 1), height=a)
    bars.set_xticks(np.arange(0, len(a), 1))
    bars.set_ylabel('Neurons')
    bars.set_xlabel('Factors')
    bars.set_title('Neurons per Factor')

    a1 = plt.subplot2grid((5, 5), (0, 2), colspan=3, rowspan=1)
    a1.plot(factors[:, 0])
    a1.set_title('Factor 0')

    a2 = plt.subplot2grid((5, 5), (1, 2), colspan=3, rowspan=1)
    a2.plot(factors[:, 1])
    a2.set_title('Factor 1')

    a3 = plt.subplot2grid((5, 5), (2, 2), colspan=3, rowspan=1)
    a3.plot(factors[:, 2])
    a3.set_title('Factor 2')

    for j in [a1, a2, a3]:
        for i in range(len(stim_df)):
            j.axvspan(stim_df.iloc[i].start, stim_df.iloc[i].stop, color=stim_df.iloc[i].color, alpha=0.3,
                      label=stim_df.iloc[i].stimulus)

    plt.tight_layout()

    if save is not None:
        pre_sve_path = save
        ind = pre_sve_path.find('.')
        mid_string = "Factors" + "age_" + str(age)
        final_save_path = pre_sve_path[:ind] + mid_string + pre_sve_path[ind:]
        plt.savefig(final_save_path, dpi=300)


def plot_factor_top6(_path, factor=0):

    _paths = utils.pathSorter(_path)

    try:
        stim_df = utils.load_stimuli(_paths['stimuli']['frame_aligned'])
    except KeyError:
        print('please align stimuli first')
        return
    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return


    goodLoadings = utils.threshold_by_variance(eigen)

    factors, loadings, x = eigen

    factorplt = factors[:, factor]
    factor_str = "FA" + str(factor)
    top6 = goodLoadings.sort_values(by=factor_str, ascending=False).iloc[:6].index

    fig = plt.figure(1, figsize=(16, 16))

    AMAIN = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    AMAIN.plot(factorplt, c='black')

    A1 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    A1.plot(x[top6].iloc[:, 0], c='black')

    A2 = plt.subplot2grid((4,4), (1,2), colspan=2 )
    A2.plot(x[top6].iloc[:, 1], c='black')

    B1 = plt.subplot2grid((4,4), (2,0), colspan=2 )
    B1.plot(x[top6].iloc[:, 2], c='black')

    B2 = plt.subplot2grid((4,4), (2,2), colspan=2 )
    B2.plot(x[top6].iloc[:, 3], c='black')

    C1 = plt.subplot2grid((4,4), (3,0), colspan=2 )
    C1.plot(x[top6].iloc[:, 4], c='black')

    C2 = plt.subplot2grid((4,4), (3,2), colspan=2 )
    C2.plot(x[top6].iloc[:, 5], c='black')

    for j in [AMAIN, A1, A2, B1, B2, C1, C2]:
        for i in range(len(stim_df)):
            j.axvspan(stim_df.iloc[i].start, stim_df.iloc[i].stop, color=stim_df.iloc[i].color, alpha=0.3,
                      label=stim_df.iloc[i].stimulus)

    handles, labels = AMAIN.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    AMAIN.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    return


def plot_avg_stimulus(_path, stim=['Rx'], sig=0.7, offset=5, threshold=2.0):
    # choices must come from stimdic
    stimdic = {"B": [0, 0], "F": [0, 1], "RR": [0, 2], "LL": [0, 3],
               "RR": [1, 0], "Rx": [1, 1], "xR": [1, 2],
               "LL": [2, 0], "Lx": [2, 1], "xL": [2, 2],
               "D": [3, 0], "C": [3, 1]}

    _paths = utils.pathSorter(_path)

    try:
        factors, loadings, x = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return

    try:
        stim_df = utils.load_stimuli(_paths['stimuli']['frame_aligned'])
    except KeyError:
        print('please align stimuli first')
        return

    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return
    plot_df = stim_df[stim_df.stimulus.isin(stim)]

    selection = []
    for nrn in range(f_cells[iscell].shape[0]):
        s = plot_df.iloc[0].start
        e = plot_df.iloc[0].stop

        neuron = f_cells[iscell][nrn]

        if np.mean(neuron[s + offset:e + offset]) / np.mean(neuron[s - 2 * offset:s]) >= threshold:
            selection.append(nrn)
    neurons = []
    for i in selection:
        neurons.append((f_cells[iscell][i] - np.mean(f_cells[iscell][i][:15])) / np.mean(f_cells[iscell][i][:15]))

    neurons = np.array(neurons)

    neurons = gaussian_filter(neurons, sigma=sig)

    fig, ax = plt.subplots(4, 4, figsize=(16, 16), sharey=True)
    for i in range(stim_df.stimulus.nunique()):
        stims = [stim_df.stimulus.unique()[i]]
        plot_df = stim_df[stim_df.stimulus.isin(stims)]

        ind1 = stimdic[stims[0]][0]
        ind2 = stimdic[stims[0]][1]

        for ind in range(len(plot_df)):
            ax[ind1, ind2].plot(
                np.mean(neurons[:, plot_df.start.values[ind] - offset:plot_df.stop.values[ind] + offset], axis=0),
                label="trial" + str(ind))

        ax[ind1, ind2].set_title("stimulus: " + stims[0])
        ax[ind1, ind2].axvspan(offset,
                               offset + abs(plot_df.start.values[ind] - offset - plot_df.stop.values[ind] + offset),
                               color='red', alpha=0.2)

    plt.show()
    return


def plot_avg_stimulus_neurons(_path, stim=['Rx'], sig=0.7, offset=5, threshold=2.0):
    # choices must come from stimdic
    stimdic = {"B": [0, 0], "F": [0, 1], "RR": [0, 2], "LL": [0, 3],
               "RR": [1, 0], "Rx": [1, 1], "xR": [1, 2],
               "LL": [2, 0], "Lx": [2, 1], "xL": [2, 2],
               "D": [3, 0], "C": [3, 1]}

    paths = utils.pathSorter(_path)

    factors, loadings, x = utils.load_eigen(paths['output']['eigenvalues'])
    stim_df = utils.load_stimuli(paths['stimuli']['frame_aligned'])
    ops, iscell, stats, f_cells = utils.load_suite2p(paths['output']['suite2p'])

    plot_df = stim_df[stim_df.stimulus.isin(stim)]

    plot_df = stim_df[stim_df.stimulus.isin(stim)]

    selection = []
    for nrn in range(f_cells[iscell].shape[0]):
        s = plot_df.iloc[0].start
        e = plot_df.iloc[0].stop

        neuron = f_cells[iscell][nrn]

        if np.mean(neuron[s + offset:e + offset]) / np.mean(neuron[s - 2 * offset:s]) >= threshold:
            selection.append(nrn)
    neurons = []
    for i in selection:
        neurons.append((f_cells[iscell][i] - np.mean(f_cells[iscell][i][:15])) / np.mean(f_cells[iscell][i][:15]))

    neurons = np.array(neurons)

    neurons = gaussian_filter(neurons, sigma=sig)

    fig, ax = plt.subplots(neurons.shape[0], figsize=(neurons.shape[0], neurons.shape[0]))

    for i in range(neurons.shape[0]):
        ax[i].plot(neurons[i])

        for x in range(len(plot_df)):
            ax[i].axvspan(plot_df.start.values[x] + offset, plot_df.stop.values[x] + offset, color='red', alpha=0.2)

    plt.show()
    return


def binocMap(frameAlignedStims):
    stims = utils.map_stimuli(frameAlignedStims)
    b = {'F': 'Forward', 'RR': 'Right', 'LL': 'Left', 'B': 'Backward'}
    stims.loc[:, 'stimulus_name'] = stims.stimulus.map(b)
    return stims


def pixelwise(base_path, stimOffset=5):
    imgpath = utils.pathSorter(base_path)['image']['move_corrected']
    frameAlignedStimsPath = utils.pathSorter(base_path)['stimuli']['frame_aligned']

    eva_weightings = {
        'BackwardRight': [1, 0, 0.25],
        'Backward': [1, 0, 1],
        'BackwardLeft': [0.25, 0, 1],
        'Left': [0, 0.25, 1],
        'ForwardLeft': [0, 0.75, 1],
        'Forward': [0, 1, 0],
        'ForwardRight': [0.75, 1, 0],
        'Right': [1, 0.25, 0]
    }

    img = cm.load(imgpath)
    frameAlignedStims = pd.read_hdf(frameAlignedStimsPath)

    try:
        frameAlignedStims.loc[:, 'stimulus'] = frameAlignedStims.stimulus_name.values
    except AttributeError:
        frameAlignedStims = binocMap(frameAlignedStims)
        frameAlignedStims.loc[:, 'stimulus'] = frameAlignedStims.stimulus_name.values

    a = frameAlignedStims[(frameAlignedStims.velocity != 0)]
    staticIndices = a[(a.velocity_0 == 0) & (a.velocity_1 == 0)].index
    stim_df = utils.stimStartStop(a.drop(staticIndices))

    statInds = []
    for i in frameAlignedStims.loc[staticIndices].img_stacks.values:
        for j in i:
            statInds.append(j)

    bg_image = np.mean(img[statInds], axis=0)

    frames, x, y = img.shape

    all_img = []
    for stim in stim_df[stim_df.stimulus.isin(eva_weightings.keys())].stimulus.unique():
        _frames = utils.arrangedArrays(stim_df[stim_df.stimulus == stim], offset=stimOffset)

        stimImage = np.mean(img[_frames], axis=0) - bg_image
        stimImage[stimImage < 0] = 0

        rgb = np.zeros((3, x, y), 'float64')

        rgb[0, :, :] = stimImage * eva_weightings[stim][0]
        rgb[1, :, :] = stimImage * eva_weightings[stim][1]
        rgb[2, :, :] = stimImage * eva_weightings[stim][2]

        r = rgb[0, :, :]
        g = rgb[1, :, :]
        b = rgb[2, :, :]

        r = r - r.min()
        b = b - b.min()
        g = g - g.min()
        mymax = np.max(np.dstack((r, g, b)))
        all_img.append(np.dstack((r ** 1.5, g ** 1.5, b ** 1.5)))

    _all_img = []
    for img in all_img:
        _all_img.append(img / np.max(all_img))

    fin_img = np.sum(_all_img, axis=0)
    fin_img /= np.max(fin_img)

    return fin_img, _all_img


def weighted_mean_angle(degs, weights):
    _sums = []
    for d in range(len(degs)):
        _sums.append(weights[d]*rect(1, radians(degs[d])))
    return degrees(phase(sum(_sums)/np.sum(weights)))


def pixelwise2(base_path, stimOffset=5):
    imgpath = utils.pathSorter(base_path)['image']['move_corrected']
    frameAlignedStimsPath = utils.pathSorter(base_path)['stimuli']['frame_aligned']

    img = cm.load(imgpath)
    frameAlignedStims = pd.read_hdf(frameAlignedStimsPath)

    frameAlignedStims = binocMap(frameAlignedStims)
    frameAlignedStims.loc[:, 'stimulus'] = frameAlignedStims.stimulus_name.values

    stim_df = utils.stimStartStop(frameAlignedStims)

    statInds = utils.arrangedArrays(stim_df[stim_df.stimulus.isna()])
    try:
        bg_image = np.mean(img[statInds], axis=0)
    except IndexError:
        bg_image = np.mean(img[statInds[:-15]], axis=0)

    frames, x, y = img.shape
    stim_df.stimulus = stim_df.stimulus.map({'F': 'Forward', 'RR': 'Right', 'LL': 'Left', 'B': 'Backward'})

    all_img = []
    for stim in stim_df[stim_df.stimulus.isin(eva_weightings.keys())].stimulus.unique():
        _frames = utils.arrangedArrays(stim_df[stim_df.stimulus == stim], offset=stimOffset)

        try:
            stimImage = np.mean(img[_frames], axis=0) - bg_image
        except IndexError:
            stimImage = np.mean(img[_frames[:-5]], axis=0) - bg_image

        stimImage[stimImage < 0] = 0

        rgb = np.zeros((3, x, y), 'float64')

        rgb[0, :, :] = stimImage * eva_weightings[stim][0]
        rgb[1, :, :] = stimImage * eva_weightings[stim][1]
        rgb[2, :, :] = stimImage * eva_weightings[stim][2]

        r = rgb[0, :, :]
        g = rgb[1, :, :]
        b = rgb[2, :, :]

        r = r - r.min()
        b = b - b.min()
        g = g - g.min()
        mymax = np.max(np.dstack((r, g, b)))
        all_img.append(np.dstack((r ** 1.5, g ** 1.5, b ** 1.5)))

    _all_img = []
    for img in all_img:
        _all_img.append(img / np.max(all_img))

    fin_img = np.sum(_all_img, axis=0)
    fin_img /= np.max(fin_img)

    return fin_img, _all_img


def spatial_neurons(base_path, x=[0,9999], y=[0,9999]):
    ops, iscell, stats, f_cells = utils.load_suite2p(utils.pathSorter(base_path)['output']['suite2p'])
    i = 0
    xs = []
    ys = []
    nrns = np.arange(0, len(stats[iscell]))
    for i in range(len(stats[iscell])):
        ys.append(np.mean(stats[iscell][i]['ypix']))
        xs.append(np.mean(stats[iscell][i]['xpix']))

    locs = pd.DataFrame({'neuron' : nrns, 'x' : xs, 'y' : ys})
    return locs[(locs.x>=x[0])&(locs.x<=x[1])&((locs.y>=y[0]))&((locs.y<=y[1]))]


def neuronColor(basePath, threshold=1.5, offset=5, figshape=(12,12), plot=False, returnTuneds=False, ret_cells=True, loc=None):
    ops, iscell, stats, f_cells = utils.load_suite2p(utils.pathSorter(basePath)['output']['suite2p'])
    image = cm.load(utils.pathSorter(basePath)['image']['move_corrected'])

    stim_df, cells = utils.stim_cell_returner(basePath)

    if loc is not None:
        cell_nums = spatial_neurons(basePath, loc[0], loc[1]).neuron.values
        cells = cells[cell_nums, :]

    monocStims = ['Backward', 'Forward', 'BackwardLeft', 'ForwardRight', 'ForwardLeft', 'BackwardRight', 'Right',
                  'Left']
    mstim_df = stim_df[stim_df.stimulus.isin(monocStims)]
    if len(mstim_df) == 0:
        stim_df.stimulus = stim_df.stimulus.map({'F': 'Forward', 'RR': 'Right', 'LL': 'Left', 'B': 'Backward'})
        mstim_df = stim_df[stim_df.stimulus.isin(monocStims)]

    stimuli = []
    neurons = []
    finVals = []
    finVals_std = []
    for w in range(len(cells)):
        x = cells[w]

        for s in mstim_df.stimulus.unique():
            a = mstim_df[mstim_df.stimulus == s]
            nrnRespones = []

            for n in range(len(a)):
                _i = np.arange(a.iloc[n].start + offset, a.iloc[n].stop + offset)
                try:
                    maxInd = np.nanargmax(x[_i])
                    nrnRespones.append(x[_i][maxInd])
                except:
                    nrnRespones.append(0)

            finVals.append(np.nanmean(nrnRespones))
            finVals_std.append(np.nanstd(nrnRespones))

            stimuli.append(s)
            if loc is None:
                neurons.append(w)
            else:
                neurons.append(cell_nums[w])

    celldf = pd.DataFrame({'neuron': neurons, 'stimulus': stimuli, 'val': finVals, 'error': finVals_std})

    tunedCells = celldf[celldf.val>=threshold]
    if returnTuneds:
        return tunedCells, celldf, iscell, stats, f_cells, mstim_df, stim_df

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in tunedCells.neuron.unique():
        ypix = stats[iscell][cell]['ypix']
        xpix = stats[iscell][cell]['xpix']

        for c in range(cell_img.shape[2]):

            _df = celldf[celldf.neuron == cell]
            _df.val = _df.val / _df.val.sum()
            for _s in _df.stimulus.unique():
                _val = _df[_df.stimulus == _s].val.values

                cell_img[ypix, xpix, c] += eva_weightings[_s][c] * _val
    if plot:
        plt.figure(figsize=figshape)
        plt.imshow(cell_img)

        _cmap = eva_weightings.values()

        fig, ax = plt.subplots(figsize=(figshape[0], 1))
        fig.subplots_adjust(bottom=0.5)

        cmap = mpl.colors.ListedColormap(_cmap)
        cmap.set_over('0.25')
        cmap.set_under('0.75')

        bounds = list(np.arange(cmap.N))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing='proportional',
                                        orientation='horizontal')

        cb2.ax.set_xticklabels(eva_weightings.keys())

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.show()
    return cell_img, ops, celldf


def plot_cell(_path, plottedCells):
    # plots all cells included in a list

    _paths = utils.pathSorter(_path)
    ops, iscell, _stats, _f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    stats = _stats[iscell]
    f_cells = _f_cells[iscell]
    cell_img = np.zeros((ops['Ly'], ops['Lx']))

    for cell in plottedCells:
        ypix = stats[cell]['ypix']
        xpix = stats[cell]['xpix']
        cell_img[ypix, xpix] = 1

    masked = np.ma.masked_where(cell_img == 0, cell_img)

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(ops['refImg'], cmap=mpl.cm.gray)
    ax.imshow(masked, cmap=mpl.cm.gist_rainbow, interpolation=None, alpha=1)


def plot_factor_corr(_path, fullCorr, minThresh=0.2):
    _paths = utils.pathSorter(_path)
    ops, iscell, _stats, _f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    stats = _stats[iscell]
    f_cells = _f_cells[iscell]

    cell_img = np.zeros((ops['Ly'], ops['Lx']))

    for nrn in range(len(fullCorr)):

        corrval = fullCorr[nrn]

        if abs(corrval) >= minThresh:
            ypix = stats[nrn]['ypix']
            xpix = stats[nrn]['xpix']
            cell_img[ypix, xpix] = corrval

    masked = np.ma.masked_where(cell_img == 0, cell_img)

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(ops['refImg'], cmap=mpl.cm.gray)
    ax.imshow(masked, cmap=mpl.cm.bwr, interpolation=None, alpha=1)

    fig, ax = plt.subplots(figsize=(10, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.bwr
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Correlation')
    fig.show()