from tkinter import Tk
from tkinter import filedialog as fd
from pathlib import Path
from nptdms import TdmsFile
from datetime import datetime as dt
from datetime import date

import datetime
import shutil
import os
import copy

try:
    import caiman as cm
except:
    print('caiman not available')

import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)


def pathSorter(parentPath=None, key=None):
    """
    finds and returns specific paths to items from a home folder

    :param parentPath: central directory to data belonging to an experiment
    :param key: can request a specific path to be returned
    :return: returns requested path or all paths
    """

    # if a path is not provided, provides a popup window to select data source
    if parentPath is None:
        root = Tk()
        root.update()
        parentPath = fd.askdirectory(parent=root, title='Please Select Data Folder')
        root.destroy()

    _paths = {'stimuli': {}, 'image': {}, 'output': {}, 'etc': {}}

    volumetric=False
    with os.scandir(parentPath) as entries:
        for entry in entries:
            if entry.is_file():

                if entry.name.endswith('Notes.txt'):
                    _paths['etc']['notes'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('Params.json'):
                    _paths['etc']['moveCorrectionParams'] = Path(parentPath).joinpath(entry.name)

                if entry.name.endswith('.tif'):
                    _paths['image']['raw'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('Timestamps.tdms'):
                    _paths['image']['timestamps'] = Path(parentPath).joinpath(entry.name)
                if 'timestamps.txt' in entry.name:
                    _paths['image']['new_timestamps'] = Path(parentPath).joinpath(entry.name)

                if 'frametimes.h5' in entry.name:
                    _paths['image']['new_timestamps_processed'] = Path(parentPath).joinpath(entry.name)


                if ('stims' in entry.name or entry.name.startswith('fish')) and entry.name.endswith('.txt'):
                    _paths['stimuli']['raw'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('.h5') and entry.name.startswith('fish'):
                    _paths['stimuli']['processed'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('aligned.h5'):
                    _paths['stimuli']['frame_aligned'] = Path(parentPath).joinpath(entry.name)

                if entry.name.endswith('_tail.tdms'):
                    _paths['stimuli']['tail'] = Path(parentPath).joinpath(entry.name)

                if 'eigen' in entry.name:
                    _paths['output']['eigenvalues'] = Path(parentPath).joinpath(entry.name)

            if 'planes' in entry.name:
                volumetric = True
                volumePaths = Path(parentPath).joinpath('planes')

                if not os.path.exists(volumePaths.joinpath('move_corrected')):
                    print('please move correct volume first')
                    return
                else:
                    volumePaths = volumePaths.joinpath('move_corrected')

    if volumetric:
        _paths['image']['volume'] = {}

        with os.scandir(volumePaths) as entries:
            for entry in entries:
                _paths['image']['volume'][entry.name] = volumePaths.joinpath(entry.name)



    moveCorrectedPath = Path(parentPath).joinpath('move_corrected')
    if moveCorrectedPath.exists():
        with os.scandir(moveCorrectedPath) as entries:
            for entry in entries:
                if entry.name.endswith('.tif'):
                    _paths['image']['move_corrected'] = Path(moveCorrectedPath).joinpath(entry.name)

    suite2pPath = Path(parentPath).joinpath('move_corrected').joinpath('suite2p')
    if suite2pPath.exists():
        _paths['output']['suite2p'] = {
            "iscell": Path(suite2pPath).joinpath('plane0/iscell.npy'),
            "stats": Path(suite2pPath).joinpath('plane0/stat.npy'),
            "ops": Path(suite2pPath).joinpath('plane0/ops.npy'),
            "f_cells": Path(suite2pPath).joinpath('plane0/F.npy'),
            "f_neuropil": Path(suite2pPath).joinpath('plane0/Fneu.npy'),
            "spikes": Path(suite2pPath).joinpath('plane0/spks.npy'),
            "data": Path(suite2pPath).joinpath('plane0/data.bin')
        }
    if key is not None:
        return _paths[key]
    else:
        return _paths


def load_suite2p(_paths):
    try:
        ops = np.load(_paths['ops'], allow_pickle=True).item()
        iscell = np.load(_paths['iscell'], allow_pickle=True)[:, 0].astype(bool)
        stats = np.load(_paths['stats'], allow_pickle=True)
        f_cells = np.load(_paths['f_cells'])
        return ops, iscell, stats, f_cells
    except KeyError:
        print('please run suite2p neuron extraction first')
    except TypeError:
        ops = np.load(_paths.joinpath('ops.npy'), allow_pickle=True).item()
        iscell = np.load(_paths.joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
        stats = np.load(_paths.joinpath('stat.npy'), allow_pickle=True)
        f_cells = np.load(_paths.joinpath('F.npy'))
        return ops, iscell, stats, f_cells


def load_eigen(eigen_path):
    return np.load(eigen_path, allow_pickle=True)['arr_0']


def load_stimuli(stimuli_path):
    # requires stimuli to be mapped onto df
    df = pd.read_hdf(stimuli_path)
    stim_df = stimStartStop(df)
    stim_df = stim_df[stim_df.stimulus.notna()]
    stim_df.reset_index()
    return colormapStimuli(stim_df)


def map_raw_stimuli(stims):
    monocDic = {
        0:'Forward',
        45:'ForwardRight',
        90:'Right',
        135:'BackwardRight',
        180:'Backward',
        225:'BackwardLeft',
        270:'Left',
        315:'ForwardLeft'
    }

    stims.loc[:, 'stimulus_name'] = '---'
    inds = stims.loc[(stims.stim_type == 's')].index
    vals = stims.loc[(stims.stim_type == 's')].angle.map(monocDic)
    stims.loc[inds, 'stimulus_name'] = vals

    angs = stims[(stims.stim_type == 'b')].angle.values
    ang_inds = stims[(stims.stim_type == 'b')].index
    rights = []
    lefts = []
    x = 0
    for a in angs:
        if a == [90, 90]:
            rights.append(x)
        elif a == [270, 270]:
            lefts.append(x)
        x += 1

    fullrights = []
    medrights = []
    latrights = []
    n = 0
    for v in stims.loc[ang_inds[rights]].velocity:
        if v[0] != 0 and v[1] != 0:
            fullrights.append(n)
        elif v[0] == 0 and v[1] != 0:
            latrights.append(n)
        elif v[0] != 0 and v[1] == 0:
            medrights.append(n)
        n += 1

    fulllefts = []
    medlefts = []
    latlefts = []
    n = 0
    for v in stims.loc[ang_inds[lefts]].velocity:
        if v[0] != 0 and v[1] != 0:
            fulllefts.append(n)
        elif v[0] == 0 and v[1] != 0:
            medlefts.append(n)
        elif v[0] != 0 and v[1] == 0:
            latlefts.append(n)
        n += 1

    for i in fullrights:
        stims.loc[ang_inds[rights][i], 'stimulus_name'] = 'RR'
    for i in medrights:
        stims.loc[ang_inds[rights][i], 'stimulus_name'] = 'Rx'
    for i in latrights:
        stims.loc[ang_inds[rights][i], 'stimulus_name'] = 'xR'

    for i in fulllefts:
        stims.loc[ang_inds[lefts][i], 'stimulus_name'] = 'LL'
    for i in medlefts:
        stims.loc[ang_inds[lefts][i], 'stimulus_name'] = 'xL'
    for i in latlefts:
        stims.loc[ang_inds[lefts][i], 'stimulus_name'] = 'Lx'

    return stims


def map_stimuli(stims):
    """
    Maps binocular stimuli

    Rx - medial right
    xR lateral right

    Lx lateral left
    xL medial left

    F forward
    B backward

    C converging
    D divering

    Easy to add different stims or replace this function

    :param stims:
    :return:
    """


    stims.loc[(stims['velocity_0'] == 0) & (stims['velocity_1'] != 0) & (stims['angle_1'] == 90), 'stimulus'] = 'Rx'
    stims.loc[(stims['velocity_0'] == 0) & (stims['velocity_1'] != 0) & (stims['angle_1'] == 270), 'stimulus'] = 'Lx'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] == 0) & (stims['angle_1'] == 90), 'stimulus'] = 'xR'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] == 0) & (stims['angle_1'] == 270), 'stimulus'] = 'xL'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 0) & (
                stims['angle_1'] == 0), 'stimulus'] = 'F'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 180) & (
                stims['angle_1'] == 180), 'stimulus'] = 'B'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 90) & (
                stims['angle_1'] == 90), 'stimulus'] = 'RR'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 270) & (
                stims['angle_1'] == 270), 'stimulus'] = 'LL'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 270) & (
                stims['angle_1'] == 90), 'stimulus'] = 'C'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 90) & (
                stims['angle_1'] == 270), 'stimulus'] = 'D'

    return stims


def colormapStimuli(stim_df):
    stims = stim_df.stimulus.unique()
    clrs = sns.color_palette(n_colors=len(stims))

    color_dict = {}
    x=0
    for stim in stims:
        color_dict[stim] = clrs[x]
        x+=1

    try:
        stim_df.loc[:, 'color'] = stim_df.stimulus.map(color_dict)
    except NotImplementedError:
        stim_df.loc[:, 'color'] = stim_df.stimulus.astype(str).map(color_dict)
    except TypeError:
        color_dict = {}
        x = 0
        for stim in stims:
            color_dict[stim] = [clrs[x]]
            x += 1
        stim_df.loc[:, 'color'] = stim_df.stimulus.map(color_dict)

    return stim_df


def stimStartStop(stims):
    '''
    agnostic to what type of stimulus, it just needs to have stimuli labeled and include start/stop

    :param stims:
    :return:
    '''
    if not 'stimuli' in stims.columns:
        try:
            stims = map_stimuli(stims)
        except:
            print('error: please map stimuli on dataframe')
            return

    stimuli = []
    starts = []
    stops = []

    for row in range(len(stims)):
        data = stims.iloc[row]

        stimuli.append(data.stimulus)
        starts.append(data.img_stacks.values[0])
        stops.append(data.img_stacks.values[-1])

    stim_dict = {
        'stimulus': stimuli,
        'start': starts,
        'stop': stops
    }

    return pd.DataFrame(stim_dict)


def arrangedArrays(df, offset=5):
    '''
    only feed a dataframe with one stimulus and itll filter the repeats and return a single array image stacks

    '''

    if not 'start' in df.columns:
        print('error: use start stop format for stimuli')
        return
    a = []
    for repeat in range(len(df)):
        s = df.iloc[repeat].start + offset
        e = df.iloc[repeat].stop + offset
        a.append(np.arange(s, e))

    b = []
    for i in a:
        for j in i:
            b.append(j)
    return np.array(b)


def arrangedArrays2(df, offset=10):
    # like the other but it gives you an offset back and forward
    a = []
    for repeat in range(len(df)):
        s = df.iloc[repeat].start - offset
        e = df.iloc[repeat].stop + offset
        a.append(np.arange(s, e))

    b = []
    for i in a:
        for j in i:
            b.append(j)
    return np.array(b)


def neuronResponses(neuronArray, isCellArray, stimulusDf):

    if not 'start' in stimulusDf.columns:
        try:
            stimulusDf = stimStartStop(stimulusDf)
        except:
            print('error: use start stop format for stimuli')
            return

    nrns = np.where(isCellArray == 1)[0]
    neuronArray = neuronArray[isCellArray]
    neuron_responses = []

    for nrn_ind in range(len(neuronArray)):

        nrn_dict = {}
        nrn = neuronArray[nrn_ind]

        background_val = np.median(nrn)
        nrn_dict['bg'] = background_val

        for stim in sorted(stimulusDf.stimulus.unique()):
            stim_indices = arrangedArrays(stimulusDf[stimulusDf.stimulus == stim])
            stim_val = np.median(nrn[stim_indices])

            nrn_dict[stim] = stim_val

        neuron_responses.append(nrn_dict)
    neuron_df = pd.DataFrame(neuron_responses)
    neuron_df.loc[:, 'neuron'] = nrns
    return neuron_df


def neuronResponsesFdff(neuronArray, isCellArray, stimulusDf):

    if not 'start' in stimulusDf.columns:
        try:
            stimulusDf = stimStartStop(stimulusDf)
        except:
            print('error: use start stop format for stimuli')
            return

    nrns = np.where(isCellArray == 1)[0]
    neuronArray = neuronArray[isCellArray]
    neuron_responses = []

    for nrn_ind in range(len(neuronArray)):

        nrn_dict = {}
        nrn = neuronArray[nrn_ind]

        background_val = np.median(nrn)
        nrn_dict['bg'] = background_val

        for stim in sorted(stimulusDf.stimulus.unique()):
            stim_indices = arrangedArrays(stimulusDf[stimulusDf.stimulus == stim])
            stim_val = (np.median(nrn[stim_indices]) - background_val) / background_val

            nrn_dict[stim] = stim_val

        neuron_responses.append(nrn_dict)
    neuron_df = pd.DataFrame(neuron_responses)
    neuron_df.loc[:, 'neuron'] = nrns

    _df = neuron_df[neuron_df.bg > 0]
    _df.drop(columns='bg', inplace=True)

    return _df


def threshold_by_variance(eigen, variance=0.5):
    factors, loadings, x = eigen
    loading_mtrx = pd.DataFrame(loadings, columns=['FA{}'.format(i) for i in range(0, 10)], index=x.columns)
    loading_mtrx['highest_loading'] = loading_mtrx.idxmax(axis=1)
    b = loading_mtrx.drop(columns='highest_loading').T
    thresh = variance
    goods = []
    for i in b.columns:
        val = b[i].max()
        if val >= thresh:
            goods.append(i)
        else:
            pass
    good_cells = loading_mtrx.loc[goods]
    return good_cells


def pandas2hdf(_path):
    paths = pathSorter(_path)

    stimPath = paths['stimuli']['raw']

    with open(stimPath) as file:
        contents = file.read()

    # separate the text file into the different stimulus lines and withdraw the stimulus dictionaries
    parsed = contents.split('\n')
    fish_details = parsed[0]
    stimulus_details = parsed[1:]

    # some tricky text splitting
    times = [i[:i.find('{')] for i in stimulus_details]
    tex_freq = False
    if 'tex_freq' in stimulus_details[0]:
        tex_freq = True
        _stimulus_dicts = []
        tex_freqs = []
        for i in stimulus_details:
            _stimulus_dicts.append(i[i.find('{'):i.find('}') + 1])
            tex_freqs.append(i[i.find('freq: '):].split(' ')[-1])

        stimulus_dicts = [eval(i[i.find('{'):]) for i in _stimulus_dicts if 'stationary_end' not in i]
        freq_fixer = []
        for i in range(len(tex_freqs)):
            if tex_freqs[i] == '}':
                freq_fixer.append(tex_freqs[i - 1])
            else:
                freq_fixer.append(tex_freqs[i])
    else:
        stimulus_dicts = [eval(i[i.find('{'):]) for i in stimulus_details if 'stationary_end' not in i]

    # mostly a binocular gratings fix, need to stack the tuples into two separate columns
    for stim in range(len(stimulus_dicts)):
        for item in stimulus_dicts[stim].copy():
            try:
                if len(stimulus_dicts[stim][item]) > 1 and type(stimulus_dicts[stim][item]) is not str:
                    for i in range(len(stimulus_dicts[stim][item])):
                        name = item + '_' + str(i)
                        stimulus_dicts[stim][name] = stimulus_dicts[stim][item][i]
                    stimulus_dicts[stim].pop(item)
            except:
                pass

    stim_df = pd.DataFrame(stimulus_dicts)

    final_stims = stim_df
    if tex_freq:
        final_stims.loc[:, 'freq'] = freq_fixer

    # interpret the times and set up an array to measure elapsed times across experiment
    ntime_array = []
    for i in range(len((times))):
        ntime_array.append(dt.strptime(times[i].split(' ')[1], '%H:%M:%S.%f:'))
    time_array = []
    rawt_array = []
    for i in range(len(times)):
        try:
            # time_array.append((ntime_array[i + 1] - ntime_array[i]).total_seconds())
            rawt_array.append(str(ntime_array[i])[11:])
        except:
            pass

    final_stims.loc[:, 'raw_t'] = rawt_array

    # save a new file (don't overwrite an existing)
    fish_details = fish_details[:fish_details.rfind(' ')]

    val_offset = 0
    new_file = Path(_path).joinpath(f'{fish_details}_{val_offset}.h5')

    while os.path.exists(new_file):
        val_offset += 1
        new_file = Path(_path).joinpath(f'{fish_details}_{val_offset}.h5')


    final_stims.to_hdf(new_file, key='df')
    print('file saved:', new_file)
    return new_file


def pandas_stim_aligner(_path):
    # requires a path to be sent in as a h5 stimuli path
    # requires an extra line or two into the pathSorter util for non-pandas stimuli
    # really just needs to be directed to a dataframe with a 'raw_t' column
    paths = pathSorter(_path)
    new_timestamps=False
    try:
        stimulusPath = paths['stimuli']['processed']
    except KeyError:
        print('trying to convert stim file')
        try:
            pandas2hdf(_path)
            paths = pathSorter(_path)
            stimulusPath = paths['stimuli']['processed']
        except:
            print('failed to process stimulus')
            return

    stims = pd.read_hdf(stimulusPath)
    stims['raw_t'] = stims['raw_t'].apply(lambda a: dt.strptime(a, '%H:%M:%S.%f').time())

    try:
        frameTimePath = paths['image']['timestamps']
        image_times = image_timings(frameTimePath)

    except KeyError:
        print('no legacy timestamps found')
        try:
            image_times = pd.read_hdf(paths['image']['new_timestamps_processed']).rename({0:'times'}, axis=1).reset_index()
            image_times = image_times.drop(columns='index')
            new_timestamps = True
        except KeyError:
            try:
                frameTimePath = paths['image']['new_timestamps']
                image_times = raw_text_frametimes_to_df(frameTimePath)
                new_timestamps = True
            except KeyError:
                print('no updated timestamps found')
                return

    # start_ind = image_times[image_times.times >= stims.loc[0].raw_t].index[0]
    if new_timestamps:
        diffs = []
        for i in range(len(image_times) - 2):
            diffs.append(dt.combine(date.today(), image_times.iloc[i + 1].values[0]) - dt.combine(date.today(), image_times.iloc[i].values[0]))
        imageHz = 1/np.mean(diffs).total_seconds()
        print('calculated imageHz at :', imageHz)
    else:
        try:
            notesPath = paths['etc']['notes']
            _notes = pd.read_csv(notesPath, sep=':', header=None)
            imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
        except KeyError:
            imageHz = 500
            print('defaulting to 2 Hz')

    try:
        end_ind_offset = (stims.iloc[-1].duration - stims.iloc[-1].stationary_time) // (imageHz / 1000)
    except AttributeError:
        _statval = max((stims.iloc[-1].duration - stims.iloc[-1].stationary_time_0), (stims.iloc[-1].duration - stims.iloc[-1].stationary_time_1))
        end_ind_offset = _statval // (imageHz / 1000)

    try:
        end_ind = image_times[image_times.times >= stims.iloc[-1].raw_t].index[0] + end_ind_offset
    except IndexError:
        end_ind = len(image_times) - 1

    imgs = []
    stim_num = len(stims)
    for i in range(stim_num):
        if i + 1 >= stim_num:
            break
        indices = image_times[
            (image_times.times >= stims.loc[i].raw_t) & (image_times.times <= stims.loc[i + 1].raw_t)].index
        imgs.append(indices)
    imgs.append(image_times[(image_times.times >= stims.loc[stim_num - 1].raw_t)].loc[:end_ind].index)
    stims.loc[:, 'img_stacks'] = imgs

    df_trimmer = []
    for row in range(len(stims)):
        if len(stims.img_stacks.values[row]) > 0:
            df_trimmer.append(True)
        else:
            df_trimmer.append(False)

    stims = stims[df_trimmer]
    stims.to_hdf(os.path.join(_path, 'frame_aligned.h5'), key='stimuli')
    return


def image_timings(frameTimePath):

    try:
        # legacy way
        time_data = TdmsFile(frameTimePath).object('2P_Frame', 'Time').data

        time_array = []
        date_array = []

        for i in range(len(time_data)):
            time_array.append(time_data[i].replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).time())
            date_array.append(time_data[i].replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).date())

    except AttributeError:
        # updated
        time_data = TdmsFile(frameTimePath)['2P_Frame']['Time'][:]
        time_array = []
        date_array = []

        for i in range(len(time_data)):
            time_array.append(pd.to_datetime(time_data[i], utc=True).tz_convert('America/New_York').time())
            date_array.append(pd.to_datetime(time_data[i], utc=True).tz_convert('America/New_York').date())

    frame_dic = {"times": time_array, 'Date': date_array}
    imageTimeDf = pd.DataFrame(frame_dic)

    return imageTimeDf


def raw_text_frametimes_to_df(time_path):
    with open(time_path) as file:
        contents = file.read()
    parsed = contents.split('\n')

    times = []
    for line in range(len(parsed) - 1):
        times.append(dt.strptime(parsed[line], '%H:%M:%S.%f').time())
    return pd.DataFrame(times)


def raw_text_logfile_to_df(log_path, frametimes=None):
    with open(log_path) as file:
        contents = file.read()
    split = contents.split('\n')

    movesteps = []
    times = []
    for line in range(len(split)):
        if 'piezo' in split[line] and 'connected' not in split[line] and 'stopped' not in split[line]:
            t = split[line].split(' ')[0][:-1]
            z = split[line].split(' ')[6]
            try:
                if isinstance(eval(z), float):
                    times.append(dt.strptime(t, '%H:%M:%S.%f').time())
                    movesteps.append(z)
            except NameError:
                continue
    else:
        # last line is blank and likes to error out
        pass
    log_steps = pd.DataFrame({'times': times, 'steps': movesteps})

    if frametimes is not None:
        log_steps = log_aligner(log_steps, frametimes)
    else:
        pass
    return log_steps



def log_aligner(logsteps, frametimes):

    trimmed_logsteps = logsteps[(logsteps.times >= frametimes.iloc[0].values[0])&(logsteps.times <= frametimes.iloc[-1].values[0])]
    return trimmed_logsteps


def sequentialVolumes(frametimePath, imgPath, steps=5, leadingFrame=None):
    frametimes = raw_text_frametimes_to_df(frametimePath)
    img = cm.load(imgPath)

    if leadingFrame is not None:
        img = img[leadingFrame:]
        frametimes = frametimes.loc[leadingFrame:]

    step = img.shape[0] / steps

    imgs = [[]] * steps
    frametime_all = [[]] * steps
    imgpaths = []
    frametime_paths = []

    for i in range(steps):

        root_path = Path(frametimePath).parents[0].joinpath('planes')

        try:
            os.mkdir(root_path)
        except FileExistsError:
            pass

        image = img[int(i*step) : int((i+1)*step)]
        frames = frametimes.loc[int(i*step) : int((i+1)*step)]

        imgs[i] = image
        frametime_all[i] = frames

        imagePath = root_path.joinpath(f'{i}.tif')
        framePath = root_path.joinpath(f'{i}_frametimes.h5')

        imgpaths.append(imagePath)
        frametime_paths.append(framePath)

        image.save(imagePath)
        frames.to_hdf(framePath, 'frametime')

    return [imgs, frametime_all], [imgpaths, frametime_paths]


def volumeSplitter(logPath, frametimePath, imgPath, leadingFrame=None, extraStep=False, intermediate_return=False):
    frametimes = raw_text_frametimes_to_df(frametimePath)
    logfile = raw_text_logfile_to_df(logPath, frametimes)
    img = cm.load(imgPath)

    if leadingFrame is not None:
        img = img[leadingFrame:]
        frametimes = frametimes.loc[leadingFrame:]

    if intermediate_return:
        return frametimes, logfile, img

    if extraStep:
        n_imgs = logfile.steps.nunique() - 1
    else:
        n_imgs = logfile.steps.nunique()

    imgs = [[]] * n_imgs
    frametime_all = [[]] * n_imgs

    imgpaths = []
    frametime_paths = []

    root_path = Path(logPath).parents[0].joinpath('planes')

    try:
        os.mkdir(root_path)
    except FileExistsError:
        pass

    x=0
    for i in range(n_imgs):
        new_img = img[i::n_imgs]
        new_img_frametime = frametimes.iloc[1::n_imgs]

        imgs[i] = new_img
        frametime_all[i] = new_img_frametime

        new_img_path = root_path.joinpath(f'{x}.tif')
        new_framet_path = root_path.joinpath(f'{x}_frametimes.h5')
        imgpaths.append(new_img_path)
        frametime_paths.append(new_framet_path)

        new_img.save(new_img_path)
        new_img_frametime.to_hdf(new_framet_path, 'frametimes')

        print(f'saved {new_img_path}')
        print(f'saved {new_framet_path}')
        x += 1

    return [imgs, frametime_all], [imgpaths, frametime_paths]


def stackHzReturner(timeframe_df):
    t_vals = timeframe_df.loc[:, 0].values
    dt_offset = date.today()

    times = []
    for i in range(len(t_vals) - 1):
        times.append(dt.combine(dt_offset, t_vals[i + 1]) - dt.combine(dt_offset, t_vals[i]))

    return 1 / np.mean(times).total_seconds()


def returnFoverF(stim_df, good_cells):
    inds = []
    for arr in stim_df[(stim_df.velocity_0 == 0) & ((stim_df.velocity_1 == 0))].img_stacks.values:
        for j in arr[5:-5]:
            inds.append(j)

    fDff = []
    for i in range(good_cells.shape[0]):
        background = np.nanmean(good_cells[i, inds])
        ff = (good_cells[i, :] - background) / background
        fDff.append(ff)
    return np.array(fDff)


def volumeFrametimeMover(frametimePaths):
    basePath = frametimePaths[0].parents[0].joinpath('move_corrected')

    for i in range(len(frametimePaths)):
        locPath = basePath.joinpath(f'plane_{i}/{frametimePaths[i].name}')
        shutil.copy(frametimePaths[i], locPath)


def stimCopier(stimPath, planes=5):
    basePath = stimPath.parents[0].joinpath('planes/move_corrected')

    for i in range(planes):
        locPath = basePath.joinpath(f'plane_{i}/{stimPath.name}')
        shutil.copy(stimPath, locPath)


def lineOffset(image, offset_amount=1):
    N_img = image.copy()
    rows, cols = (N_img.shape[1], N_img.shape[2])

    for r in range(rows):
        if r < (rows - offset_amount) and (r + offset_amount) >= 0:
            if (r % 2) == 0:
                N_img[:, r, :] = N_img[:, r + offset_amount, :]

    return N_img

def reduce_to_pi(ar):
    ar = ar * np.pi/180
    return (np.mod(ar + np.pi, np.pi * 2) - np.pi)*180/np.pi

def rad2deg(ar):
    return (ar*np.pi)/180


def nrnResponses(mstim_df, cells, allowNegs=False):
    nrn_vals = []
    stimuluses = []
    nrns = []

    for cell in range(len(cells)):
        for stimulus in mstim_df.stimulus.unique():
            nrns.append(cell)
            stimuluses.append(stimulus)
            nrn_vals.append(np.mean(cells[cell, arrangedArrays(mstim_df[mstim_df.stimulus.isin([stimulus])])]))
    neuron_df = pd.DataFrame({'neuron': nrns, 'stimulus': stimuluses, 'val': nrn_vals})

    if not allowNegs:
        neuron_df.loc[neuron_df.val < 0, 'val'] = 0

    return neuron_df


def stim_cell_returner(base_path):
    stimuli = pd.read_hdf(pathSorter(base_path)['stimuli']['frame_aligned'])
    try:
        stimuli.loc[:, 'stimulus'] = stimuli.stimulus_name.values
    except AttributeError:
        tmp = map_stimuli(stimuli)
        stimuli.loc[:, 'stimulus_name'] = tmp.stimulus.values
        stimuli.loc[:, 'stimulus'] = stimuli.stimulus_name.values

    try:
        inds_1 = stimuli[(stimuli.velocity == 0)].index
        inds_2 = stimuli[(stimuli.velocity_0 == 0) & (stimuli.velocity_1 == 0)].index
        dropped_inds = sorted(np.concatenate([inds_1, inds_2]))
    except AttributeError:
        inds_2 = stimuli[(stimuli.velocity_0 == 0) & (stimuli.velocity_1 == 0)].index
        dropped_inds = sorted(np.concatenate([inds_2]))

    stim_df = stimStartStop(stimuli.drop(dropped_inds))

    ## fix lateral medial mixup
    xrs = stim_df[stim_df.stimulus == 'xR'].index
    xls = stim_df[stim_df.stimulus == 'xL'].index
    rxs = stim_df[stim_df.stimulus == 'Rx'].index
    lxs = stim_df[stim_df.stimulus == 'Lx'].index

    stim_df.loc[xrs, 'stimulus'] = 'Rx'
    stim_df.loc[lxs, 'stimulus'] = 'Lx'
    stim_df.loc[rxs, 'stimulus'] = 'xR'
    stim_df.loc[lxs, 'stimulus'] = 'xL'

    stim_df.stimulus = stim_df.stimulus.astype('category')

    ops, iscell, stats, f_cells = load_suite2p(pathSorter(base_path)['output']['suite2p'])

    cells = returnFoverF(stimuli, f_cells[iscell])

    return stim_df, cells


def returnStimCellAligned(stims_all, cells_all, leading_frames=50):

    first_stims = []
    last_stims = []

    start_offset = 0
    end_offset = 1

    sameStart = False
    sameFinish = False

    # have to find the same start and end stimulus.
    # potentially stim started before imaging or ended after imaging
    # this finds a same-start and same-stop to excise further along

    while not sameStart and not sameFinish:
        for data in stims_all:
            first_stims.append(data.iloc[start_offset].stimulus)
            last_stims.append(data.iloc[-end_offset].stimulus)

        # make a set of the grabbed stimuli, if only one item in set they are all the same, otherwise take a step
        if len(set(first_stims)) == 1:
            sameStart = True
        else:
            start_offset += 1
        if len(set(last_stims)) == 1:
            sameFinish = True
        else:
            end_offset += 1

    # grab the start and end of the stimulation shown
    rel_inds = []
    for data in stims_all:
        rel_inds.append([data.iloc[start_offset].start, data.iloc[-end_offset].stop])

    # this is supposed to fix the lengths to be equal
    # seems to sorta work? ends up being +/- 1
    all_lens = []
    for x in rel_inds:
        all_lens.append(x[1] - x[0])

    if True in (np.diff(all_lens) > 0):
        for j in np.where(np.diff(all_lens) > 0):
            if np.diff(all_lens)[j] > 0:
                rel_inds[int(j)][1] += 1
            elif np.diff(all_lens)[j] < 0:
                rel_inds[int(j)][1] -= 1

    # maintain original stimuli (was once concerned with mutability and downstream processing)
    # this arranges the stimuli from 0-end

    fin_stims = copy.deepcopy(stims_all)

    for n in range(len(stims_all)):
        fin_stims[n].loc[:, 'start'] = stims_all[n].start.values - rel_inds[n][0]
        fin_stims[n].loc[:, 'stop'] = stims_all[n].stop.values - rel_inds[n][0]

    first_l = fin_stims[0].stop.values[-1]

    # excises the relevant part from each cell for the new 0-end
    fin_cells = []
    for b in range(len(cells_all)):
        offset_l = fin_stims[b].stop.values[-1] - first_l
        fin_cells.append(cells_all[b][:, rel_inds[b][0] - leading_frames + offset_l : rel_inds[b][1]])

    return fin_stims, fin_cells


def corrNeurons(new_variables, ncells, corrThreshold=0.45):
    # takes the path of the eigens to compare to
    # as well as the array of cells
    fctrs = pd.DataFrame(new_variables)


    if ncells.shape[1] == fctrs.shape[0]:
        df = pd.DataFrame(ncells).T
    else:
        df = pd.DataFrame(ncells)

    x = df.copy()
    x.replace([np.inf, -np.inf], np.nan)


    factor_corrs = [[]] * fctrs.shape[1]
    for i in range(fctrs.shape[1]):
        corrs = []
        for j in range(x.shape[1]):
            corrs.append(np.corrcoef(fctrs.loc[:, i], x.loc[:, j])[0][1])

        factor_corrs[i] = corrs

    fullcorr = pd.DataFrame(factor_corrs)
    asdf = abs(fullcorr).max() >= corrThreshold
    cordf = fullcorr.loc[:, asdf]

    uniqueFactorsN = cordf.idxmax().nunique()
    factor_neurons = [[]] * uniqueFactorsN
    for q in range(uniqueFactorsN):
        factor_neurons[q] = cordf.idxmax()[cordf.idxmax() == q].index
    return factor_neurons, factor_corrs


def multiAlignment(dataPaths, n_components=10):

    from sklearn.decomposition import FactorAnalysis
    from sklearn.preprocessing import StandardScaler

    stims_all = []
    cells_all = []

    for p in dataPaths:
        s, c = stim_cell_returner(p)
        stims_all.append(s)
        cells_all.append(c)

    fin_stims, fin_cells = returnStimCellAligned(stims_all, cells_all)

    celldfs = []
    for arr in fin_cells:
        celldfs.append(pd.DataFrame(arr).T)
    allneurons = pd.concat(celldfs, axis=1).T.reset_index(drop=True).T

    x = allneurons.copy()
    x.replace([np.inf, -np.inf], np.nan)
    x.dropna(axis=1, inplace=True)

    X = StandardScaler().fit_transform(x)
    transformer = FactorAnalysis(n_components=n_components, rotation='varimax', max_iter=50000, iterated_power=10, tol=0.0001)
    X_transformed = transformer.fit_transform(X)

    return allneurons, fin_cells, fin_stims, X_transformed