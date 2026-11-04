"""Prepare the OpenBMI motor-imagery data for the cross-subject experiments.

Reads sess01/sess02 *_EEG_MI.mat of every subject from ./raw, keeps the 20 motor channels, applies a
5th-order zero-phase Butterworth band-pass (8-30 Hz) at the original 1000 Hz, resamples the 0-4 s trials
to 250 Hz, normalises every trial by its maximum absolute value and writes ./npy_data_250/subjNN_data.npy
(400 trials x 20 channels x 1000 samples) and subjNN_label.npy.

The channel table and the .mat reading routine are adapted from MIN2Net
(https://github.com/IoBT-VISTEC/MIN2Net, Apache License 2.0, see THIRD_PARTY_NOTICES.md). Modifications:
20-channel selection, band-pass before resampling, 250 Hz resampling, per-trial normalisation and the
per-subject .npy export.
"""
import numpy as np
import scipy.io as sio
import os
from scipy import signal
from scipy.signal import butter, filtfilt


OpenBMI_CONSTANT = {
    'raw_path': 'datasets/OpenBMI/raw', # raw data path
    'n_subjs': 54,
    'n_trials_2_class': 100,
    'n_trials_3_class': 150, 
    'n_chs': 62,
    'orig_smp_freq': 1000,                  # Original sampling frequency  (Hz)
    'trial_len': 8,                         # 8s (cut-off)
    'MI': {
        'start': 0,                         # start at time = 0 s
        'stop': 4,                          # stop at time = 0 s
        'len': 4,                           # 4s
    },
    'orig_chs': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 
                'C3','Cz','C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 
                'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                'C2', 'C6', 'CP3','CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 
                'TP7', 'TPP9h', 'FT10','FTT10h','TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 
                'AF7', 'AF3', 'AF4', 'AF8', 'PO3','PO4'],
    'sel_chs': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']               # select: 62 -> 20
}

BANDPASS = (8, 30)      # Hz; set to None (or pass --no_bandpass) to skip the filter

current_file_path = os.path.abspath(__file__)
folder_path = os.path.dirname(current_file_path)
raw_path = os.path.join(folder_path, 'raw')      # raw data path


def chanel_selection(sel_chs):
    orig_chs = OpenBMI_CONSTANT['orig_chs']
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        # print('chosen_channel:', name_ch ,'---', 'Index_is:', ch_id)
    return chs_id


def resampling(data, new_smp_freq, data_len):
    '''降采样'''
    if len(data.shape) != 3:
        raise Exception('Dimesion error', "--> please use three-dimensional input")
    new_smp_point = int(data_len*new_smp_freq)
    data_resampled = np.zeros((data.shape[0], data.shape[1], new_smp_point))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i,j,:] = signal.resample(data[i,j,:], new_smp_point)
    return data_resampled


def read_raw(PATH, session, subject, num_class, id_ch_selected):
    '''
    Read one subject raw data from .mat file
    '''
    mat_file_name = PATH + '/sess'+str(session).zfill(2)+'_subj'+str(subject).zfill(2)+'_EEG_MI.mat'
    mat = sio.loadmat(mat_file_name)
    
    print(mat.keys())
    print('This is data from: ', mat_file_name)

    raw_train_data = mat['EEG_MI_train'][0]['smt'][0]                       # origin_raw: [time, trial, channel] -> [4000, 100, 62]
    raw_train_data = (np.swapaxes(raw_train_data, 0, 2))[id_ch_selected]    # channel_selected: [20, 100, 4000]
    raw_train_data = np.swapaxes(raw_train_data, 0, 1)                      # [100, 20, 4000]
    # print('raw_train_data_shape:', raw_train_data.shape)
    raw_test_data = mat['EEG_MI_test'][0]['smt'][0]
    raw_test_data = np.swapaxes(raw_test_data, 0, 2)[id_ch_selected]
    raw_test_data = np.swapaxes(raw_test_data, 0, 1)
    # print('raw_test_data_shape:', raw_test_data.shape)
    label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0]-1              # [100, ]   class index y_dec - 1
    label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0]-1
    return raw_train_data, label_train_data, raw_test_data, label_test_data

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    '''
    5th-butterworth bandpass filter
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def data_norm(data):
    """
    对数据进行归一化 [-1, 1]
    :param data:   ndarray ,shape[N,channel,samples]
    :return:
    """
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.max(abs(data[i]))

    return data_copy

def __load_OpenBMI_singe_subj(PATH, id_chosen_chs, subj, new_smp_freq=250, num_class=2, MI_len=4):
    '''
    Load all OpenBMI data
    return: X_train:[200, chs, new_freq * 4], y_train:[200, ], X_test, y_test
    '''

    n_trials = OpenBMI_CONSTANT['n_trials_2_class']     # 100
    sessions = [1, 2]
    idx = 0
    n_chs = len(id_chosen_chs)
    X_session, y_session = np.zeros((4, n_trials, n_chs, int(new_smp_freq * MI_len))), np.zeros((4, n_trials))
    for id_se, sess in enumerate(sessions):
        X_tr, y_tr, X_te, y_te = read_raw(PATH, sess, subj, num_class, id_chosen_chs)       # [100, 62, 4000]

        # band-pass at the original 1000 Hz (paper: 8-30 Hz, 5th-order Butterworth), then resample to 250 Hz
        if BANDPASS is not None:
            X_tr = butter_bandpass_filter(X_tr, BANDPASS[0], BANDPASS[1], OpenBMI_CONSTANT['orig_smp_freq'], 5)
            X_te = butter_bandpass_filter(X_te, BANDPASS[0], BANDPASS[1], OpenBMI_CONSTANT['orig_smp_freq'], 5)
        X_tr_resam = resampling(X_tr, new_smp_freq, MI_len)                                 # [100, 20, 1000]
        X_te_resam = resampling(X_te, new_smp_freq, MI_len)

        # 通道归一化
        X_tr_resam = data_norm(X_tr_resam)
        X_te_resam = data_norm(X_te_resam)

        X_session[idx, :, :, :] = X_tr_resam
        y_session[idx, :] = y_tr
        X_session[idx + 1, :, :, :] = X_te_resam
        y_session[idx + 1, :] = y_te
        idx = idx + 2
    
    X_session = X_session.reshape(-1, n_chs, int(new_smp_freq * MI_len))    # [400, 20, 1000]
    y_session = y_session.reshape(-1, )                                     # [400, ]

    print('X_all_norm shape:', X_session.shape)
    print('y_all shape:', y_session.shape)
    print('The preprocessing of subject {} is DONE!!!'.format(subj))

    return X_session, y_session


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--no_bandpass', action='store_true', help='skip the 8-30 Hz band-pass')
    _ap.add_argument('--bandpass', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'), help='band-pass edges in Hz (default 8 30)')
    _args = _ap.parse_args()
    if _args.no_bandpass:
        BANDPASS = None
    elif _args.bandpass is not None:
        BANDPASS = tuple(_args.bandpass)
    print('Band-pass:', BANDPASS)
    sel_chs = OpenBMI_CONSTANT['sel_chs']
    all_chs = OpenBMI_CONSTANT['orig_chs']

    # 数据存储位置
    save_path = folder_path + '/npy_data_250'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 选取 20 通道
    id_chosen_chs = chanel_selection(sel_chs)
    downsample_rate = 250

    for subj in range(1, 55):
        print('Subject: {}'.format(subj))
        X_all, y_all = __load_OpenBMI_singe_subj(PATH=raw_path,
                                                 id_chosen_chs=id_chosen_chs,
                                                 subj=subj,
                                                 new_smp_freq=downsample_rate, 
                                                 num_class=2,
                                                 MI_len=4,
                                                 )
        
        np.save(save_path + f'/subj{str(subj).zfill(2)}_data.npy', X_all)
        np.save(save_path + f'/subj{str(subj).zfill(2)}_label.npy', y_all)
        print('Subject {} save DONE'.format(subj))
        print('-----------------------------------')

