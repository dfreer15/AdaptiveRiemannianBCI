import numpy as np
from mne.io import read_raw_edf, find_edf_events
import os
import pandas as pd
from numpy import genfromtxt
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt
import math
from time import clock


def get_data_PN(dataset_dir, j, n_classes, num_channels=64, remove_rest=True):

    freq = 160

    if j == 89:
        j = 109
    # get directory name for one subject
    data_dir = dataset_dir + "S" + format(j, '03d')
    task_list = ["R04", "R06", "R08", "R10", "R12", "R14"]
    signal_out = np.empty([0, num_channels])
    label_out = np.empty(0)
    for task in task_list:
        file = data_dir + "/S" + format(j, '03d') + task + ".edf"
        print(file)
        # R04, R06 R08, R10, R12, R14: motor imagery tasks
        raw = read_raw_edf(file, preload=True, stim_channel='auto', verbose='WARNING')
        events = np.asarray(find_edf_events(raw))
        raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        signal = np.transpose(raw.get_data()[:num_channels])

        time = raw.times

        events = get_events_PN(task, events)
        labels = label_data_PN(signal, time, events, remove_rest, n_classes, freq)
        unique, counts = np.unique(labels, return_counts=True)
        print(unique, counts)
        signal_out = np.vstack([signal_out, signal])
        label_out = np.append(label_out, labels)

    return signal_out, label_out


def get_data_2a(data_name, n_classes, num_channels=22, remove_rest=False, training_data=True, reuse_data=False,
                mult_data=False, noise_data=False):

    freq = 250
    if n_classes == 5:
        remove_rest = False

    raw = read_raw_edf(data_name, preload=True, stim_channel='auto', verbose='WARNING')

    events = find_edf_events(raw)
    events.pop(0)
    time = events.pop(0)
    events1 = events.pop(0)
    events2 = events.pop(0)
    events3 = events.pop(0)

    # raw_train.plot_psd(area_mode='range', tmax=10.0)
    # raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    signal = np.transpose(raw.get_data()[:num_channels])

    events = np.transpose(np.vstack([time, events1, events2, events3]))
    time = raw.times * freq

    if training_data:
        signal_out, labels = label_data_2a(signal, time, events, remove_rest, n_classes, freq, reuse_data=reuse_data,
                                           mult_data=mult_data, noise_data=noise_data)
        # if remove_rest:
        signal = signal_out
    else:
        labels = np.zeros(signal.shape[0])

    return np.asarray(signal), labels


def get_data_2a_fb(data_name, n_classes, num_channels=22, remove_rest=True):

    freq = 250

    raw_train = read_raw_edf(data_name, preload=True)
    raw_train_4_10 = raw_train
    raw_train_7_13 = raw_train
    raw_train_10_16 = raw_train
    raw_train_13_19 = raw_train
    raw_train_16_24 = raw_train
    raw_train_20_28 = raw_train
    raw_train_24_32 = raw_train
    raw_train_28_36 = raw_train
    raw_train_32_40 = raw_train

    raw_test = read_raw_edf(data_name)

    events_train = find_edf_events(raw_train)
    events_train.pop(0)
    time = events_train.pop(0)
    events1 = events_train.pop(0)
    events2 = events_train.pop(0)
    events3 = events_train.pop(0)

    # raw_train.plot_psd(area_mode='range', tmax=10.0)
    raw_train_4_10.filter(4., 10., fir_design='firwin', skip_by_annotation='edge')
    signal_train = np.transpose(raw_train_4_10.get_data()[:num_channels])
    raw_train_7_13.filter(7., 13., fir_design='firwin', skip_by_annotation='edge')
    data_7_13 = np.transpose(raw_train_7_13.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_7_13])
    raw_train_10_16.filter(10., 16., fir_design='firwin', skip_by_annotation='edge')
    data_10_16 = np.transpose(raw_train_10_16.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_10_16])
    raw_train_13_19.filter(13., 19., fir_design='firwin', skip_by_annotation='edge')
    data_13_19 = np.transpose(raw_train_13_19.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_13_19])
    raw_train_16_24.filter(16., 24., fir_design='firwin', skip_by_annotation='edge')
    data_16_24 = np.transpose(raw_train_16_24.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_16_24])
    raw_train_20_28.filter(20., 28., fir_design='firwin', skip_by_annotation='edge')
    data_20_28 = np.transpose(raw_train_20_28.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_20_28])
    raw_train_24_32.filter(24., 32., fir_design='firwin', skip_by_annotation='edge')
    data_24_32 = np.transpose(raw_train_24_32.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_24_32])
    raw_train_28_36.filter(28., 36., fir_design='firwin', skip_by_annotation='edge')
    data_28_36 = np.transpose(raw_train_28_36.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_28_36])
    raw_train_32_40.filter(32., 40., fir_design='firwin', skip_by_annotation='edge')
    data_32_40 = np.transpose(raw_train_32_40.get_data()[:num_channels])
    signal_train = np.hstack([signal_train, data_32_40])

    events = np.transpose(np.vstack([time, events1, events2, events3]))
    time = raw_train.times * 250

    signal, labels = label_data_2a(signal_train, time, events, remove_rest, n_classes, freq)

    return np.asarray(signal), labels


def get_data_gtec(datadir, file, n_classes, num_channels=32, reuse_data=False, mult_data=False, noise_data=False, neg_data=False):
    data = genfromtxt(datadir + 'signal/' + file + '_01.csv', delimiter=';')
    labels = genfromtxt(datadir + 'labels/' + file + '_labels_01.csv', delimiter=';')

    data_y = np.zeros(len(data))
    raw_data = np.zeros((len(data), num_channels))
    time_data = np.zeros(len(data))

    for i in range(1, len(data)):
        if not math.isnan(np.amax(data[i][1:num_channels + 1])):
            raw_data[i] = data[i][1:num_channels + 1]
            time_data[i] = data[i][0]

    # # # # # # FILTER DATA # # # # # # #
    # Don't want to filter all data now. Should happen in 'real time'.
    # signal = butter_bandpass_filter(raw_data, 7, 30, 250)

    # # # # # # LABEL DATA  # # # # # # #
    stim_t_vec, stim_stop_t_vec, stim_class_vec = sort_stim_data_3class(labels)
    signal, labels = sort_EEG_data_gtec(time_data, raw_data, stim_t_vec, stim_class_vec, n_classes, reuse_data=reuse_data, mult_data=mult_data,
                                                                         noise_data=noise_data, neg_data=neg_data)

    # norm_data = norm_dataset(raw_data)

    return np.asarray(signal), labels


def get_events_PN(run, events):

    events_out = np.zeros(events.shape)
    for i in range(len(events)):
        if events[i, 2] == 'T0':
            events_out[i, 2] = 0
        elif events[i, 2] == 'T1':
            if run == 'R04' or run == 'R08' or run == 'R12':
                events_out[i, 2] = 1  # Left fist
            elif run == 'R06' or run == 'R10' or run == 'R14':
                events_out[i, 2] = 3  # Both fists
        elif events[i, 2] == 'T2':
            if run == 'R04' or run == 'R08' or run == 'R12':
                events_out[i, 2] = 2  # Right fist
            elif run == 'R06' or run == 'R10' or run == 'R14':
                events_out[i, 2] = 4  # Both feet
        events_out[i, 0] = float(events[i, 0])
        events_out[i, 1] = float(events[i, 1])

    return events_out


def label_data_PN(signal, time, events, remove_rest, n_classes, freq):

    final_labels = []
    signal_out = []
    t = 0

    min_event = 1

    if n_classes == 4:
        max_event = 4
    elif n_classes == 3:
        max_event = 3
    elif n_classes == 2:
        if remove_rest:
            max_event = 2
        else:
            max_event = 4
            min_event = 0

    for j in range(len(time)):
        while events[t, 2] < min_event or events[t, 2] > max_event:
            t = t+1
            if t == len(events):
                return np.asarray(final_labels)

        if events[t, 0] + 0.5 < time[j] < events[t, 0] + 2.5:
            if not remove_rest:
                final_labels.append(1)
            else:
                final_labels.append(events[t, 2])
                signal_out.append(signal[j])
        elif time[j] >= events[t, 0] + 2.5:
            if not remove_rest:
                final_labels.append(1)
            else:
                final_labels.append(events[t, 2])
                signal_out.append(signal[j])
            t = t+1
        else:
            if remove_rest:
                continue
            else:
                final_labels.append(0)

        if t == len(events):
            return np.asarray(final_labels)

    return signal_out, np.asarray(final_labels)


def label_data_2a(signal, time, events, remove_rest, n_classes, freq, reuse_data=False, mult_data=False, noise_data=False, neg_data=False):
    final_labels = []
    signal_out = np.zeros(signal.shape)
    t, s, j1 = 0, 0, 0
    da_mod = 1
    if reuse_data:
        da_mod = da_mod + 1
    if mult_data:
        da_mod = da_mod + 1
    if noise_data:
        da_mod = da_mod + 1
    if neg_data:
        da_mod = da_mod + 1

    min_event = 769
    if n_classes == 4 or n_classes == 5:
        max_event = 772
    elif n_classes == 3:
        max_event = 771
    elif n_classes == 2:
        if remove_rest:
            max_event = 770
        else:
            max_event = 772

    for j in range(len(time)):
        while events[t, 1] < min_event or events[t, 1] > max_event:
            t = t+1
            if t == len(events):
                signal_out = signal_out[:len(final_labels)]
                print("Signal out shape: ", signal_out.shape)
                print("Length of Labels: ", len(final_labels))
                return signal_out, np.asarray(final_labels)

        if events[t, 0] + freq/2 < time[j] < events[t, 0] + freq * (5/2):
            if not remove_rest and n_classes == 2:
                final_labels.append(1)
                signal_out[j1] = signal[j]
                j1 += 1
            else:
                final_labels.append(events[t, 1] - 768)
                signal_out[j1] = signal[j]
                j1 += 1
        elif time[j] >= events[t, 0] + freq * (5/2):
            if not remove_rest and n_classes == 2:
                final_labels.append(1)
                signal_out[j1] = signal[j]
                j1 += 1
            else:
                final_labels.append(events[t, 1] - 768)
                signal_out[j1] = signal[j]
                j1 += 1
            t = t+1
        elif events[t, 0] < time[j] < events[t, 0] + freq/2:
            continue
        else:
            if remove_rest:
                continue
            elif s < da_mod*final_labels.count(1):
                final_labels.append(0)
                signal_out[j1] = signal[j]
                s += 1
                j1 += 1
            else:
                continue

        if t == len(events):
            signal_out = signal_out[:len(final_labels)]
            print("Signal out shape: ", signal_out.shape)
            print("Length of Labels: ", len(final_labels))
            return signal_out, np.asarray(final_labels)

    signal_out = signal_out[:len(final_labels)]
    print("Signal out shape: ", signal_out.shape)
    print("Length of Labels: ", len(final_labels))

    return np.asarray(signal_out), np.asarray(final_labels)


def sort_stim_data_3class(data_stim):
    stim_t_vec = []
    stim_class_vec = []
    stim_stop_t_vec = []

    n = 0
    for j in range(1, len(data_stim)):
        if data_stim[j][1] == 770 or data_stim[j][1] == 769 or data_stim[j][1] == 780:
            stim_t_vec.append(data_stim[j][0])
            if data_stim[j][1] == 770:
                # stim_class_vec.append(1)  # Right/Event
                stim_class_vec.append(3)
            elif data_stim[j][1] == 769:
                # stim_class_vec.append(1)  # Left/Event
                stim_class_vec.append(2)
            elif data_stim[j][1] == 780:
                stim_class_vec.append(1)    # Both hands
        if data_stim[j][1] == 800:
            stim_stop_t_vec.append(data_stim[j][0])
            n = n + 1

    unique, counts = np.unique(stim_class_vec, return_counts=True)
    print(unique, counts)

    return [stim_t_vec, stim_stop_t_vec, stim_class_vec]


def sort_EEG_data_gtec(data_time, data_t, stim_vec, class_vec, n_classes, remove_rest=False, reuse_data=False, mult_data=False, noise_data=False, neg_data=False):
    da_mod = 1
    if reuse_data:
        da_mod += da_mod
    if mult_data:
        da_mod += 2*da_mod
    if noise_data:
        da_mod += da_mod

    # pix_transp = np.zeros((len(data_t), 9, 9))
    output_y = []
    signal_out = np.zeros(data_t.shape)
    max_n = len(stim_vec) - 1
    print(max_n)

    unique, counts = np.unique(class_vec, return_counts=True)
    print(unique, counts)

    n, s, j1 = 0, 0, 0
    for i in range(len(data_time)):
        # # # # # Put into 2D configuration (if desired) # # # # #

        # Label each datapoint
        if n < max_n:
            # tic = clock()
            if stim_vec[n] + 0.5 < data_time[i] < stim_vec[n] + 2.5:
                if not remove_rest and n_classes == 2:
                    output_y.append(1)
                    signal_out[j1] = data_t[i]
                    j1 += 1
                else:
                    output_y.append(class_vec[n])
                    signal_out[j1] = data_t[i]
                    j1 += 1
            elif data_time[i] > stim_vec[n] + 2.5:
                if not remove_rest and n_classes == 2:
                    output_y.append(1)
                    signal_out[j1] = data_t[i]
                    j1 += 1
                else:
                    output_y.append(class_vec[n])
                    signal_out[j1] = data_t[i]
                    j1 += 1
                n = n + 1
            elif stim_vec[n] < data_time[i] < stim_vec[n] + 0.5:
                continue
            else:
                if remove_rest:
                    continue
                elif s < da_mod*output_y.count(2):
                    # print(s, output_y.count(2))
                    output_y.append(0)
                    signal_out[j1] = data_t[i]
                    s = s + 1
                    j1 += 1
            # print(clock() - tic)
    unique, counts = np.unique(output_y, return_counts=True)
    print(unique, counts)

    output_y = np.array(output_y)
    signal_out = signal_out[:j1]
    # pix_transp, data_raw, output_y = len_pow_2(pix_transp, data_t, output_y)

    return signal_out, output_y


def process_data_2a(data, label, window_size, num_channels=22):

    data, label = segment_signal_without_transition(data, label, window_size)
    unique, counts = np.unique(label, return_counts=True)
    data = norm_dataset(data)
    data = data.reshape([label.shape[0], window_size, num_channels])

    train_data, test_data, train_y, test_y = split_data(data, label)

    return train_data, test_data, train_y, test_y


def segment_signal_without_transition(data, label, window_size, overlap=1):

    # print(data.shape)

    for (start, end) in windows(data, window_size, overlap=overlap):
        if len(data[start:end]) == window_size:
            x1_F = data[start:end]
            if start == 0:
                unique, counts = np.unique(label[start:end], return_counts=True)
                labels = unique[np.argmax(counts)]
                segments = x1_F
            else:
                try:
                    unique, counts = np.unique(label[start:end], return_counts=True)
                    labels = np.append(labels, unique[np.argmax(counts)])
                    segments = np.vstack([segments, x1_F])
                except ValueError:
                    continue

    return segments, labels


def windows(data, size, overlap=1):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += (size / overlap)


def split_data(data_in_s, label_s):
    # the first 2 trials are in training, last is testing (0.666)
    split = int(0.666 * len(label_s))
    train_x = data_in_s[0:split]
    train_y = label_s[0:split]

    test_x = data_in_s[split:]
    test_y = label_s[split:]

    return train_x, test_x, train_y, test_y


def find_cov_matrices(data):

    cov_mat = np.zeros((data.shape[0], data.shape[2], data.shape[2]))
    for i in range(data.shape[0]):
        cov_mat[i] = np.cov(np.transpose(data[i]))

    return cov_mat


def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros(dataset_1D.shape)
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    return norm_dataset_1D


def feature_normalize(data):
    if data.mean() == 0:
        data_normalized = np.zeros(data.shape)
        return data_normalized
    else:
        mean = data[data.nonzero()].mean()
        sigma = data[data.nonzero()].std()
        # print("Mean: {}      Std: {}".format(mean, sigma))
        data_normalized = data
        # print('Data:')
        # print(np.min(data_normalized), mean, np.max(data_normalized))
        # print(np.std(data_normalized))

        # data_normalized = data_normalized - np.min(data_normalized)
        # data_normalized = data_normalized / (np.max(data_normalized) - np.min(data_normalized))
        # print('Shifted:')
        # print(np.min(data_normalized), np.mean(data_normalized), np.max(data_normalized))
        # print(np.std(data_normalized))

        data_normalized = (data_normalized - mean) / sigma
        # print('Norm:')
        # print(np.min(data_normalized), np.mean(data_normalized), np.max(data_normalized))
        # print(np.std(data_normalized))
        # data_normalized = np.cbrt(data_normalized)

        return data_normalized


def dataset_1Dto1D_32_PN(dataset_1D):
    dataset_1D_32 = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        dataset_1D_32[i] = data_1Dto1D_32_PN(dataset_1D[i])
    return dataset_1D_32


def data_1Dto1D_32_PN(data):
    oneD_32 = np.zeros(32)
    oneD_32 = [data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16], data[18], data[20],
               data[21], data[23], data[25], data[27], data[29], data[31], data[33], data[35], data[37], data[40],
               data[41], data[46], data[48], data[50], data[52], data[54], data[55], data[56], data[58], data[59], data[61]]
    return oneD_32


def dataset_1Dto1D_16_PN(dataset_1D):
    dataset_1D_16 = np.zeros([dataset_1D.shape[0], 16])
    for i in range(dataset_1D.shape[0]):
        dataset_1D_16[i] = data_1Dto1D_16_PN(dataset_1D[i])
    return dataset_1D_16


def data_1Dto1D_16_PN(data):
    oneD_16 = np.zeros(16)
    oneD_16 = [data[8], data[10], data[12], data[21], data[23], data[31], data[33], data[35], data[40], data[41], data[48],
               data[50], data[52], data[55], data[59], data[61]]
    return oneD_16


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # print('Filter input: ', data)
    first_state = True
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    # print('Filter output: ', y)
    return y


def data_aug(data, labels, size, reuse_data, mult_data, noise_data):
    print('Before Data Augmentation: ', data.shape)

    if reuse_data:
        data, labels = data_reuse_f(data, labels, size)
    if mult_data:
        data, labels = data_mult_f(data, labels, size)
    if noise_data:
        data, labels = data_noise_f(data, labels, size)

    print('After Data Augmentation: ', data.shape)

    return data, labels


def data_reuse_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] > 0:
            new_data.append(data[i])
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])
    print('Added Data Shape: ', new_data_ar.shape)
    data_out = np.append(data, new_data_ar, axis=0)
    labels_out = np.append(labels, np.asarray(new_labels))

    return data_out, labels_out


def data_mult_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] > 0:
            data_t = data[i]*1.1
            new_data.append(data_t)
            new_labels.append(labels[i])
    for i in range(len(labels)):
        if labels[i] > 0:
            data_t2 = data[i] * 0.9
            new_data.append(data_t2)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])
    print('Added Data Shape: ', new_data_ar.shape)

    data_out = np.append(data, new_data_ar, axis=0)
    print('Data Out Shape: ', data_out.shape)

    labels_out = np.append(labels, np.asarray(new_labels))

    return data_out, labels_out


def data_noise_f(data, labels, size):
    return data, labels
