import numpy as np
from mne.io import read_raw_edf, find_edf_events
import os
import pandas as pd
from numpy import genfromtxt
from scipy.signal import butter, lfilter
import math
from time import clock


def get_data_2a(data_name, n_classes, num_channels=22):
    # # # Collects data from edf file and returns it as an np array along with the events
    # # # This has been specifically designed to handle data from the BCI Competition IV 2a dataset
    # Returns: signal as np array, time array, and the events/prompts
    
    freq = 250

    raw = read_raw_edf(data_name, preload=True, stim_channel='auto', verbose='WARNING')

    events = find_edf_events(raw)
    events.pop(0)
    time = events.pop(0)
    events1 = events.pop(0)
    events2 = events.pop(0)
    events3 = events.pop(0)

    # raw_train.plot_psd(area_mode='range', tmax=10.0)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    signal = np.transpose(raw.get_data()[:num_channels])

    events = np.transpose(np.vstack([time, events1, events2, events3]))
    time = raw.times * freq

    return np.asarray(signal), time, events


def label_data_2a_train(signal, time, events, freq, da_mod=1, reuse_data=False, mult_data=False, noise_data=False, neg_data=False):
    # # # Labels the training data for BCI Competition dataset 2a
    # Returns: class-balanced signal and labels as np.arrays
    
    final_labels = []
    signal_out = np.zeros(signal.shape)
    t, s, j1 = 0, 0, 0
    
    # Used for data augmentation
    if reuse_data:
        da_mod += da_mod
    if mult_data:
        da_mod += 2*da_mod
    if noise_data:
        da_mod += da_mod
    if neg_data:
        da_mod += da_mod

    min_event = 769
    max_event = 772

    # 0 - rest; 1 - left; 2 - right; 3 - foot; 4 - tongue
    for j in range(len(time)):
        # Iterates  through events until a valid event is encountered
        while events[t, 1] < min_event or events[t, 1] > max_event:
            t = t+1
            if t == len(events):
                signal_out = signal_out[:len(final_labels)]
                return signal_out, np.asarray(final_labels)

        # Adds signal to final array if it is between 0.5 and 2.5 seconds after the prompt
        if events[t, 0] + freq/2 < time[j] < events[t, 0] + freq * (5/2):
            final_labels.append(events[t, 1] - 768)
            signal_out[j1] = signal[j]
            j1 += 1
        # At 2.5 seconds after the prompt, adds the signal data and iterates to the next event
        elif time[j] >= events[t, 0] + freq * (5/2):
            final_labels.append(events[t, 1] - 768)
            signal_out[j1] = signal[j]
            j1 += 1
            t += 1
        elif events[t, 0] < time[j] < events[t, 0] + freq/2:
            continue
        else:
            # Only add resting data when it isn't overrepresented when compared to the other classes
            # Ensures a balanced training dataset
            if s < da_mod*final_labels.count(1):
                final_labels.append(0)
                signal_out[j1] = signal[j]
                s += 1
                j1 += 1
            else:
                continue

        if t == len(events):
            signal_out = signal_out[:len(final_labels)]
            return signal_out, np.asarray(final_labels)

    signal_out = signal_out[:len(final_labels)]

    return np.asarray(signal_out), np.asarray(final_labels)


def label_data_2a_val(signal, time, events, freq, remove_rest=False):
    # # # Labels the validation and test sets for the BCI Comp. IV dataset 2a
    # Returns: signal and labels
    
    final_labels = []
    t, s, j1 = 0, 0, 0

    if not remove_rest:
        signal_out = signal
    else:
        signal_out = np.zeros(signal.shape)

    min_event, max_event = 769, 772
    cc_labels = 0

    for j in range(len(time)):
        while events[t, 1] < min_event or events[t, 1] > max_event:
            t += 1
            if t == len(events):
                signal_out = signal_out[:len(final_labels)]
                return signal_out, np.asarray(final_labels)

        # Labels all data between 0.5 and 4 seconds after prompt
        if events[t, 0] + freq / 2 < time[j] < events[t, 0] + freq * 4:
            final_labels.append(events[t, 1] - 768)
            cc_labels += 1
            if remove_rest:
                signal_out[j1] = signal[j]
                j1 += 1
        elif time[j] >= events[t, 0] + freq * 4:
            final_labels.append(events[t, 1] - 768)
            t += 1
            if remove_rest:
                signal_out[j1] = signal[j]
                j1 += 1
        elif remove_rest:
            if events[t, 0] < time[j] < events[t, 0] + freq / 2:
                continue
            elif events[t, 0] + freq * (5 / 2) < time[j] < events[t, 0] + freq * 4:
                continue
            elif s < int(cc_labels/4):
                signal_out[j1] = signal[j]
                final_labels.append(events[t, 1] - 768)
                s += 1
                j1 += 1
            else:
                continue
        else:
            final_labels.append(0)

        if t == len(events):\
            signal_out = signal_out[:len(final_labels)]
            return signal_out, np.asarray(final_labels)

    return signal_out, np.asarray(final_labels)


def label_data_lsl(data, label_in, n_channels=16, classes=4):
    j = 1
    
    data_out = np.zeros((data.shape[0], n_channels))
    label_out = np.zeros(len(data))
    if classes == 3:
        for i in range(len(data)):
            data_out[i] = data[i, 1:n_channels + 1]
            if label_in[j, 0] + 500 < data[i, 0] < label_in[j, 0] + 2500:
                if label_in[j, 1] == 300:  # means left
                    label_out[i] = 1
                elif label_in[j, 1] == 200: # means right
                    label_out[i] = 2
                else:
                    label_out[i] = 0
            elif data[i, 0] > label_in[j, 0] + 2500:
                if label_in[j, 1] == 300:  # means left
                    label_out[i] = 1
                elif label_in[j, 1] == 200: # means right
                    label_out[i] = 2
                else:
                    label_out[i] = 0
                j = j + 1
            else:
                label_out[i] = 0
    elif classes == 2:
        i = 0
        for t in range(len(data)):
            if label_in[j, 0] + 500 < data[t, 0] < label_in[j, 0] + 2500:
                if label_in[j, 1] == 300:  # means left
                    data_out[i] = data[t, 1:n_channels + 1]
                    label_out[i] = 0
                    i += 1
                elif label_in[j, 1] == 200: # means right
                    data_out[i] = data[t, 1:n_channels + 1]
                    label_out[i] = 1
                    i += 1
            elif data[t, 0] > label_in[j, 0] + 2500:
                if label_in[j, 1] == 300:  # means left
                    data_out[i] = data[t, 1:n_channels + 1]
                    label_out[i] = 0
                    i += 1
                    j = j + 1
                elif label_in[j, 1] == 200:  # means right
                    data_out[i] = data[t, 1:n_channels + 1]
                    label_out[i] = 1
                    i += 1
                    j = j + 1

        data_out = data_out[:i]
        label_out = label_out[:i]

    elif classes == 4:
        i = 0
        for t in range(len(data)):
            try:
                if label_in[j, 0] + 500 < data[t, 0] < label_in[j, 0] + 2500:
                    if label_in[j, 1] == 300:  # means left
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 0
                        i += 1
                    elif label_in[j, 1] == 200:  # means right
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 1
                        i += 1
                    elif label_in[j, 1] == 400:  # means up
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 2
                        i += 1
                    elif label_in[j, 1] == 500:  # means down
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 3
                        i += 1
                elif data[t, 0] > label_in[j, 0] + 2500:
                    if label_in[j, 1] == 300:  # means left
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 0
                        i += 1
                    elif label_in[j, 1] == 200:  # means right
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 1
                        i += 1
                    elif label_in[j, 1] == 400:  # means up
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 2
                        i += 1
                    elif label_in[j, 1] == 500:  # means down
                        data_out[i] = data[t, 1:n_channels + 1]
                        label_out[i] = 3
                        i += 1
                    j += 1
            except IndexError:
                continue
        
        data_out = data_out[:i]
        label_out = label_out[:i]

    elif classes == 5:
        for i in range(len(data)):
            data_out[i] = data[i, 1:n_channels + 1]
            # print(label_in[j, 0], data[i, 0])
            if label_in[j, 0] + 500 < data[i, 0] < label_in[j, 0] + 2500:
                if label_in[j, 1] == 300:  # means left
                    label_out[i] = 1
                elif label_in[j, 1] == 200:  # means right
                    label_out[i] = 2
                elif label_in[j, 1] == 400:  # means up
                    label_out[i] = 3
                elif label_in[j, 1] == 500:  # means down
                    label_out[i] = 4
                else:
                    label_out[i] = 0
            elif data[i, 0] > label_in[j, 0] + 2500:
                if label_in[j, 1] == 300:  # means left
                    label_out[i] = 1
                elif label_in[j, 1] == 200:  # means right
                    label_out[i] = 2
                elif label_in[j, 1] == 400:  # means up
                    label_out[i] = 3
                elif label_in[j, 1] == 500:  # means down
                    label_out[i] = 4
                else:
                    label_out[i] = 0
                j = j + 1
            else:
                label_out[i] = 0

    return data_out, label_out


def process_data_2a(data, label, window_size, num_channels=22):
    # # # Processes entire dataset, segmenting and normalising it
    # Returns: Training and testing data and labels
    
    data, label = segment_signal_without_transition(data, label, window_size)
    unique, counts = np.unique(label, return_counts=True)
    data = norm_dataset(data)
    data = data.reshape([label.shape[0], window_size, num_channels])

    train_data, test_data, train_y, test_y = split_data(data, label)

    return train_data, test_data, train_y, test_y


def segment_signal_without_transition(data, label, window_size, overlap=1):
    # # # Breaks signal into overlapping windows of data
    # Returns the sequence of segments, and a single label for each segment

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
    # # # Calculates where windows begin and end
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += (size / overlap)


def split_data(data_in_s, label_s):
    # Splits data into a training and test set
    
    split = int(0.2 * len(label_s))
    train_x = data_in_s[0:split]
    train_y = label_s[0:split]

    test_x = data_in_s[split:]
    test_y = label_s[split:]

    return train_x, test_x, train_y, test_y


def find_cov_matrices(data):
    # # # Calculates covariance matrices
    # Returns: Covariance matrices
    
    cov_mat = np.zeros((data.shape[0], data.shape[2], data.shape[2]))
    for i in range(data.shape[0]):
        cov_mat[i] = np.cov(np.transpose(data[i]))

    return cov_mat


def norm_dataset(dataset_1D):
    # # # Normalises the dataset
    # Returns: normalised dataset
    norm_dataset_1D = np.zeros(dataset_1D.shape)
    for i in range(dataset_1D.shape[1]):
        norm_dataset_1D[:,i] = feature_normalize(dataset_1D[:,i])
    return norm_dataset_1D


def feature_normalize(data):
    # # # Performs z-normalization on a given segment of data
    # Returns: normalized segment
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    data_normalized = (data_normalized - np.min(data_normalized))/np.ptp(data_normalized)

    return data_normalized


def dataset_1Dto1D_32_PN(dataset_1D):
    # Maps entire Physionet dataset to the 32 channels used in this study
    dataset_1D_32 = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        dataset_1D_32[i] = data_1Dto1D_32_PN(dataset_1D[i])
    return dataset_1D_32


def data_1Dto1D_32_PN(data):
    # Maps Physionet data to the 32 channels used in this study
    oneD_32 = np.zeros(32)
    oneD_32 = [data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16], data[18], data[20],
               data[21], data[23], data[25], data[27], data[29], data[31], data[33], data[35], data[37], data[40],
               data[41], data[46], data[48], data[50], data[52], data[54], data[55], data[56], data[58], data[59], data[61]]
    return oneD_32


def dataset_1Dto1D_16_PN(dataset_1D):
    # Maps entire Physionet dataset to the 16 channels used in this study
    dataset_1D_16 = np.zeros([dataset_1D.shape[0], 16])
    for i in range(dataset_1D.shape[0]):
        dataset_1D_16[i] = data_1Dto1D_16_PN(dataset_1D[i])
    return dataset_1D_16


def data_1Dto1D_16_PN(data):
    # Maps Physionet data to the 16 channels used in this study
    oneD_16 = np.zeros(16)
    oneD_16 = [data[8], data[10], data[12], data[21], data[23], data[31], data[33], data[35], data[40], data[41], data[48],
               data[50], data[52], data[55], data[59], data[61]]
    return oneD_16


def butter_bandpass(lowcut, highcut, fs, order=5):
    # # # Creates butterworth bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # # # Applies butterworth bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def data_aug(data, labels, size, reuse_data, mult_data, noise_data, neg_data):
    # # # Performs data augmentation for enhanced classifer training
    
    n_channels = data.shape[2]

    if reuse_data:
        data, labels = data_reuse_f(data, labels, size, n_channels=n_channels)
    if mult_data:
        data, labels = data_mult_f(data, labels, size, n_channels=n_channels)
    if noise_data:
        data, labels = data_noise_f(data, labels, size, n_channels=n_channels)
    if neg_data:
        data, labels = data_neg_f(data, labels, size, n_channels=n_channels)

    return data, labels


def data_reuse_f(data, labels, size, n_channels=22):
    # # # Augments data by reusing control class data, while using unique resting data
    
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


def data_neg_f(data, labels, size, n_channels=22):
    # # # Calculates and appends vertically flipped control class data for data augmentation 
    
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] > 0:
            data_t = -1*data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])
    print('Added Data Shape: ', new_data_ar.shape)
    data_out = np.append(data, new_data_ar, axis=0)
    labels_out = np.append(labels, np.asarray(new_labels))

    return data_out, labels_out


def data_mult_f(data, labels, size, n_channels=22):
    # # # Scales control class data in the positive and negative direction for data augmentation
    
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] > 0:
            # print(data[i])
            data_t = data[i]*1.02
            new_data.append(data_t)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] > 0:
            data_t = data[i]*0.98
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])
    data_out = np.append(data, new_data_ar, axis=0)

    labels_out = np.append(labels, np.asarray(new_labels))

    return data_out, labels_out


def data_noise_f(data, labels, size, n_channels=22):
    # # # Adds noise to control class data for data augmentation
    
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] > 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / 5
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])
    data_out = np.append(data, new_data_ar, axis=0)

    labels_out = np.append(labels, np.asarray(new_labels))

    return data_out, labels_out
