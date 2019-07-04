# import os.path

import eeg_io_pp
import eeg_io_pp_2
import pyriemann
import numpy as np
from mne.io import read_raw_edf
import math

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from torch import optim

import Deep_Func
import adaptive_bci_func as ABF

import matplotlib.pyplot as plt

import LSLacquire as LSLa
from pylsl import StreamOutlet, StreamInfo
import time

def get_clf():
    if clf_method == "Riemann":
        fgda = pyriemann.tangentspace.FGDA()
        mdm = pyriemann.classification.MDM()
        clf = Pipeline([('FGDA', fgda), ('MDM', mdm)])
        return clf, False
    elif clf_method == "Braindecode":
        model = Deep_Func.create_model(n_classes, freq*window_size, in_chans=num_channels)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer
    elif clf_method == "LSTM":
        model = Deep_Func.create_model_lstm(n_classes, freq*window_size, in_chans=num_channels)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer


def transform_fit(clf, opt, train_data, train_labels):
    np.set_printoptions(precision=3)
    # print(train_data[0,0,:])
    if clf_method == "Riemann":
        # train_data = train_data + 1e-12
        cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))
        print(np.linalg.eigvals(cov_train[15]))
        clf.fit_transform(cov_train, train_labels)
        return clf
    elif clf_method == "Braindecode":
        train_data = (train_data * 1e6).astype(np.float32)
        y = train_labels.astype(np.int64)
        X = np.transpose(train_data, [0, 2, 1])
        model = Deep_Func.fit_transform(clf, opt, X, y, input_time_length=int(freq*window_size), n_channels=num_channels,
                                        num_epochs=epochs)
        return model
    elif clf_method == "LSTM":
        train_data = (train_data * 1e6).astype(np.float32)
        y = train_labels.astype(np.int64)
        X = np.transpose(train_data, [0, 2, 1])
        model = Deep_Func.fit_transform(clf, opt, X, y, input_time_length=int(freq * window_size),
                                        n_channels=num_channels, num_epochs=epochs)
        return model


def predict(clf, val_data, labels, clf_method = "Riemann"):
    if clf_method == "Riemann":
        # print(val_data.shape)
        cov_val = pyriemann.estimation.Covariances().fit_transform(np.transpose(val_data, axes=[0, 2, 1]))
        # print('Covariance Matrix Values: ')
        # print(cov_val)
        pred_val = clf.predict(cov_val)
        return pred_val, 1
    elif clf_method == "Braindecode":
        val_data = (val_data * 1e6).astype(np.float32)
        X = np.transpose(val_data, [0, 2, 1])
        pred_val, cert = Deep_Func.predict(clf, X, labels, input_time_length=int(freq*window_size), n_channels=num_channels)
        return pred_val, cert
    elif clf_method == "LSTM":
        val_data = (val_data * 1e6).astype(np.float32)
        X = np.transpose(val_data, [0, 2, 1])
        pred_val, cert = Deep_Func.predict(clf, X, labels, input_time_length=int(freq * window_size),
                                       n_channels=num_channels)
        return pred_val, cert


def get_data():

    global freq, dataset_dir

    windowed = False

    if windowed:
        data_train_i = np.load(train_data_folder + train_file + '.npy')
        label_train_i = np.load(train_data_folder + train_file + '_labels.npy')
        print("Loading from datafile: ", train_data_folder + train_file)

        train_len = int(data_train_i.shape[0] * 0.333)
        data_train = data_train_i[:train_len]
        label_train = label_train_i[:train_len]
        data_val = data_train_i[train_len:]
        label_val = label_train_i[train_len:]

        # plt.hist(data_train.flatten(), bins=1000)
        # plt.show()

        for i in range(len(data_train)):
            # print(data_train[i, 16])
            data_train[i] = eeg_io_pp.butter_bandpass_filter(data_train[i], 5, 30, freq)
            data_train_ii = data_train[i]

        data_train = eeg_io_pp.norm_dataset(data_train)
    else:
        data_dir = '/homes/df1215/Downloads/eeg_test_data/'
        # data_dir = '/data/eeg_test_data/'
        data = np.genfromtxt(data_dir + 'Fani_0/fd_NFB_001_data.csv', delimiter=',')
        labels = np.genfromtxt(data_dir + 'Fani_0/fd_NFB_001_markers.csv', delimiter=',')

        # data = np.genfromtxt(data_dir + 'Daniel_0/daniel_1211_001_data.csv', delimiter=',')
        # labels = np.genfromtxt(data_dir + 'Daniel_0/daniel_1211_001_markers.csv', delimiter=',')

        data = data[1:len(data)]
        print('data shape: ', data.shape)
        labels = labels[1:len(labels)]

        data_train_i, label_train_i = eeg_io_pp_2.label_data_lsl(data, labels, n_channels=num_channels)
        print('data_train_labeled shape: ', data_train_i.shape)
        data_train, data_val, label_train, label_val = eeg_io_pp_2.process_data_2a(data_train_i, label_train_i, 125, num_channels=num_channels)

    return data_train, label_train, data_val, label_val


def get_test_data():

    global data_test
    if dataset == "bci_comp":
        # data_test, label_test = eeg_io_pp.get_data_2a(dataset_dir + test_file, n_classes, remove_rest=False,
        #                                               training_data=False)
        raw = read_raw_edf(dataset_dir + test_file, preload=True, stim_channel='auto', verbose='WARNING')
        data_test = np.asarray(np.transpose(raw.get_data()[:num_channels]))
    elif dataset == "gtec":
        file = "daniel_WET_3class"
        # data_test, label_test = eeg_io_pp.get_data_gtec(dataset_dir, file, n_classes)
        data = np.genfromtxt(dataset_dir + 'signal/' + file + '_01.csv', delimiter=';')
        raw_data = np.zeros((len(data), num_channels))
        for i in range(1, len(data)):
            if not math.isnan(np.amax(data[i][1:num_channels + 1])):
                raw_data[i] = data[i][1:num_channels + 1]
        data_test = np.asarray(raw_data)
    return data_test


def init_globals(expInfo):
    global dataset, train_data_folder, train_file, model_file
    global dataset_dir, train_file, test_file
    global remove_rest_val, reuse_data, mult_data, noise_data, neg_data, da_mod
    global window_size, freq, num_channels, overlap, buffer_size
    global clf_method, n_classes, epochs
    global bci_iter

    bci_iter = 0

    clf_method = "Riemann"
    # clf_method = "Braindecode"
    # clf_method = "LSTM"

    n_classes = 2
    epochs = 10

    # dataset = "bci_comp"
    dataset = "gtec"

    running = True
    remove_rest_val = True
    reuse_data = False
    mult_data = False
    noise_data = False
    neg_data = False
    da_mod = 2

    subject_num = 7
    window_size = 0.5  # in seconds
    overlap = 2

    # if dataset == "bci_comp":
    #     freq, num_channels = 250, 22
    # elif dataset == "physionet":
    #     freq, num_channels = 160, 64
    # elif dataset == "gtec":
    #     freq = 250
    #     num_channels = 32

    freq = 250
    num_channels = 16

    buffer_size = int(freq * window_size)

    # File naming
    data_folder = '/homes/df1215/bci_test_venv/bin/'
    train_data_folder = data_folder + '4class_MI/data/final_data/'
    fb_data_folder = data_folder + '4class_MI_feedback/data/final_data/'
    # train_file = '%s_%s_%s' % (expInfo['participant'], '4class_MI', expInfo['session'])
    train_file = '%s_%s_%s' % (expInfo['participant'], '2class_MI', expInfo['session'])
    fb_data_file = train_file
    model_file = data_folder + 'models/' + train_file
    model_file = model_file + '_' + str(epochs) + 'epoch_model'

    return


def train_network(exp_info):
    init_globals(exp_info)

    # model_file = data_folder + 'models/' + data_file + '_' + str(epochs) + 'epoch_model'

    if clf_method == "LSTM":
        try:
            clf = Deep_Func.load_model(model_file, in_chans=num_channels, input_time_length=buffer_size)
        # except FileNotFoundError:
        except EnvironmentError:
            data_train, label_train, data_val, label_val = get_data()
            clf, opt = get_clf()
            clf = transform_fit(clf, opt, data_train, label_train)
            Deep_Func.save_model(clf, model_file)
    elif clf_method == "Riemann":
        data_train, label_train, data_val, label_val = get_data()
        print(data_train.shape)
        clf, opt = get_clf()
        clf = transform_fit(clf, opt, data_train, label_train)

        print("training data: ")
        pred_train, cert = predict(clf, data_train, label_train)
        eval_network(label_train, pred_train)

        print("validation data: ")
        pred_val, cert = predict(clf, data_val, label_val)
        eval_network(label_val, pred_val)

    return clf


def eval_network(label_val, pred_val):
    plt.hist(label_val.flatten(), bins=1000)
    # plt.show()

    unique, counts = np.unique(label_val, return_counts=True)
    print("Labels: ", unique, counts)
    print(label_val)
    unique, counts = np.unique(pred_val, return_counts=True)
    print("Predicted: ", unique, counts)
    print(pred_val)

    conf_mat = confusion_matrix(label_val, pred_val)
    print(conf_mat)
    tru_pos, prec_i, recall_i = [], [], []
    for i in range(conf_mat.shape[0]):
        tru_pos.append(conf_mat[i, i])
        prec_i.append(conf_mat[i, i]/np.sum(conf_mat[:, i]).astype(float))
        recall_i.append(conf_mat[i, i]/np.sum(conf_mat[i, :]).astype(float))

    accuracy_val = np.sum(tru_pos).astype(float) / (np.sum(conf_mat)).astype(float)
    print("accuracy: {}".format(accuracy_val))

    precision_tot = np.sum(prec_i)/conf_mat.shape[0]
    print("total precision: {}".format(precision_tot))

    precision_cc = np.sum(prec_i[1:]) / (conf_mat.shape[0]-1)
    print("control class precision: {}".format(precision_cc))

    recall_tot = np.sum(recall_i) / conf_mat.shape[0]
    print("total recall: {}".format(recall_tot))

    recall_cc = np.sum(recall_i[1:]) / (conf_mat.shape[0] - 1)
    print("control class recall: {}".format(recall_cc))

    print("# # # # # # # # # # # # # # # # # # # # # # #")
    print(" ")
    print("# # # # # # # # # # # # # # # # # # # # # # #")

    return accuracy_val





def bci_buffer(iter_num, buffer_size):
    current_data = data_test[iter_num:iter_num + buffer_size]
    return current_data


def iter_bci_buffer(current_data, iter_n):
    real_time = True
    buffer_size = current_data.shape[0]
    if real_time:
        current_data = bci_buffer_rt(current_data, buffer_size, iter_n)
    else:
        current_data = bci_buffer(iter_n, buffer_size)
        iter_n += 1

    return current_data


def bci_buffer_rt(current_data, buffer_size, iter_n):
    is_new_data = True
    iter_i = (iter_n - 1)*buffer_size
    while is_new_data:
        for [new_data, iter_i] in read_bci_data(iter_i):
            i = buffer_size
            while i > 0:
                if i == buffer_size:
                    current_data[buffer_size - 1] = new_data[0:current_data.shape[1]]
                else:
                    current_data[buffer_size - i - 1] = current_data[buffer_size - i]
                i -= 1
            if iter_i > buffer_size:
                is_new_data = False
    # print(current_data[:,1])

    return current_data


def read_bci_data(iter_n):
    # data = data_test[iter_n]
    data = data_receiver.receive()
    while data.shape[0] < 1:
        data = data_receiver.receive()
    for i in range(data.shape[0]):
        iter_n += 1
        yield data[i], iter_n


def setup_receiver():
    global data_receiver
    data_receiver = LSLa.lslReceiver(True, True)


def get_bci_class(bci_iter, clf, num_channels=32):
    filter_rt = True

    buffer_size = int(freq*window_size)
    label = [0]
    if bci_iter < 2:
        global buffer_data
        buffer_data = np.zeros((buffer_size, num_channels))
    # print('bci_iter: ', bci_iter)
    buffer_data = iter_bci_buffer(buffer_data, bci_iter)
    # save data!!!!
    if filter_rt:
        # print('filter_rt!')
        buffer_data = eeg_io_pp.butter_bandpass_filter(buffer_data, 7, 30, freq)

    x1 = buffer_data.reshape(1, buffer_data.shape[0], buffer_data.shape[1])
    x1 = eeg_io_pp.norm_dataset(x1)

    try:
        a, cert = predict(clf, x1, label)
    except ValueError:
        a, cert = 0, 0
        print('Value Error')

    # print(bci_iter, a)

    return a, cert, buffer_data


if __name__ == '__main__':

    expInfo = {'participant': 'daniel', 'session': '001'}
    clf = train_network(expInfo)

    data_train, label_train, data_val, label_val = get_data()
    unique, counts = np.unique(label_train, return_counts=True)
    print("Labels (train): ", unique, counts)
    unique, counts = np.unique(label_val, return_counts=True)
    print("Labels (val): ", unique, counts)

    pred_val, cert = predict(clf, data_val, label_val)
    # pred_val, cert = ABF.predict_own(data_val, mean_cov)
    eval_network(label_val, pred_val)

    # data_test, label_test = eeg_io_pp.get_data_2a(dataset_dir + test_file, n_classes, remove_rest=False, training_data=False)

    buffer_size = int(freq * window_size)
    buffer_data = np.zeros((buffer_size, num_channels))
    iter_num = 0
    filter_rt=False

    setup_receiver()
    info = StreamInfo(name='FeedbackEEG1', type='FeedbackEEG1', channel_count=3, channel_format='float32', source_id='myuid342345')

    class_outlet = StreamOutlet(info)
    label = 0
    use_rest = False

    cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(data_train, axes=[0, 2, 1]))
    mean_cov_i, num_in_class_i = ABF.training_data_cov_means(cov_train, label_train)
    # mean_cov, med_time = ABF.full_calc_mean_cov(cov_train, label_train, n_classes, mean_cov_i)

    while buffer_data.shape[0] == buffer_size:
        begin_tic = time.time()
        buffer_data = iter_bci_buffer(buffer_data, iter_num)
        # iter_tic = time.time() - begin_tic

        if filter_rt:
            buffer_data = eeg_io_pp.butter_bandpass_filter(buffer_data, 0.5, 30, freq)

        x1 = eeg_io_pp_2.norm_dataset(buffer_data)
        # x1 = buffer_data
        x1 = x1.reshape(1, x1.shape[0], x1.shape[1])
        x2 = pyriemann.estimation.Covariances().fit_transform(np.transpose(x1, axes=[0, 2, 1]))

        try:
            # a, cert = predict(clf, x2, label)
            a, cert = ABF.predict_own(x2, mean_cov_i)
        except ValueError:
            # print('ValueError!')
            a, cert = 0, 0
        except KeyboardInterrupt:
            data_receiver.clean_up()

        if not use_rest:
            if a == 0:
                a = 300
            elif a == 1:
                a = 200
            elif a == 2:
                a = 400
            elif a == 3:
                a = 500
            # cert = 1

        print(iter_num, a, cert)
        class_outlet.push_sample([a,cert,iter_num])
        time.sleep(0.5)
        iter_num += 1

