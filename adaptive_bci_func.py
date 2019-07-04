import numpy as np
import eeg_io_pp_2
import real_time_BCI
import pyriemann
import time
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import covariance
from pyriemann.utils.mean import mean_covariance, mean_riemann
from pyriemann.utils.distance import distance
import pandas

def predict_distances_own(covtest, covmeans):
    """Helper to predict the distance. equivalent to transform."""
    Nc = len(covmeans)

    # if self.n_jobs == 1:
    # print(covtest.shape, covmeans.shape)
    dist = [distance(covtest, covmeans[m], 'riemann')
            for m in range(Nc)]
    # else:
    #     dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
    #         covtest, self.covmeans_[m], self.metric_dist)
    #                                         for m in range(Nc))

    dist = np.concatenate(dist, axis=1)
    return dist


def predict_own(covtest, covmeans):
    """get the predictions.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_channels)
        ndarray of SPD matrices.

    Returns
    -------
    pred : ndarray of int, shape (n_trials, 1)
        the prediction for each trials according to the closest centroid.
    """
    print('covshapes: ', covtest.shape, covmeans.shape)
    dist = predict_distances_own(covtest, covmeans)
    print('Dist Shape: ', dist.shape)
    cert = (dist.mean(axis=1) - dist.min(axis=1)) * 4
    return dist.argmin(axis=1), cert



def full_calc_mean_cov(cov_train_i, label_train_i, num_classes, mean_cov_i):
    tic2 = time.clock()

    # print(cov_train_i.shape)
    # print(label_train_i.shape)

    mean_cov_n = np.zeros((num_classes, cov_train_i.shape[1], cov_train_i.shape[2]))

    for l in range(num_classes):
        # print(l)
        try:
            mean_cov_n[l] = mean_covariance(cov_train_i[label_train_i == l], metric='riemann',
                                            sample_weight=None)
        except ValueError:
            mean_cov_n[l] = mean_cov_i[l]

    toc2 = time.clock()
    time_out = toc2 - tic2
    print(toc2 - tic2)  # .4436 seconds

    return mean_cov_n, time_out


def training_data_cov_means(X, y, num_classes=4):

    mean_cov_i = np.zeros((num_classes, X.shape[1], X.shape[2]))
    num_in_class_i = np.zeros(num_classes)
    print('num_classes: ', num_classes)
    for l in range(num_classes):
        sample_weight = np.ones(X[y == l].shape[0])
        mean_cov_i[l] = mean_covariance(X[y == l], metric='riemann',
                                        sample_weight=sample_weight)
        num_in_class_i[l] = X[y == l].shape[0]

    return mean_cov_i, num_in_class_i


def alter_mean_cov(mean_cov_in, num_in_class_in, X_val, label_v):
    label_val_j = label_v

    # print(mean_cov_in)

    mean_cov_n = mean_cov_in
    mean_cov_out = mean_cov_in
    # num_in_class_n = num_in_class_in
    num_in_class_out = num_in_class_in
    tic = time.clock()
    for l in range(num_classes):
        if X_val[label_val_j == l].shape[0] > 0:
            sample_weight_n = np.ones(X_val[label_val_j == l].shape[0] + 1)
            # sample_weight_n[0] = num_in_class_in[l]
            sample_weight_n[0] = 10
            sample_weight_n = sample_weight_n/num_in_class_in[l]
            X_val_n = np.vstack(
                [mean_cov_n[l].reshape(1, num_channels, num_channels), X_val[label_val_j == l]])
            mean_cov_n[l] = mean_riemann(X_val_n,
                                            sample_weight=sample_weight_n, init=mean_cov_n[l])
            # num_in_class_n[l] = X_val[label_val_j == l].shape[0]
            # num_in_class_out[l] = num_in_class_out[l] + num_in_class_n[l]
            # mean_cov_out[l] = mean_cov_out[l] + (mean_cov_in[l] - mean_cov_n[l]) / num_in_class_out[l]
            mean_cov_out[l] = mean_cov_n[l]

    time_out = time.clock()-tic
    print(time_out)

    return mean_cov_out, num_in_class_out, time_out


def alter_mean_cov_2(mean_cov_in, num_in_class_in, X_val, label_v):
    label_val_j = label_v

    # print(mean_cov_in)

    mean_cov_n = mean_cov_in
    mean_cov_out = mean_cov_in
    num_in_class_n = np.zeros((num_in_class_in.shape))
    num_in_class_out = num_in_class_in

    # print('OLD OlD num in classes: ', num_in_class_out)
    tic = time.clock()
    for l in range(num_classes):
        if X_val[label_val_j == l].shape[0] > 0:
            sample_weight_n = np.ones(X_val[label_val_j == l].shape[0] + 1)
            # sample_weight_n[0] = num_in_class_in[l]
            sample_weight_n[0] = num_in_class_in[l]
            sample_weight_n = sample_weight_n/num_in_class_in[l]
            X_val_n = np.vstack(
                [mean_cov_n[l].reshape(1, mean_cov_in.shape[1], mean_cov_in.shape[1]), X_val[label_val_j == l]])
            mean_cov_n[l] = mean_riemann(X_val_n,
                                            sample_weight=sample_weight_n, init=mean_cov_n[l])
            num_in_class_n[l] = X_val[label_val_j == l].shape[0]
            # print('num in class N: ', l, num_in_class_n[l])
            # print('OLD num in class N: ', l, num_in_class_out[l])
            num_in_class_out[l] = num_in_class_out[l] + num_in_class_n[l]
            # print('NEW num in class N: ', l, num_in_class_out[l])
            # mean_cov_out[l] = mean_cov_out[l] + (mean_cov_in[l] - mean_cov_n[l]) / num_in_class_out[l]
            mean_cov_out[l] = mean_cov_n[l]

    time_out = time.clock()-tic
    print(time_out)

    return mean_cov_out, num_in_class_out, time_out


def get_data():
    global num_channels, num_classes
    data_type = 'gtec'

    if data_type == 'gtec':
        num_channels = 32
        num_classes = 4
        window_size = 125

        single_session = True

        data_dir = '/data/EEG_Data/adaptive_eeg_test_data/'
        # data = np.genfromtxt(data_dir + 'Daniel_0/daniel_1211_001_data.csv', delimiter=',')
        # labels = np.genfromtxt(data_dir + 'Daniel_0/daniel_1211_001_markers.csv', delimiter=',')

        data_train_i = np.genfromtxt(data_dir + 'Daniel_0/df_FB_001_data.csv', delimiter=',')
        labels_train_i = np.genfromtxt(data_dir + 'Daniel_0/df_FB_001_markers.csv', delimiter=',')

        data_train_i = data_train_i[1:len(data_train_i)]
        labels_train_i = labels_train_i[1:len(labels_train_i)]

        data_train_i, label_train_i = eeg_io_pp_2.label_data_lsl(data_train_i, labels_train_i, n_channels=num_channels, classes=num_classes)

        if single_session:
            data_train, data_val, label_train, label_val = eeg_io_pp_2.process_data_2a(data_train_i, label_train_i, 125,
                                                                                   num_channels=num_channels)
        else:
            data_val_i = np.genfromtxt(data_dir + 'Fani_0/fd_FB_001_data.csv', delimiter=',')
            labels_val_i = np.genfromtxt(data_dir + 'Fani_0/fd_FB_001_markers.csv', delimiter=',')

            data_val_i, label_val_i = eeg_io_pp_2.label_data_lsl(data_val_i, labels_val_i, n_channels=num_channels,
                                                                 classes=num_classes)

            data_train, label_train = eeg_io_pp_2.segment_signal_without_transition(data_train_i, label_train_i, window_size)
            data_train = eeg_io_pp_2.norm_dataset(data_train)
            data_val, label_val = eeg_io_pp_2.segment_signal_without_transition(data_val_i, label_val_i, window_size)
            data_val = eeg_io_pp_2.norm_dataset(data_val)

        data_train = data_train.reshape([label_train.shape[0], window_size, num_channels])
        data_val = data_val.reshape([label_val.shape[0], window_size, num_channels])

    elif data_type == "bci_comp":
        num_classes = 4
        freq, num_channels = 250, 22
        subject_num = 7

        dataset = data_type
        data_folder_i = '/data/bci_data_preprocessed/'

        dataset_dir = "/data2/bci_competition/BCICIV_2a_gdf/"
        train_file = 'A' + format(subject_num, '02d') + 'T.gdf'
        test_file = 'A' + format(subject_num, '02d') + 'E.gdf'
        data_file = dataset + '_sub' + str(subject_num)
        # model_file = data_folder_i + 'models/' + data_file
        # data_folder = data_folder_i + 'bci_comp_data/'

        sig, time, events = eeg_io_pp_2.get_data_2a(dataset_dir + train_file, num_classes)
        data, labels = eeg_io_pp_2.label_data_2a_val(sig, time, events, freq, remove_rest=True)

        labels = labels - 1

        data_train, data_val, label_train, label_val = eeg_io_pp_2.process_data_2a(data, labels, 125, num_channels=num_channels)

    return data_train, data_val, label_train, label_val


def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    #A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
    return A


def shrinkage():

        plt.imshow(cov_train_pyr[0])
        plt.colorbar()
        plt.show()
        cov_train_lasso = cov_train
        prec_train_lasso = cov_train
        cov_train_oas = cov_train
        corr_lasso = cov_train
        for i in range(len(data_train)):
            cov_train_oas[i] = covariance.OAS().fit(data_train[i]).covariance_
            # plt.imshow(cov_train[i])
            # plt.colorbar()
            # plt.show()
            GLassCV = covariance.GraphLassoCV(cv=5)
            cov_train_lasso[i] = GLassCV.fit(data_train[i]).covariance_
            prec_train_lasso[i] = GLassCV.fit(data_train[i]).precision_
            corr_lasso[i] = cov2corr(prec_train_lasso[i])
            print('sum of correlations: ', np.sum(np.abs(corr_lasso[i]), axis=1))
            myalphas = GLassCV.cv_alphas_
            print(myalphas)
            print(np.mean(GLassCV.grid_scores_, axis=1))

            plt.imshow(corr_lasso[i])
            plt.colorbar()
            plt.show()
        cov_train[i] = covariance.LedoitWolf().fit(data_train[i]).covariance_


def select_channels(data, label):
    elec_selec = pyriemann.channelselection.ElectrodeSelection(nelec=22)
    data_out = elec_selec.fit_transform(data, label)

    return data_out, elec_selec


def run_prog():
    global num_channels, num_classes

    # num_channels = 32
    # num_classes = 4

    data_train, data_val, label_train, label_val = get_data()

    fgda = pyriemann.tangentspace.FGDA()
    fgda_i = fgda
    mdm = pyriemann.classification.MDM()
    clf = Pipeline([('FGDA', fgda), ('MDM', mdm)])
    # clf = mdm

    print('Data Train Length: {}'.format(len(data_train)))

    cov_train = np.zeros((len(data_train), num_channels, num_channels))
    cov_val = np.zeros((len(data_val), num_channels, num_channels))

    cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(data_train, axes=[0, 2, 1]))
    cov_val = pyriemann.estimation.Covariances().fit_transform(np.transpose(data_val, axes=[0, 2, 1]))

    # cov_train, elec_selec = select_channels(cov_train, label_train)
    # cov_val = elec_selec.transform(cov_val)

    clf.fit_transform(cov_train, label_train)

    print("training data: ")
    pred_train = clf.predict(cov_train)
    real_time_BCI.eval_network(label_train, pred_train)

    mean_cov_i, num_in_class_i = training_data_cov_means(cov_train, label_train)

    i = 1
    rt_train_window = 2
    opt_num_trials = len(data_train) - 1
    # opt_num_trials = 100
    mean_cov = np.asarray(mean_cov_i)
    mean_cov_2 = np.asarray(mean_cov_i)
    mean_med_t0 = np.asarray(mean_cov_i)
    mean_slow_t0 = np.asarray(mean_cov_i)
    num_in_class = num_in_class_i
    num_in_class_2 = num_in_class_i
    cov_train_i = cov_train
    label_train_i = label_train
    cov_train_med = cov_train[-opt_num_trials:]
    label_train_med = label_train[-opt_num_trials:]

    plot_win = 80
    slow_adapt, med_adapt, fast_adapt = False, False, True
    acc_plt_fc = np.zeros(plot_win)
    acc_plt_med = np.zeros(plot_win)
    acc_plt_fast = np.zeros(plot_win)
    acc_plt_clf = np.zeros(plot_win)
    time_plt_fc = np.zeros(plot_win)
    time_plt_med = np.zeros(plot_win)
    time_plt_fast = np.zeros(plot_win)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.ion()
    plotted_updates = 0
    while (plotted_updates < plot_win):
        if i % rt_train_window == 0:

            if slow_adapt:
                print("Slow ADAPTIVE BCI CLASSIFIER")
                fc_tic = time.clock()
                cov_train_i = np.vstack([cov_train_i, cov_val[i - rt_train_window:i]])
                label_train_i = np.append(label_train_i, label_val[i - rt_train_window:i])
                # cov_val = covariance.OAS().fit(data_val)
                # cov_train_i = fgda_fc.fit_transform(cov_train_i, label_train_i)
                fc_mean_cov, fc_time = full_calc_mean_cov(cov_train_i, label_train_i, num_classes, mean_slow_t0)
                fc_time = time.clock() - fc_tic
                fc_val_predict, fc_cert = predict_own(cov_val[i:], fc_mean_cov)
                fc_val_acc = real_time_BCI.eval_network(label_val[i:],fc_val_predict)
                mean_slow_t0 = fc_mean_cov

            if med_adapt:
                print("Medium ADAPTIVE BCI CLASSIFIER")
                med_tic = time.clock()
                cov_train_med[:(opt_num_trials - rt_train_window)] = cov_train_med[rt_train_window:]
                label_train_med[:(opt_num_trials - rt_train_window)] = label_train_med[rt_train_window:]
                cov_train_med[(opt_num_trials - rt_train_window):] = cov_val[i - rt_train_window:i]
                label_train_med[(opt_num_trials - rt_train_window):] = label_val[i - rt_train_window:i]
                # cov_train_med = fgda_med.fit_transform(cov_train_med, label_train_med)
                med_mean_cov, med_time = full_calc_mean_cov(cov_train_med, label_train_med, num_classes, mean_med_t0)
                med_time = time.clock() - med_tic
                med_val_predict, med_cert = predict_own(cov_val[i:], med_mean_cov)
                med_val_acc = real_time_BCI.eval_network(label_val[i:], med_val_predict)
                mean_med_t0 = med_mean_cov

            if fast_adapt:
                print("Fast ADAPTIVE BCI CLASSIFIER")
                fast_tic = time.clock()
                cov_train_fast = cov_val[i - rt_train_window:i]
                # cov_train_fast = fgda_fast.fit_transform(cov_val[i - rt_train_window:i], label_val[i - rt_train_window:i])
                mean_cov_in = mean_cov
                mean_cov_out, num_in_class, fast_time = alter_mean_cov_2(mean_cov_in, num_in_class, cov_train_fast,
                                                        label_val[i - rt_train_window:i])
                # print('MC After: ', sum(sum(mean_cov_out - mean_cov_in)))
                fast_time = time.clock() - fast_tic
                val_predict, fast_cert = predict_own(cov_val[i:], mean_cov_out)
                fast_val_acc = real_time_BCI.eval_network(label_val[i:], val_predict)

            print("WITHOUT ADAPTATION")
            val_predict_clf = clf.predict(cov_val[i:])
            clf_val_acc = real_time_BCI.eval_network(label_val[i:], val_predict_clf)
            time.sleep(1)

            ax1.clear()
            ax1.set_ylim([0, 1])

            ax2.clear()
            if slow_adapt:
                acc_plt_fc[0:plot_win - 1] = acc_plt_fc[1:plot_win]
                acc_plt_fc[plot_win - 1] = fc_val_acc
                ax1.plot(acc_plt_fc, label='Slow')
                time_plt_fc[0:plot_win - 1] = time_plt_fc[1:plot_win]
                time_plt_fc[plot_win - 1] = fc_time
                ax2.plot(time_plt_fc, label='Slow')
            if med_adapt:
                acc_plt_med[0:plot_win - 1] = acc_plt_med[1:plot_win]
                acc_plt_med[plot_win - 1] = med_val_acc
                ax1.plot(acc_plt_med, label='Medium')
                time_plt_med[0:plot_win - 1] = time_plt_med[1:plot_win]
                time_plt_med[plot_win - 1] = med_time
                ax2.plot(time_plt_med, label='Medium')
            if fast_adapt:
                acc_plt_fast[0:plot_win - 1] = acc_plt_fast[1:plot_win]
                acc_plt_fast[plot_win - 1] = fast_val_acc
                ax1.plot(acc_plt_fast, label='Fast')
                time_plt_fast[0:plot_win - 1] = time_plt_fast[1:plot_win]
                time_plt_fast[plot_win - 1] = fast_time
                ax2.plot(time_plt_fast, label='Fast')

            acc_plt_clf[0:plot_win - 1] = acc_plt_clf[1:plot_win]
            acc_plt_clf[plot_win - 1] = clf_val_acc

            ax1.plot(acc_plt_clf, label='No Adaptation')
            ax1.legend(loc="lower left")

            # ax2.title("Computation Time")
            ax2.legend(loc="lower left")

            plt.draw()
            plt.pause(0.0001)

            plotted_updates += 1

        elif i > len(cov_val) - rt_train_window - 10:
            time.sleep(30)
            break
        i += 1

    # print('Slow: ')
    # print(acc_plt_fc, time_plt_fc)
    # print('Medium: ')
    # print(acc_plt_med, time_plt_med)
    # print('Fast: ')
    # print(acc_plt_fast, time_plt_fast)
    # print('No Adaptation: ')
    # print(acc_plt_clf)

    final_output_data = np.vstack((acc_plt_clf, acc_plt_fc, time_plt_fc, acc_plt_med, time_plt_med, acc_plt_fast, time_plt_fast))
    final_out = np.transpose(final_output_data)
    print(final_out.shape)
    # np.savetxt('df_FB_med_FGDA_20training.csv', final_out, delimiter=',')


if __name__ == '__main__':
    run_prog()