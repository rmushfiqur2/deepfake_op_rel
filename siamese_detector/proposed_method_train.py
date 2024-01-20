import matplotlib.pyplot as plt
from scipy import stats
from my_functions import *

net_model = 'EfficientNetAutoAttB4ST'
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 128

facedet = BlazeFace().to(device)
facedet.load_weights("./blazeface/blazeface.pth")
facedet.load_anchors("./blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)

roc_curves_arr_old = [[] for _ in range(100)]
roc_curves_arr_new = [[] for _ in range(100)]
auc_arr = [[] for _ in range(100)]
auc_arr_train = [[] for _ in range(100)]
auc_arr_valid = [[] for _ in range(100)]
valid_ids = []

flag_tune_only_last_layer = True
initial_lr = 5e-6
repeat_times = 1
epoch_num = 250
num_train_video = 3
import glob

repeat_loss_arr, repeat_valid_loss_arr = [], []
eer_arr_old, eer_arr_new, eer_arr_frame_old, eer_arr_frame_new = [], [], [], []
recons = "5-9_fpv64"  # "all" 5-9 # swapped also 64 frames per person
train_set1 = "features_real_"
train_set2 = "features_fake_"
test_set1 = "features_real_"
test_set2 = "features_swapped_"  # features_swapped_ features_fake_

# cross datasaet validation
# testing data: cacd

import pandas as pd

table = pd.read_csv('matching_table_ids.txt')
id_celebdf = table.id_celebdf.tolist()
id_cacds = table.id_cacd.tolist()

train_acc = False
valid_acc = False

exclude_ids = [14, 15, 18, 19, 42, 43, 44, 45]  # no face swapped generated

# Create the figure and axes for subplots
fig_auc, axes_auc = plt.subplots(2, 3, figsize=(10, 6))


def include_id_info(matrix, id_to_add):
    # matrix shape (frame_num, num_features) (128, 1792)
    frames_per_video_matrix = matrix.shape[0]
    id_info = np.zeros((1, 64))
    id_info[0, id_to_add] = 1
    id_info_repeated = np.repeat(id_info, frames_per_video_matrix, axis=0)
    matrix_with_id = np.concatenate((matrix.cpu(), id_info_repeated), axis=1)
    return torch.from_numpy(matrix_with_id).type(torch.float32)


def include_id_data_contrasive(contrasive_data, id_no):
    a = include_id_info(contrasive_data[0], id_no)
    b = include_id_info(contrasive_data[1], id_no)
    return [a, b]  # contrasive_data [a, b]


if __name__ == '__main__':
    auc_all_runs = {}
    R_colors = ['red', 'green', 'blue', 'orange']
    for R in range(4):  # repeat process
        all_ids = set(range(5))  # total 5 videos
        valid_index = sample(all_ids, 1)
        rest_idx = list(set(all_ids).difference(valid_index))
        train_index = rest_idx
        valid_index2, train_index2 = valid_index, train_index

        real_videos_train, real_videos_valid, real_videos_test, fake_videos_train, fake_videos_valid, fake_videos_test = [], [], [], [], [], []
        test_id_cal_real = [0]
        test_id_cal_fake = [0]
        train_id_cal_real = [0]
        train_id_cal_fake = [0]
        valid_id_cal_real = [0]
        valid_id_cal_fake = [0]
        for n in range(62):

            if n not in id_celebdf:
                continue  # not in cacd
            if n in exclude_ids:
                continue

            name = 'celebdfv2_' + str(n)
            name_cacd = 'cacd_' + str(n)

            file = glob.glob(
                '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/features_real_'
                + name + '_v_' + '*' + '_recon_' + recons + '*')
            if len(file) == 0:
                continue
            file = glob.glob(
                '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/features_real_'
                + name_cacd + '_v_*' + '_recon_' + recons + '*')
            if len(file) == 0:
                continue

            print("Celebdf id:", n)

            if n not in valid_ids:
                valid_ids.append(n)

            for idx in train_index:
                file = glob.glob(
                    '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/'
                    + train_set1 + name + '_v_' + str(idx) + '_recon_' +
                    recons + '*')[0]
                with open(file, 'rb') as f:
                    res = pickle.load(f)
                real_videos_train = real_videos_train + [
                    include_id_data_contrasive(
                        sample_frames_N(res, frames_per_video), n)
                ]
            for idx in train_index2:
                file = glob.glob(
                    '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/'
                    + train_set2 + name + '_v_' + str(idx) + '_recon_' +
                    recons + '*')[0]
                with open(file, 'rb') as f:
                    res = pickle.load(f)
                fake_videos_train = fake_videos_train + [
                    include_id_data_contrasive(
                        sample_frames_N(res, frames_per_video), n)
                ]
            for idx in valid_index:
                file = glob.glob(
                    '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/'
                    + train_set1 + name + '_v_' + str(idx) + '_recon_' +
                    recons + '*')[0]
                with open(file, 'rb') as f:
                    res = pickle.load(f)
                real_videos_valid = real_videos_valid + [
                    include_id_data_contrasive(
                        sample_frames_N(res, frames_per_video), n)
                ]
            for idx in valid_index2:
                file = glob.glob(
                    '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/'
                    + train_set2 + name + '_v_' + str(idx) + '_recon_' +
                    recons + '*')[0]
                with open(file, 'rb') as f:
                    res = pickle.load(f)
                fake_videos_valid = fake_videos_valid + [
                    include_id_data_contrasive(
                        sample_frames_N(res, frames_per_video), n)
                ]

            file = glob.glob(
                '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/'
                + test_set1 + name_cacd + '_v_' + str(0) + '_recon_' + recons +
                '*')[0]
            with open(file, 'rb') as f:
                res = pickle.load(f)
            real_videos_test = real_videos_test + [
                include_id_data_contrasive(
                    sample_frames_N(res, frames_per_video), n)
            ]
            test_id_cal_real.append(len(real_videos_test))

            file = glob.glob(
                '../required-feature-folder/saved_features_unsupervised_train_swapped_plus_effi/'
                + test_set2 + name_cacd + '_v_' + str(0) + '_recon_' + recons +
                '*')[0]
            with open(file, 'rb') as f:
                res = pickle.load(f)
            fake_videos_test = fake_videos_test + [
                include_id_data_contrasive(
                    sample_frames_N(res, frames_per_video), n)
            ]
            test_id_cal_fake.append(len(fake_videos_test))

            train_id_cal_real.append(len(real_videos_train))
            train_id_cal_fake.append(len(fake_videos_train))
            valid_id_cal_real.append(len(real_videos_valid))
            valid_id_cal_fake.append(len(fake_videos_valid))

        in_features = 3904
        net_tuned, train_loss_history, valid_loss_history = Tune_NN_with_id(
            device, initial_lr, epoch_num, real_videos_train,
            real_videos_valid, real_videos_test, fake_videos_train,
            fake_videos_valid, fake_videos_test, in_features, 3840, R)

        # test (corss datasaet valiation)

        net_tuned.eval()
        jj = 0
        valid_ids_to_plot_auc = [3, 6, 16, 27, 47, 57]
        plot_AUC_names = [
            "Ryan Reynolds\nmean=0.967\nstd=0.006",
            "Will Ferrell\nmean=0.788\nstd=0.016",
            "Chris Pine\nmean=0.969\nstd=0.001",
            "Ben Affleck\nmean=0.887\nstd=0.007",
            "Kate Beckinsale\nmean=0.907\nstd=0.011",
            "Carrie-Anne Moss\nmean=0.855\nstd=0.020"
        ]
        plot_auc_serial = 0
        for person_id in valid_ids:
            res_fake_frame, res_real_frame = [], []
            for i in range(test_id_cal_real[jj], test_id_cal_real[jj + 1]):
                out1 = real_videos_test[i][0]
                out2 = real_videos_test[i][1]
                if len(out1) != len(out2):
                    tmp = min(len(out1), len(out2))
                    out1 = out1[:tmp]
                    out2 = out2[:tmp]
                out_feature = net_tuned([out1.to(device), out2.to(device)])
                out1_feature, out2_feature = out_feature[0], out_feature[1]
                res_real_frame = res_real_frame + list(
                    np.sum(
                        np.power(
                            out1_feature.cpu().detach().numpy() -
                            out2_feature.cpu().detach().numpy(), 2), 1))
            for i in range(test_id_cal_fake[jj], test_id_cal_fake[jj + 1]):
                out1 = fake_videos_test[i][0]
                out2 = fake_videos_test[i][1]
                if len(out1) != len(out2):
                    tmp = min(len(out1), len(out2))
                    out1 = out1[:tmp]
                    out2 = out2[:tmp]
                out_feature = net_tuned([out1.to(device), out2.to(device)])
                out1_feature, out2_feature = out_feature[0], out_feature[1]
                res_fake_frame = res_fake_frame + list(
                    np.sum(
                        np.power(
                            out1_feature.cpu().detach().numpy() -
                            out2_feature.cpu().detach().numpy(), 2), 1))

            Y_test = [0] * len(res_fake_frame) + [1] * len(res_real_frame)
            false_pos_rate, true_pos_rate, _ = roc_curve(
                Y_test, res_fake_frame + res_real_frame)
            roc_auc = auc(false_pos_rate, true_pos_rate)
            if person_id in valid_ids_to_plot_auc:
                row_num = plot_auc_serial // 3
                col_num = plot_auc_serial % 3
                axes_auc[row_num,
                         col_num].plot(false_pos_rate,
                                       true_pos_rate,
                                       color='blue')  #  color=R_colors[R]
                if col_num == 0:
                    axes_auc[row_num, col_num].set_ylabel('true positive rate')
                if row_num == 1:
                    axes_auc[row_num,
                             col_num].set_xlabel('false positive rate')
                if R == 0:
                    axes_auc[row_num,
                             col_num].text(0.5,
                                           0.1,
                                           plot_AUC_names[plot_auc_serial],
                                           fontsize=12,
                                           ha='center')
                    axes_auc[row_num, col_num].set_xlim(0.0, 1.0)
                    axes_auc[row_num, col_num].set_ylim(0.0, 1.0)
                    axes_auc[row_num, col_num].grid(True)
                    axes_auc[row_num, col_num].set_xticks(
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        ['0', '.25', '.5', '.75', '1'])
                    axes_auc[row_num, col_num].set_yticks(
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        ['0', '.25', '.5', '.75', '1'])
                plot_auc_serial += 1
                print(
                    'Cross dataset validation: vaidation celebdf' +
                    str(valid_index) + '  AUC=', roc_auc)
            auc_arr[valid_ids.index(person_id)].append(roc_auc)
            jj = jj + 1

        if train_acc:
            jj = 0
            for person_id in valid_ids:
                res_fake_frame, res_real_frame = [], []
                for i in range(train_id_cal_real[jj],
                               train_id_cal_real[jj + 1]):
                    out1 = real_videos_train[i][0]
                    out2 = real_videos_train[i][1]
                    if len(out1) != len(out2):
                        tmp = min(len(out1), len(out2))
                        out1 = out1[:tmp]
                        out2 = out2[:tmp]
                    out_feature = net_tuned([out1.to(device), out2.to(device)])
                    out1_feature, out2_feature = out_feature[0], out_feature[1]
                    res_real_frame = res_real_frame + list(
                        np.sum(
                            np.power(
                                out1_feature.cpu().detach().numpy() -
                                out2_feature.cpu().detach().numpy(), 2), 1))
                for i in range(train_id_cal_fake[jj],
                               train_id_cal_fake[jj + 1]):
                    out1 = fake_videos_train[i][0]
                    out2 = fake_videos_train[i][1]
                    if len(out1) != len(out2):
                        tmp = min(len(out1), len(out2))
                        out1 = out1[:tmp]
                        out2 = out2[:tmp]
                    out_feature = net_tuned([out1.to(device), out2.to(device)])
                    out1_feature, out2_feature = out_feature[0], out_feature[1]
                    res_fake_frame = res_fake_frame + list(
                        np.sum(
                            np.power(
                                out1_feature.cpu().detach().numpy() -
                                out2_feature.cpu().detach().numpy(), 2), 1))

                Y_rain = [0] * len(res_fake_frame) + [1] * len(res_real_frame)
                false_pos_rate, true_pos_rate, _ = roc_curve(
                    Y_rain, res_fake_frame + res_real_frame)
                roc_auc = auc(false_pos_rate, true_pos_rate)
                auc_arr_train[valid_ids.index(person_id)].append(roc_auc)
                jj = jj + 1
        if valid_acc:
            jj = 0
            for person_id in valid_ids:
                res_fake_frame, res_real_frame = [], []
                for i in range(valid_id_cal_real[jj],
                               valid_id_cal_real[jj + 1]):
                    out1 = real_videos_valid[i][0]
                    out2 = real_videos_valid[i][1]
                    if len(out1) != len(out2):
                        tmp = min(len(out1), len(out2))
                        out1 = out1[:tmp]
                        out2 = out2[:tmp]
                    out_feature = net_tuned([out1.to(device), out2.to(device)])
                    out1_feature, out2_feature = out_feature[0], out_feature[1]
                    res_real_frame = res_real_frame + list(
                        np.sum(
                            np.power(
                                out1_feature.cpu().detach().numpy() -
                                out2_feature.cpu().detach().numpy(), 2), 1))
                for i in range(valid_id_cal_fake[jj],
                               valid_id_cal_fake[jj + 1]):
                    out1 = fake_videos_valid[i][0]
                    out2 = fake_videos_valid[i][1]
                    if len(out1) != len(out2):
                        tmp = min(len(out1), len(out2))
                        out1 = out1[:tmp]
                        out2 = out2[:tmp]
                    out_feature = net_tuned([out1.to(device), out2.to(device)])
                    out1_feature, out2_feature = out_feature[0], out_feature[1]
                    res_fake_frame = res_fake_frame + list(
                        np.sum(
                            np.power(
                                out1_feature.cpu().detach().numpy() -
                                out2_feature.cpu().detach().numpy(), 2), 1))

                Y_rain = [0] * len(res_fake_frame) + [1] * len(res_real_frame)
                false_pos_rate, true_pos_rate, _ = roc_curve(
                    Y_rain, res_fake_frame + res_real_frame)
                roc_auc = auc(false_pos_rate, true_pos_rate)
                auc_arr_valid[valid_ids.index(person_id)].append(roc_auc)
                jj = jj + 1
    plt.tight_layout()
    plt.show()

    valid_n = len(valid_ids)
    result = np.array(auc_arr[:valid_n], dtype=np.float32)
    result_train = np.array(auc_arr_train[:valid_n], dtype=np.float32)
    result_valid = np.array(auc_arr_valid[:valid_n], dtype=np.float32)
    #result_backed = result.copy()
    print(result.shape)
    summary = np.mean(result, axis=1)
    print(summary)
    # Calculate mean
    mean_value = np.mean(summary)
    # Calculate median
    median_value = np.median(summary)
    # Calculate trimmed mean (trimming 10% from each end)
    trimmed_mean_value = stats.trim_mean(summary, proportiontocut=0.1)
    # Display the results
    # Calculate standard deviation
    std_deviation = np.std(summary, ddof=1)
    # Calculate interquartile range (IQR)
    q3, q1 = np.percentile(summary, [75, 25])
    iqr = q3 - q1
    print(f"Mean: {mean_value}, Std: {std_deviation}")
    print(f"Median: {median_value}, IQR:{iqr}")
    print(f"Trimmed Mean (20%): {trimmed_mean_value}")

    d = {'auc_table': result, 'auc_summary': summary, 'ids': valid_ids}
    if input('want to save?') == 'y':
        filename = input('filename :')
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(d, f)
            ids_column = np.expand_dims(valid_ids, axis=1)
            csv_data = np.concatenate((ids_column, result), axis=1)
            # convert array into dataframe
            DF = pd.DataFrame(csv_data)
            DF.to_csv(filename + '.csv')
            #csv_data.tofile(filename + '.csv', sep=',', format='%10.5f')
            if valid_acc:
                csv_data = np.concatenate((ids_column, result_valid), axis=1)
                DF = pd.DataFrame(csv_data)
                DF.to_csv(filename + '_validation.csv')
            if train_acc:
                csv_data = np.concatenate((ids_column, result_train), axis=1)
                DF = pd.DataFrame(csv_data)
                DF.to_csv(filename + '_train.csv')
