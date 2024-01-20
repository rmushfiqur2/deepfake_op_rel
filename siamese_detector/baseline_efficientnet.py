from my_functions import *

net_model = 'EfficientNetAutoAttB4ST'
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print("torch cuda not available")
face_policy = 'scale'
face_size = 224
frames_per_video = 64

videoreader = VideoReader(verbose=False)


import gc
torch.cuda.empty_cache()
gc.collect()


model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
net = getattr(fornet,net_model)().eval().to(device)
net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

table = pd.read_csv('matching_table_ids.txt')
id_celebdf = table.id_celebdf.tolist()
id_cacds = table.id_cacd.tolist()


img_folder_real = "/home/mrahman7/Insync/mrahman7@ncsu.edu/Google Drive - Shared with me/faceswap-GAN/face-cacd/"
img_folder_fake = "/home/mrahman7/Insync/mrahman7@ncsu.edu/Google Drive - Shared with me/faceswap-GAN/face-cacd-swapped-2ndhalf/"

from glob import glob

valid_ids = []
auc_arr = []

exclude_id = [42, 43, 44, 45, 19, 18]
false_pos_rate_all = []
true_pos_rate_all = []

for person_id in range(0,62):

    if person_id not in id_celebdf:
        continue  # not in cacd

    if person_id in exclude_id:
        continue

    print("person id: {}".format(person_id))

    cacd_id = id_cacds[id_celebdf.index(person_id)]
    print(cacd_id)


    img_files = glob(img_folder_real + str(cacd_id) + "/*")
    img_files_fake = glob(img_folder_fake + "to_" + str(cacd_id) + "/*/*")

    print(len(img_files))
    #print(len(img_files_gen))
    print(len(img_files_fake))
    #print(len(img_files_fake_gen))

    if len(img_files)*len(img_files_fake) == 0:
        continue
    print('working')

    selected_img = np.linspace(0, len(img_files)-1, num =  frames_per_video)
    selected_img_fake = np.linspace(0, len(img_files_fake)-1, num =  frames_per_video)
    img_stack_real = [cv2.cvtColor(cv2.imread(img_files[int(i)]), cv2.COLOR_BGR2RGB) for i in selected_img]
    img_stack_fake = [cv2.cvtColor(cv2.imread(img_files_fake[int(i)]), cv2.COLOR_BGR2RGB) for i in selected_img_fake]

    faces_real_t = torch.stack( [ transf(image=frame)['image'] for frame in img_stack_real] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
    faces_fake_t = torch.stack( [ transf(image=frame)['image'] for frame in img_stack_fake] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces

    name = "cacd_"+ str(person_id)  # saved as id of original celebdf, not cacd
    vid_id = 0
    output_real_name = '../saved_features/features_real_'+name+'_v_'+str(vid_id)+'_recon_5-9_fpv'+str(frames_per_video)+'.pkl'

    output_fake_name = '../saved_features/features_swapped_'+name+'_v_'+str(vid_id)+'_recon_5-9_fpv'+str(frames_per_video)+'.pkl'

    with torch.no_grad():
        res_real = net(faces_real_t.to(device)).cpu().numpy().flatten()
        res_fake = net(faces_fake_t.to(device)).cpu().numpy().flatten()

    res_merge = np.concatenate((res_real, res_fake))
    # In my Tune_NN real video=1, fake video=0; but in this net fake video=1, real video=0
    Y_test = np.concatenate((np.array([0] * len(faces_real_t)) , np.array([1] * len(faces_fake_t))))

    false_pos_rate, true_pos_rate, _ = roc_curve(Y_test, res_merge)
    roc_auc = auc(false_pos_rate, true_pos_rate)
    auc_arr.append(roc_auc)
    valid_ids.append(person_id)

    false_pos_rate_all.append(false_pos_rate)
    false_pos_rate_all.append(true_pos_rate)
import csv
if input('want to save?')=='y':
    filename = input('filename :')
    with open(filename+'.pkl', 'wb') as f:
        ids_column = np.expand_dims(valid_ids, axis=1)
        result = np.expand_dims(auc_arr, axis=1)
        csv_data=np.concatenate((ids_column, result), axis=1)
        # convert array into dataframe
        DF = pd.DataFrame(csv_data)
        DF.to_csv(filename + '.csv')
        with open(filename + '_fp_tp.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for ddd in false_pos_rate_all:
                spamwriter.writerow([ dd for dd in ddd])

