from my_functions import *
from PIL import Image 

# In[2]:

#with open(r"before_resizing.pkl", "rb") as input_file:
    #scattered_faces = pickle.load(input_file)
    
#for i in range(len(scattered_faces)):
    #face = scattered_faces[i] # uint 8 (H,W,3)
    #fc = Image.fromarray(face)
    #fc.save("store/scattered_{}.jpg".format(i))
    
#d = fdsf + 1


net_model = 'Xception'
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print("torch cuda not available")
face_policy = 'scale'
face_size = 224
frames_per_video = 64

#facedet = BlazeFace().to(device)
#facedet.load_weights("../blazeface/blazeface.pth")
#facedet.load_anchors("../blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)


# In[3]:


#del net
#del out1, out2
import gc
torch.cuda.empty_cache()
gc.collect()


# In[4]:


# #save features

model_url = 'xception_features/Xception_DFDC_bestval-e826cdb64d73ef491e6b8ff8fce0e1e1b7fc1d8e2715bc51a56280fff17596f9.pth'
net = getattr(fornet,net_model)().eval().to(device)
# getattr(fornet,net_model) is equivalent to fornet.EfficientNetAutoAttB4ST here
net.load_state_dict(torch.load(model_url,map_location=device))
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
names = ['CH','BO', 'RF', 'KP', 'EC', 'TB', 'JP', 'LN', 'MS', 'ND']

import gc

table = pd.read_csv('matching_table_ids.txt')
id_celebdf = table.id_celebdf.tolist()
id_cacds = table.id_cacd.tolist()

from glob import glob
# ==================================================================================================
# 5-9 videos used in the reconstruction
# (real - recons real), (face swapped, recons face swapped)

img_folder_real = "/home/mrahman7/Insync/mrahman7@ncsu.edu/Google Drive - Shared with me/faceswap-GAN/face-cacd/"
#img_folder_real_gen = "/home/mrahman7/Insync/mrahman7@ncsu.edu/Google Drive - Shared with me/faceswap-GAN/face-cacd-gan-using5-9/"
img_folder_fake = "/home/mrahman7/Insync/mrahman7@ncsu.edu/Google Drive - Shared with me/faceswap-GAN/face-cacd-swapped-2ndhalf/"
#img_folder_fake_gen = "/home/mrahman7/Insync/mrahman7@ncsu.edu/Google Drive - Shared with me/faceswap-GAN/face-cacd-gan-using5-9-swapped/"

from glob import glob

valid_ids = []
auc_arr = []

exclude_id = [42, 43, 44, 45, 19, 18]

for person_id in range(0,62):

    if person_id not in id_celebdf:
        continue  # not in cacd

    if person_id in exclude_id:
        continue

    print("person id: {}".format(person_id))

    cacd_id = id_cacds[id_celebdf.index(person_id)]
    print(cacd_id)


    #print(img_folder_real + str(person_id) + "/id{}_000{}*".format(person_id, vid_id))
    img_files = glob(img_folder_real + str(cacd_id) + "/*")
    #img_files_gen = glob(img_folder_real_gen + str(cacd_id) + "/*")
    img_files_fake = glob(img_folder_fake + "to_" + str(cacd_id) + "/*/*")
    #img_files_fake_gen = glob(img_folder_fake_gen + "to_" + str(cacd_id) + "/*/*")

    print(len(img_files))
    #print(len(img_files_gen))
    print(len(img_files_fake))
    #print(len(img_files_fake_gen))

    if len(img_files)*len(img_files_fake) == 0:
        continue
    print('working')

    selected_img = np.linspace(0, len(img_files)-1, num =  frames_per_video)
    selected_img_fake = np.linspace(0, len(img_files_fake)-1, num =  frames_per_video)
    #print([int(i) for i in selected_img])
    img_stack_real = [cv2.cvtColor(cv2.imread(img_files[int(i)]), cv2.COLOR_BGR2RGB) for i in selected_img]
    #img_stack_gen = [cv2.cvtColor(cv2.imread(img_files_gen[int(i)]), cv2.COLOR_BGR2RGB) for i in selected_img]
    img_stack_fake = [cv2.cvtColor(cv2.imread(img_files_fake[int(i)]), cv2.COLOR_BGR2RGB) for i in selected_img_fake]
    #img_stack_fake_gen = [cv2.cvtColor(cv2.imread(img_files_fake_gen[int(i)]), cv2.COLOR_BGR2RGB) for i in selected_img_fake]

    faces_real_t = torch.stack( [ transf(image=frame)['image'] for frame in img_stack_real] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
    #faces_gen_t = torch.stack( [ transf(image=frame)['image'] for frame in img_stack_gen] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
    faces_fake_t = torch.stack( [ transf(image=frame)['image'] for frame in img_stack_fake] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
    #faces_fake_gen_t = torch.stack( [ transf(image=frame)['image'] for frame in img_stack_fake_gen] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces

    name = "cacd_"+ str(person_id)  # saved as id of original celebdf, not cacd
    vid_id = 0
    output_real_name = '../saved_features/features_real_'+name+'_v_'+str(vid_id)+'_recon_5-9_fpv'+str(frames_per_video)+'.pkl'

    output_fake_name = '../saved_features/features_swapped_'+name+'_v_'+str(vid_id)+'_recon_5-9_fpv'+str(frames_per_video)+'.pkl'

    with torch.no_grad():
        # net() instead of net.features()
        res_real = net(faces_real_t.to(device)).cpu().numpy().flatten()
        #out2 = net.features(faces_gen_t.to(device))
        res_fake = net(faces_fake_t.to(device)).cpu().numpy().flatten()
        #out4 = net.features(faces_fake_gen_t.to(device))
    #with open(output_real_name, 'wb') as f: pickle.dump([out1, out2], f) #save results
    #with open(output_fake_name, 'wb') as f: pickle.dump([out3, out4], f) #save results

    res_merge = np.concatenate((res_real, res_fake))
    # In my Tune_NN real video=1, fake video=0; but in this net fake video=1, real video=0
    Y_test = np.concatenate((np.array([0] * len(faces_real_t)) , np.array([1] * len(faces_fake_t))))

    false_pos_rate, true_pos_rate, _ = roc_curve(Y_test, res_merge)
    roc_auc = auc(false_pos_rate, true_pos_rate)
    auc_arr.append(roc_auc)
    valid_ids.append(person_id)

if input('want to save?')=='y':
    filename = input('filename :')
    with open(filename+'.pkl', 'wb') as f:
        ids_column = np.expand_dims(valid_ids, axis=1)
        result = np.expand_dims(auc_arr, axis=1)
        csv_data=np.concatenate((ids_column, result), axis=1)
        # convert array into dataframe
        DF = pd.DataFrame(csv_data)
        DF.to_csv(filename + '.csv')
        #csv_data.tofile(filename + '.csv', sep=',', format='%10.5f')

