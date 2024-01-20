from my_functions import *
from PIL import Image

net_model = 'EfficientNetAutoAttB4ST'
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')
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
device = torch.device('cuda:0')

model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
net = getattr(fornet, net_model)().eval().to(device)
# getattr(fornet,net_model) is equivalent to fornet.EfficientNetAutoAttB4ST here
net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
transf = utils.get_transformer(face_policy,
                               face_size,
                               net.get_normalizer(),
                               train=False)
                               
### unsupervised
import torch

# finetuned flr
from models.partially_freezed_resnet import get_resnet
model_resnet = get_resnet(device)
model_resnet.load_state_dict(torch.load('../finetuned-on-compact-train-swapped/finetuned-backbone.pth'))
model_resnet.to(device)

modules=list(model_resnet.children())[:-1]
model_unsupervised=nn.Sequential(*modules)
model_unsupervised.to(device)

import gc

table = pd.read_csv('matching_table_ids.txt')
id_celebdf = table.id_celebdf.tolist()
id_cacds = table.id_cacd.tolist()

from glob import glob
# ==================================================================================================
# 5-9 videos used in the reconstruction
# (real - recons real), (face swapped, recons face swapped)

img_folder_real = "../faces/cacd_auth/"
img_folder_real_gen = "../faces/cacd_auth_recons/"
img_folder_fake = "../faces/cacd_deepfake/"
img_folder_fake_gen = "../faces/cacd_deepfake_recons/"

from glob import glob

valid_persons = 0
for person_id in range(0, 62):

    if person_id not in id_celebdf:
        continue  # not in cacd

    if person_id == 42 or person_id == 44 or person_id == 19:
        continue

    print("person id: {}".format(person_id))

    cacd_id = id_cacds[id_celebdf.index(person_id)]
    print(cacd_id)

    img_files = glob(img_folder_real + str(person_id) + "/*")
    img_files_gen = glob(img_folder_real_gen + str(person_id) + "/*")
    img_files_fake = glob(img_folder_fake + str(person_id) + "/*")
    img_files_fake_gen = glob(img_folder_fake_gen + str(person_id) + "/*")

    print(len(img_files))
    print(len(img_files_gen))
    print(len(img_files_fake))
    print(len(img_files_fake_gen))

    if len(img_files) * len(img_files_gen) * len(img_files_fake) * len(
            img_files_fake) == 0:
        continue

    selected_img = np.linspace(0, len(img_files) - 1, num=frames_per_video)
    selected_img_fake = np.linspace(0,
                                    len(img_files_fake) - 1,
                                    num=frames_per_video)
    #print([int(i) for i in selected_img])
    img_stack_real = [
        cv2.cvtColor(cv2.imread(img_files[int(i)]), cv2.COLOR_BGR2RGB)
        for i in selected_img
    ]
    img_stack_gen = [
        cv2.cvtColor(cv2.imread(img_files_gen[int(i)]), cv2.COLOR_BGR2RGB)
        for i in selected_img
    ]
    img_stack_fake = [
        cv2.cvtColor(cv2.imread(img_files_fake[int(i)]), cv2.COLOR_BGR2RGB)
        for i in selected_img_fake
    ]
    img_stack_fake_gen = [
        cv2.cvtColor(cv2.imread(img_files_fake_gen[int(i)]), cv2.COLOR_BGR2RGB)
        for i in selected_img_fake
    ]

    faces_real_t = torch.stack([
        transf(image=frame)['image'] for frame in img_stack_real
    ])  # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
    faces_gen_t = torch.stack([
        transf(image=frame)['image'] for frame in img_stack_gen
    ])  # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
    faces_fake_t = torch.stack([
        transf(image=frame)['image'] for frame in img_stack_fake
    ])  # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
    faces_fake_gen_t = torch.stack([
        transf(image=frame)['image'] for frame in img_stack_fake_gen
    ])  # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces

    name = "cacd_" + str(
        person_id)  # saved as id of original celebdf, not cacd
    vid_id = 0
    output_real_name = '../regenerated-feature-folder/features_real_' + name + '_v_' + str(
        vid_id) + '_recon_5-9_fpv' + str(frames_per_video) + '.pkl'

    output_fake_name = '../regenerated-feature-folder/features_swapped_' + name + '_v_' + str(
        vid_id) + '_recon_5-9_fpv' + str(frames_per_video) + '.pkl'

    with torch.no_grad():
        out1 = net.features(faces_real_t.to(device))
        out2 = net.features(faces_gen_t.to(device))
        out3 = net.features(faces_fake_t.to(device))
        out4 = net.features(faces_fake_gen_t.to(device))
        out1_unsupervised = model_unsupervised(faces_real_t.to(device))
        out2_unsupervised = model_unsupervised(faces_gen_t.to(device))
        out3_unsupervised = model_unsupervised(faces_fake_t.to(device))
        out4_unsupervised = model_unsupervised(faces_fake_gen_t.to(device))
        out1 = torch.concat((out1, torch.squeeze(out1_unsupervised)), 1)
        out2 = torch.concat((out2, torch.squeeze(out2_unsupervised)), 1)
        out3 = torch.concat((out3, torch.squeeze(out3_unsupervised)), 1)
        out4 = torch.concat((out4, torch.squeeze(out4_unsupervised)), 1)
    with open(output_real_name, 'wb') as f:
        pickle.dump([out1, out2], f)  #save results
    with open(output_fake_name, 'wb') as f:
        pickle.dump([out3, out4], f)  #save results
