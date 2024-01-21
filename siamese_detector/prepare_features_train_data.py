from my_functions import *
from PIL import Image
import os
import shutil

net_model = 'EfficientNetAutoAttB4ST'
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')
if not torch.cuda.is_available():
    print("torch cuda not available")
face_policy = 'scale'
face_size = 224
frames_per_video = 64
videoreader = VideoReader(verbose=False)

import gc

torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda:0')

model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
net = getattr(fornet, net_model)().eval().to(device)
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

# ==================================================================================================
# 5-9 videos used in the reconstruction
# (real - recons real), (recons real, recons recons real)

img_folder_real = "../faces/celebdf/"
img_folder_gen = "../faces/celebdf_recons/"
img_folder_gen_gen = "../faces/celebdf_double_recons/"

from glob import glob

num_done_persons = 0
for person_id in range(0, 62):  #range(0, 8):
    print("person: {}".format(person_id))

    for vid_id in range(5):  # range(5):
        print("vid: {}".format(vid_id))
        #print(img_folder_real + str(person_id) + "/id{}_000{}*".format(person_id, vid_id))
        img_files = glob(img_folder_real + str(person_id) +
                         "/id{}_{:04d}*".format(person_id, vid_id))
        img_files_gen = glob(img_folder_gen + str(person_id) +
                             "/id{}_{:04d}*".format(person_id, vid_id))
        img_files_gen_gen = glob(img_folder_gen_gen + str(person_id) +
                                 "/id{}_{:04d}*".format(person_id, vid_id))

        if len(img_files) * len(img_files_gen) * len(img_files_gen_gen) == 0:
            continue

        num_done_persons += 1
        selected_img = np.linspace(0, len(img_files) - 1, num=frames_per_video)
        img_stack_real = [
            cv2.cvtColor(cv2.imread(img_files[int(i)]), cv2.COLOR_BGR2RGB)
            for i in selected_img
        ]
        img_stack_gen = [
            cv2.cvtColor(cv2.imread(img_files_gen[int(i)]), cv2.COLOR_BGR2RGB)
            for i in selected_img
        ]
        img_stack_gen_gen = [
            cv2.cvtColor(cv2.imread(img_files_gen_gen[int(i)]),
                         cv2.COLOR_BGR2RGB) for i in selected_img
        ]

        faces_real_t = torch.stack(
            [transf(image=frame)['image'] for frame in img_stack_real]
        )  # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
        faces_gen_t = torch.stack(
            [transf(image=frame)['image'] for frame in img_stack_gen]
        )  # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
        faces_gen_gen_t = torch.stack(
            [transf(image=frame)['image'] for frame in img_stack_gen_gen]
        )  # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces

        name = "celebdfv2_" + str(person_id)
        output_real_name = '../regenerated-feature-folder/features_real_' + name + '_v_' + str(
            vid_id) + '_recon_5-9_fpv' + str(frames_per_video) + '.pkl'

        output_fake_name = '../regenerated-feature-folder/features_fake_' + name + '_v_' + str(
            vid_id) + '_recon_5-9_fpv' + str(frames_per_video) + '.pkl'

        with torch.no_grad():
            out1 = net.features(faces_real_t.to(device))
            out2 = net.features(faces_gen_t.to(device))
            out3 = net.features(faces_gen_gen_t.to(device))
            out1_unsupervised = model_unsupervised(faces_real_t.to(device))
            out2_unsupervised = model_unsupervised(faces_gen_t.to(device))
            out3_unsupervised = model_unsupervised(faces_gen_gen_t.to(device))
            out1 = torch.concat((out1, torch.squeeze(out1_unsupervised)), 1)
            out2 = torch.concat((out2, torch.squeeze(out2_unsupervised)), 1)
            out3 = torch.concat((out3, torch.squeeze(out3_unsupervised)), 1)
        with open(output_real_name, 'wb') as f:
            pickle.dump([out1, out2], f)  #save results
        with open(output_fake_name, 'wb') as f:
            pickle.dump([out2, out3], f)  #save results
        print(person_id)
        print(num_done_persons)
        print(output_real_name)
        print(output_fake_name)
