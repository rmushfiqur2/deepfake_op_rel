# creates a dictionary of four elements => (image, reconstructed image, efficientnetAutoB4ST feature (image),
# efficientnetAutoB4ST feature (reconstructed image)
# written by Mushfiqur Rahman (with the help of chatGPT and Google)

import os
import glob
import numpy as np
from PIL import Image
import pickle
import torch
from architectures import weights, fornet
from torch.utils.model_zoo import load_url
from utils import get_transformer

# Specify the path to the folder you want to iterate over
#folder_path = '../data_deepfake/face-celebdfv2/*/*.jpg'
gen_folder_path = '../data_deepfake/face-cacd-gan-using1sthalf'
term_to_remove = '-gan-using1sthalf'
save_path = '../data_deepfake/compact-test-swapped/'

# Create a wildcard pattern to match all files in the folder
pattern = os.path.join(gen_folder_path, '*/*.jpg')
#pattern = '../data_deepfake/*'

# Use glob to retrieve a list of matching file paths
print(pattern)
file_list = glob.glob(pattern)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print("torch cuda not available")
face_policy = 'scale'
face_size = 224
frames_per_video = 64
net_model = 'EfficientNetAutoAttB4ST'
train_db = 'DFDC'
# #save features
model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
net = getattr(fornet,net_model)().eval().to(device)
# getattr(fornet,net_model) is equivalent to fornet.EfficientNetAutoAttB4ST here
net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))
transf = get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

#print(file_list)
# Iterate over the files in the folder
for file_path in file_list:
    # Check if it's a file (not a subfolder)
    file_raw = file_path.replace(term_to_remove, '')
    file_double_recons = file_path.replace(term_to_remove, term_to_remove+term_to_remove)
    print(file_raw)
    if os.path.isfile(file_path) and os.path.isfile(file_raw):
        # You can perform operations on the file here
        # For example, print the file name
        print("File:", os.path.basename(file_raw))
        image_raw = np.array(Image.open(file_raw))
        image_recons = np.array(Image.open(file_path))
        image_double_recons = np.array(Image.open(file_double_recons))

        print(image_raw.shape)
        faces_raw = torch.stack( [ transf(image=image_raw)['image']] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
        faces_gen = torch.stack( [ transf(image=image_recons)['image']] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces
        faces_double_gen = torch.stack( [ transf(image=image_double_recons)['image']] ) # torch.Size([301, 3, 224, 224]) trasf resizes and normalizes the faces

        with torch.no_grad():
            out1 = net.features(faces_raw.to(device))
            out2 = net.features(faces_gen.to(device))
            out3 = net.features(faces_double_gen.to(device))

        new_dic = {"raw_img_pil_rgb": image_raw, "recons_img_pil_rgb":image_recons, "double_recons_img_pil_rgb":image_double_recons, "raw_feature":out1, "recons_feature":out2, "double_recons_feature":out3}
        with open(save_path + os.path.basename(file_raw).replace('jpg','pickle'), 'wb') as file:
            pickle.dump(new_dic, file)