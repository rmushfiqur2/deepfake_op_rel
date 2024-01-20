import os
import shutil
#import tensorflow as tf

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
#from keras import backend as K
#from pathlib import PurePath, Path
#from moviepy.editor import VideoFileClip
#from umeyama import umeyama

#print(tf.global_variables())

#configuration = tf.ConfigProto()
#configuration.gpu_options.allow_growth = True
#session = tf.Session(config=configuration)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#sess = tf.Session(config=tf.ConfigProto(
  #allow_soft_placement=True, log_device_placement=True))
  

#import os
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#from keras import backend as K
#from pathlib import PurePath, Path
#from moviepy.editor import VideoFileClip
#from umeyama import umeyama
#import mtcnn_detect_face


def delete_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
import os
import shutil


import face_alignment
import cv2
import numpy as np
from glob import glob
from pathlib import PurePath, Path
from matplotlib import pyplot as plt


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,  flip_input=False)

# !mkdir -p binary_masks/faceA_eyes
Path(f"binary_masks/faceA_eyes").mkdir(parents=True, exist_ok=True)

imgs_processed = 0
imgs_processed_no_face = 0
ids_processed = 0

import pandas as pd
table = pd.read_csv('../siamese_detector/matching_table_ids.txt')
id_celebdf = table.id_celebdf.tolist()
id_cacds = table.id_cacd.tolist()

for idx in range(101,103):
    # train images
    dir_faceA = "../faces/celebdf/" + str(idx)
    dir_bm_faceA_eyes = '../faces/binary_masks-celebdf/'+ str(idx) +'_eyes'
    #dir_faceA = "./face-celebdfv2-gan-generated/" + str(idx)
    #dir_bm_faceA_eyes = 'binary_masks-celebdfv2-gan-generated/'+ str(idx) +'_eyes'
    fns_faceA = glob(f"{dir_faceA}/*.*")

    if not len(fns_faceA):
        continue

    # !mkdir -p binary_masks/faceA_eyes
    Path(dir_bm_faceA_eyes).mkdir(parents=True, exist_ok=True)
    delete_files(dir_bm_faceA_eyes)

    save_path = dir_bm_faceA_eyes

    
    # create binary mask for each training image
    for fn in fns_faceA:
        raw_fn = PurePath(fn).parts[-1]

        x = plt.imread(fn)
        x = cv2.resize(x, (256,256))
        preds = fa.get_landmarks(x)
        
        if preds is not None:
            preds = preds[0]
            mask = np.zeros_like(x)
            
            # Draw right eye binary mask
            pnts_right = [(preds[i,0],preds[i,1]) for i in range(36,42)]
            hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)

            # Draw left eye binary mask
            pnts_left = [(preds[i,0],preds[i,1]) for i in range(42,48)]
            hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)

            # Draw mouth binary mask
            #pnts_mouth = [(preds[i,0],preds[i,1]) for i in range(48,60)]
            #hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
            #mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)
            
            mask = cv2.dilate(mask, np.ones((13,13), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (7,7), 0)
            
        else:
            mask = np.zeros_like(x)
            print(f"No faces were detected in image '{fn}''")
            imgs_processed_no_face += 1
        
        plt.imsave(fname=f"{save_path}/{raw_fn}", arr=mask, format="jpg")
    imgs_processed += len(fns_faceA)
    
    
    # test images
    dir_faceA = "../faces/cacd_auth/" + str(idx)
    dir_bm_faceA_eyes = '../faces/binary_masks-cacd/'+ str(idx) +'_eyes'
    #dir_faceA = "./face-celebdfv2-gan-generated/" + str(idx)
    #dir_bm_faceA_eyes = 'binary_masks-celebdfv2-gan-generated/'+ str(idx) +'_eyes'
    fns_faceA = glob(f"{dir_faceA}/*.*")

    if not len(fns_faceA):
        continue

    # !mkdir -p binary_masks/faceA_eyes
    Path(dir_bm_faceA_eyes).mkdir(parents=True, exist_ok=True)
    delete_files(dir_bm_faceA_eyes)

    save_path = dir_bm_faceA_eyes

    
    # create binary mask for each training image
    for fn in fns_faceA:
        raw_fn = PurePath(fn).parts[-1]

        x = plt.imread(fn)
        x = cv2.resize(x, (256,256))
        preds = fa.get_landmarks(x)
        
        if preds is not None:
            preds = preds[0]
            mask = np.zeros_like(x)
            
            # Draw right eye binary mask
            pnts_right = [(preds[i,0],preds[i,1]) for i in range(36,42)]
            hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)

            # Draw left eye binary mask
            pnts_left = [(preds[i,0],preds[i,1]) for i in range(42,48)]
            hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)

            # Draw mouth binary mask
            #pnts_mouth = [(preds[i,0],preds[i,1]) for i in range(48,60)]
            #hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
            #mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)
            
            mask = cv2.dilate(mask, np.ones((13,13), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (7,7), 0)
            
        else:
            mask = np.zeros_like(x)
            print(f"No faces were detected in image '{fn}''")
            imgs_processed_no_face += 1
        
        plt.imsave(fname=f"{save_path}/{raw_fn}", arr=mask, format="jpg")
    imgs_processed += len(fns_faceA)
    
    ids_processed += 1

print("Nuber of processed images: "+ str(imgs_processed))
print("Number of image(s) with no face detected: " + str(imgs_processed_no_face))
print("Number of ids: " + str(ids_processed))
