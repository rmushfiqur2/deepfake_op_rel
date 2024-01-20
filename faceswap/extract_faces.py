import os
import shutil
import tensorflow as tf

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from pathlib import PurePath, Path
#from moviepy.editor import VideoFileClip
from umeyama import umeyama

print(tf.global_variables())

#configuration = tf.ConfigProto()
#configuration.gpu_options.allow_growth = True
#session = tf.Session(config=configuration)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))
  

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from pathlib import PurePath, Path
from moviepy.editor import VideoFileClip
from umeyama import umeyama
import mtcnn_detect_face

def create_mtcnn(sess, model_path):
    if not model_path:
        model_path,_ = os.path.split(os.path.realpath(__file__))

    with tf.variable_scope('pnet2'):
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = mtcnn_detect_face.PNet({'data':data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet2'):
        data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        rnet = mtcnn_detect_face.RNet({'data':data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet2'):
        data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        onet = mtcnn_detect_face.ONet({'data':data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    return pnet, rnet, onet

WEIGHTS_PATH = "./mtcnn_weights/"

sess = K.get_session()
with sess.as_default():
    global pnet, rnet, onet 
    pnet, rnet, onet = create_mtcnn(sess, WEIGHTS_PATH)

# global pnet, rnet, onet
    
pnet = K.function([pnet.layers['data']],[pnet.layers['conv4-2'], pnet.layers['prob1']])
rnet = K.function([rnet.layers['data']],[rnet.layers['conv5-2'], rnet.layers['prob1']])
onet = K.function([onet.layers['data']],[onet.layers['conv6-2'], onet.layers['conv6-3'], onet.layers['prob1']])


Path(f"faces/aligned_faces").mkdir(parents=True, exist_ok=True)
Path(f"faces/raw_faces").mkdir(parents=True, exist_ok=True)
Path(f"faces/binary_masks_eyes").mkdir(parents=True, exist_ok=True)

def get_src_landmarks(x0, x1, y0, y1, pnts):
    """
    x0, x1, y0, y1: (smoothed) bbox coord.
    pnts: landmarks predicted by MTCNN
    """    
    src_landmarks = [(int(pnts[i+5][0]-x0), 
                      int(pnts[i][0]-y0)) for i in range(5)]
    return src_landmarks

def get_tar_landmarks(img):
    """    
    img: detected face image
    """         
    ratio_landmarks = [
        (0.31339227236234224, 0.3259269274198092),
        (0.31075140146108776, 0.7228453709528997),
        (0.5523683107816256, 0.5187296867370605),
        (0.7752419985257663, 0.37262483743520886),
        (0.7759613623985877, 0.6772957581740159)
        ]   
        
    img_size = img.shape
    tar_landmarks = [(int(xy[0]*img_size[0]), 
                      int(xy[1]*img_size[1])) for xy in ratio_landmarks]
    return tar_landmarks

def landmarks_match_mtcnn(src_im, src_landmarks, tar_landmarks): 
    """
    umeyama(src, dst, estimate_scale)
    landmarks coord. for umeyama should be (width, height) or (y, x)
    """
    src_size = src_im.shape
    src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
    tar_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
    M = umeyama(np.array(src_tmp), np.array(tar_tmp), True)[0:2]
    result = cv2.warpAffine(src_im, M, (src_size[1], src_size[0]), borderMode=cv2.BORDER_REPLICATE) 
    return result

def process_mtcnn_bbox(bboxes, im_shape):
    """
    output bbox coordinate of MTCNN is (y0, x0, y1, x1)
    Here we process the bbox coord. to a square bbox with ordering (x0, y1, x1, y0)
    """
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i,0:4]
        w, h = int(y1 - y0), int(x1 - x0)
        length = (w + h)/2
        center = (int((x1+x0)/2),int((y1+y0)/2))
        new_x0 = np.max([0, (center[0]-length//2)])#.astype(np.int32)
        new_x1 = np.min([im_shape[0], (center[0]+length//2)])#.astype(np.int32)
        new_y0 = np.max([0, (center[1]-length//2)])#.astype(np.int32)
        new_y1 = np.min([im_shape[1], (center[1]+length//2)])#.astype(np.int32)
        bboxes[i,0:4] = new_x0, new_y1, new_x1, new_y0
    return bboxes

def process_video(input_img):
    global frames, save_interval
    global pnet, rnet, onet
    minsize = 30 # minimum size of face
    detec_threshold = 0.7
    threshold = [0.6, 0.7, detec_threshold]  # three steps's threshold
    factor = 0.709 # scale factor   
    
    frames += 1
    if frames % save_interval == 0:
        faces, pnts = mtcnn_detect_face.detect_face(
            input_img, minsize, pnet, rnet, onet, threshold, factor)
        faces = process_mtcnn_bbox(faces, input_img.shape)
        
        for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
            det_face_im = input_img[int(x0):int(x1),int(y0):int(y1),:]

            # get src/tar landmarks
            src_landmarks = get_src_landmarks(x0, x1, y0, y1, pnts)
            tar_landmarks = get_tar_landmarks(det_face_im)

            # align detected face
            aligned_det_face_im = landmarks_match_mtcnn(
                det_face_im, src_landmarks, tar_landmarks)

            fname = f"./faces/aligned_faces/frame{frames}face{str(idx)}.jpg"
            plt.imsave(fname, aligned_det_face_im, format="jpg")
            fname = f"./faces/raw_faces/frame{frames}face{str(idx)}.jpg"
            plt.imsave(fname, det_face_im, format="jpg")
            
            bm = np.zeros_like(aligned_det_face_im)
            h, w = bm.shape[:2]
            bm[int(src_landmarks[0][0]-h/15):int(src_landmarks[0][0]+h/15),
               int(src_landmarks[0][1]-w/8):int(src_landmarks[0][1]+w/8),:] = 255
            bm[int(src_landmarks[1][0]-h/15):int(src_landmarks[1][0]+h/15),
               int(src_landmarks[1][1]-w/8):int(src_landmarks[1][1]+w/8),:] = 255
            bm = landmarks_match_mtcnn(bm, src_landmarks, tar_landmarks)
            fname = f"./faces/binary_masks_eyes/frame{frames}face{str(idx)}.jpg"
            plt.imsave(fname, bm, format="jpg")
            break # only one face in the image
        
    return np.zeros((3,3,3))

def process_image(input_img, input_img_name):
    global pnet, rnet, onet
    minsize = 30 # minimum size of face
    detec_threshold = 0.7
    threshold = [0.6, 0.7, detec_threshold]  # three steps's threshold
    factor = 0.709 # scale factor   
    

    faces, pnts = mtcnn_detect_face.detect_face(
        input_img, minsize, pnet, rnet, onet, threshold, factor)
    faces = process_mtcnn_bbox(faces, input_img.shape)

    for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
        det_face_im = input_img[int(x0):int(x1),int(y0):int(y1),:]

        # get src/tar landmarks
        src_landmarks = get_src_landmarks(x0, x1, y0, y1, pnts)
        tar_landmarks = get_tar_landmarks(det_face_im)

        # align detected face
        aligned_det_face_im = landmarks_match_mtcnn(
            det_face_im, src_landmarks, tar_landmarks)

        fname = f"./faces/aligned_faces/{input_img_name}.jpg"
        plt.imsave(fname, aligned_det_face_im, format="jpg")
        fname = f"./faces/raw_faces/{input_img_name}.jpg"
        plt.imsave(fname, det_face_im, format="jpg")

        bm = np.zeros_like(aligned_det_face_im)
        h, w = bm.shape[:2]
        bm[int(src_landmarks[0][0]-h/15):int(src_landmarks[0][0]+h/15),
           int(src_landmarks[0][1]-w/8):int(src_landmarks[0][1]+w/8),:] = 255
        bm[int(src_landmarks[1][0]-h/15):int(src_landmarks[1][0]+h/15),
           int(src_landmarks[1][1]-w/8):int(src_landmarks[1][1]+w/8),:] = 255
        bm = landmarks_match_mtcnn(bm, src_landmarks, tar_landmarks)
        fname = f"./faces/binary_masks_eyes/{input_img_name}.jpg"
        plt.imsave(fname, bm, format="jpg")
        break # only one face in the image
        
    return len(faces)


def delete_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
            
path_face1 = './faces/aligned_faces'
path_face2 = './faces/raw_faces'
path_face3 = './faces/binary_masks_eyes'

delete_files(path_face1)
delete_files(path_face2)
delete_files(path_face3)


# CACD MTCNN -> face detection plus alignment
# AS NOW WE HAVE FACE IMAGES NOT VIDEOS, CODE NEEDS TO BE CHANGED

from moviepy.editor import *
import shutil
import os
import glob
tf.get_variable_scope().reuse_variables() #https://stackoverflow.com/questions/46056206/tensorflow-value-error-variable-already-exists-disallowed

path_face1 = './faces/aligned_faces'
path_face2 = './faces/raw_faces'
path_face3 = './faces/binary_masks_eyes'

delete_files(path_face1)
delete_files(path_face2)
delete_files(path_face3)

import pandas as pd
table = pd.read_csv('../siamese_detector/matching_table_ids.txt')
id_celebdf = table.id_celebdf.tolist()
id_cacds = table.id_cacd.tolist()

print()

for idA in range(101,103):
    print(f"id: {str(idA)}")
    
    # train images (goes to celebdf folder)
    images_with_no_face = 0
    vid_path = './new_ids/'+str(idA)+'/train'
    listA = glob.glob(vid_path + '/*')
    if not len(listA):
        print('id '+ str(idA) + ' does not exist')
        continue

    path_faceA = '../faces/celebdf/'+ str(idA) +'/'
    Path(path_faceA).mkdir(parents=True, exist_ok=True)
    delete_files(path_faceA)

    for vidA in listA: # list of image files
        filename = os.path.split(vidA)[-1].split('.')[0] # image file name
        img = cv2.imread(vidA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (96, 96))
        #print(img.shape)
        
        num_faces = process_image(img, filename)
        #print(num_faces)
        if num_faces<1:
            images_with_no_face += 1
    print(f"no face detected: {str(images_with_no_face)}")

    file_names = os.listdir(path_face1)
    for file_name in file_names:
        shutil.move(os.path.join(path_face1, file_name), path_faceA)
    delete_files(path_face2)
    delete_files(path_face3)
    
    
    # test images (goes to cacd folder)
    images_with_no_face = 0
    vid_path = './new_ids/'+str(idA) + '/test'
    listA = glob.glob(vid_path + '/*')
    if not len(listA):
        print('id '+ str(idA) + ' does not exist')
        continue

    path_faceA = '../faces/cacd_auth/'+ str(idA) +'/'
    Path(path_faceA).mkdir(parents=True, exist_ok=True)
    delete_files(path_faceA)

    for vidA in listA: # list of image files
        filename = os.path.split(vidA)[-1].split('.')[0] # image file name
        img = cv2.imread(vidA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (96, 96))
        #print(img.shape)
        
        num_faces = process_image(img, filename)
        #print(num_faces)
        if num_faces<1:
            images_with_no_face += 1
    print(f"no face detected: {str(images_with_no_face)}")

    file_names = os.listdir(path_face1)
    for file_name in file_names:
        shutil.move(os.path.join(path_face1, file_name), path_faceA)
    delete_files(path_face2)
    delete_files(path_face3)
# doesn't exist: 14, 15, 18 within (0-61)


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
table = pd.read_csv('matching_table_ids.txt')
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
