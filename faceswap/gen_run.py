# generate recons faces using trained models

import tensorflow as tf
tf.device('/GPU:0')

# conf = tf.ConfigProto()
# conf.gpu_options.per_process_gpu_memory_fraction=0.49
# session = tf.Session(config=conf)

from keras.layers import *
import keras.backend as K
import os
import cv2
import glob
import time
import numpy as np
from pathlib import PurePath, Path
from IPython.display import clear_output
import random

from data_loader.data_loader_celebdfv2 import DataLoader
from utils import showG, showG_mask, showG_eyes
from tqdm import tqdm

import matplotlib.pyplot as plt
#%matplotlib inline

from converter.face_transformer import FaceTransformer
ftrans = FaceTransformer()


freeze_B = False

# Number of CPU cores
num_cpus = os.cpu_count()

# Input/Output resolution
RESOLUTION = 64 # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

# Batch size
batchSize = 8
assert (batchSize != 1 and batchSize % 2 == 0) , "batchSize should be an even number."

# Use motion blurs (data augmentation)
# set True if training data contains images extracted from videos
use_da_motion_blur = False

# Use eye-aware training
# require images generated from prep_binary_masks.ipynb
use_bm_eyes = False

# Probability of random color matching (data augmentation)
#prob_random_color_match = 0.5

da_config = {
    #"prob_random_color_match": prob_random_color_match,
    #"use_da_motion_blur": use_da_motion_blur,
    "use_bm_eyes": use_bm_eyes
}

tf.device('/GPU:0')

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard" # standard, lite

# Loss function weights configuration
loss_weights = {}
loss_weights['w_D'] = 0.1 # Discriminator
loss_weights['w_recon'] = 1. # L1 reconstruction loss
loss_weights['w_edge'] = 0.1 # edge loss
loss_weights['w_eyes'] = 30. # reconstruction and edge loss on eyes area
loss_weights['w_pl'] = (0.01, 0.1, 0.3, 0.1) # perceptual loss (0.003, 0.03, 0.3, 0.3)

# Init. loss config.
loss_config = {}
loss_config["gan_training"] = "mixup_LSGAN" # "mixup_LSGAN" or "relativistic_avg_LSGAN"
loss_config['use_PL'] = False
loss_config["PL_before_activ"] = False
loss_config['use_mask_hinge_loss'] = False
loss_config['m_mask'] = 0.
loss_config['lr_factor'] = 1.
loss_config['use_cyclic_loss'] = False

#from networks.faceswap_gan_model import FaceswapGANModel
from networks.gan_model import OnlyGANModel
from networks.faceswap_gan_model import FaceswapGANModel
#from tensorflow.keras import regularizers
from PIL import Image
#for idx in range(27,37):


def run_for_one_person_efficient(idx):
    # Path to training images

    # original
    #img_dirA = "../faces/celebdf/" + str(idx)

    # genned images
    ####img_dirA = "../faces/celebdf-gan-generated/" + str(idx)
    #img_dirA = "../faces/celebdf-gan-using5-9/" + str(idx)

    # faceswapped images
    # img_dirA = "../faces/celebdf-swapped/to_" + str(idx)
    img_dirA = "./faces/celebdf-swapped-using0-4/to_" + str(idx)  # fake data, someone has imitated idx

    video_filter = "*" #"id*" + "_000[0-4]*"

    # Path to saved model weights
    models_dir = "./models-celebdf/gens_5_videos/id_"+str(idx)  # model used training with 5-9
    replacement = "celebdf-gan-using5-9"

    # eyes are only for loss function, no usage in inference mode
    if not os.path.exists(img_dirA):
        #continue
        print("no images found")
        return
    print("working on id "+str(idx))

    if not os.path.exists(models_dir):
        return
    assert os.path.exists(models_dir)
    model = OnlyGANModel(**arch_config)

    model.load_weights(path=models_dir)


    # https://github.com/rcmalli/keras-vggface
    #!pip install keras_vggface --no-dependencies
    #from keras_vggface.vggface import VGGFace

    # VGGFace ResNet50
    #with tf.variable_scope(tf.get_variable_scope(),reuse=False):
        #vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

    #vggface.summary()

    #model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    #model.build_train_functions(loss_weights=loss_weights, **loss_config)
    
    model.trainable = False
    ftrans.set_model(model)


    # Get filenames
    #train_A = glob.glob(img_dirA+"/*.*")  # real or genned data
    train_A = glob.glob(img_dirA+"/*/*.*") # swapped data
    #train_A = glob.glob(img_dirA+"/"+video_filter)   # gnerate of real or genned data
    #train_A = glob.glob(img_dirA+"/*/"+video_filter)   # gnerate of swapped data
    #print(img_dirA+"/*/"+video_filter)

    assert len(train_A), "No image found in " + str(img_dirA)
    print ("Number of images in folder A: " + str(len(train_A)))
    
    # for inference bm eyes is not actually used , eyes are used in loss functions only while training
    # encoder input 3 channel, encoder output 6 channel , goes to deccoder
    #if use_bm_eyes:  # eyes are only for loss function, no usage in inference mode
        #assert len(glob.glob(img_dirA_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirA_bm_eyes)
        #assert len(glob.glob(img_dirA_bm_eyes+"/*.*")) == len(train_A), \
        #"Number of faceA images does not match number of their binary masks. Can be caused by any none image file in the folder. id: {}".format(idx)


    start_iter = 0
    # Start training
    t0 = time.time()
    with tf.device('/gpu:0'):
        train_batchA = DataLoader(train_A, 8, "",  # batchsize 1
                                                  RESOLUTION, num_cpus, K.get_session(), **da_config)

    current_img = 0
    total_img = len(train_A)
    
    #while current_img < total_img:
        
        #tA, _ = train_batchA.get_next_batch() # eyes mask not used in inference
        
        # DONT USE DATALOADER, DATALOADER changes image to (-1,1) range but facetransformer expects 0-255
        
    for i in range(len(train_A)):
        single_trainA = train_A[i]
        new_filename = single_trainA.replace("celebdfv2", replacement)

        #current_img = current_img + 1

        input_img = cv2.imread(single_trainA) # arbitary size

        #tt = model.path_A([[tA[0]]]) #doesn't work
        #tt = model.path_A([tA[0:1]]) #does work

        #tt = model.path_abgr_A([tA[0]])
        #tt = model.path_abgr_A([tA[0:1]])

        print(input_img.shape)

        """for single_trainA in train_A:   # chnage here
            #showG(tA, tA, model.path_A, model.path_A, 1, once=False, save=False)
            new_filename = single_trainA.replace("celebdfv2","celebdfv2-gan-generated")
            if os.path.exists(new_filename):
                continue
            if usedd == False:
                with tf.device('/gpu:0'):
                    train_batchA = DataLoader([single_trainA], [single_trainA], 1, img_dirA_bm_eyes,  # batchsize 1
                                              RESOLUTION, num_cpus, K.get_session(), **da_config)
            else:
                train_batchA = DataLoader([single_trainA], [single_trainA], 1, img_dirA_bm_eyes,  # batchsize 1
                                          RESOLUTION, num_cpus, K.get_session(), **da_config)
            usedd = True
            wA, tA, _ = train_batchA.get_next_batch()"""
        # face transformer expects RGB input and returns RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        result_img, result_rgb, result_mask = ftrans.transform(
                    input_img, 
                    direction="BtoA", 
                    roi_coverage=0.93,
                    color_correction="adain_xyz",
                    IMAGE_SHAPE=(RESOLUTION, RESOLUTION, 3)
                    ) # result as uint8 RGB
        print(result_img.shape)

        print(new_filename)
        # PIL also expects RGB

        Path(os.path.dirname(new_filename)).mkdir(parents=True, exist_ok=True)
        #dif_image = Image.fromarray(np.abs((result_img - input_img)*20))
        result_img = Image.fromarray(result_img)
        #result_rgb = Image.fromarray(result_rgb)
        #result_mask = Image.fromarray(result_mask)
        
        result_img.save(new_filename)
        #result_rgb.save(new_filename + "_res_rgb.jpg")
        #result_mask.save(new_filename +"_res_mask.jpg")
        #dif_image.save(new_filename +"_dif_img.jpg")
        #Im2.save(dif_filename)
    del train_batchA
    del model



def run_for_one_person_efficient_cacd(idx_cacd, idx_model):
    # images from cacd, model from celebdf
    # Path to training images

    # original
    img_dirA = "../faces/cacd_auth/" + str(idx_cacd)

    # faceswapped images
    #img_dirA = "../faces/cacd-swapped-2ndhalf/to_" + str(idx_cacd)

    # Path to saved model weights
    models_dir = "./models-celebdf/gens_5_videos/id_" + str(idx_model)  # model used training with 5-9
    replacement = "cacd-gan-using5-9"

    # eyes are only for loss function, no usage in inference mode
    if not os.path.exists(img_dirA):
        # continue
        return
    print("working on id " + str(idx_cacd))

    if not os.path.exists(models_dir):
        return

    assert os.path.exists(models_dir)
    model = OnlyGANModel(**arch_config)

    model.load_weights(path=models_dir)

    # https://github.com/rcmalli/keras-vggface
    # !pip install keras_vggface --no-dependencies
    # from keras_vggface.vggface import VGGFace

    # VGGFace ResNet50
    # with tf.variable_scope(tf.get_variable_scope(),reuse=False):
    # vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

    # vggface.summary()

    # model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    # model.build_train_functions(loss_weights=loss_weights, **loss_config)

    model.trainable = False
    ftrans.set_model(model)

    # Get filenames
    train_A = glob.glob(img_dirA + "/*.*")  # real or genned data
    #train_A = glob.glob(img_dirA + "/*/*.*")  # swapped data

    assert len(train_A), "No image found in " + str(img_dirA)
    print("Number of images in folder A: " + str(len(train_A)))

    # for inference bm eyes is not actually used , eyes are used in loss functions only while training
    # encoder input 3 channel, encoder output 6 channel , goes to deccoder
    # if use_bm_eyes:  # eyes are only for loss function, no usage in inference mode
    # assert len(glob.glob(img_dirA_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirA_bm_eyes)
    # assert len(glob.glob(img_dirA_bm_eyes+"/*.*")) == len(train_A), \
    # "Number of faceA images does not match number of their binary masks. Can be caused by any none image file in the folder. id: {}".format(idx)

    for i in range(len(train_A)):
        single_trainA = train_A[i]
        new_filename = single_trainA.replace("cacd", replacement)

        input_img = cv2.imread(single_trainA)  # arbitary size
        print(input_img.shape)
        # face transformer expects RGB input and returns RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        result_img, result_rgb, result_mask = ftrans.transform(
            input_img,
            direction="BtoA",
            roi_coverage=0.93,
            color_correction="adain_xyz",
            IMAGE_SHAPE=(RESOLUTION, RESOLUTION, 3)
        )  # result as uint8 RGB
        print(result_img.shape)

        print(new_filename)
        # PIL also expects RGB

        Path(os.path.dirname(new_filename)).mkdir(parents=True, exist_ok=True)
        result_img = Image.fromarray(result_img)


        result_img.save(new_filename)
    del model


def run_swap_cacd(idx_A, idx_B, BtoA):  # if BtoA Decoder of A will be used
    # Path to training images
    # img_dirA = "./face-celebdfv2/" + str(idx)
    if BtoA:  # use data of B
        img_dirA = "./faces/cacd_auth/" + str(idx_B)
        idx = idx_B
    else:  # use data of A
        img_dirA = "./faces/cacd_auth/" + str(idx_A)
        idx = idx_A
    #video_filter = "id" + str(idx) + "_000[0-4]*"
    replacement = "cacd-swapped-2ndhalf"  # 1sthalf was mistakenly written
    # eyes are only for loss function, no usage in inference mode
    if not os.path.exists(img_dirA):
        # continue
        return
    print("working on data of id " + str(idx) + " to convert id "+ (str(idx_A) if BtoA else str(idx_B)))

    # Path to saved model weights
    models_dir = "./models-cacd/AtoB_5_1st_half/id_" + str(idx_A) + "_" + str(idx_B)
    if not os.path.exists(models_dir):
        return
    assert os.path.exists(models_dir)
    model = FaceswapGANModel(**arch_config)

    model.load_weights(path=models_dir)

    # VGGFace ResNet50
    #with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        #vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

    # vggface.summary()

    #model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    #model.build_train_functions(loss_weights=loss_weights, **loss_config)

    model.trainable = False
    ftrans.set_model(model)

    # Get filenames
    #train_A = glob.glob(img_dirA + "/*.*"+video_filter)
    #train_A = glob.glob(img_dirA + "/" + video_filter)
    train_A = glob.glob(img_dirA + "/*.*")

    assert len(train_A), "No image found in " + str(img_dirA)
    print("Number of images in folder A: " + str(len(train_A)))

    # for inference bm eyes is not actually used , eyes are used in loss functions only while training
    # encoder input 3 channel, encoder output 6 channel , goes to deccoder
    # if use_bm_eyes:  # eyes are only for loss function, no usage in inference mode
    # assert len(glob.glob(img_dirA_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirA_bm_eyes)
    # assert len(glob.glob(img_dirA_bm_eyes+"/*.*")) == len(train_A), \
    # "Number of faceA images does not match number of their binary masks. Can be caused by any none image file in the folder. id: {}".format(idx)

    start_iter = 0
    # Start training
    #t0 = time.time()
    #with tf.device('/gpu:0'):
        #train_batchA = DataLoader(train_A, 8, "",  # batchsize 1
                                  #RESOLUTION, num_cpus, K.get_session(), **da_config)

    current_img = 0
    total_img = len(train_A)

    # while current_img < total_img:

    # tA, _ = train_batchA.get_next_batch() # eyes mask not used in inference

    # DONT USE DATALOADER, DATALOADER changes image to (-1,1) range but facetransformer expects 0-255

    for i in range(len(train_A)):
        single_trainA = train_A[i]
        new_filename = single_trainA.replace("cacd", replacement+"/to_"+(str(idx_A) if BtoA else str(idx_B)))

        # current_img = current_img + 1

        input_img = cv2.imread(single_trainA)  # arbitary size

        # tt = model.path_A([[tA[0]]]) #doesn't work
        # tt = model.path_A([tA[0:1]]) #does work

        # tt = model.path_abgr_A([tA[0]])
        # tt = model.path_abgr_A([tA[0:1]])
        print(new_filename)
        print(input_img.shape)

        """for single_trainA in train_A:   # chnage here
            #showG(tA, tA, model.path_A, model.path_A, 1, once=False, save=False)
            new_filename = single_trainA.replace("celebdfv2","celebdfv2-gan-generated")
            if os.path.exists(new_filename):
                continue
            if usedd == False:
                with tf.device('/gpu:0'):
                    train_batchA = DataLoader([single_trainA], [single_trainA], 1, img_dirA_bm_eyes,  # batchsize 1
                                              RESOLUTION, num_cpus, K.get_session(), **da_config)
            else:
                train_batchA = DataLoader([single_trainA], [single_trainA], 1, img_dirA_bm_eyes,  # batchsize 1
                                          RESOLUTION, num_cpus, K.get_session(), **da_config)
            usedd = True
            wA, tA, _ = train_batchA.get_next_batch()"""
        # face transformer expects RGB input and returns RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        result_img, result_rgb, result_mask = ftrans.transform(
            input_img,
            direction="BtoA" if BtoA else "AtoB",
            roi_coverage=0.93,
            color_correction="adain_xyz",
            IMAGE_SHAPE=(RESOLUTION, RESOLUTION, 3)
        )  # result as uint8 RGB
        print(result_img.shape)


        # PIL also expects RGB

        Path(os.path.dirname(new_filename)).mkdir(parents=True, exist_ok=True)
        # dif_image = Image.fromarray(np.abs((result_img - input_img)*20))
        result_img = Image.fromarray(result_img)
        # result_rgb = Image.fromarray(result_rgb)
        # result_mask = Image.fromarray(result_mask)

        result_img.save(new_filename)
        # result_rgb.save(new_filename + "_res_rgb.jpg")
        # result_mask.save(new_filename +"_res_mask.jpg")
        # dif_image.save(new_filename +"_dif_img.jpg")
        # Im2.save(dif_filename)
    #del train_batchA
    del model

#for i in range(28,62):
    #run_for_one_person(i)
    #print(i)
import argparse
import pandas as pd
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-i', metavar='N', type=int, nargs='+',
                        help='ss')
    #args = parser.parse_args()
    #print(args.i[0])
    #run_for_one_person_efficient(args.i[0])


    # run real/genned/swapped celebdf images on celebdf model
    """for i in range(34, 63, 1):
        if i == 42 or i == 44 or i == 19 or i == 43 or i == 45:
            continue
        run_for_one_person_efficient(i)"""



    # swap inside cacd
    """for i in range(1, 3): # 2-6 linked to 1,2,3,4,6 of faceswap gan

        if i in id_models:
            cacd_i = id_cacds[id_models.index(i)]
            run_swap_cacd(cacd_i, cacd_i + 30, True)"""

    # pass real/fake cacd data to celebdf model (fake is basically swapped data while testing)
    """for i in range(0,1500,1):

        if i in id_cacds:

            print(id_cacds.index(i))
            model_i = id_models[id_cacds.index(i)]
            if model_i==42 or model_i==44:
                continue
            print("Using model {}".format(model_i))
            run_for_one_person_efficient_cacd(i, model_i)"""

# prediction on a loop memory leak issue
# https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
# https://stackoverflow.com/questions/53687165/tensorflow-memory-leak-when-building-graph-in-a-loop
# solution: use tensorflow gurbage collect or tf.reset_default_graph()


#ValueError: Variable mlp-fc/weights does not exist, or was not created with tf.get_variable()
# https://stackoverflow.com/questions/45263666/tensorflow-variable-reuse
# https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
# Quick solve: don't run any upper codeblock. just run this code block
