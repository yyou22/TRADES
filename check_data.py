from __future__ import print_function
import torch
from torch.autograd import Variable
from models.small_cnn import *
import numpy as np

def image_check(min_delta, max_delta, min_image_adv, max_image_adv):
    valid = 1.0
    if min_delta < - args.epsilon:
        print("invalid #1")
        valid -= 2.0
    elif max_delta > args.epsilon:
        print("invalid #2")
        valid -= 2.0
    elif min_image_adv < 0.0:
        print("invalid #3")
        valid -= 2.0
    elif max_image_adv > 1.0:
        print("invalid #4")
        valid -= 2.0

    if valid > 0.0:
        return True
    else:
        return False

def check_data(X_adv_data, X_data, Y_data):
    for idx in range(len(Y_data)):
        # load original image
        image = np.array(np.expand_dims(X_data[idx], axis=0), dtype=np.float32)
        image = np.array(np.expand_dims(image, axis=0), dtype=np.float32)
        # load adversarial image
        image_adv = np.array(np.expand_dims(X_adv_data[idx], axis=0), dtype=np.float32)
        image_adv = np.array(np.expand_dims(image_adv, axis=0), dtype=np.float32)
        # load label
        label = np.array(Y_data[idx], dtype=np.int64)

        # check bound
        image_delta = image_adv - image
        min_delta, max_delta = image_delta.min(), image_delta.max()
        min_image_adv, max_image_adv = image_adv.min(), image_adv.max()
        valid = image_check(min_delta, max_delta, min_image_adv, max_image_adv)
        if not valid:
            print('not valid adversarial image')
            break

def main():
    #load data
    X_adv_data = np.load('./data_attack/mnist_X_adv.npy')
    print("X_adv_data shape: ", X_adv_data.shape)
    X_data = np.load('./data_attack/mnist_X.npy')
    print("X_data shape: ", X_data.shape)
    Y_data = np.load('./data_attack/mnist_Y.npy')
