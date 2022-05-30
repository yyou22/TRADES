from __future__ import print_function
import torch
from torch.autograd import Variable
from models.small_cnn import *
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='check data')
parser.add_argument('--data', default='mnist',help='mnist/cifar')

args = parser.parse_args()

def image_check(min_delta, max_delta, min_image_adv, max_image_adv):
    valid = 1.0
    #print(min_delta)
    #print('{0:.32f}'.format(min_delta))

    if args.data == 'mnist':
        if min_delta < -0.3:
            print("invalid #1")
            valid -= 2.0
        elif max_delta > 0.3:
            print("invalid #2")
            valid -= 2.0
    else:
        if min_delta < -0.031:
            print("invalid #1")
            valid -= 2.0
        elif max_delta > 0.031:
            print("invalid #2")
            valid -= 2.0

    if min_image_adv < 0.0:
        print("invalid #3")
        print(min_image_adv)
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
    #for idx in range(20):
        # load original image
        image = np.array(np.expand_dims(X_data[idx], axis=0), dtype=np.float32)
        if args.data == 'mnist':
            image = np.array(np.expand_dims(image, axis=0), dtype=np.float32)
        else:
            image = np.transpose(image, (0, 3, 1, 2))

        # load adversarial image
        image_adv = np.array(np.expand_dims(X_adv_data[idx], axis=0), dtype=np.float32)
        if args.data == 'mnist':
            image_adv = np.array(np.expand_dims(image_adv, axis=0), dtype=np.float32)
        else:
            image_adv = np.transpose(image_adv, (0, 3, 1, 2))

        # load label
        label = np.array(Y_data[idx], dtype=np.int64)

        # check bound
        image_delta = image_adv - image

        #print(image_delta)

        min_delta, max_delta = image_delta.min(), image_delta.max()
        min_image_adv, max_image_adv = image_adv.min(), image_adv.max()
        valid = image_check(min_delta, max_delta, min_image_adv, max_image_adv)
        if not valid:
            print('not valid adversarial image')
            break
    print('check finished')

def main():

    if args.data == 'mnist':
        #load data
        X_adv_data = np.load('./data_attack/mnist_X_adv.npy')
        #print("X_adv_data shape: ", X_adv_data.shape)
        X_data = np.load('./data_attack/mnist_X.npy')
        #print("X_data shape: ", X_data.shape)
        Y_data = np.load('./data_attack/mnist_Y.npy')
    else:
        #load data
        X_adv_data = np.load('./cifar_X_adv_checkpoint.npy')
        #print("X_adv_data shape: ", X_adv_data.shape)
        X_data = np.load('./data_attack/cifar10_X.npy')
        #print("X_data shape: ", X_data.shape)
        Y_data = np.load('./data_attack/cifar10_Y.npy')

    check_data(X_adv_data, X_data, Y_data)

if __name__ == '__main__':
    main()
