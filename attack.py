from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

#Load TRADES CNN model for MNIST
from models.small_cnn import SmallCNN

device = torch.device("cuda")

#Maximum perturbation size for MNIST dataset must be smaller than 0.3
epsilon = 0.3

def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon*sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	# Return the perturbed image
	return perturbed_image

def attack(model, device, X_data, Y_data):

	with torch.no_grad():
		for idx in range(len(Y_data)):
			# load original image
            image = np.array(np.expand_dims(X_data[idx], axis=0), dtype=np.float32)
            image = np.array(np.expand_dims(image, axis=0), dtype=np.float32)
			# load label
            label = np.array(Y_data[idx], dtype=np.int64)

			# transform to torch.tensor
            data = torch.from_numpy(image).to(device)
            target = torch.from_numpy(label).to(device)

			X, y = Variable(data, requires_grad=True), Variable(target)

	return

def main():
	model = SmallCNN().to(device)
	model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))

	#load data
	X_data = np.load('./data_attack/mnist_X.npy')
	Y_data = np.load('./data_attack/mnist_Y.npy')

if __name__ == '__main__':
    main()
