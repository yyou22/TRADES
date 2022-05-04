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
model = SmallCNN().to(device)
model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))

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


