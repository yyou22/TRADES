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

    #test accuracy
    correct = 0
    adv_examples = np.empty()

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
            # output of model
            out = model(X)
            init_pred = out.data.max(1)[1]

            #if the initial prediction is wrong, don't do anything about it
            if out.data.max(1)[1] != y.data:
            continue

            #calculate the loss
            loss = F.nll_loss(out, target)

            #zero existing gradients
            model.zero_grad()

            #calculate gradients of model in backward pass
            loss.backward()

            #collect data grad
            data_grad = data.grad.data

            #call fgsm attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            #re-classify the perturbed image
            X_ = Variable(perturbed_data)
            out = model(X_)

            #check new prediction
            final_pred = out.data.max(1)[1]

            if out.data.max(1)[1] == y.data:
            correct += 1

            np.append(adv_examples, perturbed_data)

    #print out test accuracy
    final_acc = correct/float(len(Y_data))
    print("Test Accuracy: {} / {} = {}".format(correct, len(Y_data), final_acc))

    np.save('mnist_X_adv', adv_examples)

    return

def main():
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))

    #load data
    X_data = np.load('./data_attack/mnist_X.npy')
    Y_data = np.load('./data_attack/mnist_Y.npy')

    test(model, device, X_data, Y_data)
    return

if __name__ == '__main__':
    main()
