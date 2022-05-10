from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

#Load TRADES CNN model for MNIST
from models.small_cnn import SmallCNN

device = torch.device("cuda")

#Maximum perturbation size for MNIST dataset must be smaller than 0.3
eps_adjust = 0.000001
epsilon = 0.3 - eps_adjust
dim = (28, 28)
step1 = 0.001
step2 = 0.01

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    # print('{0:.64f}'.format(epsilon*sign_data_grad[0][0][10][10]))
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def overlay_attack(image, epsilon, target, model, X_data, Y_data, X, y, step1=0.001, step2=0.01):

    image_prev = image
    image_adv = image

    #select a random image
    idx_overlay = np.random.randint(0, len(Y_data))
    while Y_data[idx_overlay] == y.data:
        idx_overlay = np.random.randint(0, len(Y_data))

    while abs((image_adv-image).min()) < epsilon and abs((image_adv-image).max()) < epsilon:

        X_overlay = np.array(np.expand_dims(X_data[idx_overlay], axis=0), dtype=np.float32)
        X_overlay = np.array(np.expand_dims(X_overlay, axis=0), dtype=np.float32)
        X_overlay = np.ones(np.shape(X_overlay), dtype=np.float32) - X_overlay
        # transform to torch.tensor
        X_overlay = torch.from_numpy(X_overlay).to(device)

        image_adv = (image_prev + step1 * X_overlay) / (1 + step1)

        image_adv_ = Variable(image_adv, requires_grad = True)

        # output of model
        out = model(image_adv_)

        #calculate the loss
        loss = F.nll_loss(out, target)

        #zero existing gradients
        model.zero_grad()

        #calculate gradients of model in backward pass
        loss.backward()

        #collect data grad
        data_grad = image_adv_.grad.data

        #call fgsm attack
        image_adv = fgsm_attack(image_adv, step2, data_grad)

        if abs((image_adv-image).min()) > epsilon or abs((image_adv-image).max()) > epsilon:
            return image_prev

        image_prev = image_adv
        step1 *= 2
        step2 *= 2

def attack(model, device, X_data, Y_data):

    #test accuracy
    correct = 0
    wrong = 0 #FIXME:for testing
    #adv_examples = np.empty(np.shape(X_data), dtype=np.float32)
    adv_examples = []

    for idx in range(len(Y_data)):

        # load original image
        image = np.array(np.expand_dims(X_data[idx], axis=0), dtype=np.float32)
        image = np.array(np.expand_dims(image, axis=0), dtype=np.float32)
        # load label
        label = np.array([Y_data[idx]], dtype=np.int64)

        # transform to torch.tensor
        data = torch.from_numpy(image).to(device)
        target = torch.from_numpy(label).to(device)

        X, y = Variable(data, requires_grad = True), Variable(target)

        # output of model
        out = model(X)
        init_pred = out.data.max(1)[1]

        #if the initial prediction is wrong, don't do anything about it
        if init_pred != y.data:

            #detach the tensor from GPU
            data_ = data.detach().cpu().numpy()
            data_ = np.reshape(data_, dim)
            data_ = list(data_)
            adv_examples.append(data_)

            continue

        #call fgsm attack
        perturbed_data = overlay_attack(data, epsilon, target, model, X_data, Y_data, X, y, step1, step2)

        #re-classify the perturbed image
        X_ = Variable(perturbed_data)
        out = model(X_)

        #check new prediction
        final_pred = out.data.max(1)[1]

        if final_pred == y.data:
            correct += 1
        else:
            wrong += 1
            print("wrong: {} / {}".format(wrong, idx+1))

        #detach the tensor from GPU
        perturbed_data_ = perturbed_data.detach().cpu().numpy()
        perturbed_data_ = np.reshape(perturbed_data_, dim)
        perturbed_data_ = list(perturbed_data_)
        #print('{0:.64f}'.format((perturbed_data_ - image)[0][0][10][10]))

        #np.append(adv_examples, perturbed_data_)
        adv_examples.append(perturbed_data_)

    #print out test accuracy
    final_acc = correct/float(len(Y_data))
    print("Test Accuracy: {} / {} = {}".format(correct, len(Y_data), final_acc))

    adv_examples = np.array(adv_examples)
    np.save('mnist_X_adv', adv_examples)

    return

def main():
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))

    #load data
    X_data = np.load('./data_attack/mnist_X.npy')
    Y_data = np.load('./data_attack/mnist_Y.npy')

    attack(model, device, X_data, Y_data)
    return

if __name__ == '__main__':
    main()
