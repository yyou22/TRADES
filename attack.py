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
eps_adjust = 0.00001
epsilon = 0.3 - eps_adjust
dim = (28, 28)
w = 0.05
step = epsilon #0.005
num_step = 20
torch.manual_seed(1)

def margin_loss(logits,y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def fgsm_attack(image, image_adv, epsilon, step, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad)

    # Create the perturbed image by adjusting each pixel of the input image
    # print('{0:.64f}'.format(epsilon*sign_data_grad[0][0][10][10]))
    perturbed_image = image_adv - step*sign_data_grad #switched to minus for target attack

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.min(torch.max(perturbed_image, image - epsilon), image + epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)

    # Return the perturbed image
    return perturbed_image

def overlay_attack(image, epsilon, target, model, X, y, step=step, w=w):

    og_out = model(image)

    #generate random noise
    random_noise = torch.FloatTensor(*image.shape).uniform_(-epsilon, epsilon).to(device)
    image_adv = image + random_noise

    # output of model
    out = model(image_adv)

    #generate random noise
    random_noise = torch.FloatTensor(*image.shape).uniform_(-epsilon, epsilon).to(device)
    image_adv = image + random_noise

    #adv_target = torch.tensor([out.data.topk(2)[1][0][1]]).to(device)

    for i in range(num_step):

        image_adv = image_adv + w * torch.randn(image.shape).cuda()
        image_adv = torch.min(torch.max(image_adv, image - epsilon), image + epsilon)
        image_adv = torch.clamp(image_adv, 0.0, 1.0)
        out = model(image_adv)

        cur_pred = out.data.max(1)
        if cur_pred[1] != y.data:
            return image_adv

        adv_target = torch.tensor([out.data.topk(2)[1][0][1]]).to(device)

        image_adv_ = Variable(image_adv, requires_grad = True)

        # output of model
        out = model(image_adv_)

        loss = margin_loss(out, adv_target)

        #zero existing gradients
        model.zero_grad()

        #calculate gradients of model in backward pass
        loss.backward()

        #collect data grad
        data_grad = image_adv_.grad.data

        image_adv = fgsm_attack(image, image_adv, epsilon, step, data_grad)

        out = model(image_adv)
        cur_pred = out.data.max(1)
        if cur_pred[1] != y.data:
            return image_adv

    return image_adv

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

        #call overlay attack
        perturbed_data = overlay_attack(data, epsilon, target, model, X, y)
        #print(perturbed_data.min())
        #print(perturbed_data - data)

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
    np.save('./data_attack/mnist_X_adv', adv_examples)

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
