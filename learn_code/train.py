from __future__ import print_function, division

import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor, Resize, Compose

from learn_code.model import CNN, AngleCNN, AngleCNN1
from learn_code.prepare_data import MyDataset, ImageFolderSplitter, DatasetFromFilename, show_batch

use_gpu = torch.cuda.is_available()


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    since = time.time()

    data_dir = os.getcwd() + "\\video_hub\\train_face_finish"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['postive', 'val']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=4,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['postive', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['postive', 'val']}

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['postive', 'val']:
            if phase == 'postive':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'postive':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def creat_net():
    # get model and replace the original fc layer with your fc layer
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=25)

    torch.save(model_ft, 'net.pkl')


def train_rotation_split():
    EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
    LR = 0.001  # learning rate

    photo_hub_pwd = os.getcwd() + '\\video_hub\\train_face_finish\\postive'
    splitter = ImageFolderSplitter(photo_hub_pwd)
    transforms = Compose([Resize((100, 100)), ToTensor()])

    x_train, y_train = splitter.getTrainingDataset()
    training_dataset = DatasetFromFilename(x_train, y_train, transforms=transforms)
    training_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)

    x_valid, y_valid = splitter.getValidationDataset()
    validation_dataset = DatasetFromFilename(x_valid, y_valid, transforms=transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True)

    cnn = CNN()
    print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    try:
        from sklearn.manifold import TSNE
        HAS_SK = True
    except:
        HAS_SK = False
        print('Please install sklearn for layer visualization')

    plt.ion()
    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(
                training_dataloader):  # gives batch data, normalize x when iterate train_loader

            output = cnn(b_x)[0]  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                for test_x, test_y in validation_dataloader:
                    test_output, last_layer = cnn(test_x)
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(),
                          '| test accuracy: %.2f' % accuracy)

    plt.ioff()

    # print 10 predictions from test data
    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')


def train_rotation():
    file_name = os.getcwd() + "\\conf_hub\\face_recognize.txt"
    train_data = MyDataset(txt=file_name, transform=transforms.ToTensor())
    train_data_loader = DataLoader(train_data, batch_size=100, shuffle=True, )

    test_x, test_y = train_data.get_val_Date()
    test_x = torch.Tensor(test_x)

    cnn = CNN()
    print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    count_tmp = 0
    for epoch in range(100):
        for step, (b_x, b_y) in enumerate(train_data_loader):  # gives batch data, normalize x when iterate train_loader
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
        count_tmp = count_tmp + 1
        test_output, last_layer = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        print('Epoch: ', count_tmp, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


def train_face_gpu():
    file_name = os.getcwd() + "\\conf_hub\\face_recognize.txt"
    train_data = MyDataset(txt=file_name, transform=transforms.ToTensor())
    train_data_loader = DataLoader(train_data, batch_size=100, shuffle=True, )

    test_x_show, test_y, test_img_path = train_data.get_val_Date()
    test_y = test_y.cuda().type(torch.cuda.LongTensor)
    test_x = test_x_show.cuda()

    cnn = CNN()
    cnn = cnn.cuda()
    print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    count_tmp = 0
    for epoch in range(100):
        for step, (x, y) in enumerate(train_data_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = x.cuda()
            b_y = y.cuda()
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
        count_tmp = count_tmp + 1

        test_output, last_layer = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1]  # move the computation in GPU
        accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
        if accuracy > 0.98:
            break
    torch.save(cnn, 'net_face.pkl')  # save entire net
    torch.save(cnn.state_dict(), 'net_face_params.pkl')  # 保存参数
    test_output, last_layer = cnn(test_x[:20])
    pred_y = torch.max(test_output, 1)[1]
    print(pred_y, 'prediction number')
    print(test_y[:20], 'real number')
    show_batch(test_x_show[:20])
    plt.show()


def train_angle_gpu():
    file_name = os.getcwd() + "\\conf_hub\\face_angle.txt"
    train_data = MyDataset(txt=file_name, transform=transforms.ToTensor())
    train_data_loader = DataLoader(train_data, batch_size=100, shuffle=True, )

    test_x_show, test_y, test_img_path = train_data.get_val_Date()
    test_y = test_y.cuda().type(torch.cuda.LongTensor)
    test_x = test_x_show.cuda()

    cnn = AngleCNN1()
    cnn = cnn.cuda()
    print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    count_tmp = 0
    for epoch in range(100):
        for step, (x, y) in enumerate(train_data_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = x.cuda()
            b_y = y.cuda()
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
        count_tmp = count_tmp + 1

        test_output, last_layer = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1]  # move the computation in GPU
        accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.5f' % accuracy)
        if accuracy > 0.98:
            break
    torch.save(cnn, 'net_angle.pkl')  # 保存模型
    torch.save(cnn.state_dict(), 'net_angle_params.pkl')  # 保存参数
    test_output, last_layer = cnn(test_x[:20])
    pred_y = torch.max(test_output, 1)[1]
    print(pred_y, 'prediction number')
    print(test_y[:20] * 45, 'real number')
    show_batch(test_x_show[:20])
    plt.show()


def test_img_para():
    file_name = os.getcwd() + "\\conf_hub\\face_recognize.txt"
    train_data = MyDataset(txt=file_name, transform=transforms.ToTensor())

    test_x_show, test_y, test_img_path = train_data.get_val_Date()
    test_y = test_y.cuda().type(torch.cuda.LongTensor)
    test_x = test_x_show.cuda()
    net_angle = AngleCNN()
    net_angle.cuda()
    net_angle.load_state_dict(torch.load('net_angle_params.pkl'))
    net_angle.cuda()
    print(net_angle)
    net_face = CNN()
    net_face.cuda()
    net_face.load_state_dict(torch.load('net_face_params.pkl'))
    net_face.cuda()
    print(net_face)
    test_output, last_layer = net_face(test_x[:100])
    pred_y = torch.max(test_output, 1)[1]
    print(pred_y, 'prediction number')
    print(test_y[:100] * 45, 'real number')
    show_batch(test_x_show[:100])
    plt.show()
    test_output, last_layer = net_angle(test_x[:100])
    pred_y = torch.max(test_output, 1)[1]
    print(pred_y, 'prediction number')
    print(test_y[:100] * 45, 'real number')
    show_batch(test_x_show[:100])
    plt.show()


def train_all():
    # train_face_gpu()
    train_angle_gpu()
    # test_img_para()


if __name__ == '__main__':
    train_all()
