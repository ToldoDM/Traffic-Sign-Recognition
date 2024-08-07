import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn.functional as F
from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from torchvision.models import ResNet50_Weights
from torchvision import datasets
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile


def create_dynamic_network(num_features, num_classes, neuron_list=None, dropout_values=None):
    if neuron_list is None:
        neuron_list = []
    layers = []
    num_layers = len(neuron_list)
    # Input layer to first hidden layer
    if num_layers > 0:
        layers.append(nn.Linear(num_features, neuron_list[0]))
        layers.append(nn.ReLU())
        if dropout_values[0] != 0:
            layers.append(nn.Dropout(dropout_values[0]))

    # Additional hidden layers
    for i in range(1, num_layers):
        layers.append(nn.Linear(neuron_list[i - 1], neuron_list[i]))
        layers.append(nn.ReLU())
        if dropout_values[i] != 0:
            layers.append(nn.Dropout(dropout_values[i]))

    # Always include the final specified layer
    layers.append(nn.Linear(neuron_list[-1] if num_layers > 0 else num_features, num_classes))
    # layers.append(nn.Softmax(dim=1)) not needed cause cross entropy criterion

    return nn.Sequential(*layers)


def load_data(data_dir="~/Desktop/UniPD/VCS/Traffic-Sign-Recognition/dataset/GTSRB"):
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    trainset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    testset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    return trainset, testset


def train_TSR(config):
    neuron_layer_list = [[512, 256, 128], [512], [512, 256]]
    dropout_values = [[0.25, 0.25, 0.5], [0, 0, 0], [0.25, 0.25, 0.25]]

    # initialize model and lock backbone
    net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in net.parameters():
        param.requires_grad = False

    # modify fully connected output layer
    net.fc = create_dynamic_network(net.fc.in_features, num_classes=43,
                                    neuron_list=neuron_layer_list[config["neuron_layer_index"]],
                                    dropout_values=dropout_values[config["dropout_values_index"]])

    device = torch.device("cuda")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # Set optimizers
    if config['opt'] == 'sgd':
        optimizer = optim.SGD(net.fc.parameters(), lr=config['lr'])
    elif config['opt'] == 'adam':
        optimizer = optim.Adam(net.fc.parameters(), lr=config['lr'])
    elif config['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(net.fc.parameters(), lr=config['lr'])
    else:
        raise ValueError('Invalid optimizer provided')

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    train_data_size = len(train_subset)

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4)

    epochs = 10
    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        train_loss = 0.0
        train_acc = 0.0
        data_total_steps = len(trainloader)
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            probabilities, predictions = torch.max(softmax_outputs, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if (i + 1) % int(data_total_steps / 8) == 0:  # print every 1/8 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        train_acc = train_acc / train_data_size
        train_loss = train_loss / train_data_size

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                softmax_outputs = F.softmax(outputs, dim=1)
                pbs, predicted = torch.max(softmax_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(dict(train_loss=train_loss, train_accuracy=train_acc, val_loss=(val_loss / val_steps),
                              val_accuracy=correct / total), checkpoint=checkpoint)

    # test model on test set
    _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=True, num_workers=2)

    best_trained_model = net

    correct = 0
    total = 0
    with torch.no_grad():
        best_trained_model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            softmax_outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(softmax_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = float(correct / total)
    train.report(dict(test_accuracy=test_accuracy))

    print("Finished Training")


if __name__ == "__main__":
    # Run hyperparameter search, method used based on command line arg
    if sys.argv[1] == 'basic_search':
        # "basic search": consists of grid search, random sampling, etc...
        """
        tune.run(...) will perform the hyperparameter sweep
        The first argument is a function that will:
            - load and preprocess training/validation data
            - define the model
            - train the model
            - log desired metrics
        The config argument will define the search space.
        In the "resources_per_trial" argument, we can specify how many CPUs and GPUs
        we want to provide to each training run within the experiment.
        The "num_samples" argument defines how many training runs will be performed.
        """
        config = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "opt": tune.grid_search(['sgd', 'rmsprop', 'adam']),
            "batch_size": tune.grid_search([64]),
            "neuron_layer_index": tune.grid_search([0, 2]),
            "dropout_values_index": tune.grid_search([0, 2]),
        }
        analysis = tune.run(
            train_TSR,
            config=config,
            resources_per_trial={"cpu": 4, "gpu": 1},
            num_samples=10
        )
    else:
        print("ERROR: Invalid search type. Options: 'basic_search'")
        sys.exit(1)
