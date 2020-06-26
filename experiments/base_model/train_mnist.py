import logging
import os
from datetime import datetime

from dateutil.relativedelta import relativedelta

import torch
from clustre.helpers import delta_tostr
from clustre.helpers.datasets import mnist_testloader, mnist_trainloader
from clustre.models import mnist_cnn, mnist_resnet18
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

N_EPOCHES = 10
LR = 1e-3
LOG_FILENAME = os.path.abspath(__file__)[:-3] + "_log.txt"
SCRIPT_PATH = os.path.dirname(__file__)
FORMAT = "%(message)s"
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)
log = logging.getLogger()

torch.manual_seed(0)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

models = {"mnist_resnet18": mnist_resnet18}

for model_name, model in models.items():
    logging.info("Model: {}".format(model_name))

    model = model.to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    testing_losses = []

    for e in range(N_EPOCHES):
        move_time = relativedelta()
        forward_time = relativedelta()
        backprop_time = relativedelta()

        training_loss = 0
        testing_loss = 0
        for images, labels in mnist_trainloader:
            move_timestamp = datetime.now()
            images = images.to("cuda")
            labels = labels.to("cuda")
            input_timestamp = datetime.now()
            images = images.reshape(-1, 1, 28, 28)
            optimizer.zero_grad()
            output = model(images)
            backprop_timestamp = datetime.now()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            finish_timestamp = datetime.now()
            move_time += relativedelta(input_timestamp, move_timestamp)
            forward_time += relativedelta(backprop_timestamp, input_timestamp)
            backprop_time += relativedelta(
                finish_timestamp, backprop_timestamp
            )
            training_loss += loss.item()
        training_loss /= len(mnist_trainloader)

        with torch.no_grad():
            for images, labels in mnist_testloader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                images = images.reshape(-1, 1, 28, 28)
                output = model(images)
                testing_loss += loss.item()
            testing_loss /= len(mnist_testloader)
            testing_losses.append(testing_loss)
        logging.info(
            f"Epoch: {e}\n\tTrain: {training_loss} Test: {testing_loss}"
        )
        logging.info(f"\tMove time: {delta_tostr(move_time)}")
        logging.info(f"\tForward time: {delta_tostr(forward_time)}")
        logging.info(f"\tBackprop time: {delta_tostr(backprop_time)}")

        if testing_loss <= min(testing_losses):
            torch.save(model.state_dict(), "{}.model".format(model_name))
