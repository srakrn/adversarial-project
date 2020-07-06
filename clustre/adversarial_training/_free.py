import math
from datetime import datetime

from dateutil.relativedelta import relativedelta

import torch
from clustre.helpers import delta_tostr, get_time
from torch import nn, optim


def free_training(
    model,
    trainloader,
    n_epoches=10,
    epsilon=0.3,
    hop_step=5,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    optimizer_params={},
    device=None,
    log=None,
):
    # Move to device if desired
    if device is not None:
        model.to(device)
    # Log starting time if desired
    if log is not None:
        log.info("n_epoch,replay,move_time,input_gen,forward,backprop,update")

    # Create an optimiser instance
    optimizer = optimizer(model.parameters(), **optimizer_params)
    delta = None

    # Iterate over e times of epoches
    for e in range(math.ceil(n_epoches / hop_step)):
        # Running loss, for reference
        running_loss = 0
        move_time = relativedelta()
        time = [
            {
                "input_generation": relativedelta(),
                "forward_time": relativedelta(),
                "backprop_time": relativedelta(),
                "update_delta": relativedelta(),
            }
            for _ in range(hop_step)
        ]
        # Iterate over minibatches of trainloader
        for i, (images, labels) in enumerate(trainloader):
            # Move tensors to device if desired
            move_timestamp = datetime.now()
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)

            # Initialise attacking images
            if delta is None:
                delta = torch.zeros(
                    images.shape, requires_grad=True, device=images.device
                )
            move_fin_timestamp = datetime.now()
            move_time += relativedelta(move_fin_timestamp, move_timestamp)
            # Replay each minibatch for `hop_steps` to simulate PGD
            for i in range(hop_step):
                # Initialise attacking images
                input_generation_timestamp = datetime.now()
                attack_images = (images + delta).detach().requires_grad_()
                forward_timestamp = datetime.now()
                # Update model's parameter
                optimizer.zero_grad()
                output = model(attack_images)
                loss = criterion(output, labels)
                # Backprop
                backprop_timestamp = datetime.now()
                loss.backward()
                optimizer.step()
                # Use gradients calculated for the minimisation step
                # to update delta
                update_delta_timestamp = datetime.now()
                grad = attack_images.grad.data.detach()
                delta = delta.detach() + epsilon * torch.sign(grad)
                delta.clamp_(min=-epsilon, max=epsilon)
                finish_timestamp = datetime.now()
                time[i]["input_generation"] += relativedelta(
                    forward_timestamp, input_generation_timestamp
                )
                time[i]["forward_time"] += relativedelta(
                    backprop_timestamp, forward_timestamp
                )
                time[i]["backprop_time"] += relativedelta(
                    update_delta_timestamp, backprop_timestamp
                )
                time[i]["update_delta"] += relativedelta(
                    finish_timestamp, update_delta_timestamp
                )

            running_loss += loss.item()
        else:
            if log is not None:
                for i in range(hop_step):
                    log.info(
                        f"{e},{i},{delta_tostr(move_time/hop_step)}.\
{delta_tostr(time[i]['input_generation'])},\
{delta_tostr(time[i]['forward_time'])},\
{delta_tostr(time[i]['backprop_time'])},\
{delta_tostr(time[i]['update_delta'])}"
                    )
    if log is not None:
        log.info(f"Training ended: {get_time()}")

    return model
