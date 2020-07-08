from datetime import datetime

from dateutil.relativedelta import relativedelta
from torch import nn, optim

from clustre.attacking import pgd
from clustre.helpers import delta_tostr, get_time


def pgd_training(
    model,
    trainloader,
    n_epoches=10,
    epsilon=0.3,
    pgd_step_size=0.02,
    pgd_epoches=7,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    optimizer_params={},
    device=None,
    log=None,
):
    """Standard k-PGD Adversarial Training

    Parameters
    ----------
    model: torch.nn.model
        The model to be reinforced
    trainloader: torch.utils.data.DataLoader
        The DataLoader for the training set
    n_epoches: int
        The epoches to be trained
    epsilon: float
        Perturbation bound
    pgd_step_size: float
        PGD step size
    pgd_epoches: int
        PGD maximization iterations per minimisation step
    criterion: function
        Criterion function
    optimizer: class of torch.optim
        Optimiser to train the model
    optimizer_params: dict
        Parameters to be passed to the optimiser
    device: torch.device, str, or None
        Device to be used
    log: logger or None
        If logger, logs to the corresponding logger
    """

    # Move to device if desired
    if device is not None:
        model.to(device)
    # Log starting time if desired
    if log is not None:
        log.info(f"Training started: {get_time()}")
        log.info(
            "n_epoches,move_time,pgd_time,forward_time,backprop_time,training_loss"
        )

    # Create an optimiser instance
    optimizer = optimizer(model.parameters(), **optimizer_params)

    # Iterate over e times of epoches
    for e in range(n_epoches):
        pgd_time = relativedelta()
        move_time = relativedelta()
        forward_time = relativedelta()
        backprop_time = relativedelta()
        # Running loss, for reference
        running_loss = 0
        # Iterate over minibatches of trainloader
        for i, (images, labels) in enumerate(trainloader):
            # Move tensors to device if desired
            move_timestamp = datetime.now()
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
            # Calculate perturbations
            pgd_timestamp = datetime.now()
            adver_images = pgd(
                model,
                criterion,
                images,
                labels,
                epsilon,
                pgd_step_size,
                pgd_epoches,
                device=device,
            )
            optimizer.zero_grad()

            forward_timestamp = datetime.now()

            output = model(adver_images)

            backprop_timestamp = datetime.now()

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            finish_timestamp = datetime.now()

            running_loss += loss.item()

            move_time += relativedelta(pgd_timestamp, move_timestamp)
            pgd_time += relativedelta(forward_timestamp, pgd_timestamp)
            forward_time += relativedelta(
                backprop_timestamp, forward_timestamp
            )
            backprop_time += relativedelta(
                finish_timestamp, backprop_timestamp
            )
        else:
            if log is not None:
                log.info(
                    f"{e},\
{delta_tostr(move_time)},\
{delta_tostr(pgd_time)},\
{delta_tostr(forward_time)},\
{delta_tostr(backprop_time)},\
{running_loss/len(trainloader)}"
                )
    if log is not None:
        log.info(f"Training ended: {get_time()}")
    return model
