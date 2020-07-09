import torch


def fgsm(
    model,
    criterion,
    image,
    label,
    epsilon=0.3,
    random=False,
    alpha=0.375,
    device=None,
):
    if device is not None:
        model.to(device)
        image = image.to(device)
        label = label.to(device)

    perturb = fgsm_perturbs(
        model,
        criterion,
        image,
        label,
        epsilon=epsilon,
        random=random,
        alpha=alpha,
        device=device,
    )
    attack_image = torch.clamp(image + perturb, min=-1, max=1)

    return attack_image


def fgsm_perturbs(
    model,
    criterion,
    image,
    label,
    epsilon=0.3,
    random=False,
    alpha=0.375,
    device=None,
):
    if len(image.shape) == 3:
        image.unsqueeze_(0)
        label.unsqueeze_(0)

    model.eval()

    if device is not None:
        model.to(device)
        image = image.to(device)
        label = label.to(device)

    perturb = torch.zeros_like(image)
    if random:
        perturb.uniform_(-epsilon, epsilon)
    perturb = perturb.to(image.device)

    image.requires_grad = True
    perturb.requires_grad = True

    output = model(image + perturb)
    loss = criterion(output, label)
    loss.backward()

    grad = perturb.grad.detach()
    if random:
        perturb.data = torch.clamp(
            perturb + alpha * torch.sign(grad), -epsilon, epsilon
        )
    else:
        perturb.data = torch.clamp(torch.sign(grad), -epsilon, epsilon)

    return perturb
