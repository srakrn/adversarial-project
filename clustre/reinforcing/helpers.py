import logging

from sklearn.metrics import classification_report

log = logging.getLogger(__name__)


def accuracy_unattacked(model, testloader):
    model.eval()
    y_test = []
    y_pred = []
    for image, label in testloader:
        if not image.is_cuda:
            image = image.to("cuda")
            label = label.to("cuda")
        model.to("cuda")
        y_test.append(label.item())
        y_pred.append(model(image).argmax(axis=1).item())
    print("Original model report:")
    print(classification_report(y_test, y_pred))
    logging.info(classification_report(y_test, y_pred))


def accuracy_attacked(model, testloader, testset_perturbs, density=0.2):
    model.eval()
    y_test = []
    y_pred = []
    for (image, label), perturb in zip(testloader, testset_perturbs):
        if not image.is_cuda:
            image = image.to("cuda")
            label = label.to("cuda")
        if not perturb.is_cuda:
            perturb = perturb.to("cuda")
        model.to("cuda")
        y_test.append(label.item())
        y_pred.append(
            model(image + density * perturb.reshape(image.shape))
            .argmax(axis=1)
            .item()
        )
    print("Adversarial on original model report:")
    print(classification_report(y_test, y_pred))
    logging.info(classification_report(y_test, y_pred))
