import logging

from sklearn.metrics import classification_report

log = logging.getLogger(__name__)


def accuracy_unattacked(model, testloader):
    y_test = []
    y_pred = []
    for image, label in testloader:
        y_test.append(label.item())
        y_pred.append(model(image).argmax(axis=1).item())
    print("Original model report:")
    print(classification_report(y_test, y_pred))
    logging.info(classification_report(y_test, y_pred))


def accuracy_attacked(model, testloader, testset_perturbs, density=0.2):
    y_test = []
    y_pred = []
    for (image, label), perturb in zip(testloader, testset_perturbs):
        if perturb.device == "cuda":
            perturb = perturb.to("cpu")
        y_test.append(label.item())
        y_pred.append(
            model(image + density * perturb.reshape(image.shape).unsqueeze_(0))
            .argmax(axis=1)
            .item()
        )
    print("Adversarial on original model report:")
    print(classification_report(y_test, y_pred))
    logging.info(classification_report(y_test, y_pred))
