# %%
import unittest

# %%
import torch

# %%
from clustre.attacking import maxloss
from clustre.helpers.datasets import mnist_testloader
from clustre.models import mnist_cnn
from clustre.models.state_dicts import mnist_cnn_state
from torch import nn

# %%
mnist_cnn.load_state_dict(mnist_cnn_state)

# %%
batch = next(iter(mnist_testloader))
batch_X, batch_y = batch
point_X, point_y = batch_X[0], batch_y[0]
batchpoint_X, batchpoint_y = batch_X[0:1], batch_y[0:1]


# %%
batch_X_cuda, batch_y_cuda = batch_X.to("cuda"), batch_y.to("cuda")
point_X_cuda, point_y_cuda = batch_X_cuda[0], batch_y_cuda[0]
batchpoint_X_cuda, batchpoint_y_cuda = batch_X_cuda[0:1], batch_y_cuda[0:1]

# %%
class TestPgd(unittest.TestCase):
    def test_maxloss_shape(self):
        results = maxloss(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y)
        self.assertTupleEqual(results.shape, batch_X.shape)

    def test_maxloss_range(self):
        results = maxloss(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y)
        self.assertTrue(((results >= -1) & (results <= 1)).all())


class TestPgdCuda(unittest.TestCase):
    def test_maxloss_shape(self):
        mnist_cnn.to("cuda")
        results = maxloss(
            mnist_cnn, nn.CrossEntropyLoss(), batch_X_cuda, batch_y_cuda
        )
        self.assertTupleEqual(results.shape, batch_X.shape)

    def test_maxloss_range(self):
        mnist_cnn.to("cuda")
        results = maxloss(
            mnist_cnn, nn.CrossEntropyLoss(), batch_X_cuda, batch_y_cuda
        )
        self.assertTrue(((results >= -1) & (results <= 1)).all())

    def test_cuda(self):
        mnist_cnn.to("cuda")
        results = maxloss(
            mnist_cnn, nn.CrossEntropyLoss(), batch_X_cuda, batch_y_cuda
        )
        self.assertTrue(results.is_cuda)

    def test_force_cpu(self):
        mnist_cnn.to("cuda")
        results = maxloss(
            mnist_cnn,
            nn.CrossEntropyLoss(),
            batch_X,
            batch_y_cuda,
            device="cpu",
        )

    def test_force_cuda(self):
        mnist_cnn.to("cuda")
        results = maxloss(
            mnist_cnn,
            nn.CrossEntropyLoss(),
            batch_X,
            batch_y_cuda,
            device="cuda",
        )


# %%
if __name__ == "__main__":
    unittest.main()
