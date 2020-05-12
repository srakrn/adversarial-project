# %%
import unittest

# %%
import torch
from torch import nn

# %%
from clustre.attacking import pgd
from clustre.helpers.datasets import mnist_testloader
from clustre.models import mnist_cnn
from clustre.models.state_dicts import mnist_cnn_state

# %%
mnist_cnn.load_state_dict(mnist_cnn_state)

# %%
batch = next(iter(mnist_testloader))
batch_X, batch_y = batch
point_X, point_y = batch_X[0], batch_y[0]
batchpoint_X, batchpoint_y = batch_X[0:1], batch_y[0:1]

# %%
results = pgd(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y, n_epoches=100)

# %%
class TestPgd(unittest.TestCase):
    def test_pgd_shape(self):
        results = pgd(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y)
        self.assertTupleEqual(results.shape, batch_X.shape)

    def test_pgd_range(self):
        results = pgd(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y)
        return ((results >= -1) & (results <= 1)).all()


# %%
if __name__ == "__main__":
    unittest.main()
