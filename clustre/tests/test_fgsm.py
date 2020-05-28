# %%
import unittest

# %%
import torch
from torch import nn

# %%
from clustre.attacking import fgsm
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
class TestFgsm(unittest.TestCase):
    def test_fgsm_shape(self):
        result = fgsm(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y)
        self.assertTupleEqual(result.shape, batch_X.shape)

    def test_fgsm_range(self):
        result = fgsm(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y)
        self.assertTrue(((result >= -1) & (result <= 1)).all())


# %%
if __name__ == "__main__":
    unittest.main()
