# %%
import unittest

# %%
from torch import nn

# %%
from clustre.attacking import fgsm, fgsm_single_image
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
class TestSinglePointFgsm(unittest.TestCase):
    def test_fgsm_size_single_image(self):
        result = fgsm_single_image(mnist_cnn, nn.CrossEntropyLoss(), point_X, point_y)

        self.assertEqual(len(result.shape), 4)

    def test_fgsm_size_batch_image(self):
        result = fgsm_single_image(
            mnist_cnn, nn.CrossEntropyLoss(), batchpoint_X, batchpoint_y
        )

        self.assertEqual(len(result.shape), 4)


class TestFgsm(unittest.TestCase):
    def test_fgsm(self):
        result = fgsm(mnist_cnn, nn.CrossEntropyLoss(), batch_X, batch_y)
        self.assertTupleEqual(result.shape, batch_X.shape)


# %%
if __name__ == "__main__":
    unittest.main()
