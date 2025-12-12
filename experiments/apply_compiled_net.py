import torch
import torchvision

import mnist_dataset
from difflogic import CompiledLogicNet

torch.set_num_threads(1)

dataset = 'mnist20x20'
batch_size = 1_000

]:
    save_lib_path = 'lib/{:08d}_{}.so'.format(0, num_bits)
    compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)

    correct, total = 0, 0
    for (data, labels) in test_loader:
        data = torch.nn.Flatten()(data).bool().numpy()

        output = compiled_model.forward(data)

        correct += (output.argmax(-1) == labels).float().sum()
        total += output.shape[0]

    acc3 = correct / total
    print('COMPILED MODEL', num_bits, acc3)
