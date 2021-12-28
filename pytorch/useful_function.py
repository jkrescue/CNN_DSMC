import torch

def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

def loss_func(model, batch):
    x, y = batch
    output = model(x)
    lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3]))
    lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3]))
    loss = (lossu + lossv)
    return torch.sum(loss), output
