import paddle
import numpy as np
import matplotlib.pyplot as plt

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
    lossu = ((output[:, 0, :, :, :] - y[:, 0, :, :, :]) ** 2).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]))
    lossv = ((output[:, 1, :, :, :] - y[:, 1, :, :, :]) ** 2).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]))
    lossw = ((output[:, 2, :, :, :] - y[:, 2, :, :, :]) ** 2).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]))
    loss = lossu + lossv + lossw
    return paddle.sum(loss), output

def visualize(sample_y, out_y, error, s):
    # error = error / sample_y
    minu = np.min(sample_y[s, 0, :, :])
    maxu = np.max(sample_y[s, 0, :, :])

    minv = np.min(sample_y[s, 1, :, :])
    maxv = np.max(sample_y[s, 1, :, :])

    mineu = np.min(error[s, 0, :, :])
    maxeu = np.max(error[s, 0, :, :])

    minev = np.min(error[s, 1, :, :])
    maxev = np.max(error[s, 1, :, :])

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.subplot(2, 3, 1)
    plt.title('DSMC', fontsize=18)
    plt.imshow(np.transpose(sample_y[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower',
               extent=[0, 510, 0, 260])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Ux', fontsize=18)
    plt.subplot(2, 3, 2)
    plt.title('CNN-DSMC', fontsize=18)
    plt.imshow(np.transpose(out_y[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower',
               extent=[0, 510, 0, 260])
    plt.colorbar(orientation='horizontal')
    plt.subplot(2, 3, 3)
    plt.title('Error', fontsize=18)
    plt.imshow(np.transpose(error[s, 0, :, :]), cmap='jet', vmin=mineu, vmax=maxeu, origin='lower',
               extent=[0, 510, 0, 260])
    plt.colorbar(orientation='horizontal')

    plt.subplot(2, 3, 4)
    plt.imshow(np.transpose(sample_y[s, 1, :, :]), cmap='jet', vmin=minv, vmax=maxv, origin='lower',
               extent=[0, 510, 0, 260])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Uy', fontsize=18)
    plt.subplot(2, 3, 5)
    plt.imshow(np.transpose(out_y[s, 1, :, :]), cmap='jet', vmin=minv, vmax=maxv, origin='lower',
               extent=[0, 510, 0, 260])
    plt.colorbar(orientation='horizontal')
    plt.subplot(2, 3, 6)
    plt.imshow(np.transpose(error[s, 1, :, :]), cmap='jet', vmin=minev, vmax=maxev, origin='lower',
               extent=[0, 510, 0, 260])
    plt.colorbar(orientation='horizontal')

    plt.tight_layout()
    plt.savefig("testjpg")
    plt.show()
