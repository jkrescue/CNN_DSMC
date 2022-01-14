import numpy as np
import os
import pickle
from useful_function import *
from paddle.io import TensorDataset, DataLoader
from net import CNN_DSMC
import time


# def epoch(loader, training=False):
#     total_loss = 0
#     if training:
#         model.train()
#     else:
#         model.eval()
#     with paddle.static.device_guard('gpu'):
#         for tensors in loader:
#             loss, output = loss_func(model, tensors)
#             if training:
#                 optimizer.clear_grad()
#                 loss.backward()
#                 optimizer.step()
#             total_loss += loss.item()
#
#     return total_loss

def epoch(loader, training=False):
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    total_loss = 0
    if training:
        model.train()
    else:
        model.eval()
    with paddle.static.device_guard('gpu'):
        for tensors in loader:
            with paddle.amp.auto_cast():
                loss, output = loss_func(model, tensors)
                if training:
                    scaled = scaler.scale(loss)
                    scaled.backward()
                    scaler.minimize(optimizer, scaled)
                    optimizer.clear_grad()

                total_loss += loss.item()

    return total_loss


def train():
    for epoch_id in range(1, epochs + 1):
        begin = time.time()
        print("Epoch #" + str(epoch_id))

        # Training
        train_loss = epoch(train_loader, training=True)
        print("\tTrain Loss = " + str(train_loss))

        # Validation
        with paddle.no_grad():
            val_loss = epoch(test_loader, training=False)
        print("\tValidation Loss = " + str(val_loss))
        if (epoch_id != 0):
            print("运行1个epochs的时间为{:.2f} s".format(time.time() - begin))

        if (epoch_id % 2 == 0):
            paddle.save(model.state_dict(), os.path.join(save_path, "CNN-DSMC" + str(epoch_id) + ".pdparams"))
            print("Model saved!")


if __name__ == "__main__":
    save_path = r"./Run/test-1"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x = pickle.load(open("./dataX.pkl", "rb"))
    y = pickle.load(open("./dataY.pkl", "rb"))
    xx = x.copy()
    yy = y.copy()
    # xx = x[:, :, :, 75:175, 75:175]
    # yy = y[:, :, :, 75:175, 75:175]
    x = paddle.to_tensor(xx, dtype="float32")
    y = paddle.to_tensor(yy, dtype="float32")

    train_data, test_data = split_tensors(x, y, ratio=0.9)
    train_dataset, test_dataset = TensorDataset([train_data[0], train_data[1]]), TensorDataset(
        [test_data[0], test_data[1]])

    lr = 0.001
    kernel_size = 5
    filters = [8, 16, 32, 32, 64, 64, 128]
    bn = True
    wn = False
    model = CNN_DSMC(2, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    wd = 0.005
    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=wd)

    epochs = 10000
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train()
