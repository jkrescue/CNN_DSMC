import os
import pickle
from useful_function import *
from paddle.io import TensorDataset, DataLoader
from net import CNN_DSMC
import time

def epoch(loader, training=False):
    total_loss = 0
    if training:
        model.train()
    else:
        model.eval()
    with paddle.static.device_guard('gpu'):
        for tensors in loader:
            loss, output = loss_func(model, tensors)
            if training:
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

    return total_loss


def train():
    begin = time.time()
    for epoch_id in range(1, epochs + 1):

        print("Epoch #" + str(epoch_id))

        # Training
        train_loss = epoch(train_loader, training=True)
        print("\tTrain Loss = " + str(train_loss))

        # Validation
        with paddle.no_grad():
            val_loss = epoch(test_loader, training=False)
        print("\tValidation Loss = " + str(val_loss))
        if (epoch_id == 100):
            print("运行100个epochs的时间为{:.2f} s".format(time.time() - begin))

        if (epoch_id % 500 == 0):
            paddle.save(model.state_dict(), os.path.join(save_path, "CNN-DSMC" + str(epoch_id) + ".pdparams"))
            print("Model saved!")


if __name__ == "__main__":
    save_path = r"./Run"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x = pickle.load(open("./dataX.pkl", "rb"))
    y = pickle.load(open("./dataY.pkl", "rb"))
    x = paddle.to_tensor(x, dtype="float32")
    y = paddle.to_tensor(y, dtype="float32")

    train_data, test_data = split_tensors(x, y, ratio=0.9)
    train_dataset, test_dataset = TensorDataset([train_data[0], train_data[1]]), TensorDataset(
        [test_data[0], test_data[1]])

    lr = 0.001
    kernel_size = 5
    filters = [8, 16, 32, 32]
    bn = True
    wn = False
    model = CNN_DSMC(2, 2, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    wd = 0.005
    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=wd)

    epochs = 10000
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train()
