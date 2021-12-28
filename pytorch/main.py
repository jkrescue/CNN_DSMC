import os
import pickle
from useful_function import *
from torch.utils.data import TensorDataset
from net import CNN_DSMC
import time


def epoch(loader, training=False):
    total_loss = 0
    if training:
        model.train()
    else:
        model.eval()

    for tensors in loader:
        tensors = [tensor.to(device) for tensor in tensors]
        loss, output = loss_func(model, tensors)
        if training:
            optimizer.zero_grad()
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
        with torch.no_grad():
            val_loss = epoch(test_loader, training=False)
        print("\tValidation Loss = " + str(val_loss))
        if (epoch_id == 100):
            print("运行100个epochs的时间为{:.2f} s".format(time.time() - begin))

        if (epoch_id % 500 == 0):
            torch.save(model.state_dict(), os.path.join(save_path, "CNN-DSMC" + str(epoch_id) + ".pth"))
            print("Model saved!")


if __name__ == "__main__":
    save_path = r"./Run"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = pickle.load(open("./dataX.pkl", "rb"))
    y = pickle.load(open("./dataY.pkl", "rb"))
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    train_data, test_data = split_tensors(x, y, ratio=0.9)
    train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)

    lr = 0.001
    kernel_size = 5
    filters = [8, 16, 32, 32]
    bn = True
    wn = False
    model = CNN_DSMC(2, 2, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    model = model.to(device)
    wd = 0.005
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = 10000
    batch_size = 2
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train()
