from paddle.distributed import fleet
import os
import pickle
from useful_function import *
from paddle.io import TensorDataset, DataLoader, DistributedBatchSampler
from net import CNN_DSMC
import time
# from convnext3D import *
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

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
                fused_allreduce_gradients(list(model.parameters()), None)
                optimizer.step()
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

        if ((epoch_id <= 500 and epoch_id % 50 == 0) or epoch_id % 500 == 0):
            paddle.save(model.state_dict(), os.path.join(save_path, "CNN-DSMC" + str(epoch_id) + ".pdparams"))
            print("Model saved!")


if __name__ == "__main__":

    fleet.init(is_collective=True)
    save_path = r"./Run/test20220308"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x = pickle.load(open("/data/run01/scv6617/data_with_rho/dataX.pkl", "rb"))
    y = pickle.load(open("/data/run01/scv6617/data_with_rho/dataY.pkl", "rb"))
    x = paddle.to_tensor(x, dtype="float32")
    y = paddle.to_tensor(y, dtype="float32")

    train_data, test_data = split_tensors(x, y, ratio=27/31)
    train_dataset, test_dataset = TensorDataset([train_data[0], train_data[1]]), TensorDataset(
        [test_data[0], test_data[1]])

    lr = 0.001
    kernel_size = 5
    filters = [8, 16, 32, 32, 64, 64]
    bn = True
    wn = False
    model = CNN_DSMC(2, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    # model = convnext_tiny()
    model = fleet.distributed_model(model)
    
    wd = 0.005
    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=wd)
    optimizer = fleet.distributed_optimizer(optimizer)
    epochs = 10000
    batch_size = 8
    
    train_sampler = DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    test_sampler = DistributedBatchSampler(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    train()
