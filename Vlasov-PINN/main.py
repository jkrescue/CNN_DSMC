import paddle
import paddle.nn as nn
import numpy as np
import os
import scipy.io

save_path = r"./model_saved"
if not os.path.exists(save_path):
    os.mkdir(save_path)


class DNN(nn.Layer):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1

        self.activation = nn.Tanh

        self.layers = nn.Sequential()

        for i in range(self.depth - 1):
            self.layers.add_sublayer('layer_%d' % i, nn.Linear(layers[i], layers[i + 1]))
            self.layers.add_sublayer('activation_%d' % i, self.activation())

        self.layers.add_sublayer('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        out = self.layers(x)
        return out


class PhysicsInformedNN():
    def __init__(self, X, E, f, layers):

        # collection data
        self.t = paddle.to_tensor(X[:, 0:1], stop_gradient=False, dtype="float32")
        self.x = paddle.to_tensor(X[:, 1:2], stop_gradient=False, dtype="float32")
        self.v = paddle.to_tensor(X[:, 2:3], stop_gradient=False, dtype="float32")
        self.E = paddle.to_tensor(E, dtype="float32")
        self.f = paddle.to_tensor(f, dtype="float32")

        # parameters need to derive
        self.lambda_1 = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))
        self.lambda_2 = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))

        # MLP
        self.dnn = DNN(layers)
        self.dnn.add_parameter('lambda_1', self.lambda_1)
        self.dnn.add_parameter('lambda_2', self.lambda_2)

        # optimizers: AdamW
        self.optimizer = paddle.optimizer.Adam(parameters=self.dnn.parameters())
        self.iter = 0

    def net_f(self, t, x, v):
        f = self.dnn(paddle.concat([t, x, v], axis=1))
        return f

    def net_g(self, t, x, v, E):

        f = self.net_f(t, x, v)

        f_t = paddle.autograd.grad(f, t, grad_outputs=paddle.ones_like(f), retain_graph=True, create_graph=True)[0]
        f_x = paddle.autograd.grad(f, x, grad_outputs=paddle.ones_like(f), retain_graph=True, create_graph=True)[0]
        f_v = paddle.autograd.grad(f, v, grad_outputs=paddle.ones_like(f), retain_graph=True, create_graph=True)[0]

        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        # lambda_1 = 1.0
        # lambda_2 = -1.0

        g = f_t + lambda_1 * v * f_x + lambda_2 * E * f_v

        return g

    def loss_func(self):

        f_pred = self.net_f(self.t, self.x, self.v)
        g_pred = self.net_g(self.t, self.x, self.v, self.E)

        # set loss function
        loss = paddle.mean((self.f - f_pred) ** 2) + \
               paddle.mean(g_pred ** 2)

        return loss

    def train(self, nIter):

        self.dnn.train()
        loss_list = []
        for epoch in range(nIter):
            with paddle.static.device_guard('gpu'):

                loss = self.loss_func()
                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())
                if epoch % 100 == 0:
                    print(
                        'Epoch: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
                        (
                            epoch,
                            loss.item(),
                            self.lambda_1.item(),
                            self.lambda_2.item()
                        )
                    )
                    if (loss.item() <= min(loss_list)):
                        paddle.save(self.dnn.state_dict(),
                                    os.path.join(save_path, "PINN_" + str(epoch) + ".pdparams"))
                        print("Model saved!")
        paddle.save(self.dnn.state_dict(), os.path.join(save_path, "PINN.pdparams"))
        np.savetxt(os.path.join(save_path, "loss.txt"), np.array(loss_list), fmt='%.6f', delimiter=" ")

    def load_model(self):
        self.dnn.set_state_dict(paddle.load(os.path.join(save_path, "PINN_14400.pdparams")))


if __name__ == "__main__":
    N_u = 30000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    dataE = np.load("./dataE.npy", allow_pickle=True).item()
    dataf = np.load("./dataf.npy", allow_pickle=True).item()

    x = dataf['x'].flatten()[:, None]
    v = dataf['v'].flatten()[:, None]
    t = dataf['t'].flatten()[:, None]
    E = np.real(dataE['E'])  # t,x (500,64)
    # f = np.real(dataf['f'])  # t,x,v (500,64,64)
    f = np.real(dataf['f']) / 640000 / 63 / (63 / 0.6)  # 原数据有问题
    T, X, V = np.meshgrid(t, x, v)  # x,t,v (64,500,64)
    T = np.swapaxes(T, 0, 1)
    X = np.swapaxes(X, 0, 1)
    V = np.swapaxes(V, 0, 1)  # t,x,v (500,64,64)

    X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None], V.flatten()[:, None]))  # t,x,v
    f_star = f.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)  # t,x,v
    ub = X_star.max(0)  # t,x,v

    # create training set
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    idx_on_tx = np.unravel_index(idx, f.shape)  # return index in tx space
    E_train = E[idx_on_tx[0], idx_on_tx[1]]
    f_train = f_star[idx, :]

    # training
    model = PhysicsInformedNN(X_u_train, E_train, f_train, layers)
    model.train(20000)
