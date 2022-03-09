import paddle
import paddle.nn as nn
import numpy as np
import scipy.io
import os

save_path = "./model_saved/"
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
    def __init__(self, X, u, layers, lb, ub):

        # boundary conditions
        self.lb = paddle.to_tensor(lb, dtype="float32")
        self.ub = paddle.to_tensor(ub, dtype="float32")

        # data
        self.x = paddle.to_tensor(X[:, 0:1], dtype="float32", stop_gradient=False)
        self.t = paddle.to_tensor(X[:, 1:2], dtype="float32", stop_gradient=False)
        self.u = paddle.to_tensor(u, dtype="float32")

        # parameters of equations
        self.lambda_1 = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))
        self.lambda_2 = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=-6.0))

        # MLP
        self.dnn = DNN(layers)
        self.dnn.add_parameter('lambda_1', self.lambda_1)
        self.dnn.add_parameter('lambda_2', self.lambda_2)

        self.optimizer = paddle.optimizer.AdamW(parameters=self.dnn.parameters(), weight_decay=0.005)
        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(paddle.concat([x, t], axis=1))
        return u

    def net_f(self, x, t):
        lambda_1 = self.lambda_1
        lambda_2 = paddle.exp(self.lambda_2)
        u = self.net_u(x, t)

        u_t = paddle.autograd.grad(u, t, grad_outputs=paddle.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = paddle.autograd.grad(u, x, grad_outputs=paddle.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = paddle.autograd.grad(u_x, x, grad_outputs=paddle.ones_like(u_x), retain_graph=True, create_graph=True)[0]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def train(self, nIter):
        self.dnn.train()
        for epoch in range(nIter):
            with paddle.static.device_guard('gpu'):
                u_pred = self.net_u(self.x, self.t)
                f_pred = self.net_f(self.x, self.t)
                loss = paddle.mean((self.u - u_pred) ** 2) + paddle.mean(f_pred ** 2)

                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()

                if epoch % 1 == 0:
                    print(
                        'epoch: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
                        (
                            epoch,
                            loss.item(),
                            self.lambda_1.item(),
                            paddle.exp(self.lambda_2).item()
                        )
                    )
        paddle.save(self.dnn.state_dict(), os.path.join(save_path, "PINN.pdparams"))
        paddle.save(self.optimizer.state_dict(), os.path.join(save_path, "AdamW.pdopt"))
        print("Model saved!")

    def load_model(self):
        layer_state_dict = paddle.load(os.path.join(save_path, "PINN.pdparams"))
        self.dnn.set_state_dict(layer_state_dict)


if __name__ == "__main__":
    nu = 0.01 / np.pi

    N = 3000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('data/burgers_shock.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    U = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = U.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    idx = np.random.choice(X_star.shape[0], N, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]

    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(20000)
