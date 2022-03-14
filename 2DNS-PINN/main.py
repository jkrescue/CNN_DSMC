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
    def __init__(self, x_train, y_train, t_train, u_train, v_train, layers):

        # collection data
        self.x_train = paddle.to_tensor(x_train, stop_gradient=False, dtype="float32")
        self.y_train = paddle.to_tensor(y_train, stop_gradient=False, dtype="float32")
        self.t_train = paddle.to_tensor(t_train, stop_gradient=False, dtype="float32")

        self.u_train = paddle.to_tensor(u_train, dtype="float32")
        self.v_train = paddle.to_tensor(v_train, dtype="float32")

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
        self.optimizer = paddle.optimizer.AdamW(parameters=self.dnn.parameters(), weight_decay=0.005)
        self.iter = 0

    def net_u(self, x, y, t):
        psi_p = self.dnn(paddle.concat([x, y, t], axis=1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        u = paddle.autograd.grad(psi, y, grad_outputs=paddle.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v = -paddle.autograd.grad(psi, y, grad_outputs=paddle.ones_like(psi), retain_graph=True, create_graph=True)[0]

        return u, v, p

    def net_NS(self, x, y, t):
        psi_p = self.dnn(paddle.concat([x, y, t], axis=1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        u = paddle.autograd.grad(psi, y, grad_outputs=paddle.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v = -paddle.autograd.grad(psi, y, grad_outputs=paddle.ones_like(psi), retain_graph=True, create_graph=True)[0]

        u_t = paddle.autograd.grad(u, t, grad_outputs=paddle.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = paddle.autograd.grad(u, x, grad_outputs=paddle.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = paddle.autograd.grad(u, y, grad_outputs=paddle.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = paddle.autograd.grad(u_x, x, grad_outputs=paddle.ones_like(u), retain_graph=True, create_graph=False)[0]
        u_yy = paddle.autograd.grad(u_y, y, grad_outputs=paddle.ones_like(u), retain_graph=True, create_graph=False)[0]

        v_t = paddle.autograd.grad(v, t, grad_outputs=paddle.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_x = paddle.autograd.grad(v, x, grad_outputs=paddle.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_y = paddle.autograd.grad(v, y, grad_outputs=paddle.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = paddle.autograd.grad(v_x, x, grad_outputs=paddle.ones_like(v), retain_graph=True, create_graph=False)[0]
        v_yy = paddle.autograd.grad(v_y, y, grad_outputs=paddle.ones_like(v), retain_graph=True, create_graph=False)[0]

        p_x = paddle.autograd.grad(p, x, grad_outputs=paddle.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = paddle.autograd.grad(p, y, grad_outputs=paddle.ones_like(p), retain_graph=True, create_graph=True)[0]

        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        # lambda_1 = 1.0
        # lambda_2 = 0.01

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss_func(self):

        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x_train, self.y_train, self.t_train)

        # set loss function
        loss = paddle.mean((u_pred - self.u_train) ** 2) + \
               paddle.mean((v_pred - self.v_train) ** 2) + \
               paddle.mean(f_u_pred ** 2) + \
               paddle.mean(f_v_pred ** 2)

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

    def predict(self, x, y, t):

        x = paddle.to_tensor(x, stop_gradient=False, dtype="float32")
        y = paddle.to_tensor(y, stop_gradient=False, dtype="float32")
        t = paddle.to_tensor(t, stop_gradient=False, dtype="float32")

        psi_p = self.dnn(paddle.concat([x, y, t], axis=1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        u = paddle.autograd.grad(psi, y, grad_outputs=paddle.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v = -paddle.autograd.grad(psi, y, grad_outputs=paddle.ones_like(psi), retain_graph=True, create_graph=True)[0]

        return u.numpy(), v.numpy(), p.numpy()


if __name__ == "__main__":
    # 训练数据点样本数
    N_train = 50000
    # 神经网络结构
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    # 读取数据
    data = scipy.io.loadmat("./cylinder_nektar_wake.mat")
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2
    # 获得时间网格和空间网格数目
    N = X_star.shape[0]  # 5000
    T = t_star.shape[0]  # 200
    # 将x、y、t转换成对应u、v、p的形状
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    # x、y、t和u、v、p展成一维数组
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # 将x、y、t和u、v、p合并为一个矩阵
    data1 = np.concatenate([x, y, t, u, v, p], axis=1)  # NT x 6
    data_domain = data1[:, :]

    idx = np.random.choice(data_domain.shape[0], N_train, replace=False)

    x_train = data_domain[idx, 0].reshape(data_domain[idx, 0].shape[0], 1)
    y_train = data_domain[idx, 1].reshape(data_domain[idx, 1].shape[0], 1)
    t_train = data_domain[idx, 2].reshape(data_domain[idx, 2].shape[0], 1)

    u_train = data_domain[idx, 3].reshape(data_domain[idx, 3].shape[0], 1)
    v_train = data_domain[idx, 4].reshape(data_domain[idx, 4].shape[0], 1)

    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)

    # train
    # model.train(20000)

    # test
    model.load_model()
    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]

    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    #
    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error p: %e' % error_p)
