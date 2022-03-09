import paddle
import paddle.nn as nn
from collections import OrderedDict
from paddle import summary


class DNN(nn.Layer):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = nn.Tanh

        self.layers = nn.Sequential()

        for i in range(self.depth - 1):
            self.layers.add_sublayer('layer_%d' % i, nn.Linear(layers[i], layers[i + 1]))
            self.layers.add_sublayer('activation_%d' % i, self.activation())

        self.layers.add_sublayer('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        # layer_list = list()
        # for i in range(self.depth - 1):
        #     layer_list.append(
        #         ('layer_%d' % i, nn.Linear(layers[i], layers[i + 1]))
        #     )
        #     layer_list.append(('activation_%d' % i, self.activation()))
        #
        # layer_list.append(
        #     ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        # )
        # layerDict = OrderedDict(layer_list)
        #
        # # deploy layers
        # self.layers = nn.LayerDict(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
net1 = DNN(layers)
net2 = paddle.nn.Sequential(
    ('l1', paddle.nn.Linear(2, 20)),
    ('ac1', paddle.nn.Tanh()),
    ('l2', paddle.nn.Linear(20, 1))
)
# lambda_1 = paddle.create_parameter(
#     shape=[1],
#     dtype='float32',
#     default_initializer=nn.initializer.Constant(value=0.0))
#
# lambda_2 = paddle.create_parameter(
#     shape=[1],
#     dtype='float32',
#     default_initializer=nn.initializer.Constant(value=-6.0))
# net.add_parameter('lambda_1', lambda_1)
# net.add_parameter('lambda_2', lambda_2)

print(net1)
print(net2)
paddle.summary(net1, (2000, 2))
