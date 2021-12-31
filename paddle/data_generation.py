import paddle

x = paddle.randn((20, 2, 500, 250), dtype="float32")
y = paddle.randn((20, 2, 500, 250), dtype="float32")
print(x)
