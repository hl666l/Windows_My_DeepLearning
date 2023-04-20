from mxnet import autograd, nd
import mxnet
"""
定义变量 --> 声明要计算梯度的变量 --> 书写计算式 -->backward() 计算梯度
"""
a = nd.array([1, 2, 3], ctx=mxnet.gpu())
print(type(a))


# a = nd.arange(16).reshape((4, 4))
# print(a)
# b = nd.arange(4).reshape(4, 1)
# print(type((a)))

# b.attach_grad()  # 告诉系统我们需要对b求导，系统给我们分配内存存储b的导数
# a.attach_grad()  # 告诉系统我们需要对a求导，系统给我们分配内存存储a的导数
#
# with autograd.record():  # 在autograd.record()作用域中定义函数f(x)，便于存储f(x)，从而计算梯度
#     A = 2 * (a.T * a)
#     C = 2 * (a * b)
#
# A.backward()  # 执行反向传播，并计算梯度
# C.backward()  # 执行反向传播，并计算梯度
# """
# 由于 A，C函数中都有a所以A计算完a的梯度后存在a.grad中，C计算完a,b的梯度后，覆盖了A计算的a的梯度
# """
# print(a.grad, b.grad)
#
# with autograd.record():
#     E = 2 * (b.T * b)
# E.backward()
# print(b.grad)












