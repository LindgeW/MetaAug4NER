import torch


# Gradient Reversal Layer
class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lmbd=0.01):
        ctx.lmbd = torch.tensor(lmbd)
        return x.reshape_as(x)

    @staticmethod
    # 输入为forward输出求得的梯度
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.lmbd * grad_input.neg(), None
