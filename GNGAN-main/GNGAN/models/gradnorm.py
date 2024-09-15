import torch


def normalize_gradient(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    # 当你将一个张量的requires_grad设置为True时，表示你希望跟踪对该张量的操作，以便在反向传播过程中计算梯度。这通常用于执行基于梯度的优化，例如训练神经网络
    x.requires_grad_(True)
    # 
    f = net_D(x, **kwargs)

    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)

    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])

    f_hat = (f / (grad_norm + torch.abs(f)))  # torch.abs() 是 PyTorch 中的一个函数，用于计算张量中元素的绝对值。
    return f_hat
