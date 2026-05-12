import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterable, Optional, Callable, Tuple
from torch import nn

"""
    我们建议使用create_optimizer_lars函数并设置bn_bias_separately=True，
    而不是直接使用Lars类，这有助于LARS跳过BatchNormalization和偏置参数，
    通常能获得更好的性能表现。
    多项式预热学习率衰减对提高整体性能也很有帮助。
"""


def create_optimizer_lars(model, lr, momentum, weight_decay, bn_bias_separately, epsilon):
    """
    创建LARS优化器
    
    参数:
        model: 模型
        lr: 学习率
        momentum: 动量
        weight_decay: 权重衰减
        bn_bias_separately: 是否对BN层和偏置参数单独处理
        epsilon: 数值稳定性参数
    """
    if bn_bias_separately:
        optimizer = Lars([
            dict(params=get_common_parameters(model, exclude_func=get_norm_bias_parameters)),
            dict(params=get_norm_bias_parameters(model), weight_decay=0, lars=False)],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon)
    else:
        optimizer = Lars(model.parameters(),
                         lr=lr,
                         momentum=momentum,
                         weight_decay=weight_decay,
                         epsilon=epsilon)
    return optimizer


class Lars(Optimizer):
    r"""实现LARS优化器，来自论文 "Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>。
    
    参数:
        params (iterable): 要优化的参数迭代器或定义参数组的字典
        lr (float, 可选): 学习率
        momentum (float, 可选): 动量因子 (默认: 0)
        eeta (float, 可选): LARS系数，用于论文中 (默认: 1e-3)
        weight_decay (float, 可选): 权重衰减 (L2惩罚) (默认: 0)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr=1e-3,
            momentum=0,
            eeta=1e-3,
            weight_decay=0,
            epsilon=0.0
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("无效的学习率: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("无效的动量值: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("无效的权重衰减值: {}".format(weight_decay))
        if eeta <= 0:
            raise ValueError("无效的eeta值: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("无效的epsilon值: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True)

        super().__init__(params, defaults)

    def set_decay(self,weight_decay):
        """设置权重衰减值"""
        for group in self.param_groups:
            group['weight_decay'] = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化。
        
        参数:
            closure (callable, 可选): 重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            lars = group['lars']
            eps = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                if lars:
                    # 计算权重范数和梯度范数
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(p.grad)
                    # 计算LARS信任比率
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    trust_ratio.clamp_(0.0, 50)
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad = decayed_grad.add(p, alpha=weight_decay)
                # 梯度裁剪，防止梯度爆炸
                decayed_grad = torch.clamp(decayed_grad, -10.0, 10.0)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf

                p.add_(decayed_grad, alpha=-scaled_lr)

        return loss


"""
    帮助跳过偏置和BatchNorm层的辅助函数
"""
BN_CLS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    """从指定类型的模块中获取参数"""
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_norm_parameters(module):
    """获取所有标准化层的参数"""
    return get_parameters_from_cls(module, (nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    """获取所有偏置参数，可选择排除某些参数"""
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and 'bias' in name:
            yield param


def get_norm_bias_parameters(module):
    """获取所有标准化层和偏置参数"""
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_func=None):
    """获取除排除参数外的所有参数"""
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters:
            yield param