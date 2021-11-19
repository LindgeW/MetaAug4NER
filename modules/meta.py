from itertools import chain
import torch
from torch import nn, Tensor
from typing import List, Union, Any, Iterable
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm


class MetaModule(nn.Module):
    """
    This is a subclass of nn.Module designed for Meta-Learning.
    The key point of this class is that we will do some operations on nn.Parameter directly with gradient,
    but after doing some operation like `param - lr * grad`, the type of `param` will change from `nn.Parameter`
    to 'torch.Tensor'. If we re-cast the tensor to nn.Parameter, the grad will disappear.
    Currently, the only solution in pytroch frame is to cast the type of all parameters from `nn.Parameter` to `autograd.Variable`
    with `require_grad=True` and then call `register_buffer` function to regist the tensor so that we can use it as normal.
    Expample:
    >>> class Net(nn.Module):
    >>>     ...
    >>> class MetaNet(Net,MetaModule):
    >>>     pass # just extends two class to create MetaNet
    """

    _ignore_buffers = {

    }
    _ignore_modules = {
        _BatchNorm
    }

    def set_params(self, params: Iterable[torch.Tensor]):
        """
        set params' value with order
        """
        for piter, param in zip(MetaModule._name_params(self, with_module=True),
                                params):  # type: Any,torch.Tensor
            name, val, mmodule = piter  # type: str,nn.Parameter,nn.Module
            if param is None:
                continue
            if not isinstance(param, torch.autograd.Variable):
                param = torch.autograd.Variable(param, requires_grad=True)
            val.detach()
            # mmodule.register_buffer(name, param)
            setattr(mmodule, name, param)

    def update_params(self, lr: float, grads: Iterable[torch.Tensor]):
        """
        `param - lr*grad` param by param
        """
        nparams = []
        for param, grad in zip(self.params(), grads):
            if grad is None:
                nparams.append(None)
            else:
                nparams.append(param - lr * grad)
        self.set_params(nparams)

    def name_params(self, with_module=False):
        for val in MetaModule._name_params(self, with_module):
            yield val

    def params(self):
        for _, val in self.name_params(with_module=False):
            yield val

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        super().__setattr__(name, value)

        if isinstance(value, nn.Module):
            MetaModule._move_params_to_buffer(self)

    @staticmethod
    def _move_params_to_buffer(value: nn.Module):
        """
        cast all params' type from `Paramter` to `Variable`
        """
        od = OrderedDict()
        for k, v in value._parameters.items():  # type:str,nn.Parameter
            if v is None:
                od[k] = None
            else:
                od[k] = torch.autograd.Variable(v.data, requires_grad=True)
        for k in od:
            value._parameters.pop(k)
        for k, v in od.items():
            value.register_buffer(k, v)

        for v in value.children():
            MetaModule._move_params_to_buffer(v)

    @staticmethod
    def _name_params(module: nn.Module, with_module=False, prefix=''):
        """yield all params with raw name(without module prefix)"""
        memo = set()
        for mname, mmodule in module.named_children():
            if mmodule == module:
                continue
            for name, val, mmmodule in MetaModule._name_params(mmodule, with_module=True, prefix=prefix):
                whole_name = '.'.join([prefix, mname, name]).lstrip('.')
                memo.add(whole_name)

                # the name is reletive to module, cause when set or update params, setattr(module, name, val) is called.
                if with_module:
                    yield name, val, mmmodule
                else:
                    yield name, val

        # In MetaModule, there will be no Paramter, all Paramters will be cast to `autograd.Variable` and be registed in
        # buffers, so we only yield `named_buffers` without `named_parameters`.
        for name, val in chain(module.named_buffers(recurse=False)):
            if name in memo:
                continue

            if any([isinstance(module, cls) for cls in MetaModule._ignore_modules]):
                continue

            ignore_names = {}
            for cls in MetaModule._ignore_buffers:
                if isinstance(module, cls):
                    ignore_names = MetaModule._ignore_buffers[cls]
                    break

            pure_name = name.split('.')[-1]
            if pure_name in ignore_names:
                continue

            if with_module:
                yield pure_name, val, module
            else:
                yield pure_name, val

    @staticmethod
    def _params(module: nn.Module):
        """yield all params with raw name(without module prefix)"""
        for _, val in MetaModule._name_params(module):
            yield val

    def zero_grad(self) -> None:
        super(MetaModule, self).zero_grad()
        for name, p in self.named_buffers():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
