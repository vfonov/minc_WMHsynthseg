#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer,LoraInjectedConv3d,LoraInjectedConv2d,Linear


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def load_lora_state_dict(model: nn.Module, loaded_state_dict) -> None:
    for k in model.state_dict().keys():
        if k in loaded_state_dict.keys():
            model.state_dict()[k].copy_(loaded_state_dict[k])



def add_lora(model, device='cuda',
                    r: int = 4,
                    dropout_p: float = 0.1,
                    scale: float = 1.0,
                    **kwargs):
    """
    Apply LoRA to the model
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            add_lora(module, device=device,r=r,dropout_p=dropout_p,scale=scale,**kwargs)
        elif isinstance(module, nn.Conv3d):
            if min(module.in_channels,module.out_channels)>r:
                new=LoraInjectedConv3d(in_channels=module.in_channels, 
                                   out_channels=module.out_channels,
                                   kernel_size=module.kernel_size,
                                   stride=module.stride,padding=module.padding,
                                   dilation=module.dilation,groups=module.groups,
                                   bias=module.bias is not None,
                                   r=r,dropout_p=dropout_p,scale=scale,
                                   **kwargs)
                new.to(device)
                new.conv.weight.data = module.weight.data.to(device)
                new.conv.weight.requires_grad = False # Freeze base layer
                if module.bias is not None:
                    new.conv.bias.data = module.bias.data.to(device)
                    new.conv.bias.requires_grad = False

                setattr(model, n, new)
            #del module # ?
        elif isinstance(module, nn.Conv2d):
            if min(module.in_channels,module.out_channels)>r:
                new=LoraInjectedConv2d(in_channels=module.in_channels, 
                                    out_channels=module.out_channels,
                                    kernel_size=module.kernel_size,
                                    stride=module.stride,padding=module.padding,
                                    dilation=module.dilation,groups=module.groups,
                                    bias=module.bias is not None,
                                    r=r,dropout_p=dropout_p,scale=scale,
                                    **kwargs)
                new.to(device)
                new.conv.weight.data = module.weight.data
                new.conv.weight.requires_grad = False # Freeze base layer
                if module.bias is not None:
                    new.conv.bias.data = module.bias.data
                    new.conv.bias.requires_grad = False
                setattr(model, n, new)
        
        elif isinstance(module, nn.Linear):
            new=Linear(in_features=module.in_features, 
                                out_features=module.out_features,
                                bias=module.bias is not None,
                                device=device,
                                **kwargs).to(device)
            
            new.linear.weight.data = module.weight.data
            if module.bias is not None:
                new.linear.bias.data = module.bias.data
            setattr(model, n, new)
            #del module # ?
        # else skip
            