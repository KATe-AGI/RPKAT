'''
Build RPKAT
'''
import torch
import torch.nn as nn
from timm.models.registry import register_model
from .RPKAT import RPKATransformer

# Configuration for RPKAT model
RPKAT_cfg = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 384,
    'depths': 12,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'drop_path_rate': 0.0,
    'use_kan': True,
    'layer_scale': 1.0,
}

@register_model
def RPKAT(num_classes=1000, pretrained=False, pretrained_dataset=None, pretrained_cfg=None, model_cfg=RPKAT_cfg, **kwargs):
    model = RPKATransformer(
        num_classes=num_classes,
        **model_cfg,
        **kwargs
    )
    
    return model
