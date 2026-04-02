"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .criterion import ECCriterion
from .decoder import ECTransformer
from .ecvit import ViTAdapter
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .modeling import ECDet, ECSeg
from .postprocessor import PostProcessor
