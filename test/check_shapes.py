import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ssdlite320.model import SSD320, MobileNet

m = SSD320(backbone=MobileNet(backbone='mobilenetv4', weights='IMAGENET1K_V1'))
m.eval()

# 构造虚拟输入
img = torch.randn(1, 3, 320, 320)

# 提取 backbone 特征并构造检测用特征列表
detection_feed = m.get_detection_features(img)

# 输出每层特征图尺寸和对应先验框数
print("特征层数:", len(detection_feed))
print("每层尺寸与先验框数:")
total_anchors = 0
for i, feat in enumerate(detection_feed):
    _, c, h, w = feat.shape
    nd = m.num_defaults[i] if i < len(m.num_defaults) else None
    layer_anchors = h * w * (nd if nd is not None else 0)
    print(f" 层{i}: 尺寸={h}x{w}, 通道={c}, 每位置先验框数={nd}, 层先验框总数={layer_anchors}")
    total_anchors += layer_anchors

print("总先验框数:", total_anchors)

# 验证模型返回的 loc 输出形状
locs = m(img)[0]
print("模型 loc 输出形状:", locs.shape)
