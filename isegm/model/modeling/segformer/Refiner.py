import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.efficientnet_blocks import DepthwiseSeparableConv


class AttentionGate(nn.Module):
    """
    注意力门模块 (Attention Gate)

    该模块接收两种特征图：
    - g: Gating signal (门控信号)，通常来自较粗糙的尺度，具有较强的语义信息或空间引导信息。
    - x: Input signal (输入信号)，通常来自较精细的尺度，包含丰富的细节特征。

    注意力门会生成一个注意力系数图 (alpha)，用它来加权输入信号 x，从而突出与 g 相关的区域，
    抑制不相关的区域。
    """

    def __init__(self, F_g, F_x, F_int):
        """
        Args:
            F_g (int): 门控信号 g 的通道数。
            F_x (int): 输入信号 x 的通道数。
            F_int (int): 中间特征的通道数。
        """
        super(AttentionGate, self).__init__()

        # 对门控信号 g 进行 1x1 卷积，使其通道数与中间层匹配
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 对输入信号 x 进行 1x1 卷积，使其通道数与中间层匹配
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 生成注意力系数图
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g (torch.Tensor): 门控信号张量, shape [B, F_g, H, W]
            x (torch.Tensor): 输入信号张量, shape [B, F_x, H, W]
        Returns:
            torch.Tensor: 经过注意力加权后的输出张量, shape [B, F_x, H, W]
        """
        # g 和 x 应该具有相同的空间维度 (H, W)
        g1 = self.W_g(g)  # [B, F_int, H, W]
        x1 = self.W_x(x)  # [B, F_int, H, W]

        # 将处理后的 g 和 x 相加，并通过 ReLU 激活
        psi = self.relu(g1 + x1)

        # 通过 psi 层生成单通道的注意力图 (alpha)
        alpha = self.psi(psi)  # [B, 1, H, W]

        # 将注意力图 alpha 广播并乘以原始输入 x
        # 这会逐元素地对 x 的每个通道进行加权

        return x * alpha


# --- 主模型 ---
class Refiner(nn.Module):
    """
    基于注意力融合的局部优化网络
    """

    def __init__(self, feature_dim, image_channels=3, click_channels=2, mask_channels=1,
                 spatial_out_channels=32, feat_out_channels=128, mid_channels=64, num_classes=1):
        super(Refiner, self).__init__()

        # --- 1. 空间引导分支 (Spatial Guidance Branch) ---
        # 输入: 图像、粗糙掩码、点击图
        # 输出: 空间引导特征 (作为 Attention Gate 的门控信号 g)
        spatial_in_channels = image_channels + mask_channels + click_channels
        self.spatial_branch = nn.Sequential(
            DepthwiseSeparableConv(spatial_in_channels, spatial_out_channels, dw_kernel_size=3, stride=2),
            DepthwiseSeparableConv(spatial_out_channels, spatial_out_channels, dw_kernel_size=3)
        )

        # --- 2. 语义特征分支 (Semantic Feature Branch) ---
        # 输入: SegFormer 解码器特征
        # 输出: 经过降维和对齐的语义特征 (作为 Attention Gate 的输入信号 x)
        self.feature_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feat_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_out_channels),
            nn.ReLU(inplace=True)
        )

        # --- 3. 注意力门 ---
        # 用空间特征(g)去指导语义特征(x)
        self.attention_gate = AttentionGate(
            F_g=spatial_out_channels,
            F_x=feat_out_channels,
            F_int=feat_out_channels // 2  # 中间通道数通常设为 F_x 的一半
        )

        # --- 4. 最终的优化网络 (Refinement Network) ---
        # 输入: 原始空间特征 和 经过注意力门加权后的语义特征
        fusion_in_channels = spatial_out_channels + feat_out_channels
        self.refinement_pred = nn.Sequential(
            DepthwiseSeparableConv(fusion_in_channels, mid_channels, dw_kernel_size=3),
            DepthwiseSeparableConv(mid_channels, mid_channels, dw_kernel_size=3),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

        self.refinement_boundary = nn.Sequential(
            DepthwiseSeparableConv(fusion_in_channels, mid_channels, dw_kernel_size=3),
            DepthwiseSeparableConv(mid_channels, mid_channels, dw_kernel_size=3),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, image, click_map, feature, logits):
        mask = torch.sigmoid(logits)
        # 1. 分别通过两个分支处理输入
        spatial_input = torch.cat([image, click_map, mask], dim=1)
        spatial_features_g = self.spatial_branch(spatial_input)  # 门控信号 g

        semantic_features_x = self.feature_branch(feature)  # 输入信号 x

        semantic_features_x = F.interpolate(semantic_features_x, size=spatial_features_g.size()[2:], mode='bilinear',
                                            align_corners=True)
        # 2. 使用注意力门进行特征筛选
        gated_semantic_features = self.attention_gate(g=spatial_features_g, x=semantic_features_x)

        # 3. 拼接特征
        # 将原始的空间引导特征与被注意力加权后的语义特征拼接起来
        fused_features = torch.cat([spatial_features_g, gated_semantic_features], dim=1)

        # 4. 通过最终的优化网络得到结果
        refined_mask = self.refinement_pred(fused_features)
        refined_boundary = self.refinement_boundary(fused_features)

        # 5. 恢复原始分辨率
        refined_mask = F.interpolate(refined_mask, size=image.size()[2:], mode='bilinear',
                                     align_corners=True)
        refined_boundary = F.interpolate(refined_boundary, size=image.size()[2:], mode='bilinear',
                                         align_corners=True)

        refined_boundary_sig = torch.sigmoid(refined_boundary)

        refined_pred = refined_mask * refined_boundary_sig + logits * (1 - refined_boundary_sig)

        return refined_pred, refined_boundary
