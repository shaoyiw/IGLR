import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align
from mmcv.cnn import ConvModule

from .is_model import ISModel
from isegm.model.modifiers import LRMult
from isegm.model.ops import DistMaps
from isegm.model.ops import ScaleLayer
from isegm.utils.serialization import serialize
from .modeling.segformer.segformer_model import SegFormer
from .modeling.segformer.Refiner import Refiner


class SegFormerModel(ISModel):
    @serialize
    def __init__(self, feature_stride=4, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, pipeline_version='s1', model_version='b0',
                 **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.pipeline_version = pipeline_version
        self.model_version = model_version
        self.feature_extractor = SegFormer(self.model_version)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))

        if self.pipeline_version == 's1':
            base_radius = 3
        else:
            base_radius = 5

        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                       cpu_mode=False, use_disks=True)

        self.dist_maps_refine = DistMaps(norm_radius=5, spatial_scale=1.0,
                                         cpu_mode=False, use_disks=True)

        if self.model_version == 'b0':
            feature_indim = 256
        else:
            feature_indim = 512
        self.refiner = Refiner(feature_dim=feature_indim)

        mt_layers = [
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            ScaleLayer(init_value=0.05, lr_mult=1)
        ]
        self.maps_transform = nn.Sequential(*mt_layers)

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps_base(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    def backbone_forward(self, image, side_feature, gate):
        mask, feature = self.feature_extractor(image, side_feature, gate)
        return {'instances': mask, 'instances_aux': mask, 'feature': feature}

    def refine(self, cropped_image, cropped_points, full_feature, full_logits, bboxes):
        '''
        bboxes : [b,5]
        '''
        h1 = cropped_image.shape[-1]
        h2 = full_feature.shape[-1]
        r = h1 / h2

        cropped_feature = roi_align(full_feature, bboxes, full_feature.size()[2:], spatial_scale=1 / r, aligned=True)
        cropped_logits = roi_align(full_logits, bboxes, cropped_image.size()[2:], spatial_scale=1, aligned=True)
        click_map = self.dist_maps_refine(cropped_image, cropped_points)
        refined_mask, trimap = self.refiner(cropped_image, click_map, cropped_feature, cropped_logits)
        return {'instances_refined': refined_mask, 'trimap': trimap, 'instances_coarse': cropped_logits}

    def forward(self, image, points, gate):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:, 1:, :, :]

        if self.pipeline_version == 's1':
            small_image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=True)
            small_coord_features = F.interpolate(coord_features, scale_factor=0.5, mode='bilinear', align_corners=True)
        else:
            small_image = image
            small_coord_features = coord_features

        # small_coord_features = self.maps_transform(small_coord_features)
        outputs = self.backbone_forward(small_image, small_coord_features, gate)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                                 mode='bilinear', align_corners=True)

        return outputs

