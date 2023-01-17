import torch
import torch.nn as nn
import numpy as np


class LRMult(object):
    def __init__(self, lr_mult=1.):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult
            
class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from lib.supporters.pointwise.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor

class ISModel(nn.Module):
    def __init__(self, use_rgb_conv=True, with_aux_output=False,
                 norm_radius=260, use_disks=False, cpu_dist_maps=False,
                 clicks_groups=None, with_prev_mask=False, use_leaky_relu=False,
                 binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(in_channels=3 + self.coord_feature_ch, out_channels=6 + self.coord_feature_ch, kernel_size=1),
                norm_layer(6 + self.coord_feature_ch),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1)
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=64,
                                            kernel_size=3, stride=2, padding=1)
            self.maps_transform.apply(LRMult(0.1))
        else:
            self.rgb_conv = None
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)

        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(DistMaps(norm_radius=click_radius, spatial_scale=1.0,
                                               cpu_mode=cpu_dist_maps, use_disks=use_disks))
        else:
            self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                      cpu_mode=cpu_dist_maps, use_disks=use_disks)

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
            outputs = self.backbone_forward(x)
        else:
            coord_features = self.maps_transform(coord_features)
            outputs = self.backbone_forward(image, coord_features)

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(points, groups=(2,) + (1, ) * (len(self.clicks_groups) - 2) + (-1,))
            coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            coord_features = torch.cat(coord_features, dim=1)
        else:
            coord_features = self.dist_maps(image, points)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
