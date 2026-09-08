#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, chamfer_dist
from utils.edge_support import aggregate_multiview_edge_support
from utils.growth_budget import (
    format_demand_preserve_diag,
    format_proximity_budget_log,
    format_proximity_capacity_log,
    format_proximity_replay_log,
    format_value_promotion_diag,
    format_value_rerank_diag,
    parse_proximity_action_replay,
    requires_value_features,
    select_proximity_sources,
    resolve_proximity_selection_mode,
)
from utils.growth_diagnostics import (
    GrowthDiagnostics,
    count_proximity_proposed,
    count_split_candidates,
    format_proximity_growth_log,
)
from utils.structural_graph import (
    build_knn_graph,
    compute_continuity_defect,
    compute_geometric_turning,
    compute_redundancy,
    estimate_gaussian_normals,
)
from utils.value_allocation import compute_obdkr_value, compute_structural_value_score
from utils.value_diagnostics import (
    compute_obdkr_diagnostics,
    compute_structural_value_attribution_stats,
    format_obdkr_diagnostics_log,
    format_structural_boundary_diag,
    format_structural_norm_diag,
    format_structural_promotion_attr,
)
from torch.optim.lr_scheduler import MultiStepLR


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self.init_point = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.bg_color = torch.empty(0)
        self.confidence = torch.empty(0)
        self.visibility_history = None
        self.visible_view_count = None
        self.recent_visibility_history = None
        self.recent_visible_view_count = None
        self.camera_uid_to_train_index = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._capture_visibility_state(),
        )

    def restore(self, model_args, training_args, restore_optimizer=False):
        visibility_state = None
        if len(model_args) == 13:
            (*model_args, visibility_state) = model_args
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self._restore_visibility_state(visibility_state)
        if restore_optimizer:
            self.optimizer.load_state_dict(opt_dict)

    def _capture_visibility_state(self):
        return {
            "visibility_history": self.visibility_history,
            "visible_view_count": self.visible_view_count,
        }

    def _restore_visibility_state(self, state):
        if not isinstance(state, dict):
            self.visibility_history = None
            self.visible_view_count = None
            self.recent_visibility_history = None
            self.recent_visible_view_count = None
            return
        self.visibility_history = state.get("visibility_history")
        self.visible_view_count = state.get("visible_view_count")
        self.recent_visibility_history = None
        self.recent_visible_view_count = None
        self._sync_visibility_shape(rebuild_missing=True)

    def ensure_visibility_history(self, num_views):
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        if self.visibility_history is None:
            self.visibility_history = torch.zeros((count, int(num_views)), dtype=torch.bool, device=device)
            self.visible_view_count = torch.zeros((count,), dtype=torch.long, device=device)
            self.recent_visibility_history = torch.zeros((count, int(num_views)), dtype=torch.bool, device=device)
            self.recent_visible_view_count = torch.zeros((count,), dtype=torch.long, device=device)
            return
        if self.visibility_history.shape[0] != count:
            raise ValueError("visibility_history first dimension must match current Gaussian count.")
        if self.visibility_history.shape[1] != int(num_views):
            raise ValueError("visibility_history second dimension must match training view count.")
        self.visibility_history = self.visibility_history.to(device=device, dtype=torch.bool)
        if self.visible_view_count is None or self.visible_view_count.shape[0] != count:
            self.visible_view_count = self.visibility_history.sum(dim=1).to(dtype=torch.long)
        else:
            self.visible_view_count = self.visible_view_count.to(device=device, dtype=torch.long)
        self._ensure_recent_visibility_history(num_views)

    def update_visibility(self, view_id, visible_mask):
        view_id = int(view_id)
        if self.visibility_history is None:
            self.ensure_visibility_history(view_id + 1)
        if view_id < 0 or view_id >= self.visibility_history.shape[1]:
            raise ValueError("view_id must be within visibility_history.")
        mask = visible_mask.reshape(-1).to(device=self.get_xyz.device, dtype=torch.bool)
        if mask.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("visible_mask length must match current Gaussian count.")
        previous = self.visibility_history[:, view_id]
        newly_visible = mask & ~previous
        self.visibility_history[:, view_id] = previous | mask
        if self.visible_view_count is None or self.visible_view_count.shape[0] != self.get_xyz.shape[0]:
            self.visible_view_count = self.visibility_history.sum(dim=1).to(dtype=torch.long)
        else:
            self.visible_view_count += newly_visible.to(dtype=self.visible_view_count.dtype)
        self._update_recent_visibility(view_id, mask)

    def _sync_visibility_shape(self, rebuild_missing=False):
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        if self.visibility_history is not None:
            self.visibility_history = self.visibility_history.to(device=device, dtype=torch.bool)
            if self.visibility_history.shape[0] != count:
                if rebuild_missing:
                    self.visibility_history = None
                    self.visible_view_count = None
                else:
                    raise ValueError("visibility_history is out of sync with Gaussian count.")
        if self.visible_view_count is not None:
            self.visible_view_count = self.visible_view_count.to(device=device, dtype=torch.long)
            if self.visible_view_count.shape[0] != count:
                if self.visibility_history is not None and self.visibility_history.shape[0] == count:
                    self.visible_view_count = self.visibility_history.sum(dim=1).to(dtype=torch.long)
                elif rebuild_missing:
                    self.visible_view_count = None
                else:
                    raise ValueError("visible_view_count is out of sync with Gaussian count.")
        if getattr(self, "recent_visibility_history", None) is not None:
            self.recent_visibility_history = self.recent_visibility_history.to(device=device, dtype=torch.bool)
            if self.recent_visibility_history.shape[0] != count:
                if rebuild_missing:
                    self.recent_visibility_history = None
                    self.recent_visible_view_count = None
                else:
                    raise ValueError("recent_visibility_history is out of sync with Gaussian count.")
        if getattr(self, "recent_visible_view_count", None) is not None:
            self.recent_visible_view_count = self.recent_visible_view_count.to(device=device, dtype=torch.long)
            if self.recent_visible_view_count.shape[0] != count:
                if self.recent_visibility_history is not None and self.recent_visibility_history.shape[0] == count:
                    self.recent_visible_view_count = self.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
                elif rebuild_missing:
                    self.recent_visible_view_count = None
                else:
                    raise ValueError("recent_visible_view_count is out of sync with Gaussian count.")

    def _ensure_recent_visibility_history(self, num_views):
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        num_views = int(num_views)
        if getattr(self, "recent_visibility_history", None) is None:
            self.recent_visibility_history = torch.zeros((count, num_views), dtype=torch.bool, device=device)
            self.recent_visible_view_count = torch.zeros((count,), dtype=torch.long, device=device)
            return
        if self.recent_visibility_history.shape[0] != count:
            raise ValueError("recent_visibility_history first dimension must match current Gaussian count.")
        if self.recent_visibility_history.shape[1] != num_views:
            raise ValueError("recent_visibility_history second dimension must match training view count.")
        self.recent_visibility_history = self.recent_visibility_history.to(device=device, dtype=torch.bool)
        if getattr(self, "recent_visible_view_count", None) is None or self.recent_visible_view_count.shape[0] != count:
            self.recent_visible_view_count = self.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
        else:
            self.recent_visible_view_count = self.recent_visible_view_count.to(device=device, dtype=torch.long)

    def _update_recent_visibility(self, view_id, mask):
        if getattr(self, "recent_visibility_history", None) is None:
            self._ensure_recent_visibility_history(self.visibility_history.shape[1])
        previous = self.recent_visibility_history[:, view_id]
        newly_visible = mask & ~previous
        self.recent_visibility_history[:, view_id] = previous | mask
        if getattr(self, "recent_visible_view_count", None) is None or self.recent_visible_view_count.shape[0] != self.get_xyz.shape[0]:
            self.recent_visible_view_count = self.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
        else:
            self.recent_visible_view_count += newly_visible.to(dtype=self.recent_visible_view_count.dtype)

    def reset_recent_visibility(self):
        if getattr(self, "recent_visibility_history", None) is None:
            if self.visibility_history is not None:
                self._ensure_recent_visibility_history(self.visibility_history.shape[1])
            return
        self.recent_visibility_history.zero_()
        if getattr(self, "recent_visible_view_count", None) is None or self.recent_visible_view_count.shape[0] != self.get_xyz.shape[0]:
            self.recent_visible_view_count = self.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
        else:
            self.recent_visible_view_count.zero_()

    def _append_visibility_from_masks(self, source_mask, repeat_count=1, target_indices=None):
        if self.visibility_history is None:
            return
        source_history = self.visibility_history[source_mask]
        if target_indices is not None:
            target_history = self.visibility_history[target_indices]
            source_history = source_history[:, None, :].repeat(1, repeat_count, 1).reshape(-1, self.visibility_history.shape[1])
            source_history = source_history | target_history
        elif repeat_count != 1:
            source_history = source_history.repeat(repeat_count, 1)
        self.visibility_history = torch.cat((self.visibility_history, source_history), dim=0)
        new_counts = source_history.sum(dim=1).to(dtype=torch.long)
        if self.visible_view_count is None:
            self.visible_view_count = self.visibility_history.sum(dim=1).to(dtype=torch.long)
        else:
            self.visible_view_count = torch.cat((self.visible_view_count, new_counts), dim=0)
        if getattr(self, "recent_visibility_history", None) is not None:
            recent_source_history = self.recent_visibility_history[source_mask]
            if target_indices is not None:
                recent_target_history = self.recent_visibility_history[target_indices]
                recent_source_history = recent_source_history[:, None, :].repeat(
                    1, repeat_count, 1
                ).reshape(-1, self.recent_visibility_history.shape[1])
                recent_source_history = recent_source_history | recent_target_history
            elif repeat_count != 1:
                recent_source_history = recent_source_history.repeat(repeat_count, 1)
            self.recent_visibility_history = torch.cat((self.recent_visibility_history, recent_source_history), dim=0)
            recent_new_counts = recent_source_history.sum(dim=1).to(dtype=torch.long)
            if getattr(self, "recent_visible_view_count", None) is None:
                self.recent_visible_view_count = self.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
            else:
                self.recent_visible_view_count = torch.cat((self.recent_visible_view_count, recent_new_counts), dim=0)

    def _prune_visibility(self, valid_points_mask):
        if self.visibility_history is not None:
            self.visibility_history = self.visibility_history[valid_points_mask]
        if self.visible_view_count is not None:
            self.visible_view_count = self.visible_view_count[valid_points_mask]
        if getattr(self, "recent_visibility_history", None) is not None:
            self.recent_visibility_history = self.recent_visibility_history[valid_points_mask]
        if getattr(self, "recent_visible_view_count", None) is not None:
            self.recent_visible_view_count = self.recent_visible_view_count[valid_points_mask]

    def _resolve_value_observation_count(self, lifetime_count, recent_count):
        observation_source = getattr(self.args, "value_observation_source", "lifetime")
        if observation_source == "lifetime":
            return lifetime_count, observation_source
        if observation_source == "recent":
            if recent_count is None:
                raise ValueError("recent observation source requested but recent visibility state is unavailable")
            return recent_count, observation_source
        raise ValueError("value_observation_source must be one of lifetime, recent")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        w = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))




    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        device = self.get_xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.args.train_bg:
            l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = self.xyz_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = xyz_lr
                return xyz_lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.05))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def dist_prune(self):
        dist = chamfer_dist(self.init_point, self._xyz)
        valid_points_mask = (dist < 3.0)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._prune_visibility(valid_points_mask)


    def prune_points(self, mask, iter):
        if iter > self.args.prune_from_iter:
            valid_points_mask = ~mask
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            self.confidence = self.confidence[valid_points_mask]
            self._prune_visibility(valid_points_mask)


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)


    def compute_fsgs_proximity_candidate_value(
        self,
        train_cameras=None,
        edge_maps=None,
        candidate_mask=None,
        iteration=None,
        knn_k=12,
        **kwargs,
    ):
        visible_count = self.visible_view_count
        if visible_count is None:
            visible_count = torch.zeros((self.get_xyz.shape[0],), dtype=torch.long, device=self.get_xyz.device)
        visible_count = visible_count.reshape(-1).to(device=self.get_xyz.device, dtype=torch.float32)
        if visible_count.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("visible_view_count must match current Gaussian count.")
        raw_recent_visible_count = getattr(self, "recent_visible_view_count", None)
        recent_visible_count = raw_recent_visible_count
        if recent_visible_count is not None:
            recent_visible_count = recent_visible_count.reshape(-1).to(device=self.get_xyz.device, dtype=torch.float32)
        if recent_visible_count is not None and recent_visible_count.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("recent_visible_view_count must match current Gaussian count.")
        value_score_variant = kwargs.get("value_score_variant", getattr(self.args, "value_score_variant", "obdkr"))
        if value_score_variant not in ("obdkr", "structural"):
            raise ValueError("value_score_variant must be one of obdkr, structural")
        observation_source = getattr(self.args, "value_observation_source", "lifetime")
        active_observation_count = None
        if value_score_variant == "obdkr":
            active_observation_count, observation_source = self._resolve_value_observation_count(
                visible_count,
                recent_visible_count,
            )

        if train_cameras is not None and edge_maps is not None:
            boundary = aggregate_multiview_edge_support(
                xyz=self.get_xyz.detach(),
                cameras=train_cameras,
                edge_maps=edge_maps,
                visibility_history=self.visibility_history,
                camera_uid_to_train_index=self.camera_uid_to_train_index,
            )
        else:
            boundary = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float32, device=self.get_xyz.device)

        neighbors = build_knn_graph(self.get_xyz.detach(), k=knn_k)
        normals = estimate_gaussian_normals(self.get_scaling.detach(), self._rotation.detach())
        turning = compute_geometric_turning(self.get_xyz.detach(), normals, neighbors)
        defect = compute_continuity_defect(self.get_xyz.detach(), normals, neighbors)
        redundancy = compute_redundancy(self.get_xyz.detach(), self.get_scaling.detach(), neighbors)
        value_kwargs = dict(
            w_b=kwargs.get("w_b", getattr(self.args, "value_w_b", 1.0)),
            w_k=kwargs.get("w_k", getattr(self.args, "value_w_k", 1.0)),
            w_d=kwargs.get("w_d", getattr(self.args, "value_w_d", 1.0)),
            lambda_r=kwargs.get("lambda_r", getattr(self.args, "value_lambda_r", 1.0)),
            low_quantile=kwargs.get("low_quantile", getattr(self.args, "normalization_low_quantile", 0.05)),
            high_quantile=kwargs.get("high_quantile", getattr(self.args, "normalization_high_quantile", 0.95)),
            normalization_mask=candidate_mask,
            return_components=True,
        )
        if value_score_variant == "obdkr":
            components = compute_obdkr_value(
                active_observation_count,
                boundary,
                turning,
                defect,
                redundancy,
                tau_e=kwargs.get("tau_e", getattr(self.args, "value_tau_e", 1.0)),
                tau_s=kwargs.get("tau_s", getattr(self.args, "value_tau_s", 3.0)),
                **value_kwargs,
            )
        else:
            components = compute_structural_value_score(
                boundary,
                turning,
                defect,
                redundancy,
                **value_kwargs,
            )
        if iteration is not None:
            print(
                format_obdkr_diagnostics_log(
                    iteration,
                    compute_obdkr_diagnostics(
                        components,
                        candidate_mask,
                        observation_count=visible_count,
                        recent_observation_count=(
                            recent_visible_count
                            if recent_visible_count is not None
                            else torch.zeros((self.get_xyz.shape[0],), dtype=torch.float32, device=self.get_xyz.device)
                        ),
                        active_observation_count=active_observation_count,
                        observation_source=observation_source,
                        value_score_variant=value_score_variant,
                        num_train_views=(
                            self.recent_visibility_history.shape[1]
                            if getattr(self, "recent_visibility_history", None) is not None
                            else None
                        ),
                    ),
                ),
                flush=True,
            )
        return components

    def proximity(self, scene_extent, iteration=None, N = 3, train_cameras=None, edge_maps=None):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))
        proximity_sources = int(selected_pts_mask.sum().item())
        proximity_candidate_mask = selected_pts_mask
        proximity_proposed = count_proximity_proposed(proximity_sources, N)
        print(
            format_proximity_growth_log(
                iteration=iteration,
                num_before=self.get_xyz.shape[0],
                proximity_sources=proximity_sources,
                proximity_proposed=proximity_proposed,
            ),
            flush=True,
        )
        value_score = None
        replay_schedule = parse_proximity_action_replay(getattr(self.args, "proximity_action_replay", ""))
        replay_enabled = bool(replay_schedule)
        proximity_selection_mode = resolve_proximity_selection_mode(
            getattr(self.args, "proximity_selection_mode", "original"),
            getattr(self.args, "enable_proximity_budget", False)
            or getattr(self.args, "enable_proximity_candidate_capacity", False)
            or replay_enabled,
        )
        if requires_value_features(proximity_selection_mode):
            value_components = self.compute_fsgs_proximity_candidate_value(
                train_cameras=train_cameras,
                edge_maps=edge_maps,
                candidate_mask=selected_pts_mask,
                iteration=iteration,
                tau_e=getattr(self.args, "value_tau_e", 1.0),
                tau_s=getattr(self.args, "value_tau_s", 3.0),
                w_b=getattr(self.args, "value_w_b", 1.0),
                w_k=getattr(self.args, "value_w_k", 1.0),
                w_d=getattr(self.args, "value_w_d", 1.0),
                lambda_r=getattr(self.args, "value_lambda_r", 1.0),
                low_quantile=getattr(self.args, "normalization_low_quantile", 0.05),
                high_quantile=getattr(self.args, "normalization_high_quantile", 0.95),
                value_score_variant=getattr(self.args, "value_score_variant", "obdkr"),
                knn_k=getattr(self.args, "knn_k", 12),
            )
            value_score = value_components["U"]
        selected_pts_mask, budget_stats = select_proximity_sources(
            selected_pts_mask,
            dist,
            n=N,
            rho=getattr(self.args, "proximity_growth_ratio", 0.10),
            enabled=getattr(self.args, "enable_proximity_budget", False),
            mode=proximity_selection_mode,
            value_score=value_score,
            rerank_fraction=getattr(self.args, "value_rerank_fraction", 0.25),
            boundary_multiplier=getattr(self.args, "value_boundary_multiplier", 2.0),
            demand_ratio=getattr(self.args, "value_demand_ratio", 0.90),
            candidate_capacity_enabled=getattr(self.args, "enable_proximity_candidate_capacity", False),
            candidate_keep_ratio=getattr(self.args, "proximity_candidate_keep_ratio", 0.80),
            replay_schedule=replay_schedule,
            iteration=iteration,
        )
        print(format_proximity_budget_log(iteration, budget_stats), flush=True)
        if budget_stats.capacity_active:
            print(format_proximity_capacity_log(iteration, budget_stats), flush=True)
        if budget_stats.replay_active:
            print(format_proximity_replay_log(iteration, budget_stats), flush=True)
        print(format_value_rerank_diag(iteration, budget_stats), flush=True)
        capacity_limited = budget_stats.budget_hit or budget_stats.capacity_hit or (
            budget_stats.replay_active and budget_stats.selected_src < budget_stats.candidates
        )
        if capacity_limited and budget_stats.mode == "value_rerank":
            budget_stats.proximity_values = dist
            budget_stats.value_values = value_score
            print(format_value_promotion_diag(iteration, budget_stats), flush=True)
        if capacity_limited and budget_stats.mode == "value_demand_rerank":
            budget_stats.proximity_values = dist
            budget_stats.value_values = value_score
            print(format_demand_preserve_diag(iteration, budget_stats), flush=True)
            if getattr(self.args, "value_score_variant", "obdkr") == "structural":
                structural_stats = compute_structural_value_attribution_stats(
                    iteration,
                    value_components,
                    proximity_candidate_mask,
                    budget_stats,
                )
                print(format_structural_norm_diag(structural_stats), flush=True)
                print(format_structural_boundary_diag(structural_stats), flush=True)
                print(format_structural_promotion_attr(structural_stats), flush=True)

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask][:, None, :].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self._append_visibility_from_masks(selected_pts_mask, repeat_count=N, target_indices=new_indices)
        return proximity_sources, proximity_proposed, budget_stats.selected_src, budget_stats.selected_new



    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)
        split_gradient_mask = selected_pts_mask

        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(dist > (self.args.dist_thres * scene_extent),
                                               torch.max(self.get_scaling, dim=1).values > ( scene_extent))
        split_sparse_mask = selected_pts_mask2
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)
        split_stats = count_split_candidates(split_gradient_mask, split_sparse_mask, selected_pts_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self._append_visibility_from_masks(selected_pts_mask, repeat_count=N)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)
        return split_stats


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        clone_candidates = int(selected_pts_mask.sum().item())

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)
        self._append_visibility_from_masks(selected_pts_mask)
        return clone_candidates


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter, train_cameras=None, edge_maps=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        growth_diag = GrowthDiagnostics(iteration=iter, num_before=self.get_xyz.shape[0])

        growth_diag.clone_candidates = self.densify_and_clone(grads, max_grad, extent)
        growth_diag.num_after_clone = self.get_xyz.shape[0]
        (
            growth_diag.split_gradient_candidates,
            growth_diag.split_sparse_candidates,
            growth_diag.split_total_candidates,
        ) = self.densify_and_split(grads, max_grad, extent, iter)
        growth_diag.num_after_split = self.get_xyz.shape[0]
        if iter < 2000:
            (
                growth_diag.proximity_sources,
                growth_diag.proximity_proposed,
                growth_diag.proximity_selected_sources,
                growth_diag.proximity_selected_new,
            ) = self.proximity(extent, iteration=iter, train_cameras=train_cameras, edge_maps=edge_maps)
        growth_diag.num_after_proximity = self.get_xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask, iter)
        growth_diag.num_after_prune = self.get_xyz.shape[0]
        print(growth_diag.format_log())
        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
