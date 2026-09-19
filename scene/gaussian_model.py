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
import csv
import json
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
    format_value_timing_log,
    format_value_promotion_diag,
    format_value_rerank_diag,
    parse_proximity_action_replay,
    requires_value_features,
    resolve_value_timing_selection_mode,
    select_proximity_sources,
    resolve_proximity_selection_mode,
)
from utils.growth_diagnostics import (
    GrowthDiagnostics,
    compute_child_structure_diagnostics,
    count_proximity_proposed,
    count_split_candidates,
    format_child_structure_diag,
    format_child_target_selection_log,
    format_proximity_growth_log,
    select_structural_child_targets,
)
from utils.structural_graph import (
    build_knn_graph,
    compute_continuity_defect,
    compute_geometric_turning,
    compute_good_continuation_edge_scores,
    compute_good_continuation_score,
    compute_redundancy,
    estimate_gaussian_normals,
)
from utils.value_allocation import (
    compute_balanced_gestalt_value_score,
    compute_defect_conditioned_gestalt_value_score,
    compute_gestalt_structural_value_score,
    compute_obdkr_value,
    compute_structural_value_score,
)
from utils.value_diagnostics import (
    compute_balanced_gestalt_diagnostics,
    compute_defect_gestalt_counterfactual_diagnostics,
    compute_defect_gestalt_diagnostics,
    compute_obdkr_diagnostics,
    compute_gestalt_value_diagnostics,
    compute_structural_value_attribution_stats,
    format_balanced_gestalt_diag,
    format_balanced_gestalt_promotion_diag,
    format_defect_gestalt_counterfactual_diag,
    format_defect_gestalt_diag,
    format_defect_gestalt_promotion_diag,
    format_gestalt_promotion_diag,
    format_gestalt_value_diag,
    format_obdkr_diagnostics_log,
    format_structural_boundary_diag,
    format_structural_norm_diag,
    format_structural_promotion_attr,
)
from torch.optim.lr_scheduler import MultiStepLR


RGG_BIRTH_TYPE_ROOT = 0
RGG_BIRTH_TYPE_CLONE = 1
RGG_BIRTH_TYPE_SPLIT = 2
RGG_BIRTH_TYPE_PROXIMITY = 3
RGG_BIRTH_TYPE_NAMES = {
    RGG_BIRTH_TYPE_ROOT: "root",
    RGG_BIRTH_TYPE_CLONE: "clone",
    RGG_BIRTH_TYPE_SPLIT: "split",
    RGG_BIRTH_TYPE_PROXIMITY: "proximity",
}
RGG_BIRTH_TYPE_BY_NAME = {name: value for value, name in RGG_BIRTH_TYPE_NAMES.items()}
RGG_H1_SNAPSHOT_AGES = (50, 100, 200)
RGG_H1_FUTURE_SNAPSHOT_AGES = (350, 550)
RGG_H1_ALL_SNAPSHOT_AGES = RGG_H1_SNAPSHOT_AGES + RGG_H1_FUTURE_SNAPSHOT_AGES


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
        self.rgg_enabled = bool(getattr(args, "enable_rgg_diagnostics", False))
        self.rgg_uid = torch.empty(0, dtype=torch.long)
        self.rgg_birth_iter = torch.empty(0, dtype=torch.long)
        self.rgg_source_uid = torch.empty(0, dtype=torch.long)
        self.rgg_target_uid = torch.empty(0, dtype=torch.long)
        self.rgg_generation = torch.empty(0, dtype=torch.long)
        self.rgg_birth_type = torch.empty(0, dtype=torch.long)
        self.rgg_next_uid = 0
        self.rgg_h1_registry = {}

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
        self._ensure_rgg_state(birth_iter=-1)
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

    def update_visibility(self, view_id, visible_mask, iteration=None, is_real_view=True):
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
        self._update_rgg_h1_real_evidence(view_id, mask, iteration=iteration, is_real_view=is_real_view)

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

    def _rgg_is_enabled(self):
        return bool(getattr(self, "rgg_enabled", False) or getattr(self.args, "enable_rgg_diagnostics", False))

    def _empty_rgg_tensor(self, device=None):
        if device is None:
            device = self.get_xyz.device if self.get_xyz.numel() else torch.device("cpu")
        return torch.empty((0,), dtype=torch.long, device=device)

    def _set_empty_rgg_state(self, device=None):
        self.rgg_uid = self._empty_rgg_tensor(device)
        self.rgg_birth_iter = self._empty_rgg_tensor(device)
        self.rgg_source_uid = self._empty_rgg_tensor(device)
        self.rgg_target_uid = self._empty_rgg_tensor(device)
        self.rgg_generation = self._empty_rgg_tensor(device)
        self.rgg_birth_type = self._empty_rgg_tensor(device)
        self.rgg_next_uid = 0
        self.rgg_h1_registry = {}

    def _initialize_rgg_roots(self, birth_iter=0):
        if not self._rgg_is_enabled():
            return
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        self.rgg_uid = torch.arange(count, dtype=torch.long, device=device)
        self.rgg_birth_iter = torch.full((count,), int(birth_iter), dtype=torch.long, device=device)
        self.rgg_source_uid = torch.full((count,), -1, dtype=torch.long, device=device)
        self.rgg_target_uid = torch.full((count,), -1, dtype=torch.long, device=device)
        self.rgg_generation = torch.zeros((count,), dtype=torch.long, device=device)
        self.rgg_birth_type = torch.full((count,), RGG_BIRTH_TYPE_ROOT, dtype=torch.long, device=device)
        self.rgg_next_uid = int(count)
        self.rgg_h1_registry = {}

    def _ensure_rgg_state(self, birth_iter=-1):
        if not self._rgg_is_enabled():
            self._set_empty_rgg_state()
            return
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        fields = (
            getattr(self, "rgg_uid", None),
            getattr(self, "rgg_birth_iter", None),
            getattr(self, "rgg_source_uid", None),
            getattr(self, "rgg_target_uid", None),
            getattr(self, "rgg_generation", None),
            getattr(self, "rgg_birth_type", None),
        )
        if any((not torch.is_tensor(field)) or field.shape[0] != count for field in fields):
            self._initialize_rgg_roots(birth_iter=birth_iter)
            return
        self.rgg_uid = self.rgg_uid.to(device=device, dtype=torch.long)
        self.rgg_birth_iter = self.rgg_birth_iter.to(device=device, dtype=torch.long)
        self.rgg_source_uid = self.rgg_source_uid.to(device=device, dtype=torch.long)
        self.rgg_target_uid = self.rgg_target_uid.to(device=device, dtype=torch.long)
        self.rgg_generation = self.rgg_generation.to(device=device, dtype=torch.long)
        self.rgg_birth_type = self.rgg_birth_type.to(device=device, dtype=torch.long)
        used_next = int(self.rgg_uid.max().item()) + 1 if count else 0
        self.rgg_next_uid = max(int(getattr(self, "rgg_next_uid", 0)), used_next)

    def _allocate_rgg_uids(self, count, device):
        start = int(getattr(self, "rgg_next_uid", 0))
        uids = torch.arange(start, start + int(count), dtype=torch.long, device=device)
        self.rgg_next_uid = start + int(count)
        return uids

    def _build_rgg_children_from_indices(self, source_indices, target_indices=None, birth_iter=0, birth_type="clone"):
        if not self._rgg_is_enabled():
            return None
        self._ensure_rgg_state()
        device = self.get_xyz.device
        source_indices = source_indices.reshape(-1).to(device=device, dtype=torch.long)
        child_count = source_indices.shape[0]
        birth_type_value = self._rgg_birth_type_value(birth_type)
        source_uid = self.rgg_uid[source_indices]
        source_generation = self.rgg_generation[source_indices]
        if target_indices is None:
            target_uid = torch.full((child_count,), -1, dtype=torch.long, device=device)
            generation = source_generation + 1
        else:
            target_indices = target_indices.reshape(-1).to(device=device, dtype=torch.long)
            if target_indices.shape[0] != child_count:
                raise ValueError("RGG target_indices length must match source_indices length.")
            target_uid = self.rgg_uid[target_indices]
            target_generation = self.rgg_generation[target_indices]
            generation = torch.maximum(source_generation, target_generation) + 1
        return {
            "uid": self._allocate_rgg_uids(child_count, device),
            "birth_iter": torch.full((child_count,), int(birth_iter), dtype=torch.long, device=device),
            "source_uid": source_uid.clone(),
            "target_uid": target_uid.clone(),
            "generation": generation.clone(),
            "birth_type": torch.full((child_count,), birth_type_value, dtype=torch.long, device=device),
        }

    def _append_rgg_metadata(self, child_metadata):
        if not self._rgg_is_enabled():
            return
        if child_metadata is None:
            raise ValueError("RGG metadata is required when RGG diagnostics are enabled.")
        expected = child_metadata["uid"].shape[0]
        if self.rgg_uid.shape[0] + expected != self.get_xyz.shape[0]:
            raise ValueError("RGG parent and child metadata counts must match Gaussian count.")
        if child_metadata["uid"].shape[0] != expected:
            raise ValueError("RGG child metadata count must match appended Gaussian count.")
        self.rgg_uid = torch.cat((self.rgg_uid, child_metadata["uid"].to(self.get_xyz.device, dtype=torch.long)), dim=0)
        self.rgg_birth_iter = torch.cat((self.rgg_birth_iter, child_metadata["birth_iter"].to(self.get_xyz.device, dtype=torch.long)), dim=0)
        self.rgg_source_uid = torch.cat((self.rgg_source_uid, child_metadata["source_uid"].to(self.get_xyz.device, dtype=torch.long)), dim=0)
        self.rgg_target_uid = torch.cat((self.rgg_target_uid, child_metadata["target_uid"].to(self.get_xyz.device, dtype=torch.long)), dim=0)
        self.rgg_generation = torch.cat((self.rgg_generation, child_metadata["generation"].to(self.get_xyz.device, dtype=torch.long)), dim=0)
        self.rgg_birth_type = torch.cat((self.rgg_birth_type, child_metadata["birth_type"].to(self.get_xyz.device, dtype=torch.long)), dim=0)
        self._register_rgg_h1_births(child_metadata)
        self._assert_rgg_aligned()

    def _prune_rgg_metadata(self, valid_points_mask):
        if not self._rgg_is_enabled():
            return
        valid_points_mask = valid_points_mask.to(device=self.rgg_uid.device, dtype=torch.bool)
        self.rgg_uid = self.rgg_uid[valid_points_mask]
        self.rgg_birth_iter = self.rgg_birth_iter[valid_points_mask]
        self.rgg_source_uid = self.rgg_source_uid[valid_points_mask]
        self.rgg_target_uid = self.rgg_target_uid[valid_points_mask]
        self.rgg_generation = self.rgg_generation[valid_points_mask]
        self.rgg_birth_type = self.rgg_birth_type[valid_points_mask]
        self._assert_rgg_aligned()

    def _assert_rgg_aligned(self):
        if not self._rgg_is_enabled():
            return
        count = self.get_xyz.shape[0]
        for field_name in ("rgg_uid", "rgg_birth_iter", "rgg_source_uid", "rgg_target_uid", "rgg_generation", "rgg_birth_type"):
            field = getattr(self, field_name)
            if field.shape[0] != count:
                raise ValueError(f"{field_name} first dimension must match Gaussian count.")

    def _rgg_birth_type_value(self, birth_type):
        if isinstance(birth_type, str):
            if birth_type not in RGG_BIRTH_TYPE_BY_NAME:
                raise ValueError("unknown RGG birth type.")
            return RGG_BIRTH_TYPE_BY_NAME[birth_type]
        birth_type = int(birth_type)
        if birth_type not in RGG_BIRTH_TYPE_NAMES:
            raise ValueError("unknown RGG birth type.")
        return birth_type

    def _rgg_birth_type_name(self, birth_type):
        return RGG_BIRTH_TYPE_NAMES.get(int(birth_type), "unknown")

    def _new_rgg_h1_record(self, uid, birth_iter, source_uid, target_uid, generation, birth_type, birth_opacity=None):
        return {
            "uid": int(uid),
            "birth_iter": int(birth_iter),
            "source_uid": int(source_uid),
            "target_uid": int(target_uid),
            "generation": int(generation),
            "birth_type": self._rgg_birth_type_name(birth_type),
            "birth_opacity": float(birth_opacity) if birth_opacity is not None else None,
            "death_iter": None,
            "death_reason": None,
            "alive": True,
            "postbirth_real_opportunities": 0,
            "postbirth_real_visible_events": 0,
            "postbirth_unique_real_views": 0,
            "postbirth_real_view_ids": set(),
            "age_snapshots": {},
        }

    def _register_rgg_h1_births(self, child_metadata):
        if not self._rgg_is_enabled():
            return
        birth_types_cpu = child_metadata["birth_type"].detach().cpu()
        proximity_mask_cpu = birth_types_cpu == RGG_BIRTH_TYPE_PROXIMITY
        if not bool(proximity_mask_cpu.any().item()):
            return
        uids = child_metadata["uid"].detach().cpu()[proximity_mask_cpu].tolist()
        birth_iters = child_metadata["birth_iter"].detach().cpu()[proximity_mask_cpu].tolist()
        source_uids = child_metadata["source_uid"].detach().cpu()[proximity_mask_cpu].tolist()
        target_uids = child_metadata["target_uid"].detach().cpu()[proximity_mask_cpu].tolist()
        generations = child_metadata["generation"].detach().cpu()[proximity_mask_cpu].tolist()
        birth_types = birth_types_cpu[proximity_mask_cpu].tolist()
        child_count = int(birth_types_cpu.shape[0])
        opacity_count = int(self.get_opacity.shape[0])
        birth_opacities = [None] * len(uids)
        if child_count <= opacity_count:
            child_opacities_cpu = self.get_opacity[-child_count:].detach().reshape(-1).cpu()
            birth_opacities = child_opacities_cpu[proximity_mask_cpu].tolist()
        for uid, birth_iter, source_uid, target_uid, generation, birth_type_value, birth_opacity in zip(
            uids,
            birth_iters,
            source_uids,
            target_uids,
            generations,
            birth_types,
            birth_opacities,
        ):
            uid = int(uid)
            birth_iter = int(birth_iter)
            if birth_iter < 0:
                continue
            self.rgg_h1_registry[uid] = self._new_rgg_h1_record(
                uid,
                birth_iter,
                int(source_uid),
                int(target_uid),
                int(generation),
                birth_type_value,
                birth_opacity=birth_opacity,
            )

    def _update_rgg_h1_real_evidence(self, view_id, visible_mask, iteration=None, is_real_view=True):
        if not self._rgg_is_enabled() or not is_real_view:
            return
        if not getattr(self, "rgg_h1_registry", None):
            return
        self._assert_rgg_aligned()
        mask = visible_mask.reshape(-1).detach().to(device=self.rgg_uid.device, dtype=torch.bool)
        if mask.shape[0] != self.rgg_uid.shape[0]:
            raise ValueError("RGG post-birth visible mask length must match current Gaussian count.")
        active_indices = (self.rgg_birth_type == RGG_BIRTH_TYPE_PROXIMITY).nonzero(as_tuple=False).reshape(-1)
        active_uids = self.rgg_uid[active_indices].detach().cpu().tolist()
        active_visible = mask[active_indices].detach().cpu().tolist()
        active_opacities = self.get_opacity[active_indices].detach().reshape(-1).cpu().tolist()
        for uid, visible, opacity in zip(active_uids, active_visible, active_opacities):
            uid = int(uid)
            record = self.rgg_h1_registry.get(uid)
            if record is None or not record["alive"]:
                continue
            if iteration is not None and int(iteration) <= record["birth_iter"]:
                continue
            record["postbirth_real_opportunities"] += 1
            if bool(visible):
                record["postbirth_real_visible_events"] += 1
                record["postbirth_real_view_ids"].add(int(view_id))
                record["postbirth_unique_real_views"] = len(record["postbirth_real_view_ids"])
            self._maybe_record_rgg_h1_snapshots(record, iteration, opacity=opacity)

    def _maybe_record_rgg_h1_snapshots(self, record, iteration, opacity=None):
        if iteration is None:
            return
        age = int(iteration) - int(record["birth_iter"])
        if age < 0:
            return
        for snapshot_age in RGG_H1_ALL_SNAPSHOT_AGES:
            key = str(snapshot_age)
            if age >= snapshot_age and key not in record["age_snapshots"]:
                record["age_snapshots"][key] = {
                    "iteration": int(iteration),
                    "age": age,
                    "postbirth_real_opportunities": int(record["postbirth_real_opportunities"]),
                    "postbirth_real_visible_events": int(record["postbirth_real_visible_events"]),
                    "postbirth_unique_real_views": int(record["postbirth_unique_real_views"]),
                    "visible_rate": self._rgg_visible_rate(record),
                    "opacity": float(opacity) if opacity is not None else None,
                }

    def _rgg_visible_rate(self, record):
        opportunities = int(record["postbirth_real_opportunities"])
        if opportunities == 0:
            return 0.0
        return float(record["postbirth_real_visible_events"]) / float(opportunities)

    def _finalize_rgg_h1_pruned(self, pruned_mask, death_iter, death_reason):
        if not self._rgg_is_enabled() or not getattr(self, "rgg_h1_registry", None):
            return
        pruned_mask = pruned_mask.reshape(-1).to(device=self.rgg_uid.device, dtype=torch.bool)
        if pruned_mask.shape[0] != self.rgg_uid.shape[0]:
            raise ValueError("RGG prune mask length must match current Gaussian count.")
        pruned_indices = pruned_mask.nonzero(as_tuple=False).reshape(-1)
        pruned_uids = self.rgg_uid[pruned_indices].detach().cpu().tolist()
        for uid in pruned_uids:
            uid = int(uid)
            record = self.rgg_h1_registry.get(uid)
            if record is None:
                continue
            record["alive"] = False
            record["death_iter"] = int(death_iter) if death_iter is not None else -1
            record["death_reason"] = str(death_reason)

    def get_rgg_h1_records(self, observation_end_iter=None):
        records = []
        for uid in sorted(getattr(self, "rgg_h1_registry", {}).keys()):
            records.append(self._serialize_rgg_h1_record(self.rgg_h1_registry[uid], observation_end_iter=observation_end_iter))
        return records

    def _serialize_rgg_h1_record(self, record, observation_end_iter=None):
        output = dict(record)
        output["postbirth_real_view_ids"] = sorted(int(view_id) for view_id in record["postbirth_real_view_ids"])
        output["visible_rate"] = self._rgg_visible_rate(record)
        output["observation_end_iter"] = int(observation_end_iter) if observation_end_iter is not None else None
        output["age_at_export"] = None
        if output["observation_end_iter"] is not None and int(record["birth_iter"]) >= 0:
            output["age_at_export"] = output["observation_end_iter"] - int(record["birth_iter"])
        death_iter = record.get("death_iter")
        output["terminal_age"] = None
        if death_iter is not None and int(death_iter) >= 0 and int(record["birth_iter"]) >= 0:
            output["terminal_age"] = int(death_iter) - int(record["birth_iter"])
        output["age_snapshots"] = {
            str(age): dict(snapshot)
            for age, snapshot in record["age_snapshots"].items()
        }
        return output

    def _summarize_rgg_h1_records(self, records, observation_end_iter=None):
        summary = {
            "observation_end_iter": int(observation_end_iter) if observation_end_iter is not None else None,
            "total_proximity_records": len(records),
            "alive": 0,
            "training_prune": 0,
            "split_replaced": 0,
            "dist_prune": 0,
            "snapshot_50": 0,
            "snapshot_100": 0,
            "snapshot_200": 0,
            "snapshot_350": 0,
            "snapshot_550": 0,
        }
        for record in records:
            if record.get("alive"):
                summary["alive"] += 1
            death_reason = record.get("death_reason")
            if death_reason in ("training_prune", "split_replaced", "dist_prune"):
                summary[death_reason] += 1
            snapshots = record.get("age_snapshots", {})
            for age in RGG_H1_ALL_SNAPSHOT_AGES:
                if str(age) in snapshots:
                    summary[f"snapshot_{age}"] += 1
        return summary

    def save_rgg_diagnostics(self, directory, observation_end_iter=None):
        if not self._rgg_is_enabled():
            return
        mkdir_p(directory)
        records = self.get_rgg_h1_records(observation_end_iter=observation_end_iter)
        json_path = os.path.join(directory, "h1_proximity_postbirth_support.json")
        csv_path = os.path.join(directory, "h1_proximity_postbirth_support.csv")
        summary_path = os.path.join(directory, "h1_summary.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, sort_keys=True)
        summary = self._summarize_rgg_h1_records(records, observation_end_iter=observation_end_iter)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(
            "[RGGH1Summary] "
            f"end_iter={summary['observation_end_iter']} "
            f"total={summary['total_proximity_records']} "
            f"alive={summary['alive']} "
            f"training_prune={summary['training_prune']} "
            f"split_replaced={summary['split_replaced']} "
            f"dist_prune={summary['dist_prune']} "
            f"snap50={summary['snapshot_50']} "
            f"snap100={summary['snapshot_100']} "
            f"snap200={summary['snapshot_200']} "
            f"snap350={summary['snapshot_350']} "
            f"snap550={summary['snapshot_550']}",
            flush=True,
        )
        fieldnames = [
            "uid",
            "birth_iter",
            "source_uid",
            "target_uid",
            "generation",
            "birth_type",
            "birth_opacity",
            "death_iter",
            "death_reason",
            "alive",
            "postbirth_real_opportunities",
            "postbirth_real_visible_events",
            "postbirth_unique_real_views",
            "visible_rate",
            "observation_end_iter",
            "age_at_export",
            "terminal_age",
        ]
        for age in RGG_H1_ALL_SNAPSHOT_AGES:
            for suffix in (
                "iteration",
                "age",
                "postbirth_real_opportunities",
                "postbirth_real_visible_events",
                "postbirth_unique_real_views",
                "visible_rate",
                "opacity",
            ):
                fieldnames.append(f"age_{age}_{suffix}")
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                snapshot_prefixes = tuple(f"age_{age}_" for age in RGG_H1_ALL_SNAPSHOT_AGES)
                row = {key: record.get(key) for key in fieldnames if not key.startswith(snapshot_prefixes)}
                for age in RGG_H1_ALL_SNAPSHOT_AGES:
                    snapshot = record["age_snapshots"].get(str(age), {})
                    for suffix in (
                        "iteration",
                        "age",
                        "postbirth_real_opportunities",
                        "postbirth_real_visible_events",
                        "postbirth_unique_real_views",
                        "visible_rate",
                        "opacity",
                    ):
                        row[f"age_{age}_{suffix}"] = snapshot.get(suffix)
                writer.writerow(row)

    def capture_rgg_state(self):
        if not self._rgg_is_enabled():
            return None
        self._ensure_rgg_state()
        return {
            "enabled": True,
            "uid": self.rgg_uid.detach().cpu(),
            "birth_iter": self.rgg_birth_iter.detach().cpu(),
            "source_uid": self.rgg_source_uid.detach().cpu(),
            "target_uid": self.rgg_target_uid.detach().cpu(),
            "generation": self.rgg_generation.detach().cpu(),
            "birth_type": self.rgg_birth_type.detach().cpu(),
            "next_uid": int(self.rgg_next_uid),
            "h1_registry": self.get_rgg_h1_records(),
        }

    def restore_rgg_state(self, state):
        if not self._rgg_is_enabled():
            self._set_empty_rgg_state()
            return
        if state is None:
            self._initialize_rgg_roots(birth_iter=-1)
            return
        required = ("uid", "birth_iter", "source_uid", "target_uid", "generation")
        if not isinstance(state, dict) or any(key not in state for key in required):
            raise ValueError("paired fork checkpoint RGG state is malformed.")
        expected_count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        for key in required:
            if not torch.is_tensor(state[key]) or state[key].shape[0] != expected_count:
                raise ValueError(f"paired fork checkpoint RGG {key} first dimension must match Gaussian count.")
        self.rgg_uid = state["uid"].to(device=device, dtype=torch.long)
        self.rgg_birth_iter = state["birth_iter"].to(device=device, dtype=torch.long)
        self.rgg_source_uid = state["source_uid"].to(device=device, dtype=torch.long)
        self.rgg_target_uid = state["target_uid"].to(device=device, dtype=torch.long)
        self.rgg_generation = state["generation"].to(device=device, dtype=torch.long)
        if "birth_type" in state:
            birth_type = state["birth_type"]
            if not torch.is_tensor(birth_type) or birth_type.shape[0] != expected_count:
                raise ValueError("paired fork checkpoint RGG birth_type first dimension must match Gaussian count.")
            self.rgg_birth_type = birth_type.to(device=device, dtype=torch.long)
        else:
            self.rgg_birth_type = torch.full((expected_count,), RGG_BIRTH_TYPE_ROOT, dtype=torch.long, device=device)
        used_next = int(self.rgg_uid.max().item()) + 1 if expected_count else 0
        self.rgg_next_uid = max(int(state.get("next_uid", 0)), used_next)
        self.rgg_h1_registry = self._restore_rgg_h1_registry(state.get("h1_registry", []))
        self._assert_rgg_aligned()

    def _restore_rgg_h1_registry(self, records):
        registry = {}
        if records is None:
            return registry
        if not isinstance(records, list):
            raise ValueError("paired fork checkpoint RGG H1 registry must be a list.")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("paired fork checkpoint RGG H1 record must be a dict.")
            uid = int(record["uid"])
            restored = {
                "uid": uid,
                "birth_iter": int(record["birth_iter"]),
                "source_uid": int(record["source_uid"]),
                "target_uid": int(record["target_uid"]),
                "generation": int(record["generation"]),
                "birth_type": str(record["birth_type"]),
                "birth_opacity": record.get("birth_opacity"),
                "death_iter": record.get("death_iter"),
                "death_reason": record.get("death_reason"),
                "alive": bool(record.get("alive", True)),
                "postbirth_real_opportunities": int(record.get("postbirth_real_opportunities", 0)),
                "postbirth_real_visible_events": int(record.get("postbirth_real_visible_events", 0)),
                "postbirth_unique_real_views": int(record.get("postbirth_unique_real_views", 0)),
                "postbirth_real_view_ids": set(int(view_id) for view_id in record.get("postbirth_real_view_ids", [])),
                "age_snapshots": {
                    str(age): dict(snapshot)
                    for age, snapshot in record.get("age_snapshots", {}).items()
                },
            }
            registry[uid] = restored
        return registry

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
        self._initialize_rgg_roots(birth_iter=0)
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
        self._initialize_rgg_roots(birth_iter=0)


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
        self._finalize_rgg_h1_pruned(~valid_points_mask, death_iter=-1, death_reason="dist_prune")
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
        self._prune_rgg_metadata(valid_points_mask)


    def prune_points(self, mask, iter, rgg_death_reason="training_prune"):
        if iter > self.args.prune_from_iter:
            valid_points_mask = ~mask
            self._finalize_rgg_h1_pruned(mask, death_iter=iter, death_reason=rgg_death_reason)
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
            self._prune_rgg_metadata(valid_points_mask)


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
                              new_rotation, rgg_child_metadata=None):
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

        device = self.get_xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device=device)], 0)
        self._append_rgg_metadata(rgg_child_metadata)


    def compute_fsgs_proximity_candidate_value(
        self,
        train_cameras=None,
        edge_maps=None,
        candidate_mask=None,
        iteration=None,
        knn_k=12,
        proximity_neighbor_indices=None,
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
        if value_score_variant not in (
            "obdkr",
            "structural",
            "gestalt_structural",
            "gestalt_balanced",
            "gestalt_defect_conditioned",
        ):
            raise ValueError(
                "value_score_variant must be one of obdkr, structural, "
                "gestalt_structural, gestalt_balanced, gestalt_defect_conditioned"
            )
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
        elif value_score_variant == "structural":
            components = compute_structural_value_score(
                boundary,
                turning,
                defect,
                redundancy,
                **value_kwargs,
            )
        elif value_score_variant == "gestalt_structural":
            if proximity_neighbor_indices is None:
                raise ValueError("gestalt_structural value requires proximity_neighbor_indices.")
            good_continuation = compute_good_continuation_score(
                self.get_xyz.detach(),
                normals,
                proximity_neighbor_indices.detach().long(),
            )
            components = compute_gestalt_structural_value_score(
                boundary,
                turning,
                defect,
                redundancy,
                good_continuation,
                gestalt_lambda=kwargs.get(
                    "gestalt_lambda",
                    getattr(self.args, "gestalt_value_lambda", 1.0),
                ),
                **value_kwargs,
            )
        elif value_score_variant == "gestalt_balanced":
            if proximity_neighbor_indices is None:
                raise ValueError("gestalt_balanced value requires proximity_neighbor_indices.")
            good_continuation = compute_good_continuation_score(
                self.get_xyz.detach(),
                normals,
                proximity_neighbor_indices.detach().long(),
            )
            components = compute_balanced_gestalt_value_score(
                boundary,
                turning,
                defect,
                redundancy,
                good_continuation,
                low_quantile=value_kwargs["low_quantile"],
                high_quantile=value_kwargs["high_quantile"],
                normalization_mask=value_kwargs["normalization_mask"],
                gestalt_balance_alpha=kwargs.get(
                    "gestalt_balance_alpha",
                    getattr(self.args, "gestalt_balance_alpha", 0.5),
                ),
                return_components=True,
            )
        else:
            if proximity_neighbor_indices is None:
                raise ValueError("gestalt_defect_conditioned value requires proximity_neighbor_indices.")
            good_continuation = compute_good_continuation_score(
                self.get_xyz.detach(),
                normals,
                proximity_neighbor_indices.detach().long(),
            )
            components = compute_defect_conditioned_gestalt_value_score(
                boundary,
                turning,
                defect,
                redundancy,
                good_continuation,
                low_quantile=value_kwargs["low_quantile"],
                high_quantile=value_kwargs["high_quantile"],
                normalization_mask=value_kwargs["normalization_mask"],
                gestalt_lambda=kwargs.get(
                    "gestalt_lambda",
                    getattr(self.args, "gestalt_value_lambda", 1.0),
                ),
                return_components=True,
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
        value_components = None
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
        value_rerank_start_iter = getattr(self.args, "value_rerank_start_iter", 0)
        value_rerank_end_iter = getattr(self.args, "value_rerank_end_iter", None)
        effective_proximity_selection_mode, _ = resolve_value_timing_selection_mode(
            proximity_selection_mode,
            iteration=iteration,
            start_iter=value_rerank_start_iter,
            end_iter=value_rerank_end_iter,
        )
        if requires_value_features(effective_proximity_selection_mode):
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
                proximity_neighbor_indices=nearest_indices,
                gestalt_lambda=getattr(self.args, "gestalt_value_lambda", 1.0),
                gestalt_balance_alpha=getattr(self.args, "gestalt_balance_alpha", 0.5),
            )
            value_score = value_components["U"]
        selection_kwargs = dict(
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
            value_rerank_start_iter=value_rerank_start_iter,
            value_rerank_end_iter=value_rerank_end_iter,
        )
        selected_pts_mask, budget_stats = select_proximity_sources(
            selected_pts_mask,
            dist,
            **selection_kwargs,
        )
        print(format_proximity_budget_log(iteration, budget_stats), flush=True)
        if budget_stats.capacity_active:
            print(format_proximity_capacity_log(iteration, budget_stats), flush=True)
        if budget_stats.replay_active:
            print(format_proximity_replay_log(iteration, budget_stats), flush=True)
        if budget_stats.requested_mode == "value_demand_rerank":
            print(format_value_timing_log(iteration, budget_stats), flush=True)
        print(format_value_rerank_diag(iteration, budget_stats), flush=True)
        gestalt_stats = None
        if (
            budget_stats.mode == "value_demand_rerank"
            and getattr(self.args, "value_score_variant", "obdkr") == "gestalt_structural"
        ):
            gestalt_stats = compute_gestalt_value_diagnostics(
                iteration,
                value_components,
                proximity_candidate_mask,
                budget_stats,
            )
            print(format_gestalt_value_diag(gestalt_stats), flush=True)
        balanced_gestalt_stats = None
        if (
            budget_stats.mode == "value_demand_rerank"
            and getattr(self.args, "value_score_variant", "obdkr") == "gestalt_balanced"
        ):
            balanced_gestalt_stats = compute_balanced_gestalt_diagnostics(
                iteration,
                value_components,
                proximity_candidate_mask,
                budget_stats,
            )
            print(format_balanced_gestalt_diag(balanced_gestalt_stats), flush=True)
        defect_gestalt_stats = None
        if (
            budget_stats.mode == "value_demand_rerank"
            and getattr(self.args, "value_score_variant", "obdkr") == "gestalt_defect_conditioned"
        ):
            defect_gestalt_stats = compute_defect_gestalt_diagnostics(
                iteration,
                value_components,
                proximity_candidate_mask,
                budget_stats,
            )
            print(format_defect_gestalt_diag(defect_gestalt_stats), flush=True)
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
            if getattr(self.args, "value_score_variant", "obdkr") == "gestalt_structural":
                if gestalt_stats is None:
                    gestalt_stats = compute_gestalt_value_diagnostics(
                        iteration,
                        value_components,
                        proximity_candidate_mask,
                        budget_stats,
                    )
                print(format_gestalt_promotion_diag(gestalt_stats), flush=True)
            if getattr(self.args, "value_score_variant", "obdkr") == "gestalt_balanced":
                if balanced_gestalt_stats is None:
                    balanced_gestalt_stats = compute_balanced_gestalt_diagnostics(
                        iteration,
                        value_components,
                        proximity_candidate_mask,
                        budget_stats,
                    )
                print(format_balanced_gestalt_promotion_diag(balanced_gestalt_stats), flush=True)
            if getattr(self.args, "value_score_variant", "obdkr") == "gestalt_defect_conditioned":
                if defect_gestalt_stats is None:
                    defect_gestalt_stats = compute_defect_gestalt_diagnostics(
                        iteration,
                        value_components,
                        proximity_candidate_mask,
                        budget_stats,
                    )
                print(format_defect_gestalt_promotion_diag(defect_gestalt_stats), flush=True)
                kd_selected_mask, kd_budget_stats = select_proximity_sources(
                    proximity_candidate_mask,
                    dist,
                    **{
                        **selection_kwargs,
                        "value_score": value_components["U_base"],
                    },
                )
                counterfactual_stats = compute_defect_gestalt_counterfactual_diagnostics(
                    iteration,
                    value_components,
                    proximity_candidate_mask,
                    selected_pts_mask,
                    budget_stats,
                    kd_selected_mask,
                    kd_budget_stats,
                    dist,
                )
                print(format_defect_gestalt_counterfactual_diag(counterfactual_stats), flush=True)

        child_structure_diagnostics_enabled = getattr(self.args, "enable_child_structure_diagnostics", False)
        structural_child_target_selection_enabled = getattr(
            self.args,
            "enable_structural_child_target_selection",
            False,
        )
        current_edge_scores = None
        pool_edge_scores = None
        pool_neighbors = None
        normals = None
        if child_structure_diagnostics_enabled or structural_child_target_selection_enabled:
            normals = estimate_gaussian_normals(
                self.get_scaling.detach(),
                self._rotation.detach(),
            )
            current_edge_scores = compute_good_continuation_edge_scores(
                self.get_xyz.detach(),
                normals,
                nearest_indices.detach().long(),
            )
            pool_neighbors = build_knn_graph(
                self.get_xyz.detach(),
                k=12 if structural_child_target_selection_enabled else getattr(self.args, "knn_k", 12),
            )
            pool_edge_scores = compute_good_continuation_edge_scores(
                self.get_xyz.detach(),
                normals,
                pool_neighbors,
            )

        target_indices = nearest_indices[selected_pts_mask].long()
        if structural_child_target_selection_enabled:
            target_indices, target_selection_stats = select_structural_child_targets(
                selected_pts_mask,
                current_edge_scores,
                pool_edge_scores,
                nearest_indices,
                pool_neighbors,
                self.get_xyz.detach(),
            )
            print(format_child_target_selection_log(iteration, target_selection_stats), flush=True)

        if child_structure_diagnostics_enabled:
            print(
                format_child_structure_diag(
                    compute_child_structure_diagnostics(
                        iteration,
                        selected_pts_mask,
                        current_edge_scores,
                        pool_edge_scores,
                        nearest_indices,
                        pool_neighbors,
                        value_components=value_components,
                        xyz=self.get_xyz.detach(),
                    )
                ),
                flush=True,
            )

        new_indices = target_indices.reshape(-1).long()
        rgg_child_metadata = None
        if self._rgg_is_enabled():
            source_indices = selected_pts_mask.nonzero(as_tuple=False).reshape(-1).long()
            rgg_source_indices = source_indices[:, None].repeat(1, N).reshape(-1)
            if rgg_source_indices.numel() != new_indices.numel():
                raise ValueError("RGG proximity source/target metadata count mismatch.")
            rgg_child_metadata = self._build_rgg_children_from_indices(
                rgg_source_indices,
                target_indices=new_indices,
                birth_iter=iteration if iteration is not None else 0,
                birth_type="proximity",
            )
        source_xyz = self._xyz[selected_pts_mask][:, None, :].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        if self._rgg_is_enabled():
            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
                rgg_child_metadata=rgg_child_metadata,
            )
        else:
            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
            )
        self._append_visibility_from_masks(selected_pts_mask, repeat_count=N, target_indices=new_indices)
        return proximity_sources, proximity_proposed, budget_stats.selected_src, budget_stats.selected_new



    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        device = self.get_xyz.device
        padded_grad = torch.zeros((n_init_points), device=device)
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
        if self._rgg_is_enabled():
            source_indices = selected_pts_mask.nonzero(as_tuple=False).reshape(-1).long().repeat(N)
            rgg_child_metadata = self._build_rgg_children_from_indices(
                source_indices,
                birth_iter=iter,
                birth_type="split",
            )
            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
                rgg_child_metadata=rgg_child_metadata,
            )
        else:
            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
            )
        self._append_visibility_from_masks(selected_pts_mask, repeat_count=N)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.prune_points(prune_filter, iter, rgg_death_reason="split_replaced")
        return split_stats


    def densify_and_clone(self, grads, grad_threshold, scene_extent, iter=None):
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
        if self._rgg_is_enabled():
            source_indices = selected_pts_mask.nonzero(as_tuple=False).reshape(-1).long()
            rgg_child_metadata = self._build_rgg_children_from_indices(
                source_indices,
                birth_iter=iter if iter is not None else 0,
                birth_type="clone",
            )
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                       new_rotation, rgg_child_metadata=rgg_child_metadata)
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                       new_rotation)
        self._append_visibility_from_masks(selected_pts_mask)
        return clone_candidates


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter, train_cameras=None, edge_maps=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        growth_diag = GrowthDiagnostics(iteration=iter, num_before=self.get_xyz.shape[0])

        growth_diag.clone_candidates = self.densify_and_clone(grads, max_grad, extent, iter=iter)
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


    def add_densification_stats(self, viewspace_point_tensor, update_filter, grad_override=None):
        grad = viewspace_point_tensor.grad if grad_override is None else grad_override
        if grad is None:
            raise ValueError("viewspace_point_tensor gradient is required for densification stats")
        if grad.shape != viewspace_point_tensor.shape:
            raise ValueError(
                "densification stats gradient shape must match viewspace_point_tensor shape"
            )
        if grad.device != viewspace_point_tensor.device:
            raise ValueError(
                "densification stats gradient device must match viewspace_point_tensor device"
            )
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
