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

from argparse import ArgumentParser, Namespace
import sys
import os

from utils.growth_budget import parse_proximity_action_replay

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images_8"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False
        self.use_color = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.prune_from_iter = 500
        self.densify_until_iter = 10_000
        self.densify_grad_threshold = 0.0005
        self.prune_threshold = 0.005
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 9500
        self.sample_pseudo_interval = 10
        self.dist_thres = 10.
        self.enable_proximity_budget = False
        self.proximity_growth_ratio = 0.10
        self.enable_proximity_candidate_capacity = False
        self.proximity_candidate_keep_ratio = 0.80
        self.proximity_action_replay = ""
        self.proximity_selection_mode = "original"
        self.value_rerank_fraction = 0.25
        self.value_rerank_start_iter = 0
        self.value_rerank_end_iter = sys.maxsize
        self.value_boundary_multiplier = 2.0
        self.value_demand_ratio = 0.90
        self.value_tau_e = 1.0
        self.value_tau_s = 3.0
        self.value_observation_source = "lifetime"
        self.value_score_variant = "obdkr"
        self.value_w_b = 1.0
        self.value_w_k = 1.0
        self.value_w_d = 1.0
        self.value_lambda_r = 1.0
        self.gestalt_value_lambda = 1.0
        self.gestalt_balance_alpha = 0.5
        self.enable_child_structure_diagnostics = False
        self.enable_structural_child_target_selection = False
        self.enable_observation_evidence_diagnostics = False
        self.enable_rgg_diagnostics = False
        self.normalization_low_quantile = 0.05
        self.normalization_high_quantile = 0.95
        self.knn_k = 12
        self.depth_weight = 0.05
        self.depth_pseudo_weight = 0.5
        super().__init__(parser, "Optimization Parameters")

    def extract(self, args):
        g = super().extract(args)
        validate_optimization_params(g)
        return g


def validate_optimization_params(args):
    mode = getattr(args, "proximity_selection_mode", "original")
    valid_modes = ("original", "proximity_topk", "value_global", "value_rerank", "value_demand_rerank")
    if mode not in valid_modes:
        raise ValueError(
            "proximity_selection_mode must be one of "
            "original, proximity_topk, value_global, value_rerank, value_demand_rerank"
        )
    budget_enabled = bool(getattr(args, "enable_proximity_budget", False))
    candidate_capacity_enabled = bool(getattr(args, "enable_proximity_candidate_capacity", False))
    replay_schedule = parse_proximity_action_replay(getattr(args, "proximity_action_replay", ""))
    replay_enabled = bool(replay_schedule)
    if sum(int(flag) for flag in (budget_enabled, candidate_capacity_enabled, replay_enabled)) > 1:
        raise ValueError(
            "enable_proximity_budget, enable_proximity_candidate_capacity, "
            "and proximity_action_replay are mutually exclusive; cannot both be enabled"
        )
    if not candidate_capacity_enabled and not replay_enabled and not (
        0 < float(getattr(args, "proximity_growth_ratio", 0.10)) <= 1
    ):
        raise ValueError("proximity_growth_ratio must satisfy 0 < proximity_growth_ratio <= 1")
    if candidate_capacity_enabled and not (
        0 < float(getattr(args, "proximity_candidate_keep_ratio", 0.80)) <= 1
    ):
        raise ValueError("proximity_candidate_keep_ratio must satisfy 0 < ratio <= 1")
    if not (0 <= float(getattr(args, "value_rerank_fraction", 0.25)) <= 1):
        raise ValueError("value_rerank_fraction must satisfy 0 <= value_rerank_fraction <= 1")
    value_rerank_start_iter = int(getattr(args, "value_rerank_start_iter", 0))
    value_rerank_end_iter = int(getattr(args, "value_rerank_end_iter", sys.maxsize))
    if value_rerank_start_iter > value_rerank_end_iter:
        raise ValueError("value_rerank_start_iter must be <= value_rerank_end_iter")
    if float(getattr(args, "value_boundary_multiplier", 2.0)) < 1:
        raise ValueError("value_boundary_multiplier must be at least 1")
    if not (0 < float(getattr(args, "value_demand_ratio", 0.90)) <= 1):
        raise ValueError("value_demand_ratio must satisfy 0 < value_demand_ratio <= 1")
    if float(getattr(args, "value_tau_e", 1.0)) <= 0:
        raise ValueError("value_tau_e must be positive")
    if float(getattr(args, "value_tau_s", 3.0)) <= 0:
        raise ValueError("value_tau_s must be positive")
    observation_source = getattr(args, "value_observation_source", "lifetime")
    if observation_source not in ("lifetime", "recent"):
        raise ValueError("value_observation_source must be one of lifetime, recent")
    score_variant = getattr(args, "value_score_variant", "obdkr")
    if score_variant not in (
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
    if float(getattr(args, "value_lambda_r", 1.0)) < 0:
        raise ValueError("value_lambda_r must be non-negative")
    if float(getattr(args, "gestalt_value_lambda", 1.0)) < 0:
        raise ValueError("gestalt_value_lambda must be non-negative")
    if not (0 <= float(getattr(args, "gestalt_balance_alpha", 0.5)) <= 1):
        raise ValueError("gestalt_balance_alpha must satisfy 0 <= alpha <= 1")
    low = float(getattr(args, "normalization_low_quantile", 0.05))
    high = float(getattr(args, "normalization_high_quantile", 0.95))
    if not (0 <= low < high <= 1):
        raise ValueError("normalization quantiles must satisfy 0 <= low < high <= 1")
    if int(getattr(args, "knn_k", 12)) < 1:
        raise ValueError("knn_k must be at least 1")
    return args


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
