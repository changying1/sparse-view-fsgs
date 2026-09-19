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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import random as py_random
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from utils.depth_utils import estimate_depth
from utils.observation_evidence import (
    compute_observation_evidence_diagnostics,
    format_observation_evidence_diag,
    format_observation_evidence_edge_diag,
    format_observation_reliability_diag,
)
from utils.evidence_support import (
    compute_edge_stratified_reliability_diagnostics,
    compute_real_depth_evidence_diagnostics,
    compute_rgb_error_map,
    compute_spatial_reliability_diagnostics,
    format_real_depth_evidence_diag,
    make_edge_stratified_reliability_record,
    make_spatial_reliability_record,
    preserved_random_state,
    save_edge_stratified_reliability_summary,
    save_real_depth_evidence_summary,
    save_spatial_reliability_summary,
    should_run_edge_stratified_reliability_snapshot,
    should_run_real_depth_evidence_diagnostics,
    should_run_spatial_reliability_snapshot,
)
from utils.evidence_structural_loss import (
    compute_oe_structural_loss,
    compute_stable_mask_from_gt_rgb,
    format_oe_structural_loss_log,
    make_oe_structural_record,
    save_oe_structural_training_summary,
    should_apply_oe_structural_loss,
)
from utils.oe_densification import should_preserve_baseline_densification_stats
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.edge_support import compute_edge_map
from utils.growth_budget import requires_value_features, resolve_proximity_selection_mode
from utils.paired_fork_checkpoint import (
    load_paired_fork_checkpoint,
    next_iteration_after_paired_fork,
    save_paired_fork_checkpoint,
)
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

SPATIAL_RELIABILITY_SNAPSHOTS = (500, 1000, 1500, 2000)


def collect_observation_evidence_snapshot(gaussians, train_cameras, pipe, background, render_func=render):
    py_rng_state = py_random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        rendered_depths = []
        rendered_rgbs = []
        mono_depths = []
        visibility_masks = []
        valid_masks = []
        device = gaussians.get_xyz.device
        with torch.no_grad():
            for camera in train_cameras:
                if getattr(camera, "depth_image", None) is None:
                    raise ValueError("Observation evidence diagnostics require depth_image for every training camera.")
                render_pkg = render_func(camera, gaussians, pipe, background)
                rendered_depths.append(render_pkg["depth"][0].detach())
                rendered_rgbs.append(render_pkg["render"].detach())
                mono_depths.append(torch.as_tensor(camera.depth_image, device=device).detach())
                visibility_masks.append(render_pkg["visibility_filter"].detach())
                mask = getattr(camera, "mask", None)
                valid_masks.append(torch.as_tensor(mask, device=device).bool().detach() if mask is not None else None)
        return rendered_depths, mono_depths, visibility_masks, valid_masks, rendered_rgbs
    finally:
        py_random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def collect_spatial_reliability_snapshot(
    iteration,
    gaussians,
    train_cameras,
    pipe,
    background,
    model_path,
    camera_uid_to_train_index=None,
    save_visualizations=False,
    render_func=render,
):
    records = []
    edge_records = []
    output_dir = os.path.join(model_path, "oe_diagnostics", "spatial_reliability_png")
    with preserved_random_state():
        with torch.no_grad():
            for fallback_index, camera in enumerate(train_cameras):
                if getattr(camera, "depth_image", None) is None:
                    continue
                render_pkg = render_func(camera, gaussians, pipe, background)
                depth_diag = compute_real_depth_evidence_diagnostics(
                    render_pkg["depth"][0].detach(),
                    torch.as_tensor(camera.depth_image, device=render_pkg["depth"].device).detach(),
                    return_evidence=True,
                )
                reliability = compute_spatial_reliability_diagnostics(
                    depth_diag,
                    render_pkg["render"].detach(),
                    camera.original_image.detach().to(device=render_pkg["render"].device),
                )
                edge_reliability = compute_edge_stratified_reliability_diagnostics(
                    depth_diag,
                    render_pkg["render"].detach(),
                    camera.original_image.detach().to(device=render_pkg["render"].device),
                )
                view_id = (
                    camera_uid_to_train_index.get(getattr(camera, "uid", None), fallback_index)
                    if camera_uid_to_train_index is not None
                    else fallback_index
                )
                records.append(make_spatial_reliability_record(iteration, view_id, depth_diag["stats"], reliability))
                edge_records.append(make_edge_stratified_reliability_record(iteration, view_id, edge_reliability))
                if save_visualizations:
                    save_spatial_reliability_visualizations(
                        output_dir,
                        iteration,
                        view_id,
                        render_pkg["render"].detach(),
                        camera.original_image.detach().to(device=render_pkg["render"].device),
                        depth_diag,
                        edge_reliability=edge_reliability,
                    )
    return records, edge_records


def save_spatial_reliability_visualizations(
    output_dir,
    iteration,
    view_id,
    rendered_rgb,
    gt_rgb,
    depth_diag,
    edge_reliability=None,
):
    os.makedirs(output_dir, exist_ok=True)
    rgb_error, _ = compute_rgb_error_map(rendered_rgb, gt_rgb)
    items = {
        "rendered_rgb": _rgb_to_numpy(rendered_rgb),
        "gt_rgb": _rgb_to_numpy(gt_rgb),
        "rgb_error": _map_to_numpy(rgb_error),
        "normalized_depth_residual": _map_to_numpy(depth_diag["normalized_absolute_residual"]),
        "evidence": _evidence_to_numpy(depth_diag["evidence"]),
    }
    if edge_reliability is not None:
        items.update(
            {
                "edge_magnitude": _map_to_numpy(edge_reliability["edge_magnitude"]),
                "edge_mask": _binary_to_numpy(edge_reliability["edge_mask"]),
                "non_edge_mask": _binary_to_numpy(edge_reliability["non_edge_mask"]),
            }
        )
    for name, array in items.items():
        plt.imsave(os.path.join(output_dir, f"iter_{iteration}_view_{view_id}_{name}.png"), array)


def _rgb_to_numpy(image):
    image = image.detach().float().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image.clamp(0.0, 1.0).numpy()


def _map_to_numpy(values):
    values = values.detach().float().cpu()
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    finite = torch.isfinite(values)
    if not finite.any():
        return torch.zeros_like(values).numpy()
    result = torch.zeros_like(values)
    finite_values = values[finite]
    low = finite_values.min()
    high = finite_values.max()
    if float((high - low).item()) > 1e-8:
        result[finite] = (finite_values - low) / (high - low)
    else:
        result[finite] = 1.0
    return result.numpy()


def _evidence_to_numpy(values):
    values = values.detach().float().cpu()
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    values = torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return values.clamp(0.0, 1.0).numpy()


def _binary_to_numpy(values):
    values = values.detach().bool().cpu()
    return values.float().numpy()


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    paired_fork_resumed_from = None
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if args.paired_fork_checkpoint:
        first_iter = load_paired_fork_checkpoint(args.paired_fork_checkpoint, gaussians, opt)
        paired_fork_resumed_from = first_iter
    elif checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    train_cameras = scene.getTrainCameras()
    gaussians.ensure_visibility_history(len(train_cameras))
    gaussians.camera_uid_to_train_index = {
        getattr(camera, "uid", index): index
        for index, camera in enumerate(train_cameras)
    }
    proximity_selection_mode = resolve_proximity_selection_mode(
        getattr(args, "proximity_selection_mode", "original"),
        getattr(args, "enable_proximity_budget", False)
        or getattr(args, "enable_proximity_candidate_capacity", False),
    )
    value_edge_maps = None
    if requires_value_features(proximity_selection_mode):
        value_edge_maps = initialize_value_edge_maps(train_cameras)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    real_depth_evidence_records = []
    spatial_reliability_records = []
    edge_stratified_reliability_records = []
    oe_structural_records = []
    oe_stable_mask_cache = {}
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if paired_fork_resumed_from is not None and iteration == next_iteration_after_paired_fork(paired_fork_resumed_from):
            print(
                "[PairedForkResume] "
                f"iter={iteration} "
                f"gaussians={gaussians.get_xyz.shape[0]}",
                flush=True,
            )
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if hasattr(viewpoint_cam, "uid"):
            train_view_index = gaussians.camera_uid_to_train_index[viewpoint_cam.uid]
        else:
            train_view_index = train_cameras.index(viewpoint_cam)
        gaussians.update_visibility(train_view_index, visibility_filter, iteration=iteration, is_real_view=True)


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 =  l1_loss_mask(image, gt_image)
        base_loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))


        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
                        (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                        (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        base_loss += args.depth_weight * depth_loss

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001



        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                base_loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo

        oe_loss_active = (
            should_apply_oe_structural_loss(
                getattr(args, "enable_oe_structural_loss", False),
                iteration,
                getattr(args, "oe_structural_start_iter", 500),
                getattr(args, "oe_structural_end_iter", 2000),
                getattr(args, "oe_structural_weight", 0.0),
            )
            and getattr(viewpoint_cam, "depth_image", None) is not None
        )
        oe_weighted_loss = None
        if oe_loss_active:
            if train_view_index not in oe_stable_mask_cache:
                stable_mask, _, _ = compute_stable_mask_from_gt_rgb(
                    gt_image.detach(),
                    stable_quantile=getattr(args, "oe_stable_quantile", 0.80),
                )
                oe_stable_mask_cache[train_view_index] = stable_mask.detach()
            oe_struct_loss, oe_struct_stats, _ = compute_oe_structural_loss(
                render_pkg["depth"][0],
                torch.as_tensor(viewpoint_cam.depth_image, device=render_pkg["depth"].device).detach(),
                stable_mask=oe_stable_mask_cache[train_view_index].to(device=render_pkg["depth"].device),
                gate_mode=getattr(args, "oe_structure_gate_mode", "stable_evidence"),
                stable_quantile=getattr(args, "oe_stable_quantile", 0.80),
            )
            oe_weighted_loss = float(getattr(args, "oe_structural_weight", 0.0)) * oe_struct_loss
            if iteration % 100 == 0:
                weight = float(getattr(args, "oe_structural_weight", 0.0))
                print(format_oe_structural_loss_log(iteration, train_view_index, oe_struct_stats, weight), flush=True)
                oe_structural_records.append(
                    make_oe_structural_record(iteration, train_view_index, oe_struct_stats, weight)
                )

        loss = base_loss if oe_weighted_loss is None else base_loss + oe_weighted_loss
        preserve_oe_densification_stats = should_preserve_baseline_densification_stats(
            getattr(args, "oe_preserve_baseline_densification_stats", False),
            oe_loss_active,
            iteration,
            opt.densify_until_iter,
        )
        baseline_viewspace_grad = None
        if preserve_oe_densification_stats:
            baseline_viewspace_grad = torch.autograd.grad(
                base_loss,
                viewspace_point_tensor,
                retain_graph=True,
                allow_unused=False,
            )[0].detach()
            viewspace_point_tensor.grad = None

        loss.backward()
        with torch.no_grad():
            if preserve_oe_densification_stats and iteration % 100 == 0:
                visible_base_grad = torch.norm(baseline_viewspace_grad[visibility_filter, :2], dim=-1)
                visible_total_grad = torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1)
                visible_count = int(visible_base_grad.numel())
                if visible_count > 0:
                    base_grad_mean = float(visible_base_grad.mean().item())
                    total_grad_mean = float(visible_total_grad.mean().item())
                    base_grad_p90 = float(torch.quantile(visible_base_grad, 0.90).item())
                    total_grad_p90 = float(torch.quantile(visible_total_grad, 0.90).item())
                    ratio_mean = total_grad_mean / max(base_grad_mean, 1e-12)
                else:
                    base_grad_mean = 0.0
                    total_grad_mean = 0.0
                    base_grad_p90 = 0.0
                    total_grad_p90 = 0.0
                    ratio_mean = 0.0
                print(
                    "[OEDensifyGrad] "
                    f"iter={iteration} "
                    f"base_grad_mean={base_grad_mean:.8g} "
                    f"total_grad_mean={total_grad_mean:.8g} "
                    f"base_grad_p90={base_grad_p90:.8g} "
                    f"total_grad_p90={total_grad_p90:.8g} "
                    f"ratio_mean={ratio_mean:.8g} "
                    f"visible_count={visible_count}",
                    flush=True,
                )
            if (
                should_run_real_depth_evidence_diagnostics(
                    getattr(args, "enable_observation_evidence_diagnostics", False),
                    iteration,
                    max(int(getattr(opt, "densification_interval", 100)), 100),
                )
                and getattr(viewpoint_cam, "depth_image", None) is not None
            ):
                depth_diag = compute_real_depth_evidence_diagnostics(
                    render_pkg["depth"][0].detach(),
                    torch.as_tensor(viewpoint_cam.depth_image, device=render_pkg["depth"].device).detach(),
                    return_evidence=True,
                )
                depth_stats = depth_diag["stats"]
                record = {
                    "iteration": int(iteration),
                    "view_id": int(train_view_index),
                    **depth_stats,
                }
                real_depth_evidence_records.append(record)
                print(format_real_depth_evidence_diag(iteration, train_view_index, depth_stats), flush=True)
            if (
                should_run_spatial_reliability_snapshot(
                    getattr(args, "enable_observation_evidence_diagnostics", False),
                    iteration,
                    SPATIAL_RELIABILITY_SNAPSHOTS,
                )
                or should_run_edge_stratified_reliability_snapshot(
                    getattr(args, "enable_observation_evidence_diagnostics", False),
                    iteration,
                    SPATIAL_RELIABILITY_SNAPSHOTS,
                )
            ):
                spatial_records, edge_records = collect_spatial_reliability_snapshot(
                    iteration,
                    gaussians,
                    train_cameras,
                    pipe,
                    background,
                    scene.model_path,
                    camera_uid_to_train_index=gaussians.camera_uid_to_train_index,
                    save_visualizations=iteration == max(SPATIAL_RELIABILITY_SNAPSHOTS),
                )
                spatial_reliability_records.extend(spatial_records)
                edge_stratified_reliability_records.extend(edge_records)
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                    grad_override=baseline_viewspace_grad if preserve_oe_densification_stats else None,
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.prune_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        iteration,
                        train_cameras=train_cameras,
                        edge_maps=value_edge_maps,
                    )
                    gaussians.reset_recent_visibility()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()

            if iteration == args.paired_fork_save_iteration:
                save_paired_fork_checkpoint(
                    os.path.join(scene.model_path, f"paired_chkpnt{iteration}.pth"),
                    gaussians,
                    iteration,
                )

    if getattr(args, "enable_observation_evidence_diagnostics", False):
        save_real_depth_evidence_summary(scene.model_path, real_depth_evidence_records)
        save_spatial_reliability_summary(scene.model_path, spatial_reliability_records)
        save_edge_stratified_reliability_summary(scene.model_path, edge_stratified_reliability_records)
    if getattr(args, "enable_oe_structural_loss", False):
        save_oe_structural_training_summary(scene.model_path, oe_structural_records)
    save_final_rgg_diagnostics(args, opt, scene, gaussians)


def save_final_rgg_diagnostics(args, opt, scene, gaussians):
    if getattr(args, "enable_rgg_diagnostics", False):
        gaussians.save_rgg_diagnostics(
            os.path.join(scene.model_path, "rgg_diagnostics_final"),
            observation_end_iter=opt.iterations,
        )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def initialize_value_edge_maps(train_cameras):
    with torch.no_grad():
        return [compute_edge_map(camera.original_image.cuda()) for camera in train_cameras]



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--paired_fork_save_iteration", type=int, default=-1)
    parser.add_argument("--paired_fork_checkpoint", type=str, default=None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
