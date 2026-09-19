def should_preserve_baseline_densification_stats(
    preserve_flag,
    oe_loss_active,
    iteration,
    densify_until_iter,
):
    return bool(preserve_flag) and bool(oe_loss_active) and int(iteration) < int(densify_until_iter)
