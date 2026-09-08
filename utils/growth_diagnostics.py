from dataclasses import dataclass


def _mask_count(mask):
    if mask is None:
        return 0
    return int(mask.sum().item())


def count_split_candidates(split_gradient_mask, split_sparse_mask, split_total_mask=None):
    split_gradient_candidates = _mask_count(split_gradient_mask)
    split_sparse_candidates = _mask_count(split_sparse_mask)
    if split_total_mask is None:
        split_total_mask = split_gradient_mask.logical_or(split_sparse_mask)
    split_total_candidates = _mask_count(split_total_mask)
    return split_gradient_candidates, split_sparse_candidates, split_total_candidates


def count_proximity_proposed(proximity_sources, n=3):
    return int(proximity_sources) * int(n)


def format_proximity_growth_log(iteration, num_before, proximity_sources, proximity_proposed):
    return (
        f"[GrowthDiagProximity] iter={iteration} "
        f"before={num_before} "
        f"proximity_src={proximity_sources} "
        f"proximity_proposed={proximity_proposed}"
    )


@dataclass
class GrowthDiagnostics:
    iteration: int
    num_before: int
    clone_candidates: int = 0
    split_gradient_candidates: int = 0
    split_sparse_candidates: int = 0
    split_total_candidates: int = 0
    proximity_sources: int = 0
    proximity_proposed: int = 0
    proximity_selected_sources: int = 0
    proximity_selected_new: int = 0
    num_after_clone: int = 0
    num_after_split: int = 0
    num_after_proximity: int = 0
    num_after_prune: int = 0

    @property
    def net_growth(self):
        return self.num_after_prune - self.num_before

    def format_log(self):
        return (
            f"[GrowthDiag] iter={self.iteration} "
            f"before={self.num_before} "
            f"clone={self.clone_candidates} "
            f"split_grad={self.split_gradient_candidates} "
            f"split_sparse={self.split_sparse_candidates} "
            f"split_total={self.split_total_candidates} "
            f"proximity_src={self.proximity_sources} "
            f"proximity_proposed={self.proximity_proposed} "
            f"proximity_selected_src={self.proximity_selected_sources} "
            f"proximity_new={self.proximity_selected_new} "
            f"after_clone={self.num_after_clone} "
            f"after_split={self.num_after_split} "
            f"after_proximity={self.num_after_proximity} "
            f"after_prune={self.num_after_prune} "
            f"net={self.net_growth}"
        )
