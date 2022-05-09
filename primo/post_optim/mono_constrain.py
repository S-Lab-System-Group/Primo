import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.isotonic import IsotonicRegression


def add_monotone_constraint(primo_model, feature, direction="auto", inplace=False, visualize_changes=True):
    """Adjusts an series of features to be monotone using isotonic regression. 
    """
    # Obtain ebm model
    if "Primo" in type(primo_model).__name__:
        ebm = primo_model.prModel.gam
    elif "PrAM" in type(primo_model).__name__:
        ebm = primo_model.gam
    else:
        raise NotImplementedError

    if isinstance(feature, str) or isinstance(feature, int):
        modified_ebm = make_monotone(ebm, feature, direction, visualize_changes)
    elif isinstance(feature, list):
        modified_ebm = primo_model
        for f in feature:
            modified_ebm = make_monotone(ebm, f, direction, visualize_changes)
    else:
        raise NotImplementedError

    # Modify in place or return copy
    if inplace:
        if "Primo" in type(primo_model).__name__:
            primo_model.prModel.gam = modified_ebm
        elif "PrAM" in type(primo_model).__name__:
            primo_model.gam = modified_ebm
    else:
        # TODO: Form a Primo model.
        return modified_ebm


def make_monotone(ebm, feature, direction="auto", visualize_changes=True):
    """Adjusts an individual feature to be monotone using isotonic regression. 

    Author: Harsha Nori.
    """

    # Find feature index if passed as string
    if isinstance(feature, str):
        feature_index = ebm.feature_names.index(feature)
        feature_name = feature
    else:
        feature_index = feature
        feature_name = ebm.feature_names[feature_index]

    x = np.array(range(len(ebm.additive_terms_[feature_index])))
    y = ebm.additive_terms_[feature_index]
    w = ebm.preprocessor_.col_bin_counts_[feature_index]

    # Fit isotonic regression weighted by training data bin counts
    direction = "auto" if direction not in ["increasing", "decreasing"] else direction == "increasing"
    ir = IsotonicRegression(out_of_bounds="clip", increasing=direction)
    y_ = ir.fit_transform(x, y, sample_weight=w)

    ebm_mono = deepcopy(ebm)
    ebm_mono.additive_terms_[feature_index][1:] = y_[1:]

    # Plot changes to model
    if visualize_changes:
        fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
        ax.plot(x, y, "-", label="Origin", marker="o", markersize=6, linewidth=3)
        ax.plot(x, y_, "--", label="Monotonic", linewidth=3)
        ax.set_xlabel(f"{feature_name} Bins")
        ax.set_ylabel(f"Score")
        ax.legend()

        backend = matplotlib.get_backend()
        if "inline" not in backend:
            fig.show()

    return ebm_mono
