"""Plot operation-specific learning speed before and after the first criterion.

The first panel contains trials from each session's first curriculum stage
(``block1`` or ``stage1``). The second contains trials after the session moves
to its next stage. Exposure numbers and rolling success rates are calculated
separately for the size and shape operations, so interleaving does not mix the
two learning curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from performance import (
    CURRICULUM_COLOURS,
    CURRICULUM_ORDER,
    DEFAULT_INPUT,
    PROJECT_ROOT,
    load_trials,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "learning_speed.png"
OPERATIONS = ("size", "shape")
CRITERION_WINDOW = 15
CRITERION_RATE = 0.80
OPERATION_LINESTYLES = {
    "size": "-",
    "shape": "--",
}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot operation-specific learning curves before and after each "
            "session's first 80% curriculum transition."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Trial CSV exported from Supabase (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Operation-specific rolling window in exposures (default: 10)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.window <= 0:
        parser.error("--window must be greater than zero")
    if arguments.dpi <= 0:
        parser.error("--dpi must be greater than zero")
    return arguments


def _display_name(value: str) -> str:
    return value.replace("_", " ").title()


def _ordered_curricula(values: pd.Series) -> list[str]:
    observed = list(dict.fromkeys(values.dropna().astype(str)))
    return [name for name in CURRICULUM_ORDER if name in observed] + sorted(
        set(observed).difference(CURRICULUM_ORDER)
    )


def calculate_operation_learning(
    trials: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Add criterion period, operation exposure, and rolling success columns."""
    training = trials.loc[trials["phase"] == "training"].copy()
    if training.empty:
        raise ValueError("The data contain no included training trials")
    if training["curriculum_phase"].isna().any():
        raise ValueError("Training trials cannot have missing curriculum_phase values")

    training["operation"] = (
        training["operation"].astype("string").str.strip().str.lower()
    )
    training = training.loc[training["operation"].isin(OPERATIONS)].copy()
    if training.empty:
        raise ValueError("No size or shape training trials were found")

    training = training.sort_values(["session_id", "trial_number"])
    training["stage_index"] = training.groupby(
        "session_id", sort=False
    )["curriculum_phase"].transform(
        lambda stages: stages.ne(stages.shift()).fillna(True).cumsum()
    )
    training["criterion_period"] = np.where(
        training["stage_index"].eq(1),
        "before",
        "after",
    )

    operation_groups = training.groupby(
        ["session_id", "criterion_period", "operation"],
        sort=False,
    )
    training["operation_exposure"] = operation_groups.cumcount() + 1
    training["success_rate"] = operation_groups["is_correct"].transform(
        lambda values: values.astype(float).rolling(window, min_periods=1).mean()
    )
    return training


def _first_criterion_exposure(trials: pd.DataFrame) -> float:
    """Return the operation exposure where the 12/15 criterion is first met."""
    if trials.empty:
        return np.nan
    trials = trials.sort_values("operation_exposure")
    criterion_accuracy = trials["is_correct"].astype(float).rolling(
        CRITERION_WINDOW,
        min_periods=CRITERION_WINDOW,
    ).mean()
    reached = criterion_accuracy.ge(CRITERION_RATE)
    if not reached.any():
        return np.nan
    return float(trials.loc[reached, "operation_exposure"].iloc[0])


def calculate_rule_speed_ratios(learning: pd.DataFrame) -> pd.DataFrame:
    """Compare size-rule and subsequent shape-rule learning within sessions.

    The log2 speed ratio is log2(rule-1 trials / rule-2 trials). Therefore 0
    denotes equal speed, +1 denotes twice as fast, and -1 denotes twice as
    slow when learning rule 2.
    """
    rows: list[dict[str, object]] = []
    for session_id, participant in learning.groupby("session_id", sort=False):
        rule1 = participant.loc[
            participant["stage_index"].eq(1) & participant["operation"].eq("size")
        ]
        rule2 = participant.loc[
            participant["stage_index"].gt(1) & participant["operation"].eq("shape")
        ]
        rule1_trials = _first_criterion_exposure(rule1)
        rule2_trials = _first_criterion_exposure(rule2)
        speed_ratio = (
            float(np.log2(rule1_trials / rule2_trials))
            if np.isfinite(rule1_trials) and np.isfinite(rule2_trials)
            else np.nan
        )
        rows.append(
            {
                "session_id": session_id,
                "participant_id": participant["participant_id"].iloc[0],
                "curriculum": participant["curriculum"].iloc[0],
                "rule1_trials_to_criterion": rule1_trials,
                "rule2_trials_to_criterion": rule2_trials,
                "rule2_speed_ratio": speed_ratio,
            }
        )
    return pd.DataFrame(rows)


def _draw_period(
    axis: plt.Axes,
    learning: pd.DataFrame,
    *,
    period: str,
    window: int,
) -> None:
    subset = learning.loc[learning["criterion_period"] == period]
    curricula = _ordered_curricula(subset["curriculum"])

    if subset.empty:
        axis.text(
            0.5,
            0.5,
            f"No trials recorded {period} the first criterion",
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="#666666",
        )
    else:
        for curriculum in curricula:
            colour = CURRICULUM_COLOURS.get(curriculum, "#557A95")
            curriculum_data = subset.loc[subset["curriculum"] == curriculum]
            for operation in OPERATIONS:
                operation_data = curriculum_data.loc[
                    curriculum_data["operation"] == operation
                ]
                if operation_data.empty:
                    continue

                linestyle = OPERATION_LINESTYLES[operation]
                for _, participant in operation_data.groupby(
                    "session_id", sort=False
                ):
                    axis.plot(
                        participant["operation_exposure"],
                        participant["success_rate"],
                        color=colour,
                        linestyle=linestyle,
                        linewidth=1.2,
                        alpha=0.25,
                        zorder=2,
                    )

                mean_curve = (
                    operation_data.groupby("operation_exposure", as_index=False)[
                        "success_rate"
                    ]
                    .mean()
                    .sort_values("operation_exposure")
                )
                axis.plot(
                    mean_curve["operation_exposure"],
                    mean_curve["success_rate"],
                    color=colour,
                    linestyle=linestyle,
                    linewidth=3,
                    label=(
                        f"{_display_name(curriculum)} — "
                        f"{_display_name(operation)} mean"
                    ),
                    zorder=4,
                )

    axis.axhline(
        0.80,
        color="#444444",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="80% criterion",
        zorder=1,
    )
    if not subset.empty:
        axis.legend(frameon=False, loc="lower right")
    session_count = subset["session_id"].nunique()
    period_title = "Before first 80% criterion" if period == "before" else (
        "After first 80% criterion"
    )
    axis.set_title(f"{period_title} (n = {session_count})", pad=12)
    axis.set_xlabel("Exposure number within trial type and period")
    axis.set_ylabel(f"Success rate (last {window} exposures)")
    axis.set_ylim(-0.03, 1.03)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, color="#DDDDDD", linewidth=0.8, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def _draw_speed_ratios(axis: plt.Axes, ratios: pd.DataFrame) -> None:
    curricula = _ordered_curricula(ratios["curriculum"])
    participant_label_available = True
    mean_label_available = True

    for position, curriculum in enumerate(curricula):
        curriculum_data = ratios.loc[ratios["curriculum"] == curriculum]
        paired = curriculum_data.dropna(subset=["rule2_speed_ratio"])
        values = paired["rule2_speed_ratio"].to_numpy(dtype=float)
        colour = CURRICULUM_COLOURS.get(curriculum, "#557A95")

        if len(values):
            offsets = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else [0]
            axis.scatter(
                position + np.asarray(offsets),
                values,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                s=58,
                label="Participant" if participant_label_available else "_nolegend_",
                zorder=4,
            )
            participant_label_available = False
            axis.scatter(
                position,
                float(np.mean(values)),
                color="#222222",
                marker="D",
                s=58,
                label="Curriculum mean" if mean_label_available else "_nolegend_",
                zorder=5,
            )
            mean_label_available = False

        axis.text(
            position,
            1.82,
            f"{len(paired)}/{len(curriculum_data)} paired",
            ha="center",
            va="center",
            fontsize=9,
            color="#555555",
        )

    axis.axhline(0, color="#333333", linewidth=1.4, zorder=1)
    axis.axhline(1, color="#999999", linestyle=":", linewidth=1.0, zorder=1)
    axis.axhline(-1, color="#999999", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_title("Within-participant rule-speed ratio", pad=12)
    axis.set_ylabel("Rule 2 learning speed relative to rule 1 (log₂ ratio)")
    axis.set_xticks(range(len(curricula)))
    axis.set_xticklabels([_display_name(value) for value in curricula])
    axis.set_ylim(-2.1, 2.1)
    axis.set_yticks([-2, -1, 0, 1, 2])
    axis.set_yticklabels(
        ["4× slower", "2× slower", "Same", "2× faster", "4× faster"]
    )
    axis.grid(True, axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    if not participant_label_available:
        axis.legend(frameon=False, loc="lower right")


def plot_learning_speed(
    trials: pd.DataFrame,
    output: Path,
    *,
    window: int = 10,
    dpi: int = 200,
    show: bool = False,
) -> Path:
    learning = calculate_operation_learning(trials, window)
    ratios = calculate_rule_speed_ratios(learning)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(21, 5.8),
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.72)},
    )
    _draw_period(axes[0], learning, period="before", window=window)
    _draw_period(axes[1], learning, period="after", window=window)
    axes[1].set_ylim(axes[0].get_ylim())
    _draw_speed_ratios(axes[2], ratios)

    figure.suptitle(
        "Operation-specific learning speed around the first criterion",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        (
            "Thin lines show individual sessions and thick lines show means. "
            "Exposure counts reset after the first curriculum-stage transition "
            "and are calculated separately for size and shape. Ratios compare "
            "operation-specific trials to the 12-of-15 criterion."
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.subplots_adjust(top=0.84, bottom=0.18, wspace=0.22)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    trials = load_trials(arguments.input.resolve())
    output = plot_learning_speed(
        trials,
        arguments.output,
        window=arguments.window,
        dpi=arguments.dpi,
        show=arguments.show,
    )
    print(
        f"Loaded {len(trials):,} included trials from "
        f"{trials['session_id'].nunique():,} sessions"
    )
    print(f"Saved learning-speed plot to {output}")


if __name__ == "__main__":
    main()
