"""
Analysis scripts for performance evaluation of curriculum learning.

Plots the rolling success rate over time for each participant during the
training and test phases, separated by curriculum condition. A marginal
histogram aligned with the y-axis shows the distribution of success rates in
participants' final rolling window for each phase.

By default, data are read from ``data/db_trials.csv`` and success rate is
calculated over a rolling 10-trial window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "db_trials.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "performance.png"
INTERFERENCE_WINDOW = 15
EXCLUDED_PARTICIPANT_IDS = {
    "fdd6b46d-c12d-42a6-91cc-d825bb894a26",
}
PHASES = ("training", "test")
CURRICULUM_ORDER = (
    "blocked",
    "progressively_interleaved",
    "interleaved",
    "progressively_blocked",
)
CURRICULUM_COLOURS = {
    "blocked": "#7C5CFC",
    "progressively_interleaved": "#00B8A9",
    "interleaved": "#28A9E0",
    "progressively_blocked": "#F29E4C",
}
REQUIRED_COLUMNS = {
    "session_id",
    "participant_id",
    "curriculum",
    "curriculum_phase",
    "operation",
    "phase",
    "trial_number",
    "is_correct",
}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot rolling success rates for each curriculum and phase, with "
            "endpoint distributions."
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
        help="Number of trials in the moving window (default: 10)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.window <= 0:
        parser.error("--window must be greater than zero")
    if arguments.dpi <= 0:
        parser.error("--dpi must be greater than zero")
    return arguments


def _parse_boolean(column: pd.Series, name: str) -> pd.Series:
    values = column.astype("string").str.strip().str.lower()
    parsed = values.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "t": True,
            "f": False,
            "yes": True,
            "no": False,
        }
    )
    invalid = column.notna() & parsed.isna()
    if invalid.any():
        examples = sorted(values.loc[invalid].dropna().unique())[:5]
        raise ValueError(f"Column {name!r} contains invalid values: {examples}")
    return parsed.astype("boolean")


def load_trials(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No trial data found at {path}. Export curriculum_trials as "
            "data/db_trials.csv or pass --input."
        )

    trials = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(trials.columns))
    if missing:
        raise ValueError(f"The trial CSV is missing columns: {', '.join(missing)}")

    trials = trials.copy()
    trials["curriculum"] = trials["curriculum"].astype("string").str.strip().str.lower()
    trials["curriculum_phase"] = (
        trials["curriculum_phase"].astype("string").str.strip().str.lower()
    )
    trials["operation"] = trials["operation"].astype("string").str.strip().str.lower()
    trials["phase"] = trials["phase"].astype("string").str.strip().str.lower()
    trials["is_correct"] = _parse_boolean(trials["is_correct"], "is_correct")
    trials["trial_number"] = pd.to_numeric(trials["trial_number"], errors="raise")
    trials = trials.loc[trials["phase"].isin(PHASES)].copy()
    trials = trials.loc[
        ~trials["participant_id"].astype("string").isin(EXCLUDED_PARTICIPANT_IDS)
    ].copy()

    if trials.empty:
        raise ValueError(
            "The CSV contains no included training or test trials after exclusions"
        )
    identifiers = ["session_id", "participant_id", "curriculum"]
    if trials[identifiers].isna().any().any():
        raise ValueError("Session, participant, and curriculum values cannot be missing")
    duplicate = trials.duplicated(["session_id", "trial_number"], keep=False)
    if duplicate.any():
        examples = trials.loc[duplicate, ["session_id", "trial_number"]].head()
        raise ValueError(f"Duplicate session/trial rows found:\n{examples.to_string(index=False)}")

    return trials.sort_values(["session_id", "trial_number"]).reset_index(drop=True)


def calculate_rolling_success(trials: pd.DataFrame, window: int) -> pd.DataFrame:
    data = trials.copy()
    groups = data.groupby(["session_id", "phase"], sort=False)
    data["phase_trial"] = groups.cumcount() + 1
    data["success_rate"] = groups["is_correct"].transform(
        lambda values: values.astype(float).rolling(window, min_periods=1).mean()
    )
    return data


def calculate_interference(
    trials: pd.DataFrame,
    samples: int = INTERFERENCE_WINDOW,
) -> pd.DataFrame:
    """Calculate operation-matched retention changes for each curriculum.

    For each operation, the change score is mean accuracy in its first
    ``samples`` test exposures minus mean accuracy in its final ``samples``
    training exposures. Sessions need a full window in both periods.
    """
    rows: list[dict[str, object]] = []
    for session_id, participant in trials.groupby("session_id", sort=False):
        for operation in ("size", "shape"):
            operation_trials = participant.loc[
                participant["operation"] == operation
            ].sort_values("trial_number")
            training = operation_trials.loc[
                operation_trials["phase"] == "training"
            ].tail(samples)
            test = operation_trials.loc[operation_trials["phase"] == "test"].head(
                samples
            )
            if len(training) < samples or len(test) < samples:
                continue

            training_accuracy = float(training["is_correct"].astype(float).mean())
            test_accuracy = float(test["is_correct"].astype(float).mean())
            rows.append(
                {
                    "session_id": session_id,
                    "participant_id": participant["participant_id"].iloc[0],
                    "curriculum": participant["curriculum"].iloc[0],
                    "operation": operation,
                    "training_accuracy": training_accuracy,
                    "test_accuracy": test_accuracy,
                    "accuracy_change": test_accuracy - training_accuracy,
                }
            )

    return pd.DataFrame(
        rows,
        columns=(
            "session_id",
            "participant_id",
            "curriculum",
            "operation",
            "training_accuracy",
            "test_accuracy",
            "accuracy_change",
        ),
    )


def _ordered_curricula(values: pd.Series) -> list[str]:
    observed = list(dict.fromkeys(values.dropna().astype(str)))
    return [name for name in CURRICULUM_ORDER if name in observed] + sorted(
        set(observed).difference(CURRICULUM_ORDER)
    )


def _display_name(value: str) -> str:
    return value.replace("_", " ").title()


def _draw_phase(
    line_axis: plt.Axes,
    histogram_axis: plt.Axes,
    data: pd.DataFrame,
    *,
    curriculum: str,
    phase: str,
    window: int,
) -> None:
    subset = data.loc[
        (data["curriculum"] == curriculum) & (data["phase"] == phase)
    ].dropna(subset=["success_rate"])
    colour = CURRICULUM_COLOURS.get(curriculum, "#557A95")

    if subset.empty:
        line_axis.text(
            0.5,
            0.5,
            f"No recorded {phase} trials",
            ha="center",
            va="center",
            transform=line_axis.transAxes,
            color="#666666",
        )
        line_axis.set_xlim(0, max(window, 20))
        histogram_axis.set_xlim(0, 1)
    else:
        switch_label_available = True
        for _, participant in subset.groupby("session_id", sort=False):
            participant = participant.sort_values("phase_trial")
            line_axis.plot(
                participant["phase_trial"],
                participant["success_rate"],
                color=colour,
                linewidth=1.2,
                alpha=0.30,
            )

            if phase == "training":
                stage = participant["curriculum_phase"]
                switched = stage.ne(stage.shift()) & stage.shift().notna()
                for switch_trial in participant.loc[switched, "phase_trial"]:
                    # Draw the marker between the final trial of the old block
                    # and the first trial of the new block.
                    line_axis.axvline(
                        float(switch_trial) - 0.5,
                        color="#555555",
                        linestyle=":",
                        linewidth=1.1,
                        alpha=0.32,
                        label=(
                            "Individual block/stage switch"
                            if switch_label_available
                            else "_nolegend_"
                        ),
                        zorder=1,
                    )
                    switch_label_available = False

        mean_curve = (
            subset.groupby("phase_trial", as_index=False)["success_rate"]
            .mean()
            .sort_values("phase_trial")
        )
        line_axis.plot(
            mean_curve["phase_trial"],
            mean_curve["success_rate"],
            color=colour,
            linewidth=3,
            label="Participant mean",
            zorder=5,
        )
        endpoints = subset.groupby("session_id", sort=False).tail(1)
        line_axis.scatter(
            endpoints["phase_trial"],
            endpoints["success_rate"],
            color=colour,
            edgecolor="white",
            linewidth=0.7,
            s=34,
            zorder=6,
        )
        histogram_axis.hist(
            endpoints["success_rate"],
            bins=np.linspace(0, 1, 11),
            orientation="horizontal",
            color=colour,
            alpha=0.75,
            edgecolor="white",
        )

    if phase == "training":
        line_axis.axhline(
            0.80,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="80% criterion",
        )

    session_count = subset["session_id"].nunique()
    line_axis.set_title(
        f"{_display_name(curriculum)} — {_display_name(phase)} "
        f"(n = {session_count})",
        pad=12,
    )
    line_axis.set_xlabel(f"Trial within {_display_name(phase).lower()} phase")
    line_axis.set_ylabel(f"Success rate (last {window} trials)")
    line_axis.set_ylim(-0.03, 1.03)
    line_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    line_axis.grid(True, color="#DDDDDD", linewidth=0.8, alpha=0.7)
    line_axis.spines[["top", "right"]].set_visible(False)
    if not subset.empty:
        line_axis.legend(frameon=False, loc="lower right")

    histogram_axis.set_xlabel("Sessions")
    histogram_axis.set_ylim(line_axis.get_ylim())
    histogram_axis.tick_params(axis="y", labelleft=False, left=False)
    histogram_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    histogram_axis.grid(False)
    histogram_axis.spines[["top", "right", "left"]].set_visible(False)


def _draw_interference(
    axis: plt.Axes,
    trials: pd.DataFrame,
    *,
    samples: int = INTERFERENCE_WINDOW,
) -> None:
    changes = calculate_interference(trials, samples)
    operations = ("size", "shape")
    curricula = _ordered_curricula(changes["curriculum"])

    if changes.empty:
        axis.text(
            0.5,
            0.5,
            "No sessions have complete\ntraining and test windows",
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="#666666",
        )
        axis.set_ylim(-0.5, 0.5)
    else:
        group_width = 3.0
        for curriculum_index, curriculum in enumerate(curricula):
            curriculum_changes = changes.loc[
                changes["curriculum"] == curriculum
            ]
            colour = CURRICULUM_COLOURS.get(curriculum, "#557A95")
            group_start = curriculum_index * group_width
            session_ids = sorted(curriculum_changes["session_id"].unique())
            jitters = (
                np.linspace(-0.09, 0.09, len(session_ids))
                if len(session_ids) > 1
                else [0]
            )
            session_jitters = dict(zip(session_ids, jitters))

            paired = curriculum_changes.pivot(
                index="session_id",
                columns="operation",
                values="accuracy_change",
            ).dropna(subset=list(operations))
            for session_id, participant in paired.iterrows():
                jitter = session_jitters[session_id]
                axis.plot(
                    [group_start + jitter, group_start + 1 + jitter],
                    [participant["size"], participant["shape"]],
                    color=colour,
                    linewidth=1.0,
                    alpha=0.38,
                    zorder=2,
                )

            for position, operation in enumerate(operations):
                operation_changes = curriculum_changes.loc[
                    curriculum_changes["operation"] == operation
                ]
                if operation_changes.empty:
                    continue
                mean_change = float(operation_changes["accuracy_change"].mean())
                if np.isclose(mean_change, 0.0):
                    mean_change = 0.0
                x_position = group_start + position
                axis.bar(
                    x_position,
                    mean_change,
                    width=0.62,
                    color=colour,
                    alpha=0.72,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=1,
                )
                x_values = [
                    x_position + session_jitters[session_id]
                    for session_id in operation_changes["session_id"]
                ]
                axis.scatter(
                    x_values,
                    operation_changes["accuracy_change"],
                    color=colour,
                    edgecolor="white",
                    linewidth=0.8,
                    s=48,
                    zorder=3,
                )
                vertical_offset = 0.025 if mean_change >= 0 else -0.025
                axis.text(
                    x_position,
                    mean_change + vertical_offset,
                    f"{mean_change * 100:+.1f} pp\n"
                    f"n={len(operation_changes)}",
                    ha="center",
                    va="bottom" if mean_change >= 0 else "top",
                    fontsize=8,
                    fontweight="bold",
                    color="#333333",
                )

            curriculum_label = _display_name(curriculum).replace(
                "Progressively ", "Progressively\n"
            )
            axis.text(
                group_start + 0.5,
                -0.16,
                curriculum_label,
                ha="center",
                va="top",
                fontweight="bold",
                transform=axis.get_xaxis_transform(),
            )
            if curriculum_index:
                axis.axvline(
                    group_start - 1.0,
                    color="#CCCCCC",
                    linewidth=0.9,
                    zorder=0,
                )

        largest_change = max(float(changes["accuracy_change"].abs().max()), 0.20)
        limit = min(1.0, np.ceil((largest_change + 0.08) * 10) / 10)
        axis.set_ylim(-limit, limit)

    axis.axhline(0, color="#333333", linewidth=1.2, zorder=1)
    axis.set_title("Training-to-test retention change", pad=12)
    tick_positions = [
        curriculum_index * 3.0 + operation_index
        for curriculum_index in range(len(curricula))
        for operation_index in range(len(operations))
    ]
    tick_labels = [
        "Rule 1\n(size)" if operation == "size" else "Rule 2\n(shape)"
        for _ in curricula
        for operation in operations
    ]
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    if tick_positions:
        axis.set_xlim(-0.65, tick_positions[-1] + 0.65)
    axis.set_ylabel(
        f"First {samples} test − final {samples} training accuracy"
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(True, axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_performance(
    trials: pd.DataFrame,
    output: Path,
    *,
    window: int = 10,
    dpi: int = 200,
    show: bool = False,
) -> Path:
    rolling = calculate_rolling_success(trials, window)
    curricula = _ordered_curricula(rolling["curriculum"])
    if not curricula:
        raise ValueError("No curriculum conditions were found")

    figure = plt.figure(figsize=(21.5, max(4.5, 4.2 * len(curricula))))
    grid = figure.add_gridspec(
        len(curricula),
        7,
        width_ratios=(4.5, 1.2, 0.55, 4.5, 1.2, 0.60, 3.6),
        hspace=0.42,
        wspace=0.08,
    )
    training_axes: list[plt.Axes] = []
    for row, curriculum in enumerate(curricula):
        for phase_index, phase in enumerate(PHASES):
            column = 0 if phase == "training" else 3
            shared_training_axis = training_axes[0] if training_axes else None
            line_axis = figure.add_subplot(
                grid[row, column],
                sharex=shared_training_axis if phase == "training" else None,
            )
            if phase == "training":
                training_axes.append(line_axis)
            histogram_axis = figure.add_subplot(grid[row, column + 1], sharey=line_axis)
            _draw_phase(
                line_axis,
                histogram_axis,
                rolling,
                curriculum=curriculum,
                phase=phase,
                window=window,
            )

    interference_axis = figure.add_subplot(grid[:, 6])
    _draw_interference(interference_axis, trials)

    training_trials = rolling.loc[rolling["phase"] == "training", "phase_trial"]
    if not training_trials.empty:
        longest_training = float(training_trials.max())
        shared_right_limit = longest_training + max(1.0, longest_training * 0.05)
        training_axes[0].set_xlim(0, shared_right_limit)

    figure.suptitle(
        f"Curriculum-learning performance — rolling {window}-trial success rate",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        (
            "Thin lines show individual sessions; thick lines show the participant "
            "mean. Early values use all available trials until the rolling window is "
            "full; dotted vertical lines mark individual block/stage switches. "
            "Histograms show each session's final rolling-window success rate. "
            "Interference scores are test minus operation-matched training accuracy; "
            "negative values indicate a test decline."
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.subplots_adjust(top=0.86, bottom=0.23)

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
    output = plot_performance(
        trials,
        arguments.output,
        window=arguments.window,
        dpi=arguments.dpi,
        show=arguments.show,
    )
    print(
        f"Loaded {len(trials):,} trials from "
        f"{trials['session_id'].nunique():,} sessions"
    )
    print(f"Saved performance plot to {output}")


if __name__ == "__main__":
    main()
