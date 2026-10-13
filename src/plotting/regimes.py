"""Conservative hidden-parameter regime styling; never changes plotted data."""

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .style import PDE_TRAINING_RANGES


def in_training_range(values, pde):
    """Inclusive bounds, allowing only floating-point roundoff at endpoints."""
    values = np.asarray(values)
    lower, upper = PDE_TRAINING_RANGES[pde]
    tolerance = 4 * np.finfo(values.dtype if values.dtype.kind == "f" else float).eps
    tolerance *= max(abs(lower), abs(upper))
    return (values >= lower - tolerance) & (values <= upper + tolerance)


def style_parameter_curves(axes, values, pde):
    """Style ordered parameter traces, excluding two-point reference lines.

    Each axis must contain one curve per parameter. Existing colours, markers,
    data, ordering, scales and bandwidth reference lines are retained.
    """
    mask = in_training_range(values, pde)
    for ax in np.asarray(axes, dtype=object).ravel():
        curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
        if len(curves) != len(mask):
            raise ValueError(f"expected {len(mask)} parameter curves, found {len(curves)}")
        for line, in_train in zip(curves, mask):
            line.set_linestyle("-" if in_train else "--")
            line.set_markerfacecolor(line.get_color() if in_train else "none")
            line.set_markeredgecolor(line.get_color())


def keep_training_parameter_curves(axes, values, pde):
    """Remove extrapolation artists while retaining colours and original limits.

    Numerical arrays and bandwidth reference lines are untouched. Plotting the
    full ordered grid first preserves its original parameter-to-colour mapping.
    """
    values = np.asarray(values)
    mask = in_training_range(values, pde)
    for ax in np.asarray(axes, dtype=object).ravel():
        curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
        if len(curves) != len(values):
            raise ValueError("parameter count does not match plotted curves")
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for line, in_train in zip(curves, mask):
            if not in_train:
                line.remove()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return values[mask]


def style_parameter_sweep(ax, pde, lines=None):
    """Style samples on a hidden-parameter x axis without changing any values.

    The original trace becomes dashed/open. A solid/filled overlay covers only
    training samples; NaNs prevent connecting separate training runs. Segments
    adjoining an extrapolation sample remain dashed. Error bars stay untouched.
    Limits are retained exactly, including manually fixed limits.
    """
    limits = ax.get_xlim(), ax.get_ylim()
    if lines is None:
        lines = [line for line in ax.lines
                 if len(line.get_xdata()) > 2 and line.get_marker() not in ("_", "|")]
    for line in list(lines):
        x, y = line.get_data()
        mask = in_training_range(x, pde)
        line.set_linestyle("--")
        line.set_markerfacecolor("none")
        line.set_markeredgecolor(line.get_color())
        if not np.any(mask):
            continue
        ax.plot(np.where(mask, x, np.nan), np.where(mask, y, np.nan),
                color=line.get_color(), linestyle="-", marker=line.get_marker(),
                markerfacecolor=line.get_color(), markeredgecolor=line.get_color(),
                markersize=line.get_markersize(), markeredgewidth=line.get_markeredgewidth(),
                linewidth=line.get_linewidth(), alpha=line.get_alpha(),
                zorder=line.get_zorder() + 0.1, label="_nolegend_")
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])


def grouped_parameter_legend(ax, values, pde, *, title):
    """Two compact rows explicitly identify each parameter's regime."""
    mask = in_training_range(values, pde)
    curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
    if len(curves) != len(mask):
        raise ValueError("legend parameter count does not match plotted curves")
    groups = []
    for in_train, heading in ((True, "training range"), (False, "extrapolation")):
        handles = [Line2D([], [], linestyle="none", label=heading)]
        handles.extend(line for line, flag in zip(curves, mask) if flag == in_train)
        groups.append(handles)
    height = max(map(len, groups))
    for group in groups:
        group.extend(Line2D([], [], linestyle="none", label=" ")
                     for _ in range(height - len(group)))
    # Matplotlib fills legend columns vertically. Interleave the two groups so
    # each becomes a full row below the figure, without covering gain curves.
    handles = [handle for pair in zip(*groups) for handle in pair]
    legend = ax.figure.legend(handles=handles, title=title, ncol=height,
                              loc="upper center", bbox_to_anchor=(0.5, -0.01),
                              frameon=False, fontsize=9, title_fontsize=9,
                              labelspacing=0.5, columnspacing=1.0,
                              handlelength=2.0)
    for index in (0, 1):
        legend.get_texts()[index].set_fontweight("medium")
    return legend


def add_regime_key(fig):
    """A small grayscale-readable key outside the data panels."""
    handles = [Line2D([], [], color="0.25", linestyle=style, marker="o",
                      markerfacecolor=fill, markersize=5, linewidth=1.6, label=label)
               for style, fill, label in [
                   ("-", "0.25", "training range"),
                   ("--", "none", "extrapolation")]]
    if fig.get_constrained_layout():
        return fig.legend(handles=handles, loc="outside lower center", ncol=2,
                          frameon=False, fontsize=9)
    # The plotting routines call tight_layout at save time. Reserve its bottom
    # margin explicitly in the comparison renderer and the key stays outside.
    fig._regime_key = True
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045),
                      ncol=2, frameon=False, fontsize=9)


def comparison_filename(filename):
    """Comparison names never resolve to the manuscript's current files."""
    path = Path(filename)
    return str(path.with_name(path.stem + "_regime_style" + path.suffix))


def save_comparison(fig, filename, *, dpi=300):
    """Write both formats only after checking both for accidental overwrites."""
    path = Path(comparison_filename(filename))
    paths = [path.with_suffix(ext) for ext in (".png", ".pdf")]
    for output in paths:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite comparison {output}")
    path.parent.mkdir(parents=True, exist_ok=True)
    for output in paths:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
