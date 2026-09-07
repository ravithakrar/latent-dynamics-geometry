r"""plotstyle.py — one place that decides how every figure in this project looks.

    import plotstyle as ps
    ps.apply()                               # everyday: fast, no LaTeX needed
    ps.apply(usetex=True)                    # final renders: real LaTeX typesetting
    fig = plt.figure(figsize=(ps.WIDTH, 4.2))
    ps.save(fig, "figures/fig_name")         # writes .pdf and .png, reports the mode

The base style is `science` from SciencePlots. apply() adds `no-latex`, which renders STIX
serif text with Computer Modern maths through matplotlib and needs no LaTeX installation;
apply(usetex=True) leaves text.usetex on, so every string is typeset by LaTeX itself, which
needs a working LaTeX and is slower.

WIDTH is 6.30 in, the \textwidth of an A4 page with 25 mm margins. Author every figure at its
final printed size: a figure placed at \textwidth is scaled by LaTeX, and the font size scales
with it.

With usetex=True a literal %, &, _, # or ~ in a label must be escaped and monospace runs need
\texttt{}.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401  registers the style sheets
    HAVE_SCIENCEPLOTS = True
except ImportError:                       # degrade rather than crash
    HAVE_SCIENCEPLOTS = False

#: \textwidth of the thesis in inches (a4 210 mm - 2 x 25 mm margins).
WIDTH = 6.30
#: a comfortable height for a figure that should not dominate a page.
HALF = 3.15

# project palette, unchanged, for the places that name a colour explicitly
M1 = "#0C5DA5"        # SciencePlots blue
PMD = "#FF2C00"       # SciencePlots red
INK = "#1b1b1a"
MUTED = "#5c5c5a"
SURF = "white"
GRID = "#dcdcda"
ACCENT = "#FF9500"    # SciencePlots orange
COL = {"M1": M1, "PMd": PMD}

#: what this project changes on top of `science`.
_RC = {
    "figure.figsize": (WIDTH, 4.2),   # thesis \textwidth, not a journal column
    "font.size": 8,                   # suits a figure printed at WIDTH with no scaling
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.5,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "savefig.bbox": None,             # `science` sets "tight", which crops hand-placed captions
    "savefig.pad_inches": 0.02,
    "grid.color": GRID,
    "pdf.fonttype": 42,               # embed real glyphs, not Type 3
    "ps.fonttype": 42,
}

_MODE = "not applied"


def apply(usetex=False, extra=None):
    """Set the project style. `extra` overrides anything for one figure."""
    global _MODE
    styles = ["science"] if usetex else ["science", "no-latex"]
    if HAVE_SCIENCEPLOTS:
        plt.style.use(styles)
        _MODE = f"SciencePlots {'+'.join(styles)}"
    else:                              # SciencePlots missing: keep going, look plainer
        matplotlib.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                                    "text.usetex": bool(usetex)})
        _MODE = "fallback (SciencePlots not installed -- pip install SciencePlots)"
    matplotlib.rcParams.update(_RC)
    if extra:
        matplotlib.rcParams.update(extra)


def mode():
    return _MODE


def save(fig, stem, exts=("pdf", "png"), dpi=300, tight=False):
    """Write the same figure as a vector PDF for LaTeX and a PNG to look at."""
    import os
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in exts:
        fig.savefig(f"{stem}.{ext}", dpi=dpi, facecolor=SURF,
                    **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    print(f"wrote {stem}.{'/.'.join(exts)}  [{_MODE}]", flush=True)


if __name__ == "__main__":
    apply()
    print("mode:", mode())
    print("SciencePlots installed:", HAVE_SCIENCEPLOTS)
    print("usetex:", matplotlib.rcParams["text.usetex"],
          "| family:", matplotlib.rcParams["font.family"],
          "| mathtext:", matplotlib.rcParams["mathtext.fontset"])
