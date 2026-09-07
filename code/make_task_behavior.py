"""make_task_behavior.py -- the task and the stability of the behaviour.

  A  the recording setup, drawn only when PANEL_A below points at an existing file.
     Beside it, in the same row, hand trajectories to the eight targets on three
     sessions spanning three years, and x and y hand velocity for one target on
     those same days.
  B  trial structure, with the epoch durations measured from the trial tables, the
     eight targets, and the -100 to +400 ms analysis window drawn on the axis.
  C  correlation of direction-matched hand velocity between every pair of sessions,
     against the interval between them.
  D  the distribution of those correlations.

Reads behav_cache.npz.

Usage:  python3 make_task_behavior.py
Output: figures/fig_task_behavior.pdf/.png
"""
from __future__ import annotations
import itertools, datetime
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotstyle as ps

#: x and y here are the two axes of the HAND, not brain areas. Deliberately not
#: ps.M1 / ps.PMD, which mean primary motor and dorsal premotor everywhere else.
CX, CY = "#00B945", "#845B97"       # SciencePlots green and purple

CACHE = "behav_cache.npz"
#: three sessions spanning the full range, for panel B
SHOW = ["20131022", "20150312", "20161021"]
#: the 2016 arm used by the 14-session drift analysis
ARM2016 = ["20160909", "20160912", "20160914", "20160915", "20160919", "20160921",
           "20160923", "20160929", "20161005", "20161006", "20161007", "20161011",
           "20161013", "20161021"]
GRID_MS = np.arange(-100, 400, 20)


def load():
    z = np.load(CACHE, allow_pickle=True)
    sess = sorted({k.split("|")[0] for k in z.files})
    return z, sess


def day(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:])).toordinal()


def cond_mean_vel(z, s):
    """(8, 25, 2) direction-averaged velocity, directions in angular order."""
    v, t = z[f"{s}|vel"], z[f"{s}|targ"]
    dirs = np.unique(t)
    return np.stack([v[t == d].mean(0) for d in dirs]), dirs


#: optional image for panel A; the panel is left empty when it is absent
PANEL_A = "figures/fig_task_apparatus.png"


def main():
    ps.apply()
    z, sess = load()
    print("sessions:", len(sess))

    # panel C/D
    cm = {s: cond_mean_vel(z, s)[0] for s in sess}
    peak = {s: float(np.linalg.norm(cm[s], axis=-1).max()) for s in sess}
    pairs, dt, rx, ry = [], [], [], []
    for a, b in itertools.combinations(sess, 2):
        A, B = cm[a], cm[b]
        if A.shape != B.shape:
            continue
        pairs.append((a, b)); dt.append(abs(day(b) - day(a)))
        rx.append(np.corrcoef(A[..., 0].ravel(), B[..., 0].ravel())[0, 1])
        ry.append(np.corrcoef(A[..., 1].ravel(), B[..., 1].ravel())[0, 1])
    dt, rx, ry = np.array(dt), np.array(rx), np.array(ry)
    in2016 = np.array([a in ARM2016 and b in ARM2016 for a, b in pairs])
    print(f"pairs={len(dt)}  2016-arm={in2016.sum()}  span={dt.min()}-{dt.max()} d")
    print(f"r_x {rx.mean():.3f}+-{rx.std():.3f}   r_y {ry.mean():.3f}+-{ry.std():.3f}")
    print(f"2016 arm: r_x {rx[in2016].mean():.3f}  r_y {ry[in2016].mean():.3f}")
    pv = np.array([peak[s] for s in sess])
    print(f"peak speed {pv.min():.1f}-{pv.max():.1f} cm/s, ratio {pv.max()/pv.min():.2f}")

    fig = plt.figure(figsize=(ps.WIDTH, 4.95))
    gs = fig.add_gridspec(3, 26, height_ratios=[1.05, .78, .95],
                          hspace=.58, wspace=2.20,
                          left=.075, right=.980, top=.935, bottom=.085)

    # B: trial structure
    axA = fig.add_subplot(gs[1, :])
    delay = np.concatenate([z[f"{s}|delay"] for s in sess])
    rt = np.concatenate([z[f"{s}|rt"] for s in sess])
    dur = np.concatenate([z[f"{s}|dur"] for s in sess])
    q = lambda x: np.nanpercentile(x, [25, 50, 75]) * 1000

    # event times laid out from the median durations, target onset at 0
    d50, r50, m50 = q(delay)[1], q(rt)[1], q(dur)[1]
    xs = [0, d50, d50 + r50, d50 + r50 + m50]
    names = ["target\non", "go cue", "movement\nonset", "reward"]
    axA.plot(xs, [0] * 4, "-", color=ps.MUTED, lw=1.0, zorder=1)
    for x, n in zip(xs, names):
        axA.plot([x], [0], "|", color=ps.INK, ms=11, mew=1.4, zorder=3)
        axA.annotate(n, (x, .30), ha="center", va="bottom", fontsize=7, color=ps.INK)
    # one bar per epoch, spanning the interval it occupies on the axis, with the
    # interquartile range of its duration as a whisker on the closing edge
    for i, ((lo, md, hi), x0, x1, lab) in enumerate(
            ((q(delay), xs[0], xs[1], "instructed delay"),
             (q(rt), xs[1], xs[2], "reaction"),
             (q(dur), xs[2], xs[3], "movement"))):
        yb = -.30 if i % 2 == 0 else -.62
        axA.plot([x0 + 6, x1 - 6], [yb, yb], "-", color=ps.MUTED, lw=3.2,
                 solid_capstyle="butt", alpha=.40, zorder=2)
        axA.annotate(f"{lab}   {md:.0f} ms",
                     ((x0 + x1) / 2, yb - .10), ha="center", va="top",
                     fontsize=6.2, color=ps.MUTED)
    axA.axvspan(xs[2] - 100, xs[2] + 400, color=ps.M1, alpha=.13, zorder=0)
    axA.annotate("analysis window\n$-100$ to $+400$ ms", (xs[2] + 150, 1.02),
                 ha="center", va="bottom", fontsize=6.5, color=ps.M1)
    axA.set_xlim(-140, xs[3] + 220); axA.set_ylim(-1.30, 1.35)
    axA.set_xlabel("time from target onset (ms)")
    axA.set_title("B", loc="left", pad=14)
    axA.set_yticks([]); axA.spines[["left", "right", "top"]].set_visible(False)
    axA.tick_params(top=False, left=False, right=False, which="both")


    axS = fig.add_subplot(gs[0, 0:6])
    dirs8 = np.unique(z[f"{SHOW[-1]}|targ"])
    cmap = plt.get_cmap("twilight")
    cols = {d: cmap((i + .5) / len(dirs8)) for i, d in enumerate(dirs8)}
    if os.path.exists(PANEL_A):
        axS.imshow(plt.imread(PANEL_A), interpolation="antialiased")
    axS.set_axis_off()
    axS.set_title("A", loc="left")

    # hand trajectories, top row beside A
    d0 = day(SHOW[0])
    for j, s in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, 7 + 4*j : 11 + 4*j])
        P, t = z[f"{s}|pos"], z[f"{s}|targ"]
        for d in np.unique(t):
            for k in np.flatnonzero(t == d)[:14]:
                ax.plot(P[k, :, 0], P[k, :, 1], lw=.45, color=cols[d], alpha=.75)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"day {day(s) - d0}", loc="center", fontsize=7.5, pad=2)
        if j == 0:
            ax.annotate("hand movement", (-.14, .5), xycoords="axes fraction",
                        rotation=90, ha="center", va="center", fontsize=6.8,
                        color=ps.INK)
            ax.plot([-4, 1], [-8.5, -8.5], "-", color=ps.INK, lw=1.1)
            ax.annotate("5 cm", (-1.5, -9.4), ha="center", va="top", fontsize=6.5)

    axV = fig.add_subplot(gs[0, 20:26])
    tgt = dirs8[np.argmin(np.abs(dirs8))]   # the rightward target
    for j, s in enumerate(SHOW):
        v, t = z[f"{s}|vel"], z[f"{s}|targ"]
        m = v[t == tgt].mean(0)
        sh = ["-", "--", ":"][j]
        axV.plot(GRID_MS, m[:, 0], sh, lw=1.1, color=CX)
        axV.plot(GRID_MS, m[:, 1], sh, lw=1.1, color=CY)
    axV.axvline(0, color=ps.MUTED, lw=.6)
    axV.set_xlabel("ms from onset"); axV.set_ylabel("cm s$^{-1}$", labelpad=1.5)
    axV.legend(handles=[Line2D([], [], color=CX, lw=1.1, label="$x$ velocity"),
                        Line2D([], [], color=ps.MUTED, lw=1.0, ls="-", label="day 0"),
                        Line2D([], [], color=CY, lw=1.1, label="$y$ velocity"),
                        Line2D([], [], color=ps.MUTED, lw=1.0, ls="--", label="day 506"),
                        Line2D([], [], color=ps.MUTED, lw=1.0, ls=" ", label=" "),
                        Line2D([], [], color=ps.MUTED, lw=1.0, ls=":", label="day 1095")],
               loc="lower center", bbox_to_anchor=(.44, 1.01), ncol=3, fontsize=5.4,
               handlelength=1.1, columnspacing=.7, handletextpad=.3,
               labelspacing=.2, borderpad=.15, frameon=False)

    # C
    axC = fig.add_subplot(gs[2, 0:12])
    for r, c, lab in ((rx, CX, "$x$ velocity"), (ry, CY, "$y$ velocity")):
        axC.plot(dt, r, "o", ms=2.4, color=c, alpha=.5, mec="none", label=lab)
        k = np.polyfit(dt, r, 1)
        xx = np.linspace(dt.min(), dt.max(), 50)
        axC.plot(xx, np.polyval(k, xx), "-", color=c, lw=1.1)
    axC.set_ylim(0, 1.02); axC.set_xlabel("days between sessions")
    axC.set_ylabel("hand velocity correlation")
    axC.legend(loc="lower right", fontsize=6.5, handlelength=1.0, borderpad=.25,
               frameon=False)
    axC.set_title("C", loc="left")

    # D
    axD = fig.add_subplot(gs[2, 16:26])
    # the range must cover the whole distribution: with all 52 sessions the lowest
    # pair correlation is 0.83, and a 0.90 floor silently cropped 150 pairs
    bins = np.linspace(.80, 1.0, 41)
    for r, c, lab in ((rx, CX, "$x$"), (ry, CY, "$y$")):
        axD.hist(r, bins=bins, color=c, alpha=.55, label=lab,
                 weights=100 * np.ones_like(r) / len(r))
        axD.errorbar(r.mean(), 21.6, xerr=r.std(), fmt="o", ms=3, lw=1.0, color=c,
                     capsize=1.8, clip_on=False)
    axD.set_xlim(.80, 1.0); axD.set_ylim(0, 21)
    axD.set_xlabel("hand velocity correlation")
    axD.set_ylabel("session pairs (%)")
    axD.legend(loc="upper left", fontsize=6.5, handlelength=1.0, borderpad=.25,
               frameon=False)
    axD.set_title("D", loc="left")

    ps.save(fig, "figures/fig_task_behavior")
    print("mode:", ps.mode())


if __name__ == "__main__":
    main()
