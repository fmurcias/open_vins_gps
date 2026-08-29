#!/usr/bin/env python3
"""
Analyze the GPS fusion evaluation runs produced by run_eval.sh.

Computes, per run:
  * ATE (position) after 4-DOF posyaw alignment -- the same alignment ov_eval uses
  * loop-closure error, which is independent of whether GPS was fused
  * for dropout runs: error at outage start, peak during, at outage end, and re-convergence time
  * GPS event counts parsed out of the estimator log

HOW TO READ THE NUMBERS (this matters more than the numbers):

  A_vio_only  ATE is a real accuracy figure. GNSS is fully independent of a VIO-only run.
  B_gps       ATE is NOT an accuracy figure. That run consumed these very fixes, so its ATE measures
              how tightly it tracks its own input. Its honest signals are loop closure and the
              comparison against A.
  C/D/E       The dropout window is the strongest result in the set: GNSS is withheld from the filter
              there but still available for scoring, so it is genuine independent ground truth for
              exactly the segment under test.

Usage:
    analyze.py <results_root> [--no-plots]

AI Generated
"""

import argparse
import math
import os
import re
import sys

import numpy as np

RUN_ORDER = ["A_vio_only", "B_gps", "C_dropout30", "D_dropout60", "E_dropout90"]


def load_traj(path):
    """Load an ov_eval-format trajectory. Returns (t[N], p[N,3])."""
    t, p = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = line.split()
            if len(v) < 4:
                continue
            t.append(float(v[0]))
            p.append([float(v[1]), float(v[2]), float(v[3])])
    return np.asarray(t), np.asarray(p)


def associate(t_est, p_est, t_gt, p_gt, max_dt=0.05):
    """Nearest-neighbour timestamp association, same idea as ov_eval's ResultTrajectory."""
    idx = np.searchsorted(t_est, t_gt)
    idx = np.clip(idx, 1, len(t_est) - 1)
    pick = np.where(np.abs(t_gt - t_est[idx - 1]) < np.abs(t_gt - t_est[idx]), idx - 1, idx)
    ok = np.abs(t_gt - t_est[pick]) <= max_dt
    return t_gt[ok], p_est[pick[ok]], p_gt[ok]


def align_posyaw(p_est, p_gt):
    """4-DOF (yaw + translation) Umeyama alignment, matching AlignTrajectory::align_posyaw.

    The VIO global frame is gravity-aligned but has an arbitrary yaw and origin, so yaw+translation is
    exactly the gauge freedom to remove -- no scale, since inertial VIO is metric.
    """
    mu_e, mu_g = p_est.mean(axis=0), p_gt.mean(axis=0)
    de, dg = p_est - mu_e, p_gt - mu_g
    # Best yaw about z in closed form (same quantity AlignUtils::get_best_yaw recovers from the SVD).
    num = np.sum(dg[:, 1] * de[:, 0] - dg[:, 0] * de[:, 1])
    den = np.sum(dg[:, 0] * de[:, 0] + dg[:, 1] * de[:, 1])
    yaw = math.atan2(num, den)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = mu_g - R @ mu_e
    return (R @ p_est.T).T + t, yaw


def stats(err):
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mean=float(np.mean(err)),
                median=float(np.median(err)), max=float(np.max(err)), std=float(np.std(err)))


# Settings that invalidate a cross-run comparison if they differ. The estimator-side keys are here
# because leaving them out already caused a wrong conclusion once: an A_vio_only cohort was re-run with
# both max_cameras 2->1 and calib_imu_intrinsics on->off, and the resulting 17% ATE change was read as
# the cost of the calibration change when halving the camera count is the likelier cause. Two variables
# moved and nothing said so. Anything that changes what the filter is doing belongs in this list.
CFG_KEYS = ["gps_enabled", "gps_noise_floor", "gps_chi2_multipler", "gps_do_calib_leverarm",
            "gps_gap_threshold_secs", "gps_gap_drift_rate", "gps_gap_max_sigma",
            "gps_dropout_start_secs", "gps_dropout_end_secs",
            "max_cameras", "use_stereo", "calib_imu_intrinsics", "calib_cam_intrinsics",
            "calib_cam_extrinsics", "max_clones", "max_slam", "num_pts", "init_dyn_use", "try_zupt"]

# Reported per run rather than only on mismatch: these change the problem enough that a reader should
# see them next to the number, not have to infer them from a warning.
CFG_HEADLINE = ["max_cameras", "calib_imu_intrinsics"]


def read_cfg(rundir):
    """Key GPS settings from the config copy each run keeps.

    Runs accumulate in the results directory across parameter changes, so a stale directory sits
    happily next to fresh ones and gets compared as if it were the same experiment -- which silently
    turns a tuning difference into apparent run-to-run variance. Reading each run's own config back is
    what lets the report catch that instead of averaging over it.
    """
    out = {}
    p = os.path.join(rundir, "config", "estimator_config.yaml")
    if not os.path.isfile(p):
        return out
    with open(p, errors="replace") as f:
        for line in f:
            for k in CFG_KEYS:
                if line.startswith(k + ":"):
                    out[k] = line.split(":", 1)[1].split("#")[0].strip()
    return out


# A healthy filter touches the IMU biases on essentially every camera frame. Measured over 33 runs, the
# longest stretch of frames with a bit-identical bias was 281 in a healthy run and 1892/2401/3134 in the
# three that failed -- a clean separation. 500 frames (~32s at this bag's ~15.8 Hz) sits ~1.8x above the
# worst healthy run and ~3.8x below the best failure.
FREEZE_FRAMES = 500


def parse_log(path):
    """Pull GPS lifecycle events and VIO health out of the estimator log."""
    out = dict(init=0, accepted=0, rejected=0, resets=0, reseeds=0,
               dropout_start=None, dropout_end=None, init_time=None, frames=0,
               gaps=0, gap_recovered=0, gap_capped=0, gap_attempts=[])
    ba_seq = []
    if not os.path.isfile(path):
        out["vio"] = vio_health(ba_seq)
        return out
    with open(path, errors="replace") as f:
        for line in f:
            if "| ba = " in line:
                # Accelerometer bias, printed once per processed camera frame. Its *movement* is the
                # health signal, not its value: an EKF update always nudges it, so a bias that stops
                # moving means no update is passing at all. See vio_health().
                m = re.search(r"ba = ([-\d.]+),([-\d.]+),([-\d.]+)", line)
                if m:
                    ba_seq.append(m.group(1, 2, 3))
            elif "E-to-G transform initialized" in line:
                out["init"] += 1
                m = re.search(r"yaw = ([-\d.]+) deg", line)
                if m and out["init_time"] is None:
                    out["init_yaw_deg"] = float(m.group(1))
            elif "update accepted at t=" in line:
                out["accepted"] += 1
            elif "update rejected at t=" in line:
                out["rejected"] += 1
            elif "resetting E-to-G transform" in line:
                out["resets"] += 1
            elif "re-seeded antenna lever arm" in line:
                out["reseeds"] += 1
            elif "opening recovery" in line:
                out["gaps"] += 1
            elif "recovered from" in line:
                # How many inflated fixes it took is the headline number: 1 means the drift model was
                # sized right, several means it under-shot and only escalation saved the run, and a gap
                # with no matching "recovered" line at all is a recovery that never closed.
                out["gap_recovered"] += 1
                m = re.search(r"after (\d+) inflated fix", line)
                if m:
                    out["gap_attempts"].append(int(m.group(1)))
            elif "inflation ceiling" in line:
                out["gap_capped"] += 1
            elif "SIMULATED OUTAGE START" in line:
                m = re.search(r"t=([\d.]+)", line)
                if m:
                    out["dropout_start"] = float(m.group(1))
            elif "SIMULATED OUTAGE END" in line:
                m = re.search(r"t=([\d.]+)", line)
                if m:
                    out["dropout_end"] = float(m.group(1))
    out["vio"] = vio_health(ba_seq)
    return out


def vio_health(ba_seq):
    """Detect update starvation from the accelerometer-bias trace.

    The failure this catches: the filter's state drifts far enough that every visual feature blows the
    chi2 gate, rejected features carry no information, so nothing corrects the state and every future
    feature fails too. The filter then dead-reckons on a frozen, wrong bias while still publishing a
    plausible-looking pose. It is silent -- no warning, no error, and the run completes with a normal
    pose count -- which is exactly why it needs detecting mechanically.

    Observed once as a 31 km trajectory error: the bias froze at |ba| = 1.53 m/s^2 for ~200 s, and
    0.5 * 1.53 * 200^2 = 30.6 km accounts for essentially all of the 31.6 km of measured loop drift.

    Note that |ba| itself is NOT a usable discriminator -- it transiently reaches ~1.2 m/s^2 during
    early convergence in healthy runs too. Only the *freeze* separates cleanly.
    """
    out = dict(prints=len(ba_seq), distinct=len(set(ba_seq)), max_freeze=0, freeze_start=None,
               final_ba=None, frozen=False)
    if not ba_seq:
        return out
    best = cur = 1
    best_end = 0
    for i in range(1, len(ba_seq)):
        cur = cur + 1 if ba_seq[i] == ba_seq[i - 1] else 1
        if cur > best:
            best, best_end = cur, i
    out["max_freeze"] = best
    out["freeze_start"] = best_end - best + 1
    out["final_ba"] = math.sqrt(sum(float(x) ** 2 for x in ba_seq[-1]))
    out["frozen"] = best >= FREEZE_FRAMES
    return out


def analyze_run(name, root, t_gt, p_gt):
    d = os.path.join(root, name)
    est_path = os.path.join(d, "traj_est.txt")
    if not os.path.isfile(est_path):
        return None
    t_est, p_est = load_traj(est_path)
    if len(t_est) < 10:
        print("  %s: only %d poses, skipping" % (name, len(t_est)), file=sys.stderr)
        return None

    ta, pe, pg = associate(t_est, p_est, t_gt, p_gt)
    if len(ta) < 10:
        print("  %s: only %d associated poses, skipping" % (name, len(ta)), file=sys.stderr)
        return None

    pe_al, yaw = align_posyaw(pe, pg)
    err = np.linalg.norm(pe_al - pg, axis=1)
    err_h = np.linalg.norm(pe_al[:, :2] - pg[:, :2], axis=1)
    err_v = np.abs(pe_al[:, 2] - pg[:, 2])

    # Loop closure: the platform returns to within 0.74 m / -0.97 m of its start per GNSS, so how far
    # the *estimate* drifts over the same loop is a fusion-independent read on accumulated drift.
    est_loop = np.linalg.norm(pe_al[-1] - pe_al[0])
    gt_loop = np.linalg.norm(pg[-1] - pg[0])

    r = dict(name=name, n=len(ta), yaw_deg=math.degrees(yaw),
             ate=stats(err), ate_h=stats(err_h), ate_v=stats(err_v),
             loop_est=float(est_loop), loop_gt=float(gt_loop),
             loop_discrepancy=float(abs(est_loop - gt_loop)),
             t=ta, err=err, log=parse_log(os.path.join(d, "run.log")), cfg=read_cfg(d))

    # Dropout-window metrics: the only place GNSS is both withheld from the filter and available to
    # score it, so these are the genuinely independent accuracy numbers in the whole evaluation.
    ds, de = r["log"]["dropout_start"], r["log"]["dropout_end"]
    if ds and de:
        pre = err[ta < ds]
        during = err[(ta >= ds) & (ta <= de)]
        post = err[ta > de]
        if len(during):
            baseline = float(np.median(pre)) if len(pre) else float("nan")
            # Recovery threshold: the pre-outage 90th percentile, i.e. the top of this run's own normal
            # error band.
            #
            # NOT 1.5x the pre-outage median, which is what this used to be. That penalised accuracy: a
            # run with an unusually tight pre-outage segment set itself an unreachably low bar and was
            # reported as recovering slowly even when its post-outage error was the lowest in the cohort.
            # It ranked C_dropout30_C_no_calib_run3 -- best ATE of its group (1.18 m) and lowest
            # post-outage error of any C run (1.63 m) -- as taking 16.2s to recover, purely because its
            # median was 1.06 m instead of ~1.6 m. Using a percentile of the same distribution tracks how
            # much that run actually varies, so the bar scales with its noise rather than its skill.
            recov_thresh = float(np.percentile(pre, 90)) if len(pre) else float("nan")
            r["dropout"] = dict(
                start=ds, end=de, secs=de - ds,
                err_at_start=float(during[0]), err_peak=float(np.max(during)),
                err_at_end=float(during[-1]), baseline=baseline, recov_thresh=recov_thresh,
                growth=float(np.max(during) - baseline),
            )
            # First post-outage sample that returns under that band AND stays under it. "Stays" is
            # deliberately for the whole remainder rather than a sliding window: a run that dips back
            # briefly and then degrades again has not recovered, and a window short enough to be
            # forgiving turns D_dropout60_run2 -- which genuinely never re-converged -- into "8.2s".
            # "Never" is itself a result worth reporting.
            recov = None
            if len(post) and not math.isnan(recov_thresh):
                t_post = ta[ta > de]
                for i in range(len(post)):
                    if np.all(post[i:] <= recov_thresh):
                        recov = float(t_post[i] - de)
                        break
            r["dropout"]["recovery_secs"] = recov
    return r


def fmt(v, unit="", nd=2):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else ("%.*f%s" % (nd, v, unit))


# A run shorter than this fraction of the cohort median is treated as truncated, not as a result.
TRUNCATED_FRACTION = 0.5


def split_invalid(results):
    """Separate runs whose estimator failed from runs that produced a usable trajectory.

    Two distinct failures, both of which silently corrupt a comparison:

    *Truncated* -- the run ended early. It still writes a short but well-formed trajectory, and scoring
    it produces a *flattering* number, because a 30-second fragment has barely had time to drift. A
    segfaulted 32-pose run scored 0.55 m and became the best `E_dropout90` result in the spread table.
    The test must be cohort-relative: no absolute pose count is simultaneously "too short to score" and
    "short enough that a healthy run never hits it".

    *Frozen* -- the run completed with a full pose count but the filter stopped updating partway
    through (see vio_health). Two such runs scored 10.9 km and 15.6 km and made the C_dropout30 spread
    read as 15,580 m.

    Both tests are on the *mechanism*, never on the ATE. Excluding runs because their error looks bad
    would silently hide exactly the regressions this script exists to find; excluding them because the
    estimator is diagnosably broken is a different thing entirely. Every excluded run is still listed
    in the report with its reason.
    """
    counts = sorted(r["n"] for r in results)
    mid = len(counts) // 2
    median = counts[mid] if len(counts) % 2 else 0.5 * (counts[mid - 1] + counts[mid])
    cutoff = TRUNCATED_FRACTION * median
    for r in results:
        v = r["log"]["vio"]
        if r["n"] < cutoff:
            r["invalid"] = ("truncated", "%d poses, below the %.0f cutoff (cohort median %.0f)"
                            % (r["n"], cutoff, median))
        elif v["frozen"]:
            r["invalid"] = ("filter froze", "no update for %d consecutive frames from frame %d; "
                            "final |ba| = %.2f m/s^2" % (v["max_freeze"], v["freeze_start"], v["final_ba"]))
        else:
            r["invalid"] = None
    return ([r for r in results if not r["invalid"]],
            [r for r in results if r["invalid"]], median, cutoff)


def report(results, invalid, cohort_median, cutoff, root):
    lines = []
    w = lines.append
    w("# GPS Fusion Evaluation\n")
    w("Ground truth is the GNSS ENU track. Read these with the circularity in mind:\n")
    w("- **A_vio_only** ATE is a real accuracy number (GNSS independent of the run).")
    w("- **B_gps** ATE is *tracking tightness*, not accuracy -- that run consumed these fixes.")
    w("- **C/D/E dropout windows** are the strongest result: GNSS withheld from the filter, still scoring it.\n")

    # Surface the noise floor per run: it is the parameter that decides whether the chi2 gate tolerates
    # the real error budget, and mixing values across runs invalidates the comparison.
    # Check every tracked key, over every run -- not just the gps_* ones over GPS-enabled runs. A
    # variant like A_vio_only has GPS off but is still invalidated by a camera-count change.
    differing = {k: sorted({r["cfg"].get(k, "?") for r in results}) for k in CFG_KEYS}
    # gps_enabled and the dropout window are what *define* the variants (A has no GPS, C/D/E have
    # different outage lengths), so they differ by design and would only dilute the warning.
    by_design = ("gps_enabled", "gps_dropout_start_secs", "gps_dropout_end_secs")
    differing = {k: v for k, v in differing.items() if len(v) > 1 and k not in by_design}
    if differing:
        w("> **Warning: these runs do NOT share a configuration.** Differing settings: %s."
          % "; ".join("`%s` = %s" % (k, "/".join(v)) for k, v in differing.items()))
        w("> Rows below are not directly comparable, and any spread mixes tuning differences with")
        w("> run-to-run variance. Compare only within a configuration -- and when two settings moved")
        w("> together, neither can be credited with the difference.\n")

    # Failed runs are excluded everywhere below, but must stay visible: a run whose estimator broke is
    # a result about robustness even though its ATE is meaningless.
    if invalid:
        w("> **%d run(s) had an estimator failure and are excluded from every table below.** Their ATE "
          "is not a measure of anything -- but the failure rate is:\n>" % len(invalid))
        for r in sorted(invalid, key=lambda x: x["name"]):
            w(">   - `%s` -- **%s**: %s (scored %s)."
              % (r["name"], r["invalid"][0], r["invalid"][1], fmt(r["ate"]["rmse"], " m")))
        w(">\n> %d of %d runs failed (%.0f%%)." % (len(invalid), len(invalid) + len(results),
                                                   100.0 * len(invalid) / (len(invalid) + len(results))))
        w("> A frozen filter is the silent one: the run completes with a full pose count and keeps")
        w("> publishing a plausible pose while dead-reckoning on a stale bias. See `vio_health()`.\n")

    w("## Trajectory error (posyaw-aligned, position only)\n")
    w("| run | cams | imu calib | noise floor | poses | ATE rmse | ATE max | horiz rmse | vert rmse | loop drift (est vs GNSS) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        nf = r["cfg"].get("gps_noise_floor", "?") if r["cfg"].get("gps_enabled") == "true" else "-"
        w("| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s vs %s |" % (
            r["name"], r["cfg"].get("max_cameras", "?"), r["cfg"].get("calib_imu_intrinsics", "?"),
            nf, r["n"], fmt(r["ate"]["rmse"], " m"), fmt(r["ate"]["max"], " m"),
            fmt(r["ate_h"]["rmse"], " m"), fmt(r["ate_v"]["rmse"], " m"),
            fmt(r["loop_est"], " m"), fmt(r["loop_gt"], " m")))

    w("\n## GPS lifecycle (from estimator logs)\n")
    w("| run | init fired | accepted | rejected | reject rate | transform resets | lever-arm re-seeds |")
    w("|---|---|---|---|---|---|---|")
    for r in results:
        lg = r["log"]
        tot = lg["accepted"] + lg["rejected"]
        rate = "%.1f%%" % (100.0 * lg["rejected"] / tot) if tot else "n/a"
        w("| %s | %s | %d | %d | %s | %d | %d |" % (
            r["name"], "yes" if lg["init"] else "NO", lg["accepted"], lg["rejected"], rate,
            lg["resets"], lg["reseeds"]))

    w("\n## VIO health (update starvation)\n")
    w("`max freeze` is the longest run of consecutive camera frames over which the accelerometer bias")
    w("did not move at all -- i.e. no visual update passed the chi2 gate. Healthy runs sit in the tens;")
    w("anything approaching %d means the filter was dead-reckoning. `final |ba|` is context, not a" % FREEZE_FRAMES)
    w("verdict: it transiently reaches ~1.2 m/s2 during early convergence in healthy runs too.\n")
    w("| run | frames | bias updates | max freeze | final \\|ba\\| | verdict |")
    w("|---|---|---|---|---|---|")
    for r in sorted(results + invalid, key=lambda x: x["name"]):
        v = r["log"]["vio"]
        if not v["prints"]:
            continue
        bad = r["invalid"] is not None
        w("| %s | %d | %d | %s | %.2f m/s2 | %s |" % (
            r["name"], v["prints"], v["distinct"],
            ("**%d**" % v["max_freeze"]) if bad else str(v["max_freeze"]),
            v["final_ba"], ("**%s**" % r["invalid"][0]) if bad else "ok"))

    if any(r["log"]["gaps"] for r in results):
        w("\n## Post-outage covariance recovery\n")
        w("A gap that opens but never closes is the failure this mechanism exists to prevent: it means the\n"
          "returning fixes stayed rejected, so GPS contributed nothing after the outage.\n")
        w("| run | gaps detected | recovered | fixes needed to recover | hit inflation ceiling |")
        w("|---|---|---|---|---|")
        for r in results:
            lg = r["log"]
            if not lg["gaps"]:
                continue
            att = "/".join(str(a) for a in lg["gap_attempts"]) or "-"
            unrec = lg["gaps"] - lg["gap_recovered"]
            rec = "%d/%d" % (lg["gap_recovered"], lg["gaps"])
            if unrec:
                rec = "**" + rec + "**"
            w("| %s | %d | %s | %s | %s |" % (
                r["name"], lg["gaps"], rec, att, "**yes**" if lg["gap_capped"] else "no"))

    drop = [r for r in results if "dropout" in r]
    if drop:
        w("\n## Outage behaviour (independent ground truth)\n")
        w("| run | outage | err at start | peak during | err at end | growth vs baseline | re-converged |")
        w("|---|---|---|---|---|---|---|")
        for r in drop:
            d = r["dropout"]
            rec = "%.1f s" % d["recovery_secs"] if d.get("recovery_secs") is not None else "**never**"
            w("| %s | %.0f s | %s | %s | %s | %s | %s |" % (
                r["name"], d["secs"], fmt(d["err_at_start"], " m"), fmt(d["err_peak"], " m"),
                fmt(d["err_at_end"], " m"), fmt(d["growth"], " m"), rec))

    # Repeat spread. These runs are not reproducible at the level that matters -- identical data volume,
    # but threading in the GPS drain path lets delayed init land on different geometry -- so a single
    # run cannot separate "this setting is better" from "this run got lucky". Where repeats exist,
    # the spread is the result, not the mean.
    # Group on (variant, noise floor), not variant alone: a leftover run from an earlier setting shares
    # the variant name but is a different experiment, and folding it in reports a tuning delta as though
    # it were run-to-run noise.
    # Group by the full headline configuration, not just the run-name prefix. Two runs sharing a name
    # prefix but differing in camera count are not repeats of the same experiment, and averaging them
    # into one spread would hide exactly the confound the warning above exists to catch.
    groups = {}
    for r in results:
        base = re.sub(r"_run\d+$", "", r["name"])
        sig = (base, r["cfg"].get("gps_noise_floor", "?"),
               tuple(r["cfg"].get(k, "?") for k in CFG_HEADLINE))
        groups.setdefault(sig, []).append(r)
    repeated = {("%s (nf=%s, cams=%s, imu_calib=%s)" % (k[0], k[1], k[2][0], k[2][1])): v
                for k, v in groups.items() if len(v) > 1}
    if repeated:
        w("\n## Repeat spread (run-to-run variability)\n")
        w("| variant | n | ATE rmse: min / median / max | spread | rejected fixes | transform resets |")
        w("|---|---|---|---|---|---|")
        for k, v in sorted(repeated.items()):
            a = sorted(x["ate"]["rmse"] for x in v)
            rej = [x["log"]["rejected"] for x in v]
            res = [x["log"]["resets"] for x in v]
            med = a[len(a) // 2] if len(a) % 2 else 0.5 * (a[len(a) // 2 - 1] + a[len(a) // 2])
            w("| %s | %d | %.2f / %.2f / %.2f m | **%.2f m** | %s | %s |" % (
                k, len(v), a[0], med, a[-1], a[-1] - a[0],
                "/".join(str(x) for x in rej), "/".join(str(x) for x in res)))
        w("\nA spread comparable to the difference between configurations means single-run comparisons")
        w("are not conclusive; prefer the worst case over the mean when judging robustness.\n")

    # Reproducibility guard: SensorDataQoS is best-effort, so runs can differ in how many frames they
    # actually processed. If they do, cross-run comparison is confounded and the numbers above are not
    # comparable -- surface that rather than letting it pass silently. Truncated runs are already
    # excluded, so this now fires on genuine dropped-frame variation instead of on a crash.
    counts = [r["n"] for r in results]
    if counts and (max(counts) - min(counts)) > 0.05 * max(counts):
        w("\n> **Warning:** associated-pose counts vary by more than 5%% across completed runs (%s)." % counts)
        w("> Subscriptions use `SensorDataQoS` (best-effort), so runs may have processed different")
        w("> amounts of data and are not strictly comparable. Consider raising QoS depth/reliability.")

    text = "\n".join(lines) + "\n"
    out = os.path.join(root, "REPORT.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print("wrote %s" % out)


def make_plots(results, root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for r in results:
        t0 = r["t"][0]
        ax.plot(r["t"] - t0, r["err"], label=r["name"], lw=1.2)
        if "dropout" in r:
            ax.axvspan(r["dropout"]["start"] - t0, r["dropout"]["end"] - t0, alpha=0.10, color="red")
    ax.set_xlabel("time since first scored pose (s)")
    ax.set_ylabel("position error vs GNSS (m)")
    ax.set_title("Position error over time (shaded = simulated GPS outage)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(root, "error_vs_time.png")
    fig.savefig(p, dpi=130)
    print("wrote %s" % p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="results root (OUT_ROOT from run_eval.sh)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    gt_path = os.path.join(args.root, "groundtruth.txt")
    if not os.path.isfile(gt_path):
        sys.exit("ERROR: no groundtruth.txt in %s -- run './run_eval.sh gt' first" % args.root)
    t_gt, p_gt = load_traj(gt_path)
    print("ground truth: %d poses, %.1f s\n" % (len(t_gt), t_gt[-1] - t_gt[0]))

    names = [n for n in RUN_ORDER if os.path.isdir(os.path.join(args.root, n))]
    names += sorted(n for n in os.listdir(args.root)
                    if os.path.isdir(os.path.join(args.root, n)) and n not in RUN_ORDER)
    results = [r for r in (analyze_run(n, args.root, t_gt, p_gt) for n in names) if r]
    if not results:
        sys.exit("ERROR: no analyzable runs found in %s" % args.root)

    results, invalid, cohort_median, cutoff = split_invalid(results)
    for r in invalid:
        print("  %s: %s -- %s; excluded from the tables"
              % (r["name"], r["invalid"][0], r["invalid"][1]), file=sys.stderr)
    if not results:
        sys.exit("ERROR: every run in %s had an estimator failure -- nothing comparable to report" % args.root)

    report(results, invalid, cohort_median, cutoff, args.root)
    if not args.no_plots:
        make_plots(results, args.root)


if __name__ == "__main__":
    main()
