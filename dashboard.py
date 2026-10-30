#!/usr/bin/env python3
"""
=============================================================================
 F1 TELEMETRY & RACE ANALYSIS DASHBOARD  |  Ferrari (LEC) vs Red Bull (VER)
 Circuit: Autodromo Nazionale Monza (stylised vector layout)
-----------------------------------------------------------------------------
 Stack   : Dash, Dash Bootstrap Components, Plotly, Pandas, NumPy
 Run     : pip install dash dash-bootstrap-components plotly pandas numpy
           python dashboard.py        ->  http://127.0.0.1:8050

 Notes
 - All data is SYNTHETIC but physically modelled (see section 3):
   curvature -> corner speed limits -> braking / acceleration passes ->
   speed, throttle, brake, gear, RPM and ERS traces for each car.
 - Race data comes from a lap-by-lap simulation with tyre degradation,
   fuel burn, pit stops, dirty air and an overtaking rule.
 - Requires dash >= 2.9 (uses Patch for flicker-free animation).
=============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, Patch, ctx, dash_table, dcc, html, no_update

# =============================================================================
# 1. DESIGN TOKENS
# =============================================================================
BG = "#0B0C10"
PANEL = "#11131A"
PANEL_HI = "#171A23"
BORDER = "#262B38"
GRIDC = "#222736"
TEXT = "#E9ECF1"
MUTED = "#8D96A8"

FERRARI = "#EF1A2D"      # Scuderia Ferrari - Racing Red
RB_NAVY = "#0C2340"      # Red Bull Racing - Navy
RB_YELLOW = "#FFCC00"    # Red Bull Racing - Yellow

THROTTLE_GREEN = "#00E676"
BRAKE_RED = "#FF4D5E"
ERS_COLORS = {"DEPLOY": "#00D2FF", "HARVEST": "#B388FF", "BALANCED": MUTED}

TYRE_COLORS = {"Soft": "#FF2D2D", "Medium": "#FFD60A", "Hard": "#F5F5F5"}   # FIA colours
TYRE_TEXT = {"Soft": "#FFFFFF", "Medium": "#111111", "Hard": "#111111"}

FONT = "Titillium Web, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
MONO = "Orbitron, Consolas, monospace"

DRIVERS = {
    "LEC": dict(code="LEC", name="Charles Leclerc", team="Scuderia Ferrari",
                line=FERRARI, fill=FERRARI, edge="#FFFFFF", glow="rgba(239,26,45,0.30)"),
    "VER": dict(code="VER", name="Max Verstappen", team="Red Bull Racing",
                line=RB_YELLOW, fill=RB_NAVY, edge=RB_YELLOW, glow="rgba(255,204,0,0.28)"),
}
CODES = ("LEC", "VER")
SHORT = {"LEC": "FER", "VER": "RBR"}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =============================================================================
# 2. TRACK GEOMETRY  (closed Catmull-Rom spline, Monza-inspired)
# =============================================================================
TRACK_LENGTH_M = 5793.0
N_PTS = 1200
SECTOR_FRAC = (0.31, 0.67)

CTRL_PTS = np.array([
    (146, 230), (150, 420), (158, 620), (165, 700),                  # pit straight
    (190, 738), (168, 775), (205, 818),                              # Variante del Rettifilo
    (275, 880), (390, 915), (520, 918), (640, 896),                  # Curva Grande
    (700, 884), (728, 860), (752, 875),                              # Variante della Roggia
    (800, 850), (830, 800),                                          # Lesmo 1
    (836, 735), (800, 685),                                          # Lesmo 2
    (700, 585), (610, 470), (545, 380),                              # Serraglio straight
    (505, 335), (488, 308), (505, 282), (470, 255),                  # Variante Ascari
    (420, 205), (350, 140),
    (300, 100), (230, 78), (170, 95), (138, 145), (138, 200),        # Parabolica
], dtype=float)

CORNERS_RAW = {
    "Rettifilo": (178, 757), "Curva Grande": (400, 916), "Roggia": (728, 862),
    "Lesmo 1": (815, 826), "Lesmo 2": (822, 712), "Ascari": (490, 300),
    "Parabolica": (205, 80),
}


def catmull_rom_closed(pts: np.ndarray, per_seg: int = 80) -> np.ndarray:
    n = len(pts)
    out = []
    t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def smooth_circular(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    if w % 2 == 0:
        w += 1
    pad = w // 2
    ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
    return np.convolve(ext, np.ones(w) / w, mode="valid")


def build_track() -> dict:
    dense = catmull_rom_closed(CTRL_PTS)
    closed = np.vstack([dense, dense[:1]])
    seg = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    scale = TRACK_LENGTH_M / cum[-1]
    cum_m = cum * scale
    grid = np.linspace(0, TRACK_LENGTH_M, N_PTS, endpoint=False)
    x = np.interp(grid, cum_m, closed[:, 0] * scale)
    y = np.interp(grid, cum_m, closed[:, 1] * scale)
    ds = TRACK_LENGTH_M / N_PTS

    dx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * ds)
    dy = (np.roll(y, -1) - np.roll(y, 1)) / (2 * ds)
    ddx = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / ds ** 2
    ddy = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / ds ** 2
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
    kappa = smooth_circular(kappa, 9)

    corners = {}
    for name, (cx, cy) in CORNERS_RAW.items():
        cx, cy = cx * scale, cy * scale
        idx = int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))
        corners[name] = dict(idx=idx, s=idx * ds, x=x[idx], y=y[idx])

    return dict(N=N_PTS, ds=ds, x=x, y=y, kappa=kappa,
                xc=np.append(x, x[0]), yc=np.append(y, y[0]),
                sc=np.arange(N_PTS + 1) * ds, corners=corners)


TRACK = build_track()


# =============================================================================
# 3. VEHICLE / LAP SIMULATION MODEL
# =============================================================================
@dataclass(frozen=True)
class Car:
    code: str
    vmax_kph: float      # top speed in low-drag trim
    mu_grip: float       # mechanical lateral acceleration (m/s^2)
    downforce_k: float   # aero lateral grip growth (1/m)
    accel0: float        # launch-phase longitudinal acceleration (m/s^2)
    brake: float         # braking deceleration (m/s^2)


# Ferrari: stronger power unit / straight-line speed.  Red Bull: stronger
# mechanical + aero grip and traction in slow corners.
CARS = {
    "LEC": Car("LEC", vmax_kph=353.0, mu_grip=17.6, downforce_k=0.00455, accel0=12.4, brake=50.5),
    "VER": Car("VER", vmax_kph=346.0, mu_grip=17.8, downforce_k=0.00470, accel0=12.4, brake=52.0),
}

GEAR_EDGES = np.array([0, 72, 108, 143, 178, 216, 256, 296], dtype=float)  # km/h at which gear n starts
RACE_SPEED_SCALE = 0.955   # race-trim / fuel-heavy / tyre management factor


def gear_and_rpm(v_kph):
    v = np.atleast_1d(np.asarray(v_kph, dtype=float))
    gear = np.searchsorted(GEAR_EDGES, v, side="right").clip(1, 8)
    lo = GEAR_EDGES[gear - 1]
    hi = np.where(gear < 8, GEAR_EDGES[np.minimum(gear, 7)], 355.0)
    frac = np.clip((v - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rpm = np.where(gear < 8, 8300 + frac * (12400 - 8300), 9400 + frac * (11900 - 9400))
    return gear, rpm


def simulate_lap(car: Car, track: dict, speed_scale: float = 1.0) -> dict:
    N, ds, kappa = track["N"], track["ds"], track["kappa"]
    vmax = car.vmax_kph / 3.6
    denom = kappa - car.downforce_k
    with np.errstate(divide="ignore", invalid="ignore"):
        v_corner = np.where(denom > 1e-6, np.sqrt(car.mu_grip / denom), vmax)
    v_lim = np.clip(v_corner, 19.0, vmax)

    v = v_lim.copy()
    for _ in range(3):
        for i in range(2 * N - 1, -1, -1):                      # braking pass (backwards, 2 laps)
            j, k = i % N, (i + 1) % N
            v[j] = min(v[j], math.sqrt(v[k] ** 2 + 2 * car.brake * ds))
        for i in range(2 * N):                                   # acceleration pass (forwards, 2 laps)
            j, k = i % N, (i + 1) % N
            a = car.accel0 * (1 - (v[j] / vmax) ** 1.5) + 0.25
            v[k] = min(v[k], math.sqrt(v[j] ** 2 + 2 * a * ds), v_lim[k])
    v = v * speed_scale

    a_long = smooth_circular((np.roll(v, -1) ** 2 - v ** 2) / (2 * ds), 7)
    straight = kappa < 1 / 450
    brake = np.where(a_long < -4.0, np.clip(-a_long / (0.92 * car.brake), 0, 1), 0.0)
    thr = np.where(a_long > 0, 0.60 + 0.40 * np.clip(a_long / 2.0, 0, 1),
                   np.clip(0.60 * (1 + a_long / 4.0), 0, 0.60))
    thr = np.where(straight & (a_long > -1.5), 1.0, thr)
    thr = np.where(brake > 0.02, 0.0, thr)
    thr = np.clip(smooth_circular(thr, 5), 0, 1)
    brake = np.clip(smooth_circular(brake, 3), 0, 1)

    v_kph = v * 3.6
    gear, rpm = gear_and_rpm(v_kph)

    deploy = (thr >= 0.98) & (v_kph < 335)
    harvest = (brake > 0.05) | ((thr < 0.30) & (a_long < 0))
    ers_code = np.where(deploy, 1, np.where(harvest, 2, 0))
    delta = np.where(deploy, -0.0085, 0.0) * ds + np.where(harvest, 0.030, 0.0) * ds
    soc = np.clip(60 + np.cumsum(delta - delta.mean()), 5, 100)

    def close(arr):
        return np.append(arr, arr[0])

    v_c = close(v)
    t = np.concatenate([[0.0], np.cumsum(2 * ds / (v_c[:-1] + v_c[1:]))])
    idx1, idx2 = int(N * SECTOR_FRAC[0]), int(N * SECTOR_FRAC[1])
    sectors = np.array([t[idx1], t[idx2] - t[idx1], t[-1] - t[idx2]])

    seg = np.array_split(np.arange(N), 10)
    thr_seg = np.array([thr[s].mean() for s in seg])
    brk_seg = np.array([brake[s].mean() for s in seg])

    return dict(code=car.code, t=t, s=track["sc"], v=close(v_kph), thr=close(thr) * 100,
                brk=close(brake) * 100, gear=close(gear), rpm=close(rpm),
                ers=close(ers_code), soc=close(soc), lap_time=float(t[-1]),
                sectors=sectors, sector_frac=sectors / t[-1], top_speed=float(v_kph.max()),
                thr_seg=thr_seg, brk_seg=brk_seg)


PROFILES = {
    "quali": {c: simulate_lap(CARS[c], TRACK, 1.0) for c in CODES},
    "race": {c: simulate_lap(CARS[c], TRACK, RACE_SPEED_SCALE) for c in CODES},
}


# =============================================================================
# 4. SESSION DATA ENGINEERING  (Qualifying + Race -> one telemetry log)
# =============================================================================
RACE_LAPS = 50
FUEL_GAIN = 0.035          # s/lap faster per lap of fuel burnt
PIT_LOSS = 21.4            # s
COMPOUND_OFFSET = {"Soft": -0.65, "Medium": 0.0, "Hard": 0.48}
COMPOUND_DEG = {"Soft": 0.105, "Medium": 0.062, "Hard": 0.040}
STRATEGY = {
    "LEC": [("Soft", 13), ("Hard", 37)],                   # aggressive one-stop (undercut)
    "VER": [("Medium", 22), ("Hard", 28)],                 # one-stop
}
RACE_ADJ = {"LEC": 0.00, "VER": -0.30}     # race-pace offset (Red Bull stronger in clean air)
GRID_OFFSET = {"LEC": 0.0, "VER": 0.85}    # Ferrari on pole, Red Bull P2

QUALI_RUNS = [("Hard", "Q1", 1, 0.95), ("Medium", "Q2", 4, 0.42),
              ("Soft", "Q3 Run 1", 7, 0.14), ("Soft", "Q3 Run 2", 10, 0.0)]

SPARK = "▁▂▃▄▅▆▇█"


def format_lap(sec: float) -> str:
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:06.3f}"


def spark(vals) -> str:
    return "".join(SPARK[int(np.clip(round(v * 7), 0, 7))] for v in vals)


def _sectors_from_fraction(lap_sec, frac, rng, jitter=0.0012):
    s = lap_sec * frac * (1 + rng.normal(0, jitter, 3))
    return s * lap_sec / s.sum()


def _telemetry_strings(prof, rng, thr_bias=0.0, brk_bias=0.0):
    thr = np.clip(prof["thr_seg"] + rng.normal(0, 0.03, 10) + thr_bias, 0, 1)
    brk = np.clip(prof["brk_seg"] + rng.normal(0, 0.03, 10) + brk_bias, 0, 1)
    thr_avg = np.clip(prof["thr_seg"].mean() * 100 + rng.normal(0, 0.8) + thr_bias * 100, 0, 100)
    brk_avg = np.clip(prof["brk_seg"].mean() * 100 + rng.normal(0, 0.5) + brk_bias * 100, 0, 100)
    return f"{thr_avg:04.1f}% {spark(thr)}", f"{brk_avg:04.1f}% {spark(brk)}"


def generate_quali_log(rng) -> pd.DataFrame:
    rows = []
    for code in CODES:
        prof = PROFILES["quali"][code]
        for comp, label, first, gap in QUALI_RUNS:
            for k, kind in enumerate(["out", "push", "in"]):
                lap = first + k
                if kind == "push":
                    lap_sec = prof["lap_time"] + gap + (rng.normal(0, 0.035) if gap > 0 else 0.0)
                    top = prof["top_speed"] - gap * 3.0 + rng.normal(0, 1.2)
                    note = "PUSH LAP"
                else:
                    lap_sec = prof["lap_time"] * (1.19 if kind == "out" else 1.23) + rng.normal(0, 0.4)
                    top = prof["top_speed"] * 0.78 + rng.normal(0, 3)
                    note = "OUT LAP" if kind == "out" else "IN LAP"
                s1, s2, s3 = _sectors_from_fraction(lap_sec, prof["sector_frac"], rng)
                thr_s, brk_s = _telemetry_strings(prof, rng, -0.18 if kind != "push" else 0.0,
                                                  -0.06 if kind != "push" else 0.0)
                rows.append(dict(Session="quali", Lap=lap, Driver=code, Tyre=comp, Age=k,
                                 LapSec=lap_sec, S1=s1, S2=s2, S3=s3, TopSpeed=top,
                                 Throttle=thr_s, Brake=brk_s, Kind=kind, Note=note,
                                 Run=f"{label} · {comp}", PitIn=False, Clean=(kind == "push")))
    return pd.DataFrame(rows)


def _stint_rows(strategy):
    rows, lap = [], 1
    for si, (comp, n) in enumerate(strategy):
        nxt = strategy[si + 1][0] if si < len(strategy) - 1 else None
        for k in range(n):
            rows.append(dict(Lap=lap, Tyre=comp, Age=k, PitIn=(k == n - 1 and nxt is not None),
                             OutLap=(k == 0 and si > 0), NextTyre=nxt))
            lap += 1
    return rows


def simulate_race(rng):
    stints = {c: _stint_rows(STRATEGY[c]) for c in CODES}
    pure = {c: [] for c in CODES}
    for c in CODES:
        ref = PROFILES["race"][c]["lap_time"]
        for r in stints[c]:
            t = (ref + RACE_ADJ[c] + COMPOUND_OFFSET[r["Tyre"]] + COMPOUND_DEG[r["Tyre"]] * r["Age"]
                 - FUEL_GAIN * (r["Lap"] - 1) + rng.normal(0, 0.07))
            if r["OutLap"]:
                t += 0.9
            if r["PitIn"]:
                t += 0.5 + PIT_LOSS
            pure[c].append(t)

    cum = dict(GRID_OFFSET)
    times = {c: [] for c in CODES}
    cum_hist = {c: [] for c in CODES}
    follow = {c: 0 for c in CODES}
    for lap in range(1, RACE_LAPS + 1):
        raw = {c: pure[c][lap - 1] for c in CODES}
        pit_now = any(stints[c][lap - 1]["PitIn"] for c in CODES)
        lead = min(CODES, key=lambda c: cum[c])
        chase = "VER" if lead == "LEC" else "LEC"
        gap = cum[chase] - cum[lead]
        t_l, t_c = raw[lead], raw[chase]
        if not pit_now and gap < 1.15:
            follow[chase] += 1
            if follow[chase] >= 2 and (t_l - t_c) > 0.18:          # DRS-assisted pass
                new_l = cum[lead] + t_l + 0.45
                new_c = min(cum[chase] + t_c, new_l - 0.40)
                t_l, t_c = new_l - cum[lead], new_c - cum[chase]
                follow[chase] = 0
            else:                                                   # dirty air
                t_c = max(t_c, t_l + 0.22)
        else:
            follow[chase] = 0
        times[lead].append(t_l)
        times[chase].append(t_c)
        cum[lead] += t_l
        cum[chase] += t_c
        for c in CODES:
            cum_hist[c].append(cum[c])

    rows = []
    for c in CODES:
        prof = PROFILES["race"][c]
        for r, lt, cu in zip(stints[c], times[c], cum_hist[c]):
            base = lt - (PIT_LOSS if r["PitIn"] else 0.0)
            s1, s2, s3 = _sectors_from_fraction(base, prof["sector_frac"], rng, 0.0016)
            if r["PitIn"]:
                s3 += PIT_LOSS
            top = prof["top_speed"] + rng.normal(0, 2.5) - 0.012 * r["Age"] * 3.0
            thr_s, brk_s = _telemetry_strings(prof, rng, -0.01 * min(r["Age"], 10) / 10,
                                              0.01 if r["Tyre"] == "Soft" else 0.0)
            note = "PIT IN" if r["PitIn"] else ("OUT LAP" if r["OutLap"] else ("LAP 1" if r["Lap"] == 1 else ""))
            rows.append(dict(Session="race", Lap=r["Lap"], Driver=c, Tyre=r["Tyre"], Age=r["Age"],
                             LapSec=lt, S1=s1, S2=s2, S3=s3, TopSpeed=top, Throttle=thr_s,
                             Brake=brk_s, Kind="pit" if r["PitIn"] else "race", Note=note,
                             Run="", PitIn=r["PitIn"], NextTyre=r["NextTyre"], Cum=cu,
                             Clean=not (r["PitIn"] or r["OutLap"] or r["Lap"] == 1)))
    df = pd.DataFrame(rows)

    # ---- crossover events: sign change of cumulative gap -------------------
    piv = df.pivot(index="Lap", columns="Driver", values="Cum")
    gap = (piv["VER"] - piv["LEC"]).values          # >0 => Ferrari ahead
    prev = np.sign(GRID_OFFSET["VER"] - GRID_OFFSET["LEC"])
    events = []
    for i, g in enumerate(gap, start=1):
        sgn = np.sign(g)
        if sgn != prev and sgn != 0:
            pit = bool(df[(df.Lap == i) & df.PitIn].shape[0])
            passer = "LEC" if sgn > 0 else "VER"
            events.append(dict(lap=i, kind="pit" if pit else "overtake", passer=passer,
                               passed="VER" if passer == "LEC" else "LEC"))
        prev = sgn
    return df, events, pd.Series(gap, index=piv.index)


_rng = np.random.default_rng(2026)
QUALI_LOG = generate_quali_log(_rng)
RACE_LOG, RACE_EVENTS, RACE_GAP = simulate_race(_rng)
LOG = pd.concat([QUALI_LOG, RACE_LOG], ignore_index=True)
LOG["NextTyre"] = LOG["NextTyre"].fillna("")

SESSION_OPTIONS = [{"label": "Qualifying Session Analysis", "value": "quali"},
                   {"label": "Full Race Simulation Data", "value": "race"}]
SESSION_TITLE = {"quali": "Qualifying · Monza Pole Lap", "race": "Race Simulation · 50 Laps"}


# =============================================================================
# 5. FIGURE FACTORIES
# =============================================================================
def style_fig(fig, height, title=None):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12), margin=dict(l=52, r=20, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL_HI, bordercolor=BORDER, font=dict(color=TEXT, family=FONT)),
        title=dict(text=title, x=0.01, y=0.97, font=dict(size=15, color=TEXT)) if title else None,
    )
    fig.update_xaxes(gridcolor=GRIDC, zerolinecolor=GRIDC, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRIDC, zerolinecolor=GRIDC, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig


def empty_figure(msg, height=380):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=15, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height)


# ---------- 5a. animated track ------------------------------------------------
TRACK_IDX = dict(glow={"LEC": 5, "VER": 7}, core={"LEC": 6, "VER": 8})


def build_track_figure():
    x, y = TRACK["xc"], TRACK["yc"]
    speed = PROFILES["quali"]["LEC"]["v"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#2C3345", width=24),
                             hoverinfo="skip", showlegend=False))                       # 0 kerb / border
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#0E1017", width=18),
                             hoverinfo="skip", showlegend=False))                       # 1 asphalt
    fig.add_trace(go.Scatter(                                                            # 2 speed heat-line
        x=x, y=y, mode="markers", showlegend=False,
        marker=dict(size=4, color=speed, colorscale=[[0, "#6A00F4"], [0.25, "#2D7DFF"], [0.5, "#17E3B0"],
                                                     [0.75, "#FFD60A"], [1, "#FF3B30"]],
                    cmin=60, cmax=350, showscale=True,
                    colorbar=dict(title=dict(text="km/h", font=dict(color=MUTED, size=11)),
                                  thickness=8, len=0.55, x=1.0, tickfont=dict(color=MUTED, size=10),
                                  outlinewidth=0)),
        customdata=speed, hovertemplate="Pole-lap speed %{customdata:.0f} km/h<extra></extra>"))
    names = list(TRACK["corners"].keys())
    fig.add_trace(go.Scatter(                                                            # 3 corner labels
        x=[TRACK["corners"][n]["x"] for n in names], y=[TRACK["corners"][n]["y"] for n in names],
        mode="markers+text", text=names, textposition="top center", showlegend=False,
        marker=dict(size=6, color=MUTED, symbol="diamond"), textfont=dict(size=10, color=MUTED),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers+text", text=["S/F"], textposition="middle left",
                             marker=dict(size=11, color="#FFFFFF", symbol="square"),   # 4 start / finish
                             textfont=dict(size=10, color=TEXT), showlegend=False, hoverinfo="skip"))
    for code in CODES:                                                                    # 5-8 cars
        d = DRIVERS[code]
        fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=26, color=d["glow"])))
        fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers+text", text=[code], name=d["name"],
                                 textposition="top center", textfont=dict(size=11, color=d["line"]),
                                 marker=dict(size=12, color=d["fill"], line=dict(color=d["edge"], width=3)),
                                 hoverinfo="skip", showlegend=True))
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", uirevision="track", font=dict(family=FONT, color=TEXT),
        legend=dict(orientation="h", x=0.01, y=0.02, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    return fig


def telemetry_at(prof: dict, tt: float) -> dict:
    tt = float(np.clip(tt, 0, prof["lap_time"]))
    v = float(np.interp(tt, prof["t"], prof["v"]))
    gear, rpm = gear_and_rpm(v)
    i = int(np.clip(np.searchsorted(prof["t"], tt), 0, len(prof["t"]) - 1))
    s = float(np.interp(tt, prof["t"], prof["s"]))
    return dict(speed=v, gear=int(gear[0]), rpm=float(rpm[0]),
                thr=float(np.interp(tt, prof["t"], prof["thr"])),
                brk=float(np.interp(tt, prof["t"], prof["brk"])),
                ers=int(prof["ers"][i]), soc=float(np.interp(tt, prof["t"], prof["soc"])),
                s=s, x=float(np.interp(s, TRACK["sc"], TRACK["xc"])),
                y=float(np.interp(s, TRACK["sc"], TRACK["yc"])), t=tt)


# ---------- 5b. lap-time analysis ----------------------------------------------
def fig_lap_race(tyres):
    df = LOG[LOG.Session == "race"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30], vertical_spacing=0.07)
    clean = df[~df.PitIn & df.Tyre.isin(tyres)] if tyres else df[~df.PitIn]
    ref = clean if len(clean) else df[~df.PitIn]
    lo, hi = ref.LapSec.min() - 0.45, ref.LapSec.max() + 1.15
    ytop = hi - 0.22
    all_on = len(tyres) == 3

    for code in CODES:
        d = DRIVERS[code]
        sub = df[df.Driver == code].sort_values("Lap")
        show = sub.Tyre.isin(tyres) & ~sub.PitIn
        y = sub.LapSec.where(show)
        fig.add_trace(go.Scatter(
            x=sub.Lap, y=y, mode="lines+markers", name=f"{d['name']}", connectgaps=all_on,
            line=dict(color=d["line"], width=2.2),
            marker=dict(size=7, symbol="circle" if code == "LEC" else "diamond",
                        color=[TYRE_COLORS[t] for t in sub.Tyre], line=dict(color=d["line"], width=1.6)),
            customdata=np.stack([sub.Tyre, sub.Age + 1], axis=-1),
            hovertemplate=f"<b>{code}</b> L%{{x}} · %{{y:.3f}}s<br>%{{customdata[0]}} · tyre age %{{customdata[1]}}<extra></extra>"),
            row=1, col=1)
        pits = sub[sub.PitIn]
        if len(pits):
            fig.add_trace(go.Scatter(
                x=pits.Lap, y=[ytop] * len(pits), mode="markers", showlegend=False,
                marker=dict(symbol="triangle-down", size=13, color=d["line"], line=dict(color="#FFFFFF", width=1.2)),
                text=[f"PIT STOP {code} · L{l} · {a} → {b}" for l, a, b in zip(pits.Lap, pits.Tyre, pits.NextTyre)],
                hovertemplate="%{text}<extra></extra>"), row=1, col=1)
            for k, (_, p) in enumerate(pits.iterrows()):
                fig.add_annotation(x=p.Lap, y=ytop, xref="x", yref="y", showarrow=True, arrowhead=0,
                                   arrowcolor=d["line"], ax=0, ay=-26 - (14 if code == "VER" else 0),
                                   text=f"<b>PIT {code}</b> {p.Tyre[0]}→{p.NextTyre[0]}",
                                   font=dict(size=10, color=d["line"]), bgcolor=rgba(BG, 0.85),
                                   bordercolor=d["line"], borderwidth=1, borderpad=2)

    # gap panel (cumulative delta)
    gap = RACE_GAP
    pos = np.where(gap.values >= 0, gap.values, 0)
    neg = np.where(gap.values < 0, gap.values, 0)
    fig.add_trace(go.Scatter(x=gap.index, y=pos, mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor=rgba(FERRARI, 0.30), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=gap.index, y=neg, mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor=rgba(RB_YELLOW, 0.26), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=gap.index, y=gap.values, mode="lines", line=dict(color="#FFFFFF", width=1.6),
                             name="Gap (RBR − FER)", showlegend=False,
                             hovertemplate="L%{x} · gap %{y:+.2f}s<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1)

    ev_ot = [e for e in RACE_EVENTS if e["kind"] == "overtake"]
    ev_pit = [e for e in RACE_EVENTS if e["kind"] == "pit"]
    for kind_events, symbol, label, size in ((ev_ot, "star", "On-track overtake", 17),
                                              (ev_pit, "diamond-open", "Position swap via pit cycle", 12)):
        if not kind_events:
            continue
        laps = [e["lap"] for e in kind_events]
        ys_gap = [float(gap.loc[l]) for l in laps]
        fig.add_trace(go.Scatter(x=laps, y=ys_gap, mode="markers", name=label,
                                 marker=dict(symbol=symbol, size=size, color="#FFFFFF" if symbol == "star" else TEXT,
                                             line=dict(color=TEXT, width=1.5)),
                                 hovertemplate=label + " · L%{x}<extra></extra>"), row=2, col=1)
    for e in ev_ot:
        sub = df[(df.Driver == e["passer"]) & (df.Lap == e["lap"])]
        if len(sub) and sub.Tyre.iloc[0] in tyres:
            yy = float(sub.LapSec.iloc[0])
            fig.add_trace(go.Scatter(x=[e["lap"]], y=[yy], mode="markers", showlegend=False,
                                     marker=dict(symbol="star", size=18, color="#FFFFFF",
                                                 line=dict(color=DRIVERS[e["passer"]]["line"], width=2)),
                                     hoverinfo="skip"), row=1, col=1)
            fig.add_annotation(x=e["lap"], y=yy, xref="x", yref="y", showarrow=True, ax=34, ay=30,
                               arrowcolor="#FFFFFF", arrowhead=2,
                               text=f"<b>{e['passer']} passes {e['passed']}</b> · L{e['lap']}",
                               font=dict(size=10, color=TEXT), bgcolor=rgba(BG, 0.9), bordercolor="#FFFFFF",
                               borderwidth=1, borderpad=3)
    if not len(clean):
        fig.add_annotation(text="No laps for the selected tyre compounds", xref="paper", yref="paper",
                           x=0.5, y=0.6, showarrow=False, font=dict(size=14, color=MUTED))

    style_fig(fig, 560, "Lap-Time Delta Analysis · 50-Lap Race Distance")
    fig.update_yaxes(title_text="Lap time (s)", range=[lo, hi], row=1, col=1)
    fig.update_yaxes(title_text="Gap (s)", row=2, col=1)
    fig.update_xaxes(title_text="Lap", row=2, col=1, dtick=5, range=[0.5, RACE_LAPS + 0.5])
    fig.update_xaxes(range=[0.5, RACE_LAPS + 0.5], row=1, col=1)
    fig.update_layout(hovermode="closest")
    fig.add_annotation(text="Gap >0: Ferrari ahead · Gap <0: Red Bull ahead", xref="paper", yref="paper",
                       x=0.995, y=0.30, xanchor="right", yanchor="bottom", showarrow=False,
                       font=dict(size=10, color=MUTED))
    return fig


def fig_lap_quali(tyres):
    df = LOG[(LOG.Session == "quali") & (LOG.Kind == "push") & LOG.Tyre.isin(tyres)]
    if df.empty:
        return empty_figure("No push laps for the selected tyre compounds", 560)
    order = [f"{lbl} · {comp}" for comp, lbl, _, _ in QUALI_RUNS if comp in tyres]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.08)
    for code in CODES:
        d = DRIVERS[code]
        sub = df[df.Driver == code].set_index("Run").reindex(order).dropna(subset=["LapSec"]).reset_index()
        fig.add_trace(go.Scatter(
            x=sub.Run, y=sub.LapSec, mode="lines+markers", name=d["name"], line=dict(color=d["line"], width=2.6),
            marker=dict(size=11, color=[TYRE_COLORS[t] for t in sub.Tyre], line=dict(color=d["line"], width=2),
                        symbol="circle" if code == "LEC" else "diamond"),
            text=[format_lap(v) for v in sub.LapSec],
            hovertemplate=f"<b>{code}</b> %{{x}}<br>%{{text}}<extra></extra>"), row=1, col=1)
    piv = df.pivot_table(index="Run", columns="Driver", values="LapSec").reindex(order).dropna()
    if len(piv):
        delta = piv["VER"] - piv["LEC"]
        fig.add_trace(go.Bar(x=delta.index, y=delta.values, showlegend=False,
                             marker=dict(color=[rgba(FERRARI, 0.75) if v > 0 else rgba(RB_YELLOW, 0.75) for v in delta],
                                         line=dict(color=[FERRARI if v > 0 else RB_YELLOW for v in delta], width=1.5)),
                             text=[f"{v:+.3f}s" for v in delta], textposition="outside", textfont=dict(color=TEXT, size=11),
                             hovertemplate="%{x}<br>RBR − FER %{y:+.3f}s<extra></extra>"), row=2, col=1)
    best = df.loc[df.LapSec.idxmin()]
    fig.add_annotation(x=best.Run, y=best.LapSec, xref="x", yref="y", showarrow=True, ax=0, ay=-48,
                       arrowcolor="#FFFFFF", text=f"<b>POLE LAP · {best.Driver}</b> {format_lap(best.LapSec)}",
                       font=dict(size=11, color=TEXT), bgcolor=rgba(BG, 0.9), bordercolor="#FFFFFF",
                       borderwidth=1, borderpad=3)
    style_fig(fig, 560, "Lap-Time Delta Analysis · Qualifying Push Laps by Run")
    fig.update_yaxes(title_text="Lap time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Δ RBR − FER (s)", row=2, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.add_annotation(text="Δ >0: Ferrari faster · Δ <0: Red Bull faster", xref="paper", yref="paper",
                       x=0.995, y=0.0, xanchor="right", yanchor="top", showarrow=False, yshift=-26,
                       font=dict(size=10, color=MUTED))
    return fig


# ---------- 5c. speed vs distance telemetry cross-section -----------------------
def _zones(mask, min_len):
    zones, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if not m else i + 1
            if end - start >= min_len:
                zones.append((start, end - 1))
            start = None
    return zones


def fig_telemetry(session):
    prof = PROFILES[session]
    s = TRACK["sc"]
    rng = np.random.default_rng(7)
    vL, vV = prof["LEC"]["v"], prof["VER"]["v"]
    nL, nV = vL + rng.normal(0, 0.35, len(vL)), vV + rng.normal(0, 0.35, len(vV))   # sensor jitter
    diff = vL - vV
    delta_t = prof["LEC"]["t"] - prof["VER"]["t"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    for z0, z1 in _zones(diff > 2.5, 12):
        fig.add_vrect(x0=s[z0], x1=s[z1], fillcolor=rgba(FERRARI, 0.13), line_width=0, layer="below", row="all", col=1)
    for z0, z1 in _zones(diff < -2.5, 12):
        fig.add_vrect(x0=s[z0], x1=s[z1], fillcolor=rgba(RB_YELLOW, 0.10), line_width=0, layer="below", row="all", col=1)

    for code, arr in (("LEC", nL), ("VER", nV)):
        d = DRIVERS[code]
        fig.add_trace(go.Scatter(x=s, y=arr, mode="lines", name=f"{d['name']}",
                                 line=dict(color=d["line"], width=2.0),
                                 hovertemplate=f"<b>{code}</b> %{{x:.0f}} m · %{{y:.1f}} km/h<extra></extra>"), row=1, col=1)

    # annotate the 2 strongest zones for each car
    for sign, code, txt in ((1, "LEC", "Ferrari: top-speed gain"), (-1, "VER", "Red Bull: corner dominance")):
        zs = _zones(sign * diff > 2.5, 12)
        zs = sorted(zs, key=lambda z: -(sign * diff[z[0]:z[1] + 1]).max())[:2]
        for z0, z1 in zs:
            k = z0 + int(np.argmax(sign * diff[z0:z1 + 1]))
            fig.add_annotation(x=s[k], y=max(vL[k], vV[k]) + 6, xref="x", yref="y", showarrow=True, ax=0, ay=-34,
                               arrowcolor=DRIVERS[code]["line"], arrowwidth=1.2,
                               text=f"<b>{txt}</b><br>{abs(diff[k]):.1f} km/h",
                               font=dict(size=10, color=DRIVERS[code]["line"]), bgcolor=rgba(BG, 0.9),
                               bordercolor=DRIVERS[code]["line"], borderwidth=1, borderpad=2, align="left")
    for name, c in TRACK["corners"].items():
        fig.add_vline(x=c["s"], line=dict(color=MUTED, width=1, dash="dot"), row="all", col=1)
        fig.add_annotation(x=c["s"], y=0.02, xref="x", yref="y domain", text=name, showarrow=False,
                           textangle=-90, xanchor="right", yanchor="bottom", font=dict(size=10, color=MUTED))

    pos = np.where(delta_t >= 0, delta_t, 0)
    neg = np.where(delta_t < 0, delta_t, 0)
    fig.add_trace(go.Scatter(x=s, y=neg, mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor=rgba(FERRARI, 0.32), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=s, y=pos, mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor=rgba(RB_YELLOW, 0.28), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=s, y=delta_t, mode="lines", line=dict(color="#FFFFFF", width=1.6), showlegend=False,
                             hovertemplate="%{x:.0f} m · Δt %{y:+.3f}s<extra></extra>"), row=2, col=1)
    ttl = ("Telemetry Cross-Section · Speed vs Distance · Optimal Qualifying Lap" if session == "quali"
           else "Telemetry Cross-Section · Speed vs Distance · Race-Pace Lap (fuel-heavy trim)")
    style_fig(fig, 560, ttl)
    fig.update_yaxes(title_text="Speed (km/h)", range=[30, 385], row=1, col=1)
    fig.update_yaxes(title_text="Δt FER−RBR (s)", row=2, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=2, col=1, range=[0, TRACK_LENGTH_M])
    fig.update_xaxes(range=[0, TRACK_LENGTH_M], row=1, col=1)
    fig.update_layout(hovermode="x unified")
    fig.add_annotation(text="Δt <0: Ferrari ahead · Δt >0: Red Bull ahead", xref="paper", yref="paper",
                       x=0.995, y=0.29, xanchor="right", yanchor="bottom", showarrow=False,
                       font=dict(size=10, color=MUTED))
    return fig


# ---------- 5d. sector breakdown --------------------------------------------------
def sector_frame(session, tyres):
    df = LOG[(LOG.Session == session) & LOG.Clean & LOG.Tyre.isin(tyres)]
    out = {}
    for code in CODES:
        sub = df[df.Driver == code]
        if sub.empty:
            return None
        row = sub.loc[sub.LapSec.idxmin()] if session == "quali" else sub[["S1", "S2", "S3"]].mean()
        out[code] = np.array([row.S1, row.S2, row.S3], dtype=float)
    return out


def fig_sector(session, tyres):
    data = sector_frame(session, tyres)
    if data is None:
        return empty_figure("No data for the selected tyre compounds", 560)
    cats = ["Sector 1", "Sector 2", "Sector 3"]
    fig = go.Figure()
    for code in CODES:
        d = DRIVERS[code]
        fig.add_trace(go.Bar(
            y=cats, x=data[code], orientation="h", name=d["name"],
            marker=dict(color=d["fill"], line=dict(color=d["edge"] if code == "VER" else FERRARI, width=2)),
            text=[f"{v:.3f}s" for v in data[code]], textposition="inside", insidetextanchor="start",
            textfont=dict(color=TEXT, size=12),
            hovertemplate=f"<b>{code}</b> %{{y}} · %{{x:.3f}}s<extra></extra>"))
    xmax = max(data["LEC"].max(), data["VER"].max()) * 1.55
    for i, cat in enumerate(cats):
        diff = data["VER"][i] - data["LEC"][i]
        pct = diff / data["LEC"][i] * 100
        faster = "LEC" if diff > 0 else "VER"
        fig.add_annotation(x=xmax * 0.995, y=cat, xref="x", yref="y", xanchor="right", showarrow=False, yshift=0,
                           text=f"<b>{SHORT[faster]} +{abs(pct):.2f}%</b><br>{abs(diff):.3f}s faster",
                           font=dict(size=11, color=DRIVERS[faster]["line"]), align="right",
                           bgcolor=rgba(BG, 0.85), bordercolor=DRIVERS[faster]["line"], borderwidth=1, borderpad=3)
    ttl = "Sector Performance · Best Lap" if session == "quali" else "Sector Performance · Race Average (clean laps)"
    style_fig(fig, 560, ttl)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(title_text="Sector time (s)", range=[0, xmax])
    fig.update_yaxes(autorange="reversed")
    return fig


# =============================================================================
# 6. STATISTICS + KPI HELPERS
# =============================================================================
def welch(a, b):
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return None
    t = (a.mean() - b.mean()) / se
    dfree = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, dfree, math.erfc(abs(t) / math.sqrt(2))      # normal approximation of the p-value


def within_compound_slope(sub):
    if len(sub) < 6:
        return None
    ft = sub.LapSec + FUEL_GAIN * (sub.Lap - 1)
    ft = ft - ft.groupby(sub.Tyre).transform("mean")
    age = sub.Age - sub.Age.groupby(sub.Tyre).transform("mean")
    den = float((age ** 2).sum())
    return float((age * ft).sum() / den) if den > 0 else None


def stats_card(session, tyres):
    df = LOG[(LOG.Session == session) & LOG.Clean & LOG.Tyre.isin(tyres)]
    head = html.Thead(html.Tr([html.Th(h) for h in ["", "n", "Mean", "σ", "CV %", "95% CI ±", "Best", "Deg s/lap"]]))
    rows, series = [], {}
    for code in CODES:
        sub = df[df.Driver == code]
        series[code] = sub.LapSec.values
        if sub.empty:
            rows.append(html.Tr([html.Td(code, style={"color": DRIVERS[code]["line"], "fontWeight": 700})] +
                                [html.Td("—")] * 7))
            continue
        x = sub.LapSec.values
        n = len(x)
        sd = x.std(ddof=1) if n > 1 else float("nan")
        ci = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
        slope = within_compound_slope(sub) if session == "race" else None
        rows.append(html.Tr([
            html.Td(code, style={"color": DRIVERS[code]["line"], "fontWeight": 700}),
            html.Td(n), html.Td(f"{x.mean():.3f}"), html.Td("—" if n < 2 else f"{sd:.3f}"),
            html.Td("—" if n < 2 else f"{sd / x.mean() * 100:.2f}"), html.Td("—" if n < 2 else f"{ci:.3f}"),
            html.Td(f"{x.min():.3f}"), html.Td("—" if slope is None else f"{slope:+.3f}")]))
    w = welch(series["LEC"], series["VER"]) if len(series["LEC"]) and len(series["VER"]) else None
    if w:
        t, dfree, p = w
        mean_gap = series["VER"].mean() - series["LEC"].mean()
        faster = "Ferrari" if mean_gap > 0 else "Red Bull"
        verdict = (f"{faster} faster by {abs(mean_gap):.3f}s/lap · Welch t = {t:+.2f} (df ≈ {dfree:.0f}) · "
                   f"p {'< 0.001' if p < 0.001 else '= ' + format(p, '.3f')} → "
                   f"{'statistically significant' if p < 0.05 else 'not significant'} at α = 0.05")
    else:
        verdict = "Too few samples for a Welch t-test with the current filters (need ≥ 3 laps per driver)."
    return html.Div([
        dbc.Table([head, html.Tbody(rows)], bordered=False, size="sm", className="f1-stats-table"),
        html.Div(verdict, className="f1-verdict"),
        html.Div("Clean laps only (push laps in qualifying; no pit / out / lap-1 in race). "
                 "Deg = fuel-corrected, within-compound slope of lap time vs tyre age. "
                 "p-value uses a normal approximation.", className="f1-footnote"),
    ])


def kpi_tile(label, value, sub, color):
    return dbc.Col(html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value", style={"color": color}),
        html.Div(sub, className="kpi-sub")], className="f1-card kpi-tile",
        style={"borderTop": f"3px solid {color}"}), xs=6, lg=3)


def kpi_strip(session, tyres):
    df = LOG[(LOG.Session == session) & LOG.Tyre.isin(tyres)]
    if session == "quali":
        df = df[df.Kind == "push"]
    else:
        df = df[~df.PitIn]
    tiles = []
    best = {}
    for code in CODES:
        sub = df[df.Driver == code]
        best[code] = sub.LapSec.min() if len(sub) else None
        d = DRIVERS[code]
        if len(sub):
            row = sub.loc[sub.LapSec.idxmin()]
            tiles.append(kpi_tile(f"{code} · {'POLE LAP' if session == 'quali' else 'FASTEST LAP'}",
                                  format_lap(row.LapSec), f"{row.Tyre} · {'run ' + row.Run.split(' · ')[0] if session == 'quali' else 'lap ' + str(int(row.Lap))}",
                                  d["line"]))
        else:
            tiles.append(kpi_tile(f"{code} · BEST LAP", "—", "no laps for this filter", d["line"]))
    if best["LEC"] is not None and best["VER"] is not None:
        gap = best["VER"] - best["LEC"]
        who = "Ferrari" if gap > 0 else "Red Bull"
        tiles.append(kpi_tile("BEST-LAP GAP", f"{abs(gap):.3f}s", f"{who} quicker",
                              FERRARI if gap > 0 else RB_YELLOW))
    else:
        tiles.append(kpi_tile("BEST-LAP GAP", "—", "n/a", MUTED))
    if session == "quali":
        ts = {c: PROFILES["quali"][c]["top_speed"] for c in CODES}
        tiles.append(kpi_tile("TOP SPEED (POLE LAP)", f"{max(ts.values()):.0f} km/h",
                              f"FER {ts['LEC']:.0f} · RBR {ts['VER']:.0f}", TEXT))
    else:
        n_ot = sum(e["kind"] == "overtake" for e in RACE_EVENTS)
        n_pit = int(LOG[(LOG.Session == "race") & LOG.PitIn].shape[0])
        tiles.append(kpi_tile("PIT STOPS · OVERTAKES", f"{n_pit} · {n_ot}",
                              f"{len(RACE_EVENTS)} position changes in total", TEXT))
    return tiles


# =============================================================================
# 7. UI COMPONENTS
# =============================================================================
def f1_logo_uri():
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 130 36'>"
        "<g transform='skewX(-14) translate(10 0)' fill='#E10600'>"
        "<path d='M6 4 H34 V11 H15 V14 H30 V21 H15 V32 H6 Z'/>"
        "<path d='M40 12 L49 4 H58 V32 H49 V14 Z'/>"
        "<path d='M62 4 H112 L110 9 H60 Z'/><path d='M60 14 H108 L106 19 H58 Z'/>"
        "<path d='M58 24 H104 L102 29 H56 Z'/></g></svg>")
    return "data:image/svg+xml;utf8," + quote(svg)


def header():
    return html.Div([
        html.Div([
            html.Img(src=f1_logo_uri(), className="f1-logo"),
            html.Div([
                html.Div("TELEMETRY & RACE ANALYSIS", className="hdr-title"),
                html.Div("Autodromo Nazionale Monza · Italian GP · Lap-by-lap performance intelligence",
                         className="hdr-sub")]),
        ], className="hdr-left"),
        html.Div([
            html.Div([html.Span("●", style={"color": FERRARI}), " SCUDERIA FERRARI · LECLERC"], className="team-chip",
                     style={"borderColor": FERRARI, "boxShadow": f"0 0 14px {rgba(FERRARI, 0.35)}"}),
            html.Div([html.Span("●", style={"color": RB_YELLOW}), " RED BULL RACING · VERSTAPPEN"],
                     className="team-chip", style={"borderColor": RB_YELLOW, "background": rgba(RB_NAVY, 0.65),
                                                   "boxShadow": f"0 0 14px {rgba(RB_YELLOW, 0.25)}"}),
        ], className="hdr-right"),
    ], className="f1-header")


def control_panel():
    tyre_opts = [{"label": html.Span([html.Span("●", style={"color": TYRE_COLORS[t], "marginRight": "6px",
                                                            "textShadow": f"0 0 6px {TYRE_COLORS[t]}"}), t]),
                  "value": t} for t in ("Soft", "Medium", "Hard")]
    return html.Div([
        html.Div("CONTROL PANEL", className="section-label"),
        dbc.Row([
            dbc.Col([html.Label("Session", className="ctl-label"),
                     dbc.Select(id="session-select", options=SESSION_OPTIONS, value="quali", className="f1-select")],
                    xs=12, md=4, lg=3),
            dbc.Col([html.Label("Tyre compounds", className="ctl-label"),
                     dcc.Checklist(id="tyre-filter", options=tyre_opts, value=["Soft", "Medium", "Hard"],
                                   inline=True, className="f1-checklist",
                                   labelStyle={"display": "inline-flex", "alignItems": "center", "marginRight": "18px"})],
                    xs=12, md=4, lg=3),
            dbc.Col([html.Label("Animation playback", className="ctl-label"),
                     dbc.ButtonGroup([
                         dbc.Button("▶ Start", id="btn-start", n_clicks=0, className="pb-btn pb-start"),
                         dbc.Button("❚❚ Pause", id="btn-pause", n_clicks=0, className="pb-btn pb-pause"),
                         dbc.Button("↺ Reset", id="btn-reset", n_clicks=0, className="pb-btn pb-reset")])],
                    xs=12, md=4, lg=3),
            dbc.Col([html.Label("Simulation speed", className="ctl-label"),
                     dbc.RadioItems(id="speed-select", options=[{"label": "1×", "value": 1}, {"label": "2×", "value": 2},
                                                                {"label": "4×", "value": 4}],
                                    value=1, inline=True, className="f1-radio"),
                     html.Div(id="playback-status", className="pb-status")],
                    xs=12, md=6, lg=3),
        ], className="g-3"),
    ], className="f1-card control-panel")


def card(children, title=None, extra_class=""):
    kids = ([html.Div(title, className="section-label")] if title else []) + [children]
    return html.Div(kids, className=f"f1-card {extra_class}")


def bar(label, pct, color, text):
    return html.Div([
        html.Div([html.Span(label, className="bar-label"), html.Span(text, className="bar-text")], className="bar-head"),
        html.Div(html.Div(style={"width": f"{np.clip(pct, 0, 100):.1f}%", "background": color,
                                 "boxShadow": f"0 0 10px {rgba(color[:7], 0.55)}" if color.startswith('#') else "none"},
                          className="bar-fill"), className="bar-track")], className="bar-wrap")


def hud_card(code, st, lap_str, done):
    d = DRIVERS[code]
    ers_name = {0: "BALANCED", 1: "DEPLOY", 2: "HARVEST"}[st["ers"]]
    ers_col = ERS_COLORS[ers_name]
    return html.Div([
        html.Div([
            html.Div([html.Div(code, className="hud-code", style={"color": d["line"]}),
                      html.Div(d["name"].upper(), className="hud-name"),
                      html.Div(d["team"], className="hud-team")]),
            html.Div([html.Div("LAP", className="mini-label"),
                      html.Div(lap_str + (" ✓" if done else ""), className="hud-lap",
                               style={"color": d["line"]})], style={"textAlign": "right"}),
        ], className="hud-top"),
        html.Div([
            html.Div([html.Div(f"{st['speed']:.0f}", className="hud-speed", style={"color": TEXT}),
                      html.Div("KM/H", className="mini-label")], className="hud-speed-box"),
            html.Div([html.Div("GEAR", className="mini-label"),
                      html.Div(str(st["gear"]), className="hud-gear",
                               style={"color": d["line"], "borderColor": d["line"]})], className="hud-gear-box"),
            html.Div([html.Div("RPM", className="mini-label"),
                      html.Div(f"{st['rpm']:,.0f}", className="hud-rpm")], className="hud-gear-box"),
        ], className="hud-main"),
        bar("RPM", st["rpm"] / 125.0, d["line"] if code == "LEC" else RB_YELLOW, f"{st['rpm']:,.0f} / 12,500"),
        bar("THROTTLE", st["thr"], THROTTLE_GREEN, f"{st['thr']:.0f}%"),
        bar("BRAKE", st["brk"], BRAKE_RED, f"{st['brk']:.0f}%"),
        html.Div([
            html.Span("ERS", className="mini-label", style={"marginRight": "8px"}),
            html.Span(f"⚡ {ers_name}", className="ers-badge",
                      style={"color": ers_col, "borderColor": ers_col, "boxShadow": f"0 0 10px {rgba(ers_col, 0.35)}"}),
            html.Span(f"BATTERY {st['soc']:.0f}%", className="bar-text", style={"marginLeft": "auto"}),
        ], className="ers-row"),
        bar("ERS STATE OF CHARGE", st["soc"], ers_col if ers_name != "BALANCED" else "#6C7A96", ""),
    ], className="hud-card", style={"borderColor": d["line"],
                                    "boxShadow": f"0 0 0 1px {rgba(d['line'], 0.20)}, 0 0 26px {rgba(d['line'], 0.14)}",
                                    "background": f"linear-gradient(160deg, {rgba(d['fill'], 0.30)} 0%, {PANEL} 55%)"})


LOOP_PERIOD = max(PROFILES["quali"][c]["lap_time"] for c in CODES) + 2.0
SIM_DT = 0.25           # seconds of sim time per 80 ms tick at 1x


def build_layout():
    return dbc.Container([
        header(),
        control_panel(),
        dbc.Row(id="kpi-strip", className="g-3 mt-1"),
        dbc.Row([
            dbc.Col(card(dcc.Graph(id="track-graph", figure=build_track_figure(), config={"displayModeBar": False}),
                         "LIVE LAP SIMULATOR · POLE-LAP GHOST RUN", "pad-0"), lg=7),
            dbc.Col([
                html.Div(id="timing-strip", className="f1-card timing-strip"),
                html.Div(id="hud-LEC", className="mt-3"),
                html.Div(id="hud-VER", className="mt-3"),
            ], lg=5),
        ], className="g-3 mt-1"),
        dbc.Row([
            dbc.Col(card(dcc.Graph(id="lap-graph", config={"displaylogo": False}), None), lg=8),
            dbc.Col(card(html.Div(id="stats-body"), "PACE STATISTICS · WELCH T-TEST"), lg=4),
        ], className="g-3 mt-1"),
        dbc.Row([
            dbc.Col(card(dcc.Graph(id="telemetry-graph", config={"displaylogo": False}), None), lg=8),
            dbc.Col(card(dcc.Graph(id="sector-graph", config={"displaylogo": False}), None), lg=4),
        ], className="g-3 mt-1"),
        html.Div([
            html.Div([html.Span("LAP-BY-LAP TELEMETRY LOG", className="section-label", style={"marginBottom": 0}),
                      html.Span(id="table-caption", className="table-caption")], className="table-head"),
            dash_table.DataTable(
                id="log-table", page_size=14, page_action="native", sort_action="native", filter_action="native",
                sort_mode="multi", style_as_list_view=True,
                columns=TABLE_COLUMNS,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": PANEL_HI, "color": TEXT, "fontWeight": 700, "border": "none",
                              "borderBottom": f"2px solid {FERRARI}", "fontFamily": FONT, "fontSize": 13,
                              "textTransform": "uppercase", "letterSpacing": "0.04em"},
                style_cell={"backgroundColor": PANEL, "color": TEXT, "border": "none",
                            "borderBottom": f"1px solid {GRIDC}", "fontFamily": FONT, "fontSize": 13,
                            "padding": "8px 12px", "textAlign": "center", "minWidth": 70},
                style_cell_conditional=[{"if": {"column_id": c}, "fontFamily": "Consolas, monospace", "textAlign": "left",
                                         "minWidth": 150} for c in ("Throttle", "Brake")],
                style_filter={"backgroundColor": BG, "color": TEXT, "border": f"1px solid {BORDER}"},
                style_data_conditional=table_conditional_styles(),
                css=[{"selector": ".dash-filter input", "rule": f"color: {TEXT} !important; background: {BG} !important;"},
                     {"selector": ".dash-filter input::placeholder", "rule": f"color: {MUTED} !important;"},
                     {"selector": ".previous-next-container button, .page-number", "rule": f"color: {TEXT} !important;"},
                     {"selector": ".current-page", "rule": f"color: {TEXT} !important; background: {BG} !important;"}],
            ),
        ], className="f1-card mt-3"),
        html.Div("Synthetic, physics-informed data for portfolio demonstration · Stylised logo, not an official asset · "
                 "Not affiliated with Formula 1, Ferrari or Red Bull Racing.", className="f1-footer"),
        dcc.Interval(id="anim-interval", interval=250, n_intervals=0, disabled=False),
    ], fluid=True, className="f1-root")


# =============================================================================
# 8. DATA TABLE CONFIG
# =============================================================================
TABLE_COLUMNS = [
    {"name": "Lap", "id": "Lap", "type": "numeric"},
    {"name": "Driver", "id": "Driver"},
    {"name": "Tyre Compound", "id": "Tyre"},
    {"name": "Lap Time", "id": "LapTime"},
    {"name": "S1", "id": "S1", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "S2", "id": "S2", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "S3", "id": "S3", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "Top Speed (km/h)", "id": "TopSpeed", "type": "numeric", "format": {"specifier": ".0f"}},
    {"name": "Throttle Profile", "id": "Throttle"},
    {"name": "Brake Profile", "id": "Brake"},
    {"name": "Note", "id": "Note"},
]


def table_conditional_styles():
    styles = [{"if": {"row_index": "odd"}, "backgroundColor": "#0F1118"}]
    for tyre, col in TYRE_COLORS.items():
        styles.append({"if": {"filter_query": f'{{Tyre}} = "{tyre}"', "column_id": "Tyre"},
                       "backgroundColor": col, "color": TYRE_TEXT[tyre], "fontWeight": 700})
    styles += [
        {"if": {"filter_query": '{Driver} = "LEC"', "column_id": "Driver"}, "color": FERRARI, "fontWeight": 700},
        {"if": {"filter_query": '{Driver} = "VER"', "column_id": "Driver"}, "color": RB_YELLOW, "fontWeight": 700,
         "backgroundColor": rgba(RB_NAVY, 0.7)},
        {"if": {"filter_query": '{Note} contains "PIT"', "column_id": "Note"}, "color": "#00D2FF", "fontWeight": 700},
        {"if": {"filter_query": '{Note} = "PUSH LAP"', "column_id": "Note"}, "color": THROTTLE_GREEN, "fontWeight": 700},
    ]
    return styles


def table_frame(session, tyres):
    df = LOG[(LOG.Session == session) & LOG.Tyre.isin(tyres)].sort_values(["Lap", "Driver"]).copy()
    df["LapTime"] = df.LapSec.map(format_lap)
    return df[["Lap", "Driver", "Tyre", "LapTime", "S1", "S2", "S3", "TopSpeed", "Throttle", "Brake", "Note"]]


# =============================================================================
# 9. APP + CALLBACKS
# =============================================================================
CSS = f"""
:root {{ --bg:{BG}; --panel:{PANEL}; --panel-hi:{PANEL_HI}; --border:{BORDER}; --text:{TEXT}; --muted:{MUTED}; }}
html, body {{ background: var(--bg) !important; color: var(--text); font-family: {FONT}; }}
body {{ background-image: radial-gradient(1200px 500px at 10% -10%, {rgba(FERRARI, 0.10)}, transparent 60%),
                           radial-gradient(1000px 500px at 100% -10%, {rgba(RB_YELLOW, 0.07)}, transparent 60%); }}
.f1-root {{ max-width: 1680px; padding: 18px 22px 40px; }}
::-webkit-scrollbar {{ height:8px; width:8px; }} ::-webkit-scrollbar-thumb {{ background:#2A3042; border-radius:8px; }}
.f1-header {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;
  padding:14px 22px; margin-bottom:16px; border:1px solid var(--border); border-radius:14px;
  background: linear-gradient(100deg, {rgba(FERRARI, 0.16)}, {PANEL} 38%, {PANEL} 62%, {rgba(RB_NAVY, 0.85)});
  box-shadow: 0 0 0 1px {rgba(FERRARI, 0.10)}, 0 0 34px {rgba(FERRARI, 0.12)}; position:relative; overflow:hidden; }}
.f1-header::after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:2px;
  background: linear-gradient(90deg, {FERRARI}, transparent 45%, transparent 55%, {RB_YELLOW}); }}
.hdr-left {{ display:flex; align-items:center; gap:18px; }}
.f1-logo {{ height:52px; filter: drop-shadow(0 0 10px rgba(225,6,0,.55)); }}
.hdr-title {{ font-family:{MONO}; font-weight:700; letter-spacing:.14em; font-size:22px; }}
.hdr-sub {{ color:var(--muted); font-size:13px; letter-spacing:.04em; margin-top:2px; }}
.hdr-right {{ display:flex; gap:12px; flex-wrap:wrap; }}
.team-chip {{ border:1px solid; border-radius:999px; padding:7px 16px; font-size:12px; font-weight:700; letter-spacing:.08em; }}
.f1-card {{ background: linear-gradient(180deg, {PANEL_HI} 0%, {PANEL} 100%); border:1px solid var(--border);
  border-radius:14px; padding:16px 18px; height:100%; transition: box-shadow .25s, border-color .25s;
  box-shadow: 0 0 0 1px rgba(255,255,255,.01), 0 8px 24px rgba(0,0,0,.35); }}
.f1-card:hover {{ border-color:#3A4156; box-shadow: 0 0 0 1px {rgba(FERRARI, 0.18)}, 0 0 24px {rgba(FERRARI, 0.10)}; }}
.pad-0 {{ padding:12px 6px 0 6px; }}
.section-label {{ font-family:{MONO}; font-size:11px; letter-spacing:.2em; color:var(--muted); margin-bottom:10px; font-weight:700; }}
.ctl-label {{ display:block; font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; font-weight:700; }}
.control-panel {{ height:auto; }}
.f1-select, .form-select {{ background-color:{BG} !important; color:var(--text) !important; border:1px solid #333A4D !important; border-radius:10px; }}
.f1-select:focus {{ box-shadow:0 0 0 2px {rgba(FERRARI, 0.35)} !important; border-color:{FERRARI} !important; }}
.f1-checklist label, .f1-radio label {{ color:var(--text); cursor:pointer; font-weight:600; font-size:14px; margin-right:14px; }}
.f1-checklist input, .f1-radio input {{ margin-right:6px; accent-color:{FERRARI}; }}
.pb-btn {{ font-weight:700; letter-spacing:.06em; border-radius:10px !important; background:{BG}; border:1px solid #333A4D; color:var(--text); }}
.pb-btn:hover {{ color:#fff; border-color:#56607A; background:#161A24; }}
.pb-start.active, .pb-start:active {{ background:{rgba('#00E676', 0.18)} !important; border-color:#00E676 !important; color:#00E676 !important; }}
.pb-pause.active, .pb-pause:active {{ background:{rgba(RB_YELLOW, 0.16)} !important; border-color:{RB_YELLOW} !important; color:{RB_YELLOW} !important; }}
.pb-reset.active, .pb-reset:active {{ background:{rgba(FERRARI, 0.18)} !important; border-color:{FERRARI} !important; color:{FERRARI} !important; }}
.pb-status {{ margin-top:6px; font-size:12px; font-weight:700; letter-spacing:.1em; }}
.kpi-tile {{ padding:12px 16px; }} .kpi-label {{ font-size:11px; letter-spacing:.14em; color:var(--muted); font-weight:700; }}
.kpi-value {{ font-family:{MONO}; font-size:26px; font-weight:700; margin:4px 0 2px; }} .kpi-sub {{ font-size:12px; color:var(--muted); }}
.timing-strip {{ height:auto; display:flex; justify-content:space-between; align-items:center; gap:10px; padding:12px 18px; }}
.ts-block {{ text-align:center; }} .ts-time {{ font-family:{MONO}; font-size:22px; font-weight:700; }}
.ts-gap {{ font-family:{MONO}; font-size:15px; font-weight:700; padding:6px 14px; border:1px solid; border-radius:999px; }}
.hud-card {{ border:1px solid; border-radius:14px; padding:14px 18px; }}
.hud-top {{ display:flex; justify-content:space-between; align-items:flex-start; }}
.hud-code {{ font-family:{MONO}; font-size:26px; font-weight:900; line-height:1; }}
.hud-name {{ font-weight:700; letter-spacing:.1em; font-size:13px; margin-top:4px; }} .hud-team {{ font-size:12px; color:var(--muted); }}
.hud-lap {{ font-family:{MONO}; font-size:19px; font-weight:700; }}
.mini-label {{ font-size:10px; letter-spacing:.18em; color:var(--muted); font-weight:700; }}
.hud-main {{ display:flex; align-items:flex-end; gap:22px; margin:8px 0 10px; }}
.hud-speed {{ font-family:{MONO}; font-size:54px; font-weight:900; line-height:1; letter-spacing:.02em; min-width:130px; }}
.hud-gear {{ font-family:{MONO}; font-size:30px; font-weight:900; border:2px solid; border-radius:10px; width:52px; text-align:center; line-height:44px; }}
.hud-rpm {{ font-family:{MONO}; font-size:18px; font-weight:700; line-height:44px; }}
.bar-wrap {{ margin:6px 0; }} .bar-head {{ display:flex; justify-content:space-between; }}
.bar-label {{ font-size:10px; letter-spacing:.16em; color:var(--muted); font-weight:700; }}
.bar-text {{ font-family:{MONO}; font-size:11px; color:var(--text); }}
.bar-track {{ height:8px; border-radius:8px; background:#0A0B10; border:1px solid #20242F; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:8px; transition: width .07s linear; }}
.ers-row {{ display:flex; align-items:center; margin:10px 0 4px; }}
.ers-badge {{ border:1px solid; border-radius:999px; padding:2px 12px; font-size:12px; font-weight:700; letter-spacing:.1em; }}
.f1-stats-table {{ --bs-table-bg: transparent; --bs-table-color: {TEXT}; color:{TEXT}; font-size:13px; }}
.f1-stats-table th {{ color:var(--muted); font-size:11px; letter-spacing:.1em; border-color:var(--border); }}
.f1-stats-table td {{ border-color:var(--border); font-family:{MONO}; font-size:12px; }}
.f1-verdict {{ margin-top:12px; padding:10px 12px; border-left:3px solid {RB_YELLOW}; background:#0C0E14; border-radius:6px; font-size:13px; }}
.f1-footnote {{ margin-top:10px; font-size:11px; color:var(--muted); line-height:1.5; }}
.table-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
.table-caption {{ font-size:12px; color:var(--muted); }}
.f1-footer {{ text-align:center; color:#5B6275; font-size:11px; margin-top:22px; letter-spacing:.06em; }}
.dash-spreadsheet-container .dash-spreadsheet-inner td, .dash-spreadsheet-container .dash-spreadsheet-inner th {{ font-family:{FONT}; }}
"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>F1 Telemetry & Race Analysis · Ferrari vs Red Bull</title>
    {%favicon%}
    {%css%}
    <style>__CSS__</style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Titillium+Web:wght@400;600;700&display=swap",
    ],
    title="F1 Telemetry & Race Analysis",
    suppress_callback_exceptions=True,
)
app.index_string = INDEX_TEMPLATE.replace("__CSS__", CSS)
app.layout = build_layout()
server = app.server


@app.callback(
    Output("anim-interval", "disabled"), Output("anim-interval", "n_intervals"),
    Output("btn-start", "active"), Output("btn-pause", "active"), Output("btn-reset", "active"),
    Output("playback-status", "children"),
    Input("btn-start", "n_clicks"), Input("btn-pause", "n_clicks"), Input("btn-reset", "n_clicks"),
    State("anim-interval", "n_intervals"),
)
def playback_controls(_start, _pause, _reset, n_now):
    trig = ctx.triggered_id
    live = html.Span("● LIVE", style={"color": "#00E676"})
    paused = html.Span("❚❚ PAUSED", style={"color": RB_YELLOW})
    stopped = html.Span("■ RESET · press Start", style={"color": FERRARI})
    if trig == "btn-pause":
        return True, no_update, False, True, False, paused
    if trig == "btn-reset":
        return True, 0, False, False, True, stopped
    return False, no_update, True, False, False, live          # initial load or Start


CAR_LATERAL_OFFSET_M = 22.0   # visual offset so overlapping cars stay distinguishable


def car_xy(state: dict, side: int):
    s = state["s"]
    sa, sb = max(s - 15, 0.0), min(s + 15, TRACK_LENGTH_M)
    x0, y0 = np.interp(sa, TRACK["sc"], TRACK["xc"]), np.interp(sa, TRACK["sc"], TRACK["yc"])
    x1, y1 = np.interp(sb, TRACK["sc"], TRACK["xc"]), np.interp(sb, TRACK["sc"], TRACK["yc"])
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy) or 1.0
    return (state["x"] - side * dy / n * CAR_LATERAL_OFFSET_M,
            state["y"] + side * dx / n * CAR_LATERAL_OFFSET_M)


@app.callback(
    Output("track-graph", "figure"), Output("hud-LEC", "children"), Output("hud-VER", "children"),
    Output("timing-strip", "children"),
    Input("anim-interval", "n_intervals"), Input("speed-select", "value"),
)
def animate(n, speed):
    speed = speed or 1
    t_loop = (n * SIM_DT * speed) % LOOP_PERIOD
    prof = PROFILES["quali"]
    states, done = {}, {}
    for c in CODES:
        tt = min(t_loop, prof[c]["lap_time"])
        states[c] = telemetry_at(prof[c], tt)
        done[c] = t_loop >= prof[c]["lap_time"]

    patch = Patch()
    for c, side in (("LEC", 1), ("VER", -1)):
        px, py = car_xy(states[c], side)
        for key in ("glow", "core"):
            i = TRACK_IDX[key][c]
            patch["data"][i]["x"] = [px]
            patch["data"][i]["y"] = [py]

    gap_m = states["LEC"]["s"] - states["VER"]["s"]
    if abs(gap_m) < 0.5:
        gap_label, gap_col = "LEVEL", MUTED
    else:
        lead = "LEC" if gap_m > 0 else "VER"
        gap_label, gap_col = f"{SHORT[lead]} +{abs(gap_m):.0f} m", DRIVERS[lead]["line"]
    strip = [
        html.Div([html.Div("FERRARI · LEC", className="mini-label", style={"color": FERRARI}),
                  html.Div(format_lap(states["LEC"]["t"]), className="ts-time", style={"color": FERRARI})],
                 className="ts-block"),
        html.Div([html.Div("ON-TRACK GAP", className="mini-label", style={"marginBottom": "4px"}),
                  html.Div(gap_label, className="ts-gap", style={"color": gap_col, "borderColor": gap_col})],
                 className="ts-block"),
        html.Div([html.Div("RED BULL · VER", className="mini-label", style={"color": RB_YELLOW}),
                  html.Div(format_lap(states["VER"]["t"]), className="ts-time", style={"color": RB_YELLOW})],
                 className="ts-block"),
    ]
    huds = [hud_card(c, states[c], format_lap(states[c]["t"]), done[c]) for c in CODES]
    return patch, huds[0], huds[1], strip


@app.callback(
    Output("lap-graph", "figure"), Output("telemetry-graph", "figure"), Output("sector-graph", "figure"),
    Output("log-table", "data"), Output("table-caption", "children"),
    Output("kpi-strip", "children"), Output("stats-body", "children"),
    Input("session-select", "value"), Input("tyre-filter", "value"),
)
def update_analytics(session, tyres):
    session = session or "quali"
    tyres = list(tyres or [])
    lap_fig = fig_lap_race(tyres) if session == "race" else fig_lap_quali(tyres)
    df = table_frame(session, tyres)
    caption = f"{SESSION_TITLE[session]} · {len(df)} laps shown · sort, filter (e.g. > 300) and paginate natively"
    return (lap_fig, fig_telemetry(session), fig_sector(session, tyres), df.to_dict("records"), caption,
            kpi_strip(session, tyres), stats_card(session, tyres))


if __name__ == "__main__":
    print(f"Pole lap  FER {format_lap(PROFILES['quali']['LEC']['lap_time'])}  |  "
          f"RBR {format_lap(PROFILES['quali']['VER']['lap_time'])}")
    app.run(debug=False, host="127.0.0.1", port=8050)
