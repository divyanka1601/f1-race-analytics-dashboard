# F1 Telemetry & Race Analysis Dashboard

An interactive Formula 1 analytics dashboard comparing **Charles Leclerc (Scuderia Ferrari)** and **Max Verstappen (Red Bull Racing)** at Monza. It combines a live lap simulator, a telemetry HUD, a 50-lap race simulation, statistical pace analysis and a filterable lap-by-lap log, all built in Python.

**Live demo:** https://f1-race-analytics-dashboard-bgbk.onrender.com

> The demo runs on a free Render instance. If it has been idle, the first load can take up to a minute while the server wakes up.

<!-- To add a screenshot: upload an image (e.g. screenshot.png) to this repo, then replace this comment with:
![Dashboard screenshot](screenshot.png)
-->

---

## What it shows

| Section | What you can do |
|---|---|
| **Control panel** | Switch between *Qualifying* and *Race* views, filter by tyre compound (Soft / Medium / Hard), and control the animation (Start, Pause, Reset, 1x / 2x / 4x). |
| **KPI strip** | Best laps, best-lap gap and top speed, recalculated for the current filters. |
| **Live lap simulator** | Both cars run the pole lap on a Monza-style circuit coloured by speed, with an on-track gap indicator. |
| **Telemetry HUD** | Real-time speed, gear, RPM, throttle vs. brake and ERS state (deploy / harvest / balanced, battery level) for each driver. |
| **Lap-time delta analysis** | 50-lap race chart with automatically annotated pit stops and overtakes, plus a cumulative gap panel. |
| **Speed vs. distance** | Where Ferrari gains on the straights and where Red Bull gains in slow corners, with time-delta shading. |
| **Sector breakdown** | Grouped horizontal bars for S1 / S2 / S3 with the percentage delta per sector. |
| **Pace statistics** | Mean, standard deviation, coefficient of variation, 95% confidence interval, tyre degradation per lap and a Welch t-test. |
| **Telemetry log** | Sortable, filterable, paginated table with FIA tyre colours (Soft red, Medium yellow, Hard white). |

## Key findings in the simulated data

- **Qualifying:** Ferrari takes pole at **1:26.012**, about **0.16 s** ahead of Red Bull, thanks to higher top speed (about 349 vs 344 km/h).
- **Slow corners:** Red Bull carries more speed through low-speed corners, which closes part of the gap.
- **Race:** Red Bull has the better race pace. An early Ferrari stop changes the order through the pit cycle, and Verstappen later makes an on-track pass.

## How the data works

All data is **synthetic but physically modelled**, so the patterns have a real cause:

1. **Track:** a closed Catmull-Rom spline of a Monza-inspired layout. Curvature is computed along the whole lap.
2. **Lap physics:** curvature sets each corner's maximum speed. A backward braking pass and a forward acceleration pass then produce the speed trace. Throttle, brake, gear, RPM and ERS are derived from it.
3. **Race simulation:** tyre degradation per compound, fuel burn, pit-stop time loss, dirty air and a DRS-style overtaking rule.
4. **Telemetry log:** one Pandas DataFrame covering qualifying runs and 50 race laps, with sector times, top speed and throttle/brake profiles.
5. **Statistics:** Welch's t-test for unequal variances, confidence intervals and fuel-corrected degradation slopes.

The data layer is separate from the charts, so real telemetry (for example from the [FastF1](https://github.com/theOehrly/Fast-F1) library) could be swapped in with the same column structure.

## Tech stack

- **Python 3.12**
- **Dash** and **Dash Bootstrap Components** for the app and layout
- **Plotly** for all charts and the animated track
- **Pandas** and **NumPy** for the simulation and analysis
- **Gunicorn** and **Render** for hosting

## Run it locally

```bash
git clone https://github.com/divyanka1601/f1-race-analytics-dashboard.git
cd f1-race-analytics-dashboard
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python dashboard.py
```

Then open http://127.0.0.1:8050 in your browser.

## Project structure

```
.
├── dashboard.py        # Whole app: data model, figures, layout and callbacks
├── requirements.txt    # Python dependencies
└── README.md
```

## Design notes

- **Animation runs in the browser.** The server sends the pole-lap telemetry once, and a small clientside JavaScript callback moves the cars and updates the HUD. This keeps the animation smooth on a low-resource host, because no server round trips are needed per frame.
- **Branding:** Ferrari `#EF1A2D`, Red Bull navy `#0C2340` and yellow `#FFCC00`, on a carbon `#0B0C10` background. Red Bull's yellow is used for lines and text because navy is hard to see on a dark background.
- **Stylised F1 logo** drawn in SVG for the header. It is not an official asset.

## Limitations and next steps

- Data is synthetic, and the circuit is a stylised shape, not survey-accurate.
- The animation always shows the qualifying pole lap, regardless of the selected session.
- The Welch t-test p-value uses a normal approximation.
- Next steps: load real telemetry with FastF1, add more circuits and drivers, and add a race-replay mode.

## Disclaimer

This is an independent portfolio project. It is not affiliated with or endorsed by Formula 1, Scuderia Ferrari or Red Bull Racing. All names and colours are used for illustration only.
