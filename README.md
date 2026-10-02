# Boxing AI Performance Analyzer

An AI-powered combat sports analytics system that tracks fighters, detects
strikes, and generates live performance analytics from boxing video.

This repo is built **phase by phase**, so at every stage you have something
that actually runs, before adding the next layer of complexity.

---

## Build Roadmap

| Phase | What | Needs |
|---|---|---|
| **1 (this repo, done)** | Pose tracking + tracking IDs (foundation) | Pretrained model, any video, CPU is fine |
| 2 | Custom glove detector + player-lock re-ID | Labeled dataset, GPU recommended |
| 3 | Strike classification (jab/hook/uppercut/kick) | Phase 1 keypoint data |
| 4 | Landed/Blocked/Missed resolution + defense detection | Phase 2 + 3 outputs |
| 5 | Footwork/stance analysis + live analytics panel | Phase 1 + 3 data |
| 6 | Round-by-round CSV + dashboard + "Final Stats" outro | All previous phases |

We just built **Phase 1**. Everything else stacks on top of it.

---

## Why start here?

Strike classification, hit-zone detection, and footwork analysis are all
*useless* if the underlying skeleton tracking loses or swaps fighter IDs
during a clinch. Phase 1 gives you a way to **see and measure** how stable
your tracking is before you invest time in the harder custom-training work.

---

## Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/<your-username>/boxing_ai_analyzer.git
cd boxing_ai_analyzer
```

### 1. You don't have a GPU yet — use Google Colab (free)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. New notebook → Runtime → Change runtime type → GPU (T4 is fine to start)
3. Upload this project folder, or `git clone` it once it's on GitHub
4. Run:
   ```bash
   pip install -r requirements.txt
   ```

   Always start Streamlit through the same Python environment where the
   dependencies were installed:

   ```powershell
   python -m streamlit run src/app.py
   ```

   If Streamlit reports `Descriptors cannot be created directly`, the
   environment has an old Streamlit installation paired with a newer protobuf.
   From the project folder, repair it with:

   ```powershell
   python -m pip install --upgrade -r requirements.txt
   python -m streamlit run src/app.py
   ```

### 2. Local setup (CPU testing, slower but works for short clips)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run may download the YOLO model weights used by Ultralytics. Large
local videos, generated analysis files, training runs, caches, and model
weights are intentionally ignored by Git; keep those files locally or store
them separately (for example, in Git LFS or cloud storage).

---

## You don't have footage yet — where to get some

For **development/testing only** (not redistribution):
- Record your own sparring/shadowboxing on a phone — this is honestly the
  best option since you control angle, lighting, and can iterate fast.
- Search for Creative Commons or public-domain boxing clips (Pixabay,
  Pexels, Internet Archive have some sports footage).
- Amateur boxing gyms/clubs are often happy to share footage if you ask
  and offer to share the analytics back — great way to get real messy data
  (bad angles, occlusion, refs in frame) which is what you actually need
  to stress-test the system.

Avoid using copyrighted broadcast footage (e.g., ripped from a PPV) for
anything beyond quick private testing — don't publish demos built on it.

---

## Running Phase 1

```bash
python src/pose_tracker.py --video data/videos/your_clip.mp4 --out data/output/tracked.mp4
```

Optional flags:
- `--model yolov8s-pose.pt` — use a bigger/more accurate model (slower)
- `--show` — preview live while processing

### What to look for
The script prints how many **unique track IDs** it saw across the video.

- **Ideally: 2 IDs** (one per fighter), stable throughout.
- **If you see many more** (5, 10, 20+): the tracker is losing and
  re-assigning IDs — expected at this stage! This is precisely the problem
  Phase 2 (glove detector + appearance-based re-ID) is designed to solve.
  Referees and cornermen wandering into frame are a common cause of extra
  IDs, since right now the tracker treats every detected person the same.

This number is your baseline metric — write it down. When we build Phase 2,
you'll re-run this same video and should see that number drop to 2.

## Running the web app

To use the upload-and-analysis interface locally:

```bash
streamlit run src/app.py
```

The app writes uploaded videos and generated results to `data/videos/` and
`data/output/`. Those directories are kept in the repository with placeholder
files, but their contents are not committed.

The app groups analytics into configurable time-based rounds (3 minutes by
default) and exports an additional `<name>_round_summary.csv` file. The CLI
supports the same setting:

```bash
python src/run_pipeline.py --video data/videos/your_clip.mp4 --round_duration 180
```

The results page also includes interactive charts comparing punch volume,
punch outcomes, punch types, accuracy by round, and landed target zones.
It also provides an event review timeline: select a detected punch or
defensive action to generate a two-second-before/two-second-after preview at
half speed.
Punches close together in time are grouped into combinations and exported as
`<name>_combinations.csv`, including the punch sequence and landed/blocked/
missed counts.
Each strike also includes a transparent confidence estimate based on motion
strength and resolution quality. The strike table is editable, allowing punch
type, result, and target-zone corrections to be exported as
`<name>_strikes_corrected.csv`. Saving corrections also rebuilds the derived
summary, round-summary, combinations, performance, and summary-image files
with a `_corrected` suffix, and refreshes the dashboard from those reports.
Corrections do not rewrite the original annotated video.
The pipeline also exports `<name>_performance.csv` with an explainable
0-100 score, metric breakdown, and rule-based coaching feedback. It is a
training summary, not medical advice or an official judging score.

Strike detection also applies a minimum speed and arm-extension gate so
unrelated body motion, including leg movement, is less likely to be reported
as a punch. Events where the opponent cannot be resolved are excluded from
the user-facing punch totals rather than being reported as hits. The focused
regression tests can be run with:

```powershell
python -m unittest src/test_strike_classifier.py -v
```

Punch speed is normalized by the video's FPS, so the same movement is treated
consistently in 24, 30, and 60 FPS footage. The threshold is learned from the
video's motion distribution with only broad safety bounds, rather than one
fixed punch-speed cutoff.

---

## Project Structure

```
boxing_ai_analyzer/
├── src/
│   └── pose_tracker.py     # Phase 1: pose + tracking
├── data/
│   ├── videos/             # put your input clips here
│   └── output/             # annotated outputs land here
├── models/                 # custom-trained models will go here (Phase 2+)
├── notebooks/              # for Colab experimentation
├── requirements.txt
└── README.md
```

## GitHub notes

This repository includes a GitHub Actions workflow that checks that all
Python files compile on every push and pull request. Before pushing, review
`git status` and confirm that personal footage, generated outputs, and model
weights are not being added accidentally.

---

## Next step (Phase 2 preview)

Once you've run Phase 1 on a few clips and have a feel for where tracking
breaks down (clinches? refs entering frame? fighters leaving frame edge?),
we'll build:
1. A small labeled dataset of glove crops (a few hundred images goes a long way
   with transfer learning)
2. A lightweight glove classifier/detector fine-tuned on top of YOLOv8
3. Logic to use glove detections to confirm "this is a fighter" and filter
   out referees/cornermen from the tracked IDs

Bring your Phase 1 baseline numbers and we'll tackle that next.
