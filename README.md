# Genuine ESFP — **E**stimate → **S**mooth → **F**ilter → **P**ose‑Mapping

*Real‑time monocular 3D pose extraction and robotic imitation for the SwiftPro desktop arm*

![CI](https://github.com/Qifei-C/Genuine-ESFP/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/Qifei-C/Genuine-ESFP)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## What is ESFP?

**Genuine ESFP** turns a single RGB camera stream into smooth, physically‑valid commands for a 4‑DOF uArm SwiftPro.



| Stage             | Folder               | Default impl.    | Purpose                                |
| ----------------- | -------------------- | ---------------- | -------------------------------------- |
| **E**stimate      | `src/m2s/pose/`      | MediaPipe Pose   | Fast per‑frame 3D joints               |
| **S**mooth        | `src/m2s/smoothing/` | SmoothNet‑light  | Temporal denoising & bone‑length fix   |
| **F**ilter        | `src/m2s/filter/`    | Unscented KF     | Fuse smoothed pose with robot dynamics |
| **P**ose‑Tracking | `src/m2s/control/`   | SwiftPro USB API | Low‑latency joint commands             |

The repo also contains **unsupervised training code** that teaches the smoothing network using only raw video, enforcing temporal & skeletal consistency—no mocap labels needed.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/genuine-esfp.git
cd genuine-esfp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
```

### 2. Download test data (optional)

```bash
./scripts/download_datasets.sh       # Human3.6M + demo videos (≈5 GB)
```

### 3. Run the live demo ⚡

> Requirements: a USB‑connected SwiftPro arm on `/dev/ttyUSB0` and a webcam at index `0`.

```bash
python scripts/live_demo.py \
    --camera 0 \
    --pose_backend mediapipe \
    --smooth_weights models/smoothnet/epoch_050.pt \
    --filter_cfg configs/filter_ukf.yaml \
    --serial_port /dev/ttyUSB0
```

Wave your hand within view of the camera—the robot should mirror your motion with < 200 ms latency.

---

## Repository layout (TL;DR)

```text
monocular-to-swiftpro/
│
├── docs/                → architecture + deep dives
├── configs/             → Ω-style YAMLs for every experiment
├── data/{raw,interim,...}
├── models/{pose,smoothnet,filter}
├── notebooks/           → sanity‑check & analysis
├── scripts/             → CLI entry‑points (train, demo, export)
└── src/m2s/             → the actual library
        ├── pose/        → MediaPipe / ROMP wrappers
        ├── smoothing/   → SmoothNet + losses
        ├── filter/      → UKF / PF implementations
        ├── control/     → IK + SwiftPro SDK wrapper
        └── calibration/ → camera ↔ robot transforms
```

More detail in [`docs/architecture.md`](docs/architecture.md).

---

## Training the smoother (unsupervised)

1. **Pre‑cache raw detections**

   ```bash
   python scripts/run_pose_estimation.py \
         --input data/raw/my_recording.mp4 \
         --output data/interim/my_recording/
   ```

2. **Train**

   ```bash
   python scripts/train_smoothing.py \
         --cfg configs/smoothnet_small.yaml \
         data.interim_dir=data/interim \
         trainer.max_epochs=60
   ```

3. **Evaluate**

   ```bash
   python scripts/eval_smoothing.py \
         --weights models/smoothnet/epoch_060.pt
   ```

Customise loss weights (`λ_smooth`, `λ_bone`, `λ_2D`) directly in the config.

---

## Key dependencies

| Library              | Why we use it                         |
| -------------------- | ------------------------------------- |
| **PyTorch 2.x**      | pose backbone, smoother, UKF in Torch |
| **MediaPipe / ROMP** | off‑the‑shelf 3D pose estimation      |
| **NumPy & SciPy**    | kinematics math, UKF sigma points     |
| **hydra‑core**       | hierarchical experiment configs       |
| **uFactory SDK**     | SwiftPro serial / USB control         |

See `requirements.txt` for pinned versions.

---

## Citation

If ESFP helped your research, cite:

```text
@misc{genuine-esfp,
  title   = {Genuine ESFP: Real-time Monocular 3D Pose Smoothing and Robotic Imitation},
  author  = {Qifei, Ruichen, Yuang},
  year    = {2025},
  howpublished = {\url{https://github.com/<your-org>/genuine-esfp}}
}
```

---

## Contributing

Issues and PRs are welcome!
Please read [`CONTRIBUTING.md`](docs/contributing.md) and run `make lint && make test` before submitting.

---

## License

Distributed under the MIT License.
See [`LICENSE`](LICENSE) for full text.

---

