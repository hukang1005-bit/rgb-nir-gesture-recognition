# RGB–NIR Gesture Recognition

An offline gesture-recognition experiment combining paired RGB/ToF infrared frames with measured four-wavelength device responses. The pipeline performs gray-energy temporal sampling, device-response encoding, HOG extraction, PCA retaining 90% variance, and a linear SVM readout.

## Reference results

| Split | Correct / total | Accuracy |
| --- | --- | --- |
| Validation | 188 / 216 | 87.04% |
| Test | 244 / 288 | 84.72% |

These values come from the supplied reference model and recorded experiment, using 1,440 Briareo clips and subject-separated splits of 936/216/288. The PCA has 1,201 components; the folded deployment readout has 338,700 affine parameters. Newly trained results are written separately and are never replaced with these reference scores.

## Setup

Use 64-bit Python 3.10–3.12; Python 3.11 is recommended. The pinned PyTorch build supports Apple Silicon macOS, Windows, and Linux. Intel macOS requires a separately compatible environment.

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python run.py self-test
python run.py predict-current --current examples/clip30_currents.npz
```

The bundled current example comes from training clip 30 and should predict `clockwise`. It checks the deployment interface; it does not measure independent test accuracy.

## Reproduce from frames

Obtain the original Briareo dataset separately under its distribution terms. Raw subject images are not included. Update `config.json` or pass `--dataset-root`:

```text
Briareo/
  rgb/train/025/g00/00/rgb/000_rgb.png
  rgb/val/...
  rgb/test/...
  tof/train/025/g00/00/tof/ir/000_ir.png
  tof/val/...
  tof/test/...
```

```sh
python run.py check --dataset-root /path/to/Briareo
python run.py run --dataset-root /path/to/Briareo
```

`run` checks paired frames and subject splits, prepares 3,312 feature views, fits preprocessing on training data only, freezes the model, evaluates validation/test splits, and runs numerical checks. It requires CPU compute, roughly 16 GB RAM, and at least 5 GB free output space, excluding dataset storage. Floating-point implementations can affect HOG/PCA agreement across platforms.

Individual commands: `check`, `prepare`, `train`, `evaluate`, `verify`, `predict`, `predict-current`, and `self-test`. Run stages in that order when splitting the workflow. For example:

```sh
python run.py predict --dataset-root /path/to/Briareo --clip-id 966
```

Clip 966 is a retained reference error: the true class is clockwise and the reference model predicts counterclockwise.

The optional `bootstrap.py` creates an environment and runs the experiment. Configure the dataset first, then use `python bootstrap.py`, `bash start.command`, or `start_windows.bat`. Use `python bootstrap.py --setup-only` to install and self-test without the dataset.

## Files

- `gray_experiment/`: preprocessing, encoding, training, evaluation
- `protocol/`: fixed splits, image-relative paths, crop geometry, reference sampling and feature hashes
- `device/`: measured response table and original 450/520/660/808 nm CSV traces
- `pretrained/`: reference deployment parameters, predictions, and results
- `examples/`: a current-domain prediction example
- `reports/experiment_report.html`: offline reference report
- `reports/reference-validation/`: supplied numerical validation evidence with machine paths removed
- `ARTIFACTS_SHA256.json`: scientific-asset checksums

Generated caches and models are stored under `outputs/<signature>/`. Changes in dataset location or algorithm files create a separate cache signature. Local datasets, virtual environments, and generated outputs are ignored by Git.

## Interpretation

This experiment re-encodes RGB/ToF IR frames using measured device responses in software. Full NIR sequences select frames; eight windows are encoded as independent five-pulse sequences. It does not demonstrate a physical array, whole-system energy consumption, or continuous 40-pulse dynamics. The reference split and fixed seeds require further evaluation on independent data and repeated runs.
