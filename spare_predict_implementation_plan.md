# Implementation Plan: spare_predict Task/Model Management & Distribution

This plan covers updates to the existing `spare_predict` package to support:
1. A task registry mapping short task names to model files/versions
2. A robust CLI for `train.py` (config-driven) and `test.py` (task/model-driven)
3. Experiment tracking for training runs
4. Model distribution via Hugging Face Hub, with offline/firewall fallback
5. Packaging guidance for pip, Docker, and Singularity distribution

Implement these items against the existing codebase — do not scaffold a new project. Adapt paths, naming, and existing conventions in the current package rather than overwriting them.

---

## 1. Task Registry

**Goal:** Let users run `spare_predict --task AD` as a convenient shortcut, while still supporting explicit model paths and specific versions for advanced users.

### 1.1 Create `task_registry.yaml`
A config file (not hardcoded Python) mapping task name -> default version -> available versions -> Hugging Face repo info.

```yaml
AD:
  default_version: v1.1
  versions:
    v1.1:
      hf_repo: "yourorg/spare-AD-v1.1"
      hf_revision: "main"
      filename: "model_AD_v1.1.lib"
    v1.0:
      hf_repo: "yourorg/spare-AD-v1.0"
      hf_revision: "main"
      filename: "model_AD_v1.0.lib"

MCI:
  default_version: v2.0
  versions:
    v2.0:
      hf_repo: "yourorg/spare-MCI-v2.0"
      hf_revision: "main"
      filename: "model_MCI_v2.0.lib"
```

(Replace placeholder repo names/filenames with actual values from existing models.)

### 1.2 Implement `task_registry.py`
- Load `task_registry.yaml` at runtime.
- Provide a `resolve_model_path(args)` function:
  - If `--model <path>` is given, return it directly (always wins, no registry lookup).
  - Else if `--task <name>` is given:
    - Look up task in registry; raise clear error listing valid tasks if not found.
    - Use `--version` if provided, else the task's `default_version`.
    - Raise clear error listing valid versions if the requested version isn't found.
    - Resolve to a local file path (see Section 3 — Hugging Face download logic).
  - Else raise an error indicating either `--task` or `--model` is required.

### 1.3 Update CLI args in `test.py` / `spare_predict` entry point
- Add `--task` (short task name)
- Add `--version` (optional, modifies `--task`, not mutually exclusive with it)
- Add `--model` (explicit path; mutually exclusive *group* with `--task`, per argparse `add_mutually_exclusive_group(required=True)`)
- Add `--list-tasks`: prints all available tasks, their default version, and other available versions.
- Add `--model-info`: given `--task` (and optional `--version`), prints:
  - Task name, resolved version
  - Hugging Face repo
  - Direct HF download URL (`https://huggingface.co/{repo}/resolve/{revision}/{filename}`)
  - Expected local filename
  - Example `--model` invocation for manual/offline use

---

## 2. Hugging Face Model Download Integration

**Goal:** Resolve task/version to a local model file, downloading from Hugging Face if not already cached, with a clear fallback path for firewall/offline users.

### 2.1 Add `huggingface_hub` as a dependency
Add to `pyproject.toml` / `setup.py` / `requirements.txt`.

### 2.2 Implement download logic
In `task_registry.py` (or a new `model_loader.py`):

```python
from huggingface_hub import hf_hub_download

def get_model_path(task_entry):
    try:
        return hf_hub_download(
            repo_id=task_entry["hf_repo"],
            filename=task_entry["filename"],
            revision=task_entry.get("hf_revision", "main"),
        )
    except Exception as e:
        repo = task_entry["hf_repo"]
        revision = task_entry.get("hf_revision", "main")
        filename = task_entry["filename"]
        raise RuntimeError(
            f"Failed to download model from Hugging Face ({e}).\n"
            f"If you're behind a firewall, download manually from:\n"
            f"  https://huggingface.co/{repo}/resolve/{revision}/{filename}\n"
            f"Then run with: spare_predict --model /path/to/{filename}"
        )
```

- Print a one-line notice before downloading on first use, e.g.:
  `Downloading model_AD_v1.1.lib from Hugging Face (this may take a moment)...`
- Rely on `huggingface_hub`'s default local cache (`~/.cache/huggingface`) so repeated calls don't re-download.

### 2.3 Offline / firewall documentation
Add a "Offline / Firewall-Restricted Environments" section to `README.md`:
- Steps to manually download a model file from the printed HF URL
- Steps to transfer it to the target machine
- Example invocation using `--model /path/to/<filename>`

---

## 3. Training Experiment Tracking

**Goal:** Make it easy to run and compare multiple training experiments without losing track of which config/code produced which model.

### 3.1 Update `train.py` to create a self-contained run directory per experiment
On each run:
- Generate a `run_id` (timestamp + experiment name, e.g. `20260630_143200_baseline`)
- Create `experiments/<run_id>/`
- Copy the config file used into `experiments/<run_id>/config.yaml`
- Write `experiments/<run_id>/meta.json` containing:
  - `run_id`, `timestamp`
  - `git_commit` (via `git rev-parse HEAD`, wrapped in try/except in case not in a git repo)
  - `seed` (if applicable)
  - free-text `notes` field (optional CLI arg `--notes`)
- Save checkpoint(s) into `experiments/<run_id>/model.pt` (or existing checkpoint naming convention)
- Log per-epoch metrics into `experiments/<run_id>/metrics.json` or `train_log.txt` (train loss, val loss, any other tracked metrics)

### 3.2 Add a comparison utility (optional, lower priority)
A small script (`compare_experiments.py`) that walks `experiments/`, reads each `meta.json` + `metrics.json` + `config.yaml`, and builds a sortable summary table (pandas DataFrame) for comparing runs by hyperparameter or final metric.

### 3.3 (Optional / future) Integrate with an experiment tracker
Not required for this pass, but leave a clean integration point (e.g., a `--tracker wandb|mlflow|none` flag) so Weights & Biases or MLflow logging can be added later without restructuring `train.py` again.

---

## 4. Test Script Design

**Goal:** Keep `test.py` interface simple, but ensure it has access to whatever config/preprocessing info the model needs.

### 4.1 Confirm test.py inputs
- `--input` (data to run inference on)
- `--output_dir` (not just `--output` — predictions, metrics, and any plots should all go into one directory, not a single file path)
- Model resolved via Section 1 (`--task`/`--version` or `--model`)

### 4.2 Ensure architecture/preprocessing consistency
- If any preprocessing (normalization stats, expected input shape, etc.) depends on how the model was trained, save that info alongside the model checkpoint (or in the HF model repo as a config/JSON file) and load it automatically in `test.py` — do not require the user to separately pass training config at test time.

---

## 5. Packaging & Distribution

**Goal:** Support three distribution channels with consistent versioning.

### 5.1 Pip package
- Ensure `pyproject.toml` (or `setup.py`) defines `spare_predict` as a console script entry point.
- Models are **not** bundled in the package and **not** pre-downloaded at install time.
- Models download lazily, per-task, on first actual use (see Section 2).

### 5.2 Docker / Singularity containers
- Update Dockerfile (and Singularity def file) to **download and bundle all task model files at build time**, baking them into the image so no internet access is needed at runtime.
- Use `task_registry.yaml` as the source of truth for which models to fetch and bundle during the image build step (loop over all tasks/default versions, or all versions if you want every version available offline).

### 5.3 Version alignment
- Keep GitHub release tags, Hugging Face repo revisions/tags, and `task_registry.yaml` version keys aligned (e.g., GitHub tag `v1.1` <-> HF revision `v1.1` <-> registry key `v1.1`) so it's always traceable which code version pairs with which model version.

---

## 6. Documentation Updates (README.md)

Add or update the following sections:
1. **Quick usage**: `spare_predict --task AD`
2. **Advanced usage**: `spare_predict --task AD --version v1.0`
3. **Explicit model path**: `spare_predict --model /path/to/model.lib`
4. **Listing available tasks**: `spare_predict --list-tasks`
5. **Model info / manual download**: `spare_predict --task AD --model-info`
6. **Offline / firewall-restricted environments** (manual download + `--model` fallback steps)
7. **Distribution options**: pip vs Docker vs Singularity, and when to use each
8. **Training**: how to launch `train.py --config <path>`, where experiment outputs are saved
9. **Model weights hosting**: note that trained models are hosted on Hugging Face, code is on GitHub, and license terms may differ between the two

---

## Suggested Implementation Order

1. Task registry (`task_registry.yaml` + `task_registry.py`) — foundational, everything else depends on it
2. Hugging Face download integration + offline fallback messaging
3. Update `test.py` CLI (`--task`, `--version`, `--model`, `--list-tasks`, `--model-info`)
4. Update `train.py` for experiment run directories + metadata logging
5. Update Dockerfile / Singularity def to bundle models at build time
6. Update `pyproject.toml`/`setup.py` for pip distribution (lazy download confirmed, no bundling)
7. README updates covering all of the above
