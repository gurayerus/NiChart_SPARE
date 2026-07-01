"""
Task registry for NiChart_SPARE.

Maps short task names (e.g. "AD", "BA") to model versions and Hugging Face
repo information.  Models are downloaded lazily on first use via
huggingface_hub and cached in ~/.cache/huggingface.
"""

from pathlib import Path
from typing import Optional

import yaml

_REGISTRY_PATH = Path(__file__).parent / 'task_registry.yaml'


def _load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def list_tasks() -> None:
    """Print all available tasks with their default and available versions."""
    registry = _load_registry()
    print(f"\nAvailable tasks ({len(registry)}):\n")
    col = max(len(k) for k in registry) + 2
    for task, info in registry.items():
        default = info['default_version']
        versions = ', '.join(sorted(info['versions']))
        print(f"  {task:<{col}}  default: {default:<14}  versions: {versions}")
        print(f"  {'':<{col}}  {info['description']}")
    print()


def get_model_info(task: str, version: Optional[str] = None) -> dict:
    """Return an info dict for a task/version."""
    registry = _load_registry()
    if task not in registry:
        valid = sorted(registry)
        raise ValueError(
            f"Unknown task '{task}'.\n"
            f"Run 'NiChart_SPARE test --list-tasks' to see available tasks.\n"
            f"Valid tasks: {valid}"
        )

    task_entry = registry[task]
    ver = version or task_entry['default_version']
    if ver not in task_entry['versions']:
        valid = sorted(task_entry['versions'])
        raise ValueError(
            f"Version '{ver}' not found for task '{task}'.\n"
            f"Available versions: {valid}"
        )

    ve = task_entry['versions'][ver]
    revision = ve.get('hf_revision', 'main')
    repo = ve['hf_repo']
    filename = ve['filename']
    return {
        'task': task,
        'version': ver,
        'description': task_entry['description'],
        'spare_type': task_entry['spare_type'],
        'hf_repo': repo,
        'hf_revision': revision,
        'filename': filename,
        'hf_url': f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
    }


def print_model_info(task: str, version: Optional[str] = None) -> None:
    """Print detailed model info to stdout."""
    info = get_model_info(task, version)
    print(f"\nModel info — task '{info['task']}', version '{info['version']}':")
    print(f"  Description  : {info['description']}")
    print(f"  SPARE type   : {info['spare_type']}")
    print(f"  HF repo      : {info['hf_repo']}")
    print(f"  HF revision  : {info['hf_revision']}")
    print(f"  Filename     : {info['filename']}")
    print(f"  Download URL : {info['hf_url']}")
    print(f"\nFor manual/offline use:")
    print(f"  1. Download the model file from the URL above.")
    print(f"  2. Run inference with:")
    print(f"     NiChart_SPARE test -i input.csv -m /path/to/{info['filename']} -o ./results/")
    print()


def resolve_model_path(
    task: Optional[str] = None,
    version: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Return a local path to the model file.

    If *model* is given, return it directly (always wins — no registry lookup).
    Otherwise resolve *task* (and optional *version*) via the registry,
    downloading from Hugging Face if not already cached.
    """
    if model is not None:
        return model

    if task is None:
        raise ValueError(
            "Either --task <name> or --model <path> is required.\n"
            "Run 'NiChart_SPARE test --list-tasks' to see available tasks."
        )

    info = get_model_info(task, version)
    return _download_model(info)


def _download_model(info: dict) -> str:
    """Download model from Hugging Face if not already cached; return local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with:\n"
            "  pip install huggingface_hub\n"
            f"Or download the model manually from:\n"
            f"  {info['hf_url']}\n"
            f"Then run with: NiChart_SPARE test -m /path/to/{info['filename']} ..."
        )

    print(f"Downloading {info['filename']} from Hugging Face (this may take a moment)...")
    try:
        return hf_hub_download(
            repo_id=info['hf_repo'],
            filename=info['filename'],
            revision=info['hf_revision'],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model from Hugging Face: {e}\n"
            f"If you're behind a firewall, download manually from:\n"
            f"  {info['hf_url']}\n"
            f"Then run with: NiChart_SPARE test -m /path/to/{info['filename']} ..."
        )


def download_all_default_models() -> None:
    """Download the default model version for every task. Used at Docker build time."""
    registry = _load_registry()
    for task, task_entry in registry.items():
        ver = task_entry['default_version']
        info = get_model_info(task, ver)
        print(f"[{task}] {info['filename']}")
        _download_model(info)
