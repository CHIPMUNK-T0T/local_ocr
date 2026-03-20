import os
from pathlib import Path


def configure_hf_caches(cache_root: str | Path = ".cache/huggingface") -> Path:
    root = Path(cache_root).expanduser().resolve()
    hf_home = root / "home"
    hub_cache = root / "hub"
    modules_cache = root / "modules"

    for path in (hf_home, hub_cache, modules_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)

    try:
        from transformers import dynamic_module_utils

        dynamic_module_utils.HF_MODULES_CACHE = str(modules_cache)
    except Exception:
        pass

    return root


def resolve_local_model_path(model_path: str | Path) -> str:
    return str(Path(model_path).expanduser().resolve())


def resolve_cached_hf_snapshot(repo_id: str) -> str | None:
    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    cache_roots = []
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        cache_roots.append(Path(env_cache))
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for cache_root in cache_roots:
        repo_dir = cache_root / repo_dir_name
        if not repo_dir.exists():
            continue

        ref_file = repo_dir / "refs" / "main"
        if ref_file.exists():
            snapshot_dir = repo_dir / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
            if snapshot_dir.exists():
                return str(snapshot_dir.resolve())

        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
            if snapshots:
                return str(snapshots[-1].resolve())

    return None
