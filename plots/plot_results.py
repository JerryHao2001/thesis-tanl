"""Plot metrics from result JSON files saved by the project.

Files are expected to contain the substring "-<split>-checkpoint-<stage>.json"
for example: results-8beams-len512-scierc_joint_er-dev-checkpoint-2330.json

Usage (example):
    python plots/plot_results.py --folder plots/all_20 --split dev --metrics entity_f1,relation_f1 --out out.png

Functions:
    plot_metrics(folder, split, metrics, episode_index=0, aggregate='first', show=True, out_path=None)

The function will:
 - find files in `folder` that contain `-<split>-checkpoint-` and end with `.json`
 - extract the numeric stage from the filename using the `checkpoint-<num>.json` suffix
 - read the metric values from each JSON file; if the value is a list it will take the
   `episode_index`-th element by default (or average if aggregate=='mean')
 - plot each requested metric as a line with stage on the x-axis

"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def find_result_files(folder: str, split: str, pattern: Optional[str] = None) -> List[str]:
    """Return list of file paths in folder that match the split/checkpoint pattern.

    If `pattern` is provided it must be a substring of the filename (for example the task
    name like 'scierc_coref' or 'scierc_joint_er').
    """
    out = []
    for name in os.listdir(folder):
        if not name.endswith('.json'):
            continue
        if f'-{split}-checkpoint-' not in name:
            continue
        if pattern and pattern not in name:
            continue
        out.append(os.path.join(folder, name))
    return out


def extract_stage_from_filename(path: str) -> Optional[int]:
    """Extract numeric stage from filename like '...-checkpoint-2330.json'."""
    m = re.search(r'checkpoint-(\d+)\.json$', path)
    return int(m.group(1)) if m else None


def read_metric_value(data: dict, metric: str, episode_index: int = 0, aggregate: str = 'first') -> Optional[float]:
    """Read metric value from JSON dict.

    - If the metric is missing, return None.
    - If the metric is a list: return element at episode_index (if available)
      or the mean if aggregate == 'mean'.
    - If the metric is a single numeric value, return it.
    """
    # Support dotted keys for nested values, e.g. 'scores.entity_f1' -> data['scores']['entity_f1']
    def get_nested(d, key_path):
        cur = d
        for k in key_path.split('.'):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur

    val = get_nested(data, metric)
    if val is None:
        return None
    if isinstance(val, list):
        if aggregate == 'mean':
            try:
                return float(np.mean([float(x) for x in val]))
            except Exception:
                return None
        else:
            # take episode_index if exists, otherwise first element
            idx = episode_index if episode_index < len(val) else 0
            try:
                return float(val[idx])
            except Exception:
                return None
    else:
        try:
            return float(val)
        except Exception:
            return None


def plot_metrics(folder: str,
                 split: str,
                 metrics: List[str],
                 episode_index: int = 0,
                 aggregate: str = 'first',
                 show: bool = True,
                 out_path: Optional[str] = None,
                 pattern: Optional[str] = None) -> Dict[str, Dict[int, float]]:
    """Plot metrics found in JSON files under `folder` for the chosen `split`.

    Returns a dict: {metric: {stage: value, ...}, ...}
    """
    files = find_result_files(folder, split, pattern=pattern)
    if not files:
        raise FileNotFoundError(f"No result files found in '{folder}' for split '{split}'")

    # Collect values per metric keyed by stage
    results: Dict[str, Dict[int, float]] = {m: {} for m in metrics}

    for path in files:
        stage = extract_stage_from_filename(path)
        if stage is None:
            # skip files without a checkpoint number
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")
            continue

        for metric in metrics:
            val = read_metric_value(data, metric, episode_index=episode_index, aggregate=aggregate)
            if val is not None:
                results[metric][stage] = val

    # Plot
    plt.figure(figsize=(8, 4 + 0.5 * len(metrics)))
    for metric in metrics:
        series = results.get(metric, {})
        if not series:
            print(f"Warning: no values found for metric '{metric}'")
            continue
        stages = sorted(series.keys())
        values = [series[s] for s in stages]
        plt.plot(stages, values, marker='o', label=metric)

    plt.xlabel('stage (checkpoint number)')
    plt.ylabel('metric')
    plt.title(f"Metrics in '{os.path.basename(folder)}' (split='{split}')")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot metrics from results JSON files')
    p.add_argument('--folder', '-f', required=True, help='Folder containing result JSON files')
    p.add_argument('--split', '-s', required=True, help="Split name in filenames (e.g. 'dev' or 'test')")
    p.add_argument('--metrics', '-m', required=True, help="Comma-separated metric names to plot, e.g. 'entity_f1,relation_f1'")
    p.add_argument('--episode_index', type=int, default=0, help='Episode index to pick from metric lists (default 0)')
    p.add_argument('--aggregate', choices=['first', 'mean'], default='first', help="How to aggregate list-valued metrics: 'first' or 'mean'")
    p.add_argument('--pattern', help="Optional substring to further filter filenames (e.g. 'scierc_coref')")
    p.add_argument('--list-metrics', action='store_true', help='List available metric keys in matching files and exit')
    p.add_argument('--out', help='Optional output image path to save the plot (png/pdf)')
    p.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    if not metrics:
        raise SystemExit('No metrics provided')

    # If user wants to list available metrics, scan files and print union of keys
    if args.list_metrics:
        files = find_result_files(args.folder, args.split, pattern=getattr(args, 'pattern', None))
        keys = set()
        for path in files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Collect top-level keys; if nested dicts exist, include their keys as dotted names
                for k, v in data.items():
                    if isinstance(v, dict):
                        for subk in v.keys():
                            keys.add(f"{k}.{subk}")
                    else:
                        keys.add(k)
            except Exception as e:
                print(f"Warning: could not read {path}: {e}")
        print('\n'.join(sorted(keys)))
        raise SystemExit(0)

    plot_metrics(args.folder, args.split, metrics, episode_index=args.episode_index,
                 aggregate=args.aggregate, show=(not args.no_show), out_path=args.out, pattern=getattr(args, 'pattern', None))
