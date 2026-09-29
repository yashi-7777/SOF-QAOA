"""python run_experiment.py [--suite all|baselines|ablation|scaling|shots|benchmark|tables] [--size 128] [--quick]"""
import argparse
from src.utils import load_config, ensure_dirs, sample_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all"); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--size", type=int, default=None); ap.add_argument("--out", default="outputs")
    ap.add_argument("--quick", action="store_true", help="smoke run: fewer images / instances")
    a = ap.parse_args(); cfg = load_config(a.config); ensure_dirs(a.out)
    imgs = sample_images(a.size or cfg["experiments"]["image_size"])
    if a.quick: imgs = dict(list(imgs.items())[:2]); cfg["experiments"]["npcr_trials"] = 2
    from experiments import baselines, ablation, qaoa_scaling, shot_study, benchmark, tables
    s = a.suite
    if s in ("baselines", "all"): baselines.run(cfg, imgs, a.out)
    if s in ("scaling", "all"): qaoa_scaling.run(cfg, a.out, a.quick)
    if s in ("shots", "all"): shot_study.run(cfg, a.out, a.quick)
    if s in ("ablation", "all"): ablation.run(cfg, imgs, a.out)
    if s in ("benchmark", "all"): benchmark.run(cfg, imgs, a.out, a.quick)
    tables.build(a.out); print(f"[done] tables -> {a.out}/tables/ALL_TABLES.md")


if __name__ == "__main__":
    main()
