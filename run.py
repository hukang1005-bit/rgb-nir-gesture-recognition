#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run this single experiment from original Briareo images."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="RGB+NIR gray-energy temporal-memory experiment")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=[
            "run",
            "check",
            "prepare",
            "train",
            "evaluate",
            "verify",
            "self-test",
            "predict",
            "predict-current",
        ],
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--clip-id", type=int, default=966)
    parser.add_argument("--current", default="examples/clip30_currents.npz")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        from gray_experiment import pipeline as p
    except ImportError as e:
        raise SystemExit(
            "缺少环境依赖。请先运行 python -m pip install -r requirements.txt\n详细信息：" + str(e)
        )
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=args.threads):
        if args.command == "self-test":
            p.self_test()
            return
        if args.command == "predict-current":
            print(
                json.dumps(
                    p.predict_current(
                        p.local_path(args.current), p.local_path(args.model) if args.model else None
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        ctx = p.Context(args)
        if args.command == "predict":
            if not 0 <= args.clip_id < len(ctx.clips):
                parser.error("clip-id must be between 0 and1439")
            c = ctx.clips[args.clip_id]
            full = p.encoder.load_clip(ctx.dataset, c)
            plan, _ = p.encoder.sampling_plan(full, c)
            feature, _ = p.encoder.feature_for(full, c, plan[0], 0, ctx.response)
            import numpy as np

            model = dict(
                np.load(
                    p.local_path(args.model) if args.model else p.ROOT / "pretrained/deployment.npz"
                )
            )
            scores = p.score_features(feature, model)
            print(
                json.dumps(
                    {
                        "clip_id": args.clip_id,
                        "true_label": c["label"],
                        "prediction": int(scores.argmax()),
                        "gesture": ctx.names[scores.argmax()],
                        "source_indices": plan[0].tolist(),
                        "scores": scores.tolist(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        ctx.check()
        if args.command == "check":
            p.log("环境：" + json.dumps(p.versions(), ensure_ascii=False))
            return
        if args.command in ["run", "prepare"]:
            ctx.prepare()
        if args.command in ["run", "train"]:
            ctx.train()
        if args.command in ["run", "evaluate"]:
            ctx.evaluate()
        if args.command in ["run", "verify"]:
            ctx.verify()
        if args.command == "run":
            p.write(
                ctx.output / "latest_run.json",
                {
                    "signature": ctx.signature,
                    "relative_model_dir": str(ctx.model.relative_to(ctx.output)),
                },
            )
            p.log("完成。结果目录：" + str(ctx.model))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print("运行失败：" + str(e), file=sys.stderr)
        sys.exit(1)
