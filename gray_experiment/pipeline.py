import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import PIL
import scipy
import skimage
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from . import encoder

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(message):
    print(time.strftime("%H:%M:%S") + " " + message, flush=True)


def versions():
    return dict(
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        skimage=skimage.__version__,
        sklearn=sklearn.__version__,
        torch=torch.__version__,
        pillow=PIL.__version__,
    )


def local_path(value, base=ROOT):
    p = Path(os.path.expandvars(value)).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


class Context:
    def __init__(self, args, require_dataset=True):
        self.settings = read(ROOT / "protocol/settings.json")
        self.clips = read(ROOT / "protocol/clips.json")
        self.response = dict(np.load(ROOT / "device/response_table.npz"))
        self.feature_hashes = read(ROOT / "protocol/feature_hashes.json")
        self.reference_plan = np.load(ROOT / "protocol/sampling_indices.npy")
        self.split = np.array([c["split"] for c in self.clips])
        self.labels = np.array([c["label"] for c in self.clips])
        config_path = local_path(str(args.config))
        config = read(config_path)
        root = (
            args.dataset_root if args.dataset_root is not None else config.get("dataset_root", "")
        )
        self.dataset = local_path(root, config_path.parent) if root else None
        torch.set_num_threads(args.threads)
        torch.set_grad_enabled(False)
        if require_dataset and (self.dataset is None or not self.dataset.is_dir()):
            raise ValueError(
                "请在config.json中把dataset_root改为本机Briareo根目录；该目录应包含rgb/和tof/。当前值："
                + str(root)
            )
        digest = hashlib.sha256()
        for p in [
            ROOT / "protocol/clips.json",
            ROOT / "protocol/settings.json",
            ROOT / "device/response_table.npz",
            Path(encoder.__file__),
            Path(__file__),
            ROOT / "requirements.txt",
        ]:
            digest.update(sha(p).encode())
        digest.update(str(self.dataset).encode())
        self.signature = digest.hexdigest()
        self.output = local_path(str(args.output_dir))
        self.work = self.output / self.signature[:12]
        self.cache = self.work / "cache"
        self.model = self.work / "model"
        self.names = self.settings["classes"]

    def check(self, all_files=True):
        assert len(self.clips) == 1440 and self.reference_plan.shape == (1440, 3, 40)
        groups = {
            s: set(c["subject"] for c in self.clips if c["split"] == s)
            for s in ["train", "val", "test"]
        }
        assert [int((self.split == s).sum()) for s in groups] == [936, 216, 288]
        assert [len(groups[s]) for s in groups] == [26, 6, 8]
        assert all(
            not groups[a] & groups[b]
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        )
        assert np.array_equal(self.response["wavelengths"], [660, 520, 450, 808])
        count = 0
        for i, c in enumerate(self.clips):
            assert c["clip_id"] == i and len(c["rgb"]) == len(c["nir"]) == c["n_frames"] >= 40
            for rp, ip in zip(c["rgb"], c["nir"]):
                for p in [rp, ip]:
                    assert not Path(p).is_absolute() and ".." not in Path(p).parts
                assert Path(rp).stem.split("_")[0] == Path(ip).stem.split("_")[0]
                if all_files:
                    for p in [rp, ip]:
                        if not (self.dataset / p).is_file():
                            raise FileNotFoundError("缺少原始图片：" + str(self.dataset / p))
                count += 1
        log(f"数据检查通过：1440段、{count}对RGB/NIR原帧，受试者不交叉。")
        return count

    def feature_array(self):
        return np.load(self.cache / "features.npy", mmap_mode="r")

    def prepare(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        if (self.cache / "manifest.json").exists():
            m = read(self.cache / "manifest.json")
            assert m["signature"] == self.signature and m["complete"]
            for name, digest in m["cache_sha256"].items():
                assert sha(self.cache / name) == digest
            log("复用本独立实验包生成的特征缓存。")
            return m
        progress = self.cache / "progress.json"
        state = (
            read(progress)
            if progress.exists()
            else {"done": 0, "exact_views": 0, "views": 0, "diagnostics": []}
        )
        done = state["done"]
        mode = "r+" if done else "w+"
        features = (
            np.load(self.cache / "features.npy", mmap_mode="r+")
            if done
            else np.lib.format.open_memmap(
                self.cache / "features.npy",
                mode=mode,
                dtype=np.float32,
                shape=(1440, 3, encoder.DIM),
            )
        )
        plans = (
            np.load(self.cache / "sampling_indices.npy", mmap_mode="r+")
            if done
            else np.lib.format.open_memmap(
                self.cache / "sampling_indices.npy", mode=mode, dtype=np.int16, shape=(1440, 3, 40)
            )
        )
        for i in range(done, 1440):
            clip = self.clips[i]
            full = encoder.load_clip(self.dataset, clip)
            plan, diagnostic = encoder.sampling_plan(full, clip)
            if not np.array_equal(plan, self.reference_plan[i]):
                log(f"提示：clip{i}采样位置与记录有差异，请查看核验记录。")
            plans[i] = plan
            state["diagnostics"].append(diagnostic)
            for view in range(3 if clip["split"] == "train" else 1):
                f, _ = encoder.feature_for(full, clip, plan[view], view, self.response)
                features[i, view] = f
                state["views"] += 1
                state["exact_views"] += int(
                    hashlib.sha256(f.tobytes()).hexdigest() == self.feature_hashes[i][view]
                )
            if (i + 1) % 30 == 0 or i + 1 == 1440:
                features.flush()
                plans.flush()
                state["done"] = i + 1
                write(progress, state)
                log(
                    f"原始图片 → 灰度采样 → 器件/HOG：{i + 1}/1440，已核对{state['exact_views']}/{state['views']}个精确特征视图。"
                )
        write(self.cache / "energy_diagnostics.json", state["diagnostics"])
        m = {
            "complete": True,
            "signature": self.signature,
            "dataset_root": str(self.dataset),
            "clips": 1440,
            "computed_views": state["views"],
            "exact_reference_views": state["exact_views"],
            "sampling_exact_reference": bool(np.array_equal(plans, self.reference_plan)),
            "cache_sha256": {
                p.name: sha(p)
                for p in [self.cache / "features.npy", self.cache / "sampling_indices.npy"]
            },
        }
        write(self.cache / "manifest.json", m)
        return m

    def train(self):
        self.model.mkdir(parents=True, exist_ok=True)
        if (self.model / "validation_before_test.json").exists():
            log("复用已完成训练并冻结的同一灰度SVM。")
            return read(self.model / "validation_before_test.json")
        tr, va = np.where(self.split == "train")[0], np.where(self.split == "val")[0]
        a = self.feature_array()
        log("仅用936×3训练视图拟合标准化和PCA90%。")
        x = torch.from_numpy(np.array(a[tr], dtype=np.float64).reshape(-1, encoder.DIM))
        mean, scale = x.mean(0), x.std(0, correction=0)
        scale[scale < 1e-8] = 1.0
        factor = torch.full((encoder.DIM,), 1 / math.sqrt(encoder.DIM), dtype=torch.float64)
        z = (x - mean) / scale * factor / math.sqrt(3)
        eigen, vec = torch.linalg.eigh(z @ z.T)
        assert float(eigen.min()) > -1e-7
        eigen = eigen.clamp(min=0).flip(0)
        vec = vec.flip(1)
        cumulative = eigen.cumsum(0) / eigen.sum()
        rank = int(torch.searchsorted(cumulative, torch.tensor(0.9, dtype=torch.float64))) + 1
        basis = z.T @ vec[:, :rank] / torch.sqrt(eigen[:rank])[None]
        pc = (z * math.sqrt(3)) @ basis
        val = (
            (torch.from_numpy(np.array(a[va, 0], dtype=np.float64)) - mean) / scale * factor
        ) @ basis
        clf = LinearSVC(
            C=1.0,
            loss="squared_hinge",
            penalty="l2",
            dual=False,
            tol=1e-6,
            max_iter=50000,
            multi_class="ovr",
            fit_intercept=True,
            intercept_scaling=1.0,
            random_state=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(
                pc.numpy(), np.repeat(self.labels[tr], 3), sample_weight=np.full(len(pc), 1 / 3)
            )
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError("SVM未收敛，保留输出并检查环境。")
        assert np.array_equal(clf.classes_, np.arange(12))
        coef, bias_pc = (
            torch.from_numpy(clf.coef_.T.copy()),
            torch.from_numpy(clf.intercept_.copy()),
        )
        weight = (basis @ coef) / scale[:, None] * factor[:, None]
        bias = bias_pc - mean @ weight
        val_scores = val @ coef + bias_pc
        correct = int((val_scores.argmax(1).numpy() == self.labels[va]).sum())
        np.savez(
            self.model / "PCA_training.npz",
            mean=mean.numpy(),
            scale=scale.numpy(),
            factor=factor.numpy(),
            basis=basis.numpy(),
            eigenvalues=eigen[:rank].numpy(),
            rank=rank,
            variance=float(cumulative[rank - 1]),
            train_ids=tr,
        )
        np.savez(
            self.model / "deployment.npz",
            weight=weight.numpy(),
            bias=bias.numpy(),
            PC_weight=coef.numpy(),
            PC_bias=bias_pc.numpy(),
            wavelengths=self.response["wavelengths"],
            thresholds=self.response["thresholds"],
            mean_nA=self.response["mean_nA"],
            samples_nA=self.response["samples_nA"],
            class_names=np.array(self.names),
        )
        frozen = {
            "experiment": "gray_energy_temporal_svm",
            "signature": self.signature,
            "C": 1.0,
            "rank": rank,
            "retained_variance": float(cumulative[rank - 1]),
            "val_correct": correct,
            "val_n": 216,
            "val_acc": correct / 216,
            "parameters": 338700,
            "PC_coefficients": (rank + 1) * 12,
            "svm_iterations": int(clf.n_iter_),
            "test_not_evaluated_in_training": True,
            "versions": versions(),
        }
        write(self.model / "validation_before_test.json", frozen)
        log(f"测试前冻结：PCA{rank}维，验证{correct}/216（{correct / 216 * 100:.2f}%）。")
        return frozen

    def evaluate(self):
        frozen = read(self.model / "validation_before_test.json")
        a = self.feature_array()
        model = dict(np.load(self.model / "deployment.npz"))
        scores = score_features(np.array(a[:, 0], dtype=np.float64), model)
        pred = scores.argmax(1)
        result = dict(frozen)
        for split in ["train", "val", "test"]:
            ids = np.where(self.split == split)[0]
            cm = np.zeros((12, 12), np.int64)
            np.add.at(cm, (self.labels[ids], pred[ids]), 1)
            correct = int((pred[ids] == self.labels[ids]).sum())
            result.update(
                {
                    split + "_correct": correct,
                    split + "_n": len(ids),
                    split + "_acc": correct / len(ids),
                    split + "_confusion": cm.tolist(),
                }
            )
        assert result["val_correct"] == frozen["val_correct"]
        np.savez(
            self.model / "predictions.npz",
            scores=scores,
            predictions=pred,
            labels=self.labels,
            split=self.split,
        )
        with (self.model / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "split", "subject", "true_label", "prediction", "correct"])
            for i, c in enumerate(self.clips):
                w.writerow(
                    [
                        i,
                        c["split"],
                        c["subject"],
                        c["label"],
                        int(pred[i]),
                        bool(pred[i] == c["label"]),
                    ]
                )
        reference = np.load(ROOT / "pretrained/predictions.npz")
        result["same_predictions_as_packaged_reference"] = int(
            (pred == reference["predictions"]).sum()
        )
        result["reference_score_max_error"] = float(np.abs(scores - reference["scores"]).max())
        write(self.model / "result.json", result)
        self.report(result)
        log(
            f"本轮灰度SVM：验证{result['val_acc'] * 100:.2f}%，测试{result['test_acc'] * 100:.2f}%（{result['test_correct']}/288）。"
        )
        return result

    def verify(self):
        model = dict(np.load(self.model / "deployment.npz"))
        saved = np.load(self.model / "predictions.npz")
        p = np.load(self.model / "PCA_training.npz")
        a = self.feature_array()
        mean, scale, factor, basis = [
            torch.from_numpy(p[k]) for k in ["mean", "scale", "factor", "basis"]
        ]
        tr = np.where(self.split == "train")[0]
        x = torch.from_numpy(np.array(a[tr], dtype=np.float64).reshape(-1, encoder.DIM))
        assert torch.allclose(x.mean(0), mean, atol=1e-12, rtol=1e-12)
        pc = (x - mean) / scale * factor @ basis
        orth = float((basis.T @ basis - torch.eye(basis.shape[1])).abs().max())
        assert orth < 1e-7
        cov = pc.T @ pc / 3
        offdiag = float((cov - torch.diag(torch.diagonal(cov))).abs().max())
        assert offdiag < 1e-7
        coef, bias = [torch.from_numpy(model[k]) for k in ["PC_weight", "PC_bias"]]
        y = torch.from_numpy(2 * np.eye(12)[np.repeat(self.labels[tr], 3)] - 1)
        residual = (1 - y * (pc @ coef + bias)).clamp(min=0)
        grad = coef - 2 / 3 * (pc.T @ (y * residual))
        gb = bias - 2 / 3 * (y * residual).sum(0)
        scale0 = max(
            float((-2 / 3 * pc.T @ y).abs().max()), float((-2 / 3 * y.sum(0)).abs().max()), 1
        )
        optimal = max(float(grad.abs().max()), float(gb.abs().max())) / scale0
        assert optimal < 1e-4
        center = torch.from_numpy(np.array(a[:, 0], dtype=np.float64))
        ref = (((center - mean) / scale * factor) @ basis @ coef + bias).numpy()
        error = float(np.abs(ref - saved["scores"]).max())
        assert error < 1e-7
        assert np.array_equal(ref.argmax(1), saved["predictions"])
        probes = [
            int(np.where((self.split == s) & (self.labels == c))[0][0])
            for s in ["train", "val", "test"]
            for c in range(12)
        ] + [36, 38]
        count = 0
        for i in probes:
            clip = self.clips[i]
            full = encoder.load_clip(self.dataset, clip)
            plan, _ = encoder.sampling_plan(full, clip)
            assert np.array_equal(
                plan, np.load(self.cache / "sampling_indices.npy", mmap_mode="r")[i]
            )
            for v in range(3 if clip["split"] == "train" else 1):
                feature, _ = encoder.feature_for(full, clip, plan[v], v, self.response)
                assert np.array_equal(feature, a[i, v])
                count += 1
        result = {
            "passed": True,
            "raw_clips": len(probes),
            "raw_views": count,
            "all1440_PC_vs_folded_max_error": error,
            "PCA_orthogonality_max_error": orth,
            "PCA_covariance_offdiag_max_error": offdiag,
            "SVM_stationarity_relative_error": optimal,
            "cache": read(self.cache / "manifest.json"),
        }
        write(self.model / "verification.json", result)
        log(f"核验通过：{len(probes)}段/{count}视图原图重建及1440段部署预测。")
        return result

    def report(self, r):
        names = self.settings["class_names_cn"]
        cm = np.array(r["test_confusion"])
        lines = [
            "# 灰度等能量采样器件时间记忆实验结果",
            "",
            f"训练确定性视图：{r['train_correct']}/936（{r['train_acc'] * 100:.2f}%）；验证：{r['val_correct']}/216（{r['val_acc'] * 100:.2f}%）；测试：{r['test_correct']}/288（{r['test_acc'] * 100:.2f}%）。",
            "",
            f"PCA：{r['rank']}维，保留{r['retained_variance'] * 100:.4f}%方差；线性SVM C1；部署338,700个仿射参数。",
            "",
            "| 手势 | 正确数 | 测试召回率 |",
            "|---|---:|---:|",
        ]
        for k, name in enumerate(names):
            lines.append(f"| {name} | {cm[k, k]}/24 | {cm[k, k] / 24 * 100:.2f}% |")
        lines += [
            "",
            "数据：936/216/288跨受试者划分。训练三视图；验证/测试一个确定性视图。完整NIR选帧为离线软件处理；器件状态来自实测响应表，不代表实体阵列已部署。",
            "",
            "不同系统的浮点库可能引起HOG分箱和PCA微小差异；本次实际结果及与随包记录的一致性写入result.json。",
        ]
        (self.model / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_features(feature, model):
    return (
        torch.from_numpy(np.asarray(feature, dtype=np.float64)) @ torch.from_numpy(model["weight"])
        + torch.from_numpy(model["bias"])
    ).numpy()


def predict_current(path, model_path=None):
    model = dict(np.load(model_path or ROOT / "pretrained/deployment.npz"))
    data = np.load(path)
    feature = encoder.hog_currents(data["temporal_nA"])
    scores = score_features(feature, model)
    if not np.isfinite(scores).all():
        raise ValueError("分类分数包含非有限值，请检查电流输入和模型。")
    names = read(ROOT / "protocol/settings.json")["classes"]
    result = {
        "prediction": int(scores.argmax()),
        "gesture": names[scores.argmax()],
        "scores": scores.tolist(),
        "scores_are_probabilities": False,
    }
    if "scores" in data and model_path is None:
        error = float(np.abs(scores - data["scores"]).max())
        result.update(reference_score_max_error=error, reference_scores_close=error < 1e-7)
    return result


def self_test():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    r = predict_current(ROOT / "examples/clip30_currents.npz")
    assert r["prediction"] == 10
    error = r["reference_score_max_error"]
    log(f"自检通过：无需数据集，固定电流示例预测clockwise；参考分数最大差{error:.3g}。")
    if not r["reference_scores_close"]:
        log("提示：分数存在跨平台数值差异；类别正确，差异已报告。")
    return r
