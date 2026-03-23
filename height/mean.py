import os
import json
import glob
from collections import defaultdict

def score_to_rank(scores, reverse=True):
    """
    单文件内 score → rank
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)
    return {k: i + 1 for i, (k, _) in enumerate(sorted_items)}


def load_rank_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    scores = data.get("scores", {})
    if not scores:
        return None

    # 关键：只在单摄像头内使用 score
    return score_to_rank(scores)


def merge_ranks(rank_dicts):
    """
    多摄像头 / 多 GPU rank 合并（Borda）
    """
    merged = defaultdict(list)
    for ranks in rank_dicts:
        for k, r in ranks.items():
            merged[k].append(r)

    return {k: sum(v) / len(v) for k, v in merged.items()}


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ========== 训练集 ==========
    train_rank_all = []

    analysis_root = os.path.join(script_dir, "..", "analysis_runs")
    for gpu_id in range(4):
        rank_dir = os.path.join(analysis_root, f"gpu{gpu_id}", "dataset_rank")
        pattern = os.path.join(rank_dir, "camera_*_rank.json")

        for fp in glob.glob(pattern):
            ranks = load_rank_file(fp)
            if ranks is not None:
                train_rank_all.append(ranks)

    if train_rank_all:
        trainrank = merge_ranks(train_rank_all)
        out = os.path.join(analysis_root, "trainrank.json")
        with open(out, "w") as f:
            json.dump({"rank": trainrank}, f, indent=2)
        print(f"已生成训练集 rank: {out}")
    else:
        print("未找到任何训练集 rank 文件")

    # ========== 测试集 ==========
    test_rank_all = []

    test_rank_dir = os.path.join(script_dir, "..", "height", "labels_rank")
    pattern = os.path.join(test_rank_dir, "camera_*_rank.json")

    for fp in glob.glob(pattern):
        ranks = load_rank_file(fp)
        if ranks is not None:
            test_rank_all.append(ranks)

    if test_rank_all:
        testrank = merge_ranks(test_rank_all)
        out = os.path.join(test_rank_dir, "testrank.json")
        with open(out, "w") as f:
            json.dump({"rank": testrank}, f, indent=2)
        print(f"已生成测试集 rank: {out}")
    else:
        print("未找到任何测试集 rank 文件")