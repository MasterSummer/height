import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser("Label by rank scores (same camera group)")
    parser.add_argument("--train_rank", type=str, required=True, help="训练集 camera_*_rank.json")
    parser.add_argument("--test_rank", type=str, required=True, help="测试集 camera_*_rank.json")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径（仅测试集，兼容旧用法）")
    parser.add_argument("--output_train", type=str, default=None, help="训练集输出 JSON 路径")
    parser.add_argument("--output_test", type=str, default=None, help="测试集输出 JSON 路径")
    parser.add_argument(
        "--higher_is_higher",
        action="store_true",
        default=True,
        help="分数越大表示身高越高（默认 True）",
    )
    return parser.parse_args()


def extract_person_id(key):
    tokens = str(key).split("_")
    if len(tokens) >= 2 and tokens[0].isdigit():
        return f"{tokens[0]}_{tokens[1]}"
    return tokens[0] if tokens else str(key)


def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"{path} 缺少 scores 字段")
    return scores


def mean_scores(scores):
    buckets = defaultdict(list)
    for key, val in scores.items():
        try:
            score = float(val)
        except (TypeError, ValueError):
            continue
        pid = extract_person_id(key)
        buckets[pid].append(score)
    return {pid: sum(vals) / len(vals) for pid, vals in buckets.items() if vals}


def select_boundary(means, higher_is_higher):
    ordered = sorted(means.items(), key=lambda item: item[1], reverse=higher_is_higher)
    if not ordered:
        return None, None
    mid = len(ordered) // 2
    return ordered[mid]


def build_labels(means, boundary_score, higher_is_higher):
    ordered = sorted(means.items(), key=lambda item: item[1], reverse=higher_is_higher)
    out = []
    for idx, (pid, score) in enumerate(ordered, start=1):
        if higher_is_higher:
            label = "high" if score >= boundary_score else "low"
        else:
            label = "high" if score <= boundary_score else "low"
        out.append({"name": pid, "mean_score": score, "label": label, "rank": idx})
    return out


def build_labels_with_global_rank(means, global_rank_dict, boundary_score, higher_is_higher):
    out = []
    for pid, score in means.items():
        if higher_is_higher:
            label = "high" if score >= boundary_score else "low"
        else:
            label = "high" if score <= boundary_score else "low"
        rank = global_rank_dict.get(pid)
        out.append({"name": pid, "mean_score": score, "label": label, "rank": rank})
    # 按全局rank排序
    out.sort(key=lambda x: x["rank"])
    return out


def write_payload(path, boundary_pid, boundary_score, labels):
    payload = {
        "boundary_person": boundary_pid,
        "boundary_score": boundary_score,
        "labels": labels,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    train_scores = load_scores(args.train_rank)
    test_scores = load_scores(args.test_rank)

    train_means = mean_scores(train_scores)
    test_means = mean_scores(test_scores)
    # 合并所有pid和分数
    all_means = {**train_means, **test_means}
    # 全局排名
    ordered = sorted(all_means.items(), key=lambda item: item[1], reverse=args.higher_is_higher)
    global_rank_dict = {pid: idx+1 for idx, (pid, _) in enumerate(ordered)}

    # 仍然用训练集分界
    boundary_pid, boundary_score = select_boundary(train_means, args.higher_is_higher)
    if boundary_pid is None:
        raise SystemExit("训练集无法确定分界人。")

    train_labels = build_labels_with_global_rank(train_means, global_rank_dict, boundary_score, args.higher_is_higher)
    test_labels = build_labels_with_global_rank(test_means, global_rank_dict, boundary_score, args.higher_is_higher)

    if args.output_train:
        write_payload(args.output_train, boundary_pid, boundary_score, train_labels)
    if args.output_test:
        write_payload(args.output_test, boundary_pid, boundary_score, test_labels)
    if args.output and not args.output_test:
        write_payload(args.output, boundary_pid, boundary_score, test_labels)


if __name__ == "__main__":
    main()
