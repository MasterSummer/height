import os
import json
import glob

def process_file(filepath, output_path=None):
    with open(filepath, 'r') as f:
        data = json.load(f)

    scores = data.get("scores", {})
    if not scores:
        print(f"文件 {filepath} 没有 scores 字段，跳过。")
        return

    values = list(scores.values())
    mean_score = sum(values) / len(values)

    new_scores = {k: v - mean_score for k, v in scores.items()}
    data["scores_demean"] = new_scores
    data["mean_score"] = mean_score

    if output_path is None:
        output_path = filepath.replace(".json", "train.json")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"已处理并保存: {output_path}")

if __name__ == "__main__":
    base_dir = "analysis_runs"

    for gpu_id in range(4):
        rank_dir = os.path.join(base_dir, f"gpu{gpu_id}", "dataset_rank")

        # 匹配所有 camera_xxx_xxx_rank.json
        pattern = os.path.join(rank_dir, "camera_*_rank.json")
        rank_files = glob.glob(pattern)

        if not rank_files:
            print(f"gpu{gpu_id} 未找到匹配文件: {pattern}")
            continue

        for filepath in rank_files:
            process_file(filepath)