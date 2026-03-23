import json

# 你的测试集名单
TEST_PIDS = [
    "0324_man1","0313_man2","0326_woman1","0319_woman2","0313_man3","0326_woman2",
    "0326_woman4","0319_woman1","0324_woman4","0321_woman2","0324_woman3",
    "0327_woman1","0318_woman3","0327_woman2","0320_woman3","0320_woman4"
]

def get_pid_prefix(full_pid):
    parts = full_pid.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return full_pid

input_path = "/data1/zhaoyd/analysis_runs/300_inside2/dataset_rank/camera_300cm_inside_rank.json"
output_path = "/data1/zhaoyd/height/test_labels.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

scores = data["scores"]

train_scores = {}
test_scores = {}
for k, v in scores.items():
    prefix = get_pid_prefix(k)
    if prefix in TEST_PIDS:
        test_scores.setdefault(prefix, []).append(float(v))
    else:
        train_scores.setdefault(prefix, []).append(float(v))

train_means = {pid: sum(vals)/len(vals) for pid, vals in train_scores.items()}
test_means = {pid: sum(vals)/len(vals) for pid, vals in test_scores.items()}

ordered = sorted(train_means.items(), key=lambda x: x[1], reverse=True)
mid = len(ordered) // 2
boundary_pid, boundary_score = ordered[mid]

# 计算全体均值并全局排名
all_means = {**train_means, **test_means}
global_ordered = sorted(all_means.items(), key=lambda x: x[1], reverse=True)
global_rank = {pid: idx+1 for idx, (pid, _) in enumerate(global_ordered)}

test_labels = []
for pid, mean_score in test_means.items():
    label = "high" if mean_score >= boundary_score else "low"
    rank = global_rank[pid]
    test_labels.append({"name": pid, "mean_score": mean_score, "label": label, "rank": rank})

test_labels = sorted(test_labels, key=lambda x: x["rank"])

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "boundary_person": boundary_pid,
        "boundary_score": boundary_score,
        "labels": test_labels
    }, f, ensure_ascii=False, indent=4)

print(f"已输出 {output_path}")
