import os
import json
import re

def extract_person_id(det_id):
    # 训练集格式："0313_man4_UpChange_phonecall_400cm_inside_male_day_yhjz100_id25045"
    # 测试集格式："0313_man2"
    m = re.match(r"(\d{4}_[a-zA-Z0-9]+)", det_id)
    if m:
        return m.group(1)
    else:
        return det_id

def load_person_scores_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    person_scores = {}
    for det_id, score in data.get('scores', {}).items():
        person_id = extract_person_id(det_id)
        person_scores.setdefault(person_id, []).append(score)
    # 聚合（均值）
    person_mean_scores = {pid: sum(scores)/len(scores) for pid, scores in person_scores.items()}
    return person_mean_scores

def main(train_json, test_json, output_json):
    # 1. 训练集人物均值分数
    train_scores = load_person_scores_from_json(train_json)
    # 2. 排序并找分界线
    sorted_train = sorted(train_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_train)
    if n == 0:
        raise ValueError("训练集没有有效人物分数")
    split_idx = n // 2
    # 分界线分数
    if n % 2 == 0:
        split_score = (sorted_train[split_idx-1][1] + sorted_train[split_idx][1]) / 2
    else:
        split_score = sorted_train[split_idx][1]
    # 3. 测试集人物均值分数
    test_scores = load_person_scores_from_json(test_json)
    # 4. 打标签
    test_labels = {}
    for pid, score in test_scores.items():
        label = "high" if score > split_score else "low"
        test_labels[pid] = {"score": score, "label": label}
    # 5. 写入新测试集height文件
    output = {
        "split_score": split_score,
        "test_labels": test_labels
    }
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"已写入: {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("测试集人物高低身高区自动标注")
    parser.add_argument("--train_json", type=str, required=True, help="训练集height排名json")
    parser.add_argument("--test_json", type=str, required=True, help="测试集height排名json")
    parser.add_argument("--output_json", type=str, required=True, help="输出带标签的测试集json")
    args = parser.parse_args()
    main(args.train_json, args.test_json, args.output_json)