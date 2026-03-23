import json
import os
import random


def collect_person_ids(processed_dir: str, rank_dir: str):
    persons_with_tracks = {
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
    }
    persons_ranked = set()
    for name in os.listdir(rank_dir):
        if not name.endswith("_rank.json"):
            continue
        path = os.path.join(rank_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pid in data.get("ranking", []):
            persons_ranked.add(pid)
    return sorted(persons_with_tracks.intersection(persons_ranked))


def main():
    processed_dir = "height/processed_labels"
    rank_dir = "height/labels_rank"
    train_list_path = "height_rank/train_list.txt"
    test_list_path = "height_rank/test_list.txt"
    test_ratio = 0.25
    seed = 42

    persons = collect_person_ids(processed_dir, rank_dir)
    random.seed(seed)
    random.shuffle(persons)

    split_idx = int(len(persons) * (1 - test_ratio))
    train_ids = persons[:split_idx]
    test_ids = persons[split_idx:]

    with open(train_list_path, "w", encoding="utf-8") as f:
        for pid in train_ids:
            f.write(f"{pid}\n")
    with open(test_list_path, "w", encoding="utf-8") as f:
        for pid in test_ids:
            f.write(f"{pid}\n")

    print(f"Total: {len(persons)} | Train: {len(train_ids)} | Test: {len(test_ids)}")
    print(f"Saved: {train_list_path}, {test_list_path}")


if __name__ == "__main__":
    main()
