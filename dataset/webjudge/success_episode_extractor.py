import argparse
import os
import json
from collections import defaultdict
# import sys
# sys.path.append('/Users/chaeraemong/Desktop/cyoh/MS. 2nd Semester/Samsung_UI')

def aggregate_episode_labels(input_json_path, output_json_path):
    """
    JSONL 파일을 한 줄씩 읽어서,
      - episode_id의 prefix (MAP-A) 별로
      - 에피소드 개수와 predicted_label 합계를 계산해 반환.
    
    반환값: dict
      {
        "MAP-1466028412": {"count": 5, "sum": 13},
        "MAP-1069716234": {"count": 2, "sum": 2},
        ...
      }
    """
    counts = defaultdict(int)
    sums   = defaultdict(int)
    
    with open(input_json_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[JSONDecodeError] line {lineno}: {e}\n  >> {line}")
                continue
            
            eid = rec.get("episode_id")
            lbl = rec.get("predicted_label")
            if eid is None or lbl is None:
                continue
            
            # MAP-A-B → MAP-A 로 자르기
            base_id = eid.rsplit("-", 1)[0]
            
            counts[base_id] += 1
            sums[base_id]   += int(lbl)

    with open(output_json_path, "w", encoding="utf-8") as f_out:
        for base_id in sorted(counts):
            line = f"{base_id}: episodes={counts[base_id]}, predicted_label_sum={sums[base_id]}"
            print(line)
            f_out.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument("--trajectories_dir", type=str, default='./dataset/AITA/train/map', help="Path to trajectories directory")
    parser.add_argument("--dataset_type", type=str, default='train', help="train or test")
    parser.add_argument("--category", type=str, default="map", help="Category for application")
    parser.add_argument('--score_threshold', type=int, default=3)
    parser.add_argument("--app_name", type=str, default="google_maps", help="App name for planner prompt")
    args = parser.parse_args()

    input_json_path = os.path.join(args.trajectories_dir, f"AITA_{args.dataset_type}_{args.category}_{args.app_name}_autoeval_results(score_threshold_{args.score_threshold}).json")
    output_json_path = os.path.join(args.trajectories_dir, f"AITA_{args.dataset_type}_{args.category}_{args.app_name}_autoeval_success_results(score_threshold_{args.score_threshold}).json")
    
    if not os.path.exists(input_json_path):
        print("No webjudge result json file!")
    else:
        agg = aggregate_episode_labels(input_json_path, output_json_path)
