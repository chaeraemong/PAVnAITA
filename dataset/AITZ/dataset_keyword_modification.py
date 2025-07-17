import argparse
import os
import json

    
def modify_key(args, cat, cat_dir, epi_set):
    for epi_id in epi_set:
        json_path = os.path.join(cat_dir, epi_id, f'{epi_id}.json')
        with open(json_path, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)

        if isinstance(data, list):      # 리스트 형태라면 배열의 모든 요소를 순회하며 키 교체
            for obj in data:
                if isinstance(obj, dict) and args.target_keyword in obj:
                    obj[args.new_keyword] = obj.pop(args.target_keyword)  
        elif isinstance(data, dict):    # 혹시 단일 dict 형태라면 그 자체를 교체
            if args.target_keyword in data:
                data[args.new_keyword] = data.pop(args.target_keyword)

        with open(json_path, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)
    
    print(f"[UPDATED KEY] {cat}: '{args.target_keyword}' → '{args.new_keyword}'")


def modify_keyword_in_key(args, cat, cat_dir, epi_set):
    for epi_id in epi_set:
        json_path = os.path.join(cat_dir, epi_id, f'{epi_id}.json')
        with open(json_path, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)

        if isinstance(data, list):      # 리스트 형태라면 배열의 모든 요소를 순회하며 키 교체
            for obj in data:
                if isinstance(obj, dict) and args.key in obj:
                    original = obj[args.key]
                    obj[args.key] = original.replace(args.target_keyword, args.new_keyword)    # 부분 문자열 바꾸기
        elif isinstance(data, dict):    # 혹시 단일 dict 형태라면 그 자체를 교체
            if args.key in data:
                data[args.key] = data[args.key].replace(args.target_keyword, args.new_keyword)

        with open(json_path, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)
    
    print(f"[UPDATED KEYWORD] {cat}: '{args.target_keyword}' → '{args.new_keyword}'")
    

def auto_extract(args):
    dataset_folder = os.path.join(args.dataset_dir, args.dataset_type)

    category_dirs = [d for d in sorted(os.listdir(dataset_folder)) if os.path.isdir(os.path.join(dataset_folder, d))]
    print(f"Modifying {len(category_dirs)} categories in total.")

    category_set = category_dirs[0:len(category_dirs)]
    for category in category_set:
        category_dir = os.path.join(dataset_folder, category)
        episode_dirs = [d for d in sorted(os.listdir(category_dir)) if os.path.isdir(os.path.join(category_dir, d))]
        print(f"Modifying {len(episode_dirs)} episodes in total.")
        if args.mode == 'key':
            modify_key(args, category, category_dir, episode_dirs)
        elif args.mode == 'keyword':
            modify_keyword_in_key(args, category, category_dir, episode_dirs)

    print("Keyword modification complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument("--dataset_dir", type=str, default='./AITA', help="Path to dataset directory")
    parser.add_argument("--dataset_type", type=str, default='train', help="train or test")
    parser.add_argument("--target_keyword", type=str, default='android-in-the-wild/aitw_with_gpt', help="keyword that you want to be changed")
    parser.add_argument("--new_keyword", type=str, default='AITA', help="keyword that you want to change into")
    parser.add_argument("--mode", type=str, default='keyword', help="change 'key' or 'keyword' in key")
    parser.add_argument("--key", type=str, default='image_full_path', help="the 'key' where you want to change the 'keyword'")
    args = parser.parse_args()

    auto_extract(args)
