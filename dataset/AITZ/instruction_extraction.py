import argparse
import os
import json

    
def extract_instruction(args, cat, cat_dir, epi_set):
    output_json_path = os.path.join(args.dataset_dir, f"{args.dataset_type}_{cat}_instruction_list.json")

    instruction = None
    for epi_id in epi_set:
        with open(os.path.join(cat_dir, epi_id, f'{epi_id}.json')) as f:
            for line in f:
                line = line.strip()
                result = json.loads(line)

            instruction = result[0]["instruction"]
            print("instruction : ", instruction)
            print("="*20)

        with open(output_json_path, "a+") as f_out:
            f_out.write(json.dumps(instruction) + "\n")
    

def auto_extract(args):
    dataset_folder = os.path.join(args.dataset_dir, args.dataset_type)

    category_dirs = [d for d in sorted(os.listdir(dataset_folder)) if os.path.isdir(os.path.join(dataset_folder, d))]
    print(f"Extracting {len(category_dirs)} categories in total.")

    category_set = category_dirs[0:len(category_dirs)]
    for category in category_set:
        category_dir = os.path.join(dataset_folder, category)
        episode_dirs = [d for d in sorted(os.listdir(category_dir)) if os.path.isdir(os.path.join(category_dir, d))]
        extract_instruction(args, category, category_dir, episode_dirs)

    print("Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument("--dataset_dir", type=str, default='./android_in_the_zoo', help="Path to dataset directory")
    parser.add_argument("--dataset_type", type=str, default='test', help="train or test")
    args = parser.parse_args()

    auto_extract(args)
