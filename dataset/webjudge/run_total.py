import argparse
import os

import sys
sys.path.append('/Users/chaeraemong/Desktop/cyoh/MS. 2nd Semester/Samsung_UI')

from dataset.webjudge.webjudge_general_eval import *
from dataset.webjudge.utils import OpenaiEngine, extract_predication
import json
import copy
import asyncio
import multiprocessing

def extract_step(fname):
    name, _ = os.path.splitext(fname)   # "GENERAL-1826..._1"
    try:
        return int(name.split('_')[-1]) # "_" 뒤 숫자 부분
    except ValueError:
        return float('inf') 
    
def auto_eval(args, task_subset, final_predicted_labels, model):
    ################## get the already done task id ###############
    output_json_path = os.path.join(args.output_path, f"AITA_{args.dataset_type}_{args.category}_{args.app_name}_autoeval_results(score_threshold_{args.score_threshold}).json")
    if not os.path.exists(output_json_path):
        open(output_json_path, "w", encoding="utf-8").close()
    already_ids = []
    if os.path.exists(output_json_path):
        with open(output_json_path,"r") as f:
            already_data = f.read()
        already_tasks = already_data.splitlines()
        for item in already_tasks:
            item = json.loads(item)
            if isinstance(item, dict):
                already_ids.append(item["episode_id"])
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                already_ids.append(item[0])

    print(f"The number of already done tasks: {len(already_ids)}")

    for task_id in task_subset:

        # Skip already done task
        if task_id in already_ids:
            continue
        
        task_folder_name = task_id
        inst_id = task_folder_name.rsplit("-", 1)[0]
        input_json_path = os.path.join(args.trajectories_dir, task_folder_name, f"{inst_id}.json")

        trajectory_images_path = os.path.join(args.trajectories_dir, task_id)
        screenshot_paths = []
        result_action_type = []
        result_action_text = []
        input_image_paths = []
        task_description = None
        output_results = {}

        # Load results
        with open(input_json_path) as f:
            results = json.load(f)
        
        for result in results:
            task_description = result.get("instruction")
            result_action_type.append(result.get("result_action_type"))
            result_action_text.append(result.get("result_action_text"))
            input_image_paths.append(result.get("image_path"))

        print(f"Start evaluation for {task_description}")

        # Do the auto-eval
        if args.mode == "WebJudge_general_eval":
            files = [f for f in os.listdir(trajectory_images_path) if f.lower().endswith('.png')]
            for image in sorted(files, key=extract_step):   #(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, record, key_points = asyncio.run(WebJudge_general_eval(task_description, input_image_paths, result_action_type, result_action_text, screenshot_paths, model, args.score_threshold))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        response = model.generate(messages)[0]
        predicted_label = extract_predication(response, args.mode)
        print("predicted_label : ", predicted_label)
        
        # Store evaluation details
        output_results["episode_id"] = task_folder_name
        output_results["evaluation_details"] = response
        output_results["predicted_label"] = predicted_label

        final_predicted_labels.append(predicted_label)

        print(f"Finish evaluation for {task_description}")
        print("="*20)
        os.makedirs(args.output_path, exist_ok=True)
        with open(output_json_path, "a+") as f_out:
            f_out.write(json.dumps(output_results) + "\n")
    
    return final_predicted_labels

def parallel_eval(args):
    # Evaluate in parallel based on num of works
    task_dirs = [
        d for d in sorted(os.listdir(args.trajectories_dir)) 
        if os.path.isdir(os.path.join(args.trajectories_dir, d))
    ]
    print(f"Evaluating {len(task_dirs)} tasks in total.")

    # Load model
    model = OpenaiEngine(
        model=args.model,
        api_key=args.api_key
    )

    args.output_path = args.trajectories_dir
    
    task_set = task_dirs[0:len(task_dirs)]
    final_predicted_labels = []
    final_predicted_labels = auto_eval(args, task_set, final_predicted_labels, model)

    print("final_predicted_labels : ", final_predicted_labels)
    success_num = sum(label or 0 for label in final_predicted_labels)

    print("Evaluation complete.")
    print(f"The success rate is {(success_num / len(task_dirs)) * 100}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument('--mode', type=str, default='WebJudge_general_eval', help='the mode of evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument("--trajectories_dir", type=str, default='./dataset/AITA/train/map', help="Path to trajectories directory")
    parser.add_argument("--dataset_type", type=str, default='train', help="train or test")
    parser.add_argument("--category", type=str, default="map", help="Category for application")
    parser.add_argument("--api_key", type=str, default="", help="The api key")
    parser.add_argument("--output_path", type=str, default='./dataset/AITA', help="The output path")
    parser.add_argument('--score_threshold', type=int, default=3)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument("--app_name", type=str, default="google_maps", help="App name for planner prompt")
    args = parser.parse_args()

    parallel_eval(args)
