import os, re, ast, json, random, asyncio, argparse
from typing import List
from utils import OpenaiEngine, OpenaiResponse, application_description, extract_instruction_response


def extract_epi_id(epi_dir_path):                       # epi_dir_path 폴더 내 모든 하위 폴더의 이름을 리스트 형태로 추출
    epi_set = [d for d in sorted(os.listdir(epi_dir_path)) if os.path.isdir(os.path.join(epi_dir_path, d))]
    return epi_set


def extract_number_str(id_str: str) -> str:             # 각 episode 폴더의 숫자 id만 추출(category 정보 제외)
    return id_str.split("-", 1)[1]


def sample_example_instructions(args, k: int = 10):
    json_path = os.path.join("./android_in_the_zoo", f"{args.dataset_type}_general_instruction_list.json")
    instructions = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                instr = ast.literal_eval(s)
            except Exception:                           # 혹시 따옴표가 빠져있다면, 그냥 s 자체를 사용
                instr = s.strip('"')
            instructions.append(instr)

    if len(instructions) < k:
        raise ValueError(f"Number of the total instructions is smaller than {k}: {len(instructions)}")
    
    return random.sample(instructions, k)


def episode_folder_generator(args, inst_json_path: str, list: List[str]):
    epi_dir_path = os.path.join(args.dataset_dir, args.dataset_type, args.category)     # episode별 폴더를 생성할 위치
    key = ""                                                                            # json 파일에 추가/업데이트 할 요소

    if args.mode == 'task':
        key = "instruction"
        id = ""
        with open(inst_json_path, 'r', encoding='utf-8') as f:    
            instructions = json.load(f)

        for i in range(len(instructions)):
            id = f"{args.category.upper()}-{random.randint(10**9, 10**10 - 1)}"         # instruction 별로 '(category)-(랜덤한 10자 숫자)' 형태의 id 할당
            epi_path = os.path.join(epi_dir_path, id)                                   # instruction 별로 할당된 id 이름의 episode 폴더 경로
            os.makedirs(epi_path, exist_ok=True)                                        # instruction 별로 할당된 id 이름의 episode 폴더 생성
        
        epi_dirs = extract_epi_id(epi_dir_path)                                         # instruction 별 에피소드 폴더의 이름을 리스트 형태로 추출

        if len(epi_dirs)==len(instructions):                                            # instruction 별 에피소드 폴더 정상 생성 확인
            print(f"{len(epi_dirs)} episode folders generated in total.") 
        else:
            print(f"The number of episode directories({len(epi_dirs)}) doesn`t match with the number of instructions({len(instructions)})!")
    
    for epi_id, instr in zip(epi_dirs, instructions):                                   # episode(instruction)마다 수행
        json_path = os.path.join(epi_dir_path, epi_id, f"{epi_id}.json")                # episode 폴더 내 json 파일 경로
        
        if not os.path.exists(json_path):                                               # episode 폴더 내 json 파일 없으면 생성
            initial = [{
                "episode_id": "",                                                       # 나중에 채울 자리
                key: ""                                                                 # 나중에 채울 자리
            }]
            with open(json_path, "w", encoding="utf-8") as f_new:
                json.dump(initial, f_new, ensure_ascii=False, indent=2)

        with open(json_path, "r+", encoding="utf-8") as f_in:                           # episode 폴더 내 json 파일 내용 load
            data = json.load(f_in)

            if isinstance(data, list) and data:
                for obj in data:
                    if obj.get("episode_id", "") == "":
                        obj["episode_id"] = extract_number_str(epi_id)
                    obj[key] = instr                                                    # 'key'에 원하는 내용 할당
            elif isinstance(data, dict):
                if data.get("episode_id", "") == "":
                    data["episode_id"] = extract_number_str(epi_id)
                data[key] = instr                                                       # 'key'에 원하는 내용 할당
            else:
                print(f"[WARNING] {json_path}: JSON is not list/dict.")
                data = {
                    "episode_id": extract_number_str(epi_id),
                    key: instr
                }
                continue

        with open(json_path, "w", encoding="utf-8") as f_out:                           # json 파일 업데이트
            json.dump(data, f_out, ensure_ascii=False, indent=2)

        print(f"[UPDATED] {json_path}: Updated {args.mode}")

    print("Json folder generation completed.")
    

def task_generation(args, model):
    app_desc = application_description(args)
    num_instruction_examples = 10                                                       # sample할 예시 instruction 개수
    inst_json_path = os.path.join(args.dataset_dir, f"{args.dataset_type}_{args.category}_instruction_list.json")    # 생성한 instruction 리스트의 json 파일 경로
    inst_dir = os.path.dirname(inst_json_path)
    os.makedirs(inst_dir, exist_ok=True)
    
    if args.json_gen == 0:
        instruction_examples = sample_example_instructions(args, k=num_instruction_examples)
        # print(instruction_examples)

        messages = OpenaiResponse(args, app_desc, instruction_examples)
        # print("message : ", messages)

        response = model.generate(messages)[0]
        # print("response : ", response)

        brief_rationale, instructions = extract_instruction_response(response)
        print("brief_rationale : ", brief_rationale)
        # print("instructions : ", instructions)

        with open(inst_json_path, "w", encoding="utf-8") as f_out:
            json.dump(instructions, f_out, ensure_ascii=False, indent=2)

    elif args.json_gen == 1:
        episode_folder_generator(args, inst_json_path, list)
        
    else:
        instruction_examples = sample_example_instructions(args, k=num_instruction_examples)
        messages = OpenaiResponse(args, app_desc, instruction_examples)
        response = model.generate(messages)[0]
        brief_rationale, instructions = extract_instruction_response(response)
        print("brief_rationale : ", brief_rationale)
        with open(inst_json_path, "w", encoding="utf-8") as f_out:
            json.dump(instructions, f_out, ensure_ascii=False, indent=2)
        episode_folder_generator(args, inst_json_path, list)

    print("Task generation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument("--mode", type=str, default='task', help="Want to generate 'task' or 'macro_action'?")
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument("--api_key", type=str, default="", help="The api key")
    parser.add_argument("--dataset_dir", type=str, default='./dataset/AITA', help="Path to dataset directory")
    parser.add_argument("--dataset_type", type=str, default='train', help="train or test")
    parser.add_argument("--category", type=str, default="map", help="Category for application")
    parser.add_argument("--app_name", type=str, default="google_maps", help="App name for task generation")
    parser.add_argument("--task_gen_num", type=str, default=20, help="Number of tasks to generate per application")
    parser.add_argument("--json_gen", type=int, default=2, help="0: Only generate instructions/macro_action lists, 1: Only generate new episode folders based on generated instructions/macro_action lists, 2: Generate both new instructions/macro_action lists and corresponding episode folders")
    parser.add_argument("--verb_dir", type=str, default='./dataset/AITZ/single_verb_list.json', help="Path to dataset directory")
    args = parser.parse_args()

    model = OpenaiEngine(model=args.model,api_key=args.api_key)     # Load model

    if args.mode == 'task':
        task_generation(args, model)
