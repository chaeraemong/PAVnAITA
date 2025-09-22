import os, json, glob, base64
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

SYSTEM_PROMPT = """You are a vision-language analyst.
Check if this task requires the macro action 'Add to cart'.
If yes, return the earliest screenshot index where 'Add to cart' can start.
If no, return task_id=null and screenshot_idx=-1.
Return output strictly in JSON format:
{"single_instruction": "...", "task_id": "...", "screenshot_idx": ...}
"""

def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_inputs(macro_action, task_json, json_file_path):
    episode_id = task_json[0]["episode_id"]
    parts = []
    context = (
        f"[SINGLE MACRO ACTION]\n{macro_action}\n\n"
        f"[TASK]\n"
        f"task_id: {episode_id}\n"
        f"trajectory length: {len(task_json)}\n\n"
        + "\n".join([f"{step.get('step_id', i)}: {os.path.basename(step['image_path'])}"
                     for i, step in enumerate(task_json)])
    )
    parts.append({"role": "system", "content": SYSTEM_PROMPT})
    parts.append({"role": "user", "content": context})

    base_dir = os.path.dirname(json_file_path)

    for i, step in enumerate(task_json):
        sid = step.get("step_id", i)
        filename = os.path.basename(step["image_path"])
        full_path = os.path.join(base_dir, filename)

        parts.append({"role": "user", "content": f"[idx={sid}] screenshot"})
        try:
            b64img = encode_image_to_base64(full_path)
            parts.append({
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{b64img}"}]
                        })
        except FileNotFoundError:
            parts.append({"role": "user", "content": f"(missing file {full_path})"})
    return parts

def call_model(macro_action, task_json, json_file_path, model="gpt-5-nano"):
    inputs = build_inputs(macro_action, task_json, json_file_path)
    resp = client.responses.create(
        model=model,
        input=inputs,
        max_output_tokens=300
    )
    raw_text = resp.output_text
    try:
        return json.loads(raw_text)
    except Exception:
        return {"single_instruction": macro_action, "task_id": None, "screenshot_idx": -1}

def process_dataset(dataset_root, macro_action="Add to cart"):
    results = []
    json_files = glob.glob(os.path.join(dataset_root, "**/*.json"), recursive=True)
    for jf in tqdm(json_files, desc="processing"):
        with open(jf, "r") as f:
            task_json = json.load(f)
        try:
            out = call_model(macro_action, task_json, jf)
            #if out["task_id"] is not None and out["screenshot_idx"] >= 0:
            results.append(out)
        except Exception as e:
            print("Error on", jf, e)
    return results

if __name__ == "__main__":
    dataset_root = "PAVnAITA/dataset/AITZ/test/general"
    results = process_dataset(dataset_root, macro_action="Add to cart")
    with open("add_to_cart_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved results:", len(results))
