from dataset.webjudge.utils import encode_image
from PIL import Image
import re
import asyncio
MAX_IMAGE =50

async def identify_key_points(task, input_image_paths, model):
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
    
    prompt = """Task: {task}"""
    text = prompt.format(task=task)

    input_images_msg = []

    if input_image_paths != None:
        for input_image_path in input_image_paths:
            input_images_jpg_base64_str = encode_image(Image.open(input_image_path))
            input_images_msg.append(
                                        {
                                            'type': 'image_url',
                                            'image_url': {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
                                        }
                                    )

    messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]+ input_images_msg,
            }
        ]
    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]

async def judge_image(task, input_image_paths, image_path, key_points, model):
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:  
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.  
- **Score**: Assign a score based on the reasoning, using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:  
### Reasoning**: [Your explanation]  
### Score**: [1-5]"""


    prompt = """**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""
    text = prompt.format(task=task,key_points=key_points)

    input_images_msg = []
    if input_image_paths != None:
        for input_image_path in input_image_paths:
            input_images_jpg_base64_str = encode_image(Image.open(input_image_path))
            input_images_msg.append(
                                        {
                                            'type': 'image_url',
                                            'image_url': {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
                                        }
                                    )
    messages = [{"role": "system", "content": system_msg}]

    if input_images_msg:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "The input images are:"}] + input_images_msg
        })
    
    jpg_base64_str = encode_image(Image.open(image_path))
    messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}", "detail": "high"},
                    },
                ]
            }
        )

    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]


async def WebJudge_general_eval(task, input_image_paths, result_action_type, result_action_text, screenshot_paths, model, score_threshold):
    system_msg = """You are an expert in evaluating the performance of a **mobile map navigation agent**.  
The agent is designed to help a human user accomplish a navigation-related task (e.g., “Find a route to the nearest gas station”).  
You will receive the user’s task description, the agent’s action history, and several key screenshots/pages with brief justifications.  
Your goal is to decide **whether the agent has successfully completed the navigation task** and met every requirement.

Your response must strictly follow the evaluation criteria below!
*Important Evaluation Criteria*  
1. **Correct Destination or POI Displayed**  
    • The requested place (or the clearly closest / correct match) must be highlighted on the map (pin, list entry, etc.).  
2. **Proper Route Mode & Options Applied**  
    • The map must show the specific transport mode or option the user asked for (driving, walking, transit, cycling, toll-free, fastest, avoid highways, etc.).  
3. **Route Details Visible**  
    • At least one key detail—distance, ETA, real-time traffic, transfers, step list—must be shown on screen.  
4. **Navigation Started (when required)**  
    • If the user explicitly says “start navigation” (or similar), a live turn-by-turn UI or “navigation started” screen must appear.  
5. **Exact Match for Quantitative Constraints (when required)**  
    • If the task specifies a distance/price/time range (e.g., “within 5 km”, “under 10 min”), the chosen result or route must satisfy it precisely; otherwise, mark as failure.  
6. **Meaningful Progress — No Loops / Stagnation**  
    • Repeated or irrelevant actions that do not advance toward showing a valid route count as failure.  
7. **Final Confirmation / Result Displayed (when required)**  
    • Tasks that require a final confirmation (e.g., “share route”, “save place”) must show evidence that the action succeeded.

*Failure signals* include (but are not limited to):  
- Wrong destination selected or ambiguous location.  
- Requested transport mode / option not applied.  
- Map open but no route shown.  
- App shows error/blank state or user interface other than the maps screen at the end.  
- Quantitative constraint not satisfied.  
- Endless loops, missing “Start” tap, unsaved changes, etc.

---

**Format your response in exactly two lines**:
Thoughts: <your concise reasoning based on the criteria above, double-checking every key point>  
Status: "success" or "failure"
"""
    prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the mobile screenshots in the agent's trajectory and their reasons:
{thoughts}"""


    key_points = await identify_key_points(task, input_image_paths, model)
    key_points = key_points.replace("\n\n", "\n")

    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    
    tasks = [judge_image(task, input_image_paths, image_path, key_points, model) for image_path in screenshot_paths]
    image_responses = await asyncio.gather(*tasks)
    # print("image_responses : ", image_responses)

    input_images_msg = []
    whole_content_img = []
    whole_thoughts = []
    record = []
    pattern = r"[1-5]"
    for response, image_path in zip(image_responses, screenshot_paths):
        try:
            score_text = response.split("### Score")[1]
            thought = response.split("### Reasoning:")[-1].strip().lstrip("\n").split("### Score")[0].replace('\n',' ')
            score = re.findall(pattern, score_text)[0]
            record.append({"Response": response, "Score": int(score)})
        except Exception as e:
            print(f"Error processing response: {e}")
            score = 0
            record.append({"Response": response, "Score": 0})

        if int(score) >= score_threshold:
            jpg_base64_str = encode_image(Image.open(image_path))
            whole_content_img.append(
                {
                    'type': 'image_url',
                    'image_url': {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"}
                }
            )
            if thought != "":
                whole_thoughts.append(thought)

    whole_content_img = whole_content_img[:MAX_IMAGE]
    whole_thoughts = whole_thoughts[:MAX_IMAGE]
    if len(whole_content_img) == 0:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""

    if result_action_type == "type":
        last_actions = f"type '{result_action_text}'"
    else:
        last_actions = result_action_type
        
    text = prompt.format(task=task, last_actions="\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions)), key_points=key_points, thoughts = "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)))

    input_images_msg = []
    if input_image_paths is not None:
        for path in input_image_paths:
            input_images_jpg_base64_str = encode_image(Image.open(path))
            input_images_msg.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
            })

    messages = [{"role": "system", "content": system_msg}]

    if input_images_msg:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "The input images are:"}] + input_images_msg
        })

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": text}] + whole_content_img
    })
    
    return messages, record, key_points