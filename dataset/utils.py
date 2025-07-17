import os, re, backoff
from typing import Tuple, List
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)

class OpenaiEngine():
    def __init__(
        self,
        api_key=None,
        stop=[],
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri = "",
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.client = OpenAI(api_key=api_key)

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=log_error
    )
    def generate(self, messages, max_new_tokens=512, temperature=0, model=None, **kwargs):
        model = model if model else self.model
        response = self.client.chat.completions.create(
            model=model if model else self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return [choice.message.content for choice in response.choices]

def OpenaiResponse(args, text, sub_text):
    if args.mode == 'task':
        system_msg = """
**System Prompt — Instruction-Generation Agent (Mobile Apps)**

You are an *Instruction Generator* specialized in mobile-application domains.
Your job is to produce clear, feasible, and fully specified task instructions that a mobile agent can execute inside a given Android app.

**You will always receive two inputs**

1. **Application Description**
   A textual overview of the app’s features and services. From this, infer the app’s domain characteristics and deduce what kinds of tasks are possible.

2. **Instruction Examples**
   Sample directives (written for the web) that illustrate the required style and structure. Use them as a template, but create instructions that make sense for a *mobile* environment—*not* a website.

**Generation Rules**

- Produce **exactly** the number of instructions requested.
- Each instruction must be:
  - **Unambiguous** - the agent should know precisely what to do.
  - **Executable** - every step is realistically achievable within the described app.
  - **Aligned** - the agent’s actions should match the wording of the instruction.
- You may make *reasonable* assumptions based on the Application Description, but avoid speculative or unsupported features.
- Compound instructions are allowed (e.g., “Locate the nearest garage, then summarize its reviews”), but all subtasks must be completed for the instruction to count as successful.

**Output Format**

1. **Brief Rationale** - Explain, in a few sentences, how you derived the instruction list from the Application Description.
2. **Instruction List** - Provide the tasks exactly in the following form: [instruction 1, instruction 2, instruction 3, …]

Follow this template strictly.
"""
        prompt = """Application Description: {app_desc}

Instruction Examples: {inst_ex}

Generate {task_gen_num} instructions.
"""

        text = prompt.format(app_desc=text, inst_ex = sub_text, task_gen_num=args.task_gen_num)

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": text}] + [{'type': 'text', 'text': "Your instructions generated:\n"}]
            }
        ]
    
    ################################################################################################################################

    elif args.mode == 'macro_action':
        system_msg = """As an macro-action generator, you will be presented with four primary components to assist you in your role:

1. Android Application Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'FAILURE'.



Only use the verbs in [{text}].




"""
    
        prompt = """TASK: {task}

Result Response: {response}

{num} screenshots at the end: """

        whole_content_img = []
        images_path = images_path[:MAX_IMAGE]
        text = prompt.format(task=task, response=response, num = len(images_path) if k == 0 else k)

        for image in images_path[-k:]:
            jpg_base64_str = encode_image(Image.open(image))
            whole_content_img.append(
                {
                    'type': 'image_url',
                    'image_url': {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"}
                }
            )
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": text}] 
                + whole_content_img
                + [{'type': 'text', 'text': "Your verdict:\n"}]
            }
        ]

    ################################################################################################################################
    
    return messages


def application_description(args):
    if args.app_name == "google_maps":
        app_desc = """This is a map and navigation application. Its primary functions are to search for specific places and provide routes for driving, public transit, or walking. It displays real-time trafficand allows searching for nearby points of interest like restaurants and cafes. Additionally, you can check menu,photo,review or other details of specific place."""
    elif args.app_name == "aliexpress":
        app_desc = """This is a shopping application. Its primary functions are to search for products, add them to the cart or wishlist, and make purchases. Users can apply filters such as price or category to find the most affordable or relevant items. The application also allows browsing through detailed product information, reviews, and seller ratings. Additionally, it provides access to user account data including purchase history, saved payment methods, saved items in the wishlist, and shipping information."""
    else:
        print("Invalid application! Description required for the new app!")
    return app_desc


def extract_instruction_response(response: str) -> Tuple[str, List[str]]:
    # Brief Rationale part
    rationale_pattern = (
        r"\*\*Brief Rationale\*\*\s*:?\s*"
        r"(.*?)"                                                # 최소 탐욕(non-greedy)으로 캡쳐
        r"(?=\s*\*\*Instruction List\*\*)"                   # Instruction List 전까지
    )
    m1 = re.search(rationale_pattern, response, re.DOTALL)
    brief_rationale = m1.group(1).strip() if m1 else ""

    # Instruction List part
    instr_block_pattern = r"\*\*Instruction List\*\*:\s*(.*)"
    parts = re.split(instr_block_pattern, response, maxsplit=1)
    instr_block = parts[1] if len(parts) == 2 else response
    instructions = re.findall(r"^\s*\d+\.\s*(.+)$", instr_block, flags=re.MULTILINE)

    return brief_rationale, instructions