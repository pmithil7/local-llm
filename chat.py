from typing import List
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
)


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. Give short and concise answers."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


history = []

question = "Which city is the captial of India?"

answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "And which is it for Canada?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
