import openai
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import os
import math
from prompt import question_parsing_prompt, llm_sum_prompt

def openai_inference(sys_input, user_input):
        openai.api_key = "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"
        res = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_input},
                {"role": "user", "content": user_input},
            ],
        )
        output = res.choices[0].message.content
        return output

def llama_inference(model_path, sys_input, user_input):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": sys_input},
        {"role": "user", "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id = tokenizer.eos_token_id,
    )

    response = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    return generated_text

def parse_question(question, model_type='openai'):
        sys_prompt = question_parsing_prompt['sys_prompt']
        user_prompt = question_parsing_prompt['user_prompt']
        user_prompt = user_prompt.replace('<question>', question)
        output = None
        try:
            if model_type == 'openai':
                res = openai_inference(sys_prompt, user_prompt)
            else:
                raise Exception("Unsupported LLM type.")
            output = eval(res)
        except Exception as e:
            print(e)
        return output

def llm_evdence_gen(question, llm_type='api'):
    sys_input = llm_sum_prompt['sys_input']
    user_input = llm_sum_prompt['user_input']
    user_input = user_input.replace('<question>', question)
    if llm_type == 'api':
        output = openai_inference(sys_input, user_input)
    elif llm_type == 'local':
        output = llama_inference(sys_input, user_input)
    else:
        raise Exception("Unsupported LLM type.")
    return output

def process_question(input_file, output_file):
    parsed_questions = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            parsed_questions = json.load(f)

    with open(input_file, 'r') as f:
       train_data = json.load(f)
    for i in tqdm(range(len(train_data['questions']))):
        item = train_data['questions'][i]
        if item["id"] not in parsed_questions:
            parsed_questions[item["id"]] = {"question": item["body"]}
            parsed_q = parse_question(item["body"])
            parsed_questions[item["id"]]["parsed_question"] = parsed_q
            if i % 100 == 0:
                with open(output_file, 'w') as f:
                    json.dump(parsed_questions, f)
    with open(output_file, 'w') as f:
        json.dump(parsed_questions, f)

def process_llm_evidence(input_file, output_file):
    llm_self_evidence = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            llm_self_evidence = json.load(f)
    with open(input_file, 'r') as f:
       train_data = json.load(f)
    for i in tqdm(range(len(train_data['questions']))):
        item = train_data['questions'][i]
        if item["id"] not in llm_self_evidence:
            llm_self_evidence[item["id"]] = {"question": item["body"]}
            evidence = llm_evdence_gen(item["body"])
            llm_self_evidence[item["id"]]["llm_evidence"] = evidence
            if i % 100 == 0:
                with open(output_file, 'w') as f:
                    json.dump(llm_self_evidence, f)
    with open(output_file, 'w') as f:
        json.dump(llm_self_evidence, f)


def adjust_learning_rate(param_group, LR, epoch, warmup_epochs, num_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < warmup_epochs:
        lr = LR * epoch / warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    input_file = "/Users/zongchang/Desktop/科研/DT-MLM/QA dataset/BioASQ/test/12B4_golden.json"
    parsed_question_file = "/Users/zongchang/Desktop/科研/DT-MLM/QA dataset/BioASQ/parsed_question_train.json"
    llm_evidence_file = "/Users/zongchang/Desktop/科研/DT-MLM/QA dataset/BioASQ/llm_evidence_test_4.json"
    # process_question(input_file, parsed_question_file)
    process_llm_evidence(input_file, llm_evidence_file)
