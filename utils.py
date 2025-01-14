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

def process_paper_evidence(input_file, data_file, output_file):
    with open(data_file, 'r') as f:
        raw_data = json.load(f)
    with open(input_file, 'r') as f:
        tmp = json.load(f)
    item_list = raw_data["questions"]
    for item in item_list:
        paper_list = [x["text"] for x in item["snippets"]]
        tmp[item["id"]]["paper_evidence"] = paper_list
    with open(output_file, 'w') as f:
        json.dump(tmp, f)

def process_pubmed_evidence(data_path, test_path, evidence_train_path, evidence_test_path):
    with open(test_path, 'r') as f:
        test_json = json.load(f)
        test_id_list_full = list(test_json.keys())
    test_id_list = random.sample(test_id_list_full, 100)

    test_data = {}
    train_data = {}
    with open(data_path, 'r') as f:
        origin_data = json.load(f)
        progress_bar = tqdm(range(len(list(origin_data.keys()))))
        for idx, item in origin_data.items():
            if "QUESTION" in item and "CONTEXTS" in item:
                if idx in test_id_list:
                    test_data[idx] = {}
                    test_data[idx]["question"] = item["QUESTION"]
                    test_data[idx]["llm_evidence"] = llm_evdence_gen(item["QUESTION"])
                    test_data[idx]["paper_evidence"] = item["CONTEXTS"]
                else:
                    train_data[idx] = {}
                    train_data[idx]["question"] = item["QUESTION"]
                    train_data[idx]["llm_evidence"] = llm_evdence_gen(item["QUESTION"])
                    train_data[idx]["paper_evidence"] = item["CONTEXTS"]
            progress_bar.update(1)

    with open(evidence_train_path, 'w') as f:
        json.dump(train_data, f)

    with open(evidence_test_path, 'w') as f:
        json.dump(test_data, f)


def adjust_learning_rate(param_group, LR, epoch, warmup_epochs, num_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-5
    if epoch < warmup_epochs:
        lr = LR * epoch / warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    input_file = "/Users/zongchang/Desktop/code/EvidenceMap/dataset/BioASQ/test/llm_evidence_test_4.json"
    data_file = "/Users/zongchang/Desktop/code/EvidenceMap/dataset/BioASQ/test/12B4_golden.json"
    output_file = "/Users/zongchang/Desktop/code/EvidenceMap/dataset/BioASQ/test/evidence_test_4.json"
    pm_data_path = "./dataset/PubMedQA/ori_pqal.json"
    pm_test_path = "./dataset/PubMedQA/test_ground_truth.json"
    pm_evidence_train_path = "./dataset/PubMedQA/evidence_train.json"
    pm_evidence_test_path = "./dataset/PubMedQA/test/evidence_test.json"
    process_paper_evidence(input_file, data_file, output_file)
    # parsed_question_file = "/Users/zongchang/Desktop/科研/DT-MLM/QA dataset/BioASQ/parsed_question_train.json"
    # llm_evidence_file = "/Users/zongchang/Desktop/科研/DT-MLM/QA dataset/BioASQ/llm_evidence_test_4.json"
    # process_question(input_file, parsed_question_file)
    # process_llm_evidence(input_file, llm_evidence_file)
    # process_pubmed_evidence(pm_data_path, pm_test_path, pm_evidence_train_path, pm_evidence_test_path)
