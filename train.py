import os
import wandb
import random
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from framework import framework_selector
from config import set_argument

def data_preprocess(dataset_dir, dataset_name):
    if dataset_name == 'bioasq':
        train_data_file = dataset_dir + '/BioASQ/training.json'
        test_data_file = dataset_dir + '/BioASQ/test/12B1_golden.json'
        with open(train_data_file, 'r', encoding='utf-8') as f:
            train_dataset = json.load(f)
            item_list = train_dataset['questions']
            questions_train = [[item['id'], item['body']] for item in item_list]
            answers_train = [item['ideal_answer'][0] for item in item_list]
            # qtype_train = [item['type'] for item in item_list]
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
            item_list = test_dataset['questions']
            questions_test = [[item['id'], item['body']] for item in item_list]
            answers_test = [item['ideal_answer'][0] for item in item_list]
            # qtype_test = [item['type'] for item in item_list]
        return questions_train, answers_train, questions_test, answers_test
    else:
        raise Exception("Unsupported dataset.")

def get_parsed_question(dataset_dir, dataset_name, questions_train, questions_test):
    if dataset_name == 'bioasq':
        questions_parsed_train = []
        questions_parsed_test = []
        parsed_question_train_file = dataset_dir + '/BioASQ/parsed_question_train.json'
        parsed_question_test_file = dataset_dir + '/BioASQ/test/parsed_question_test_1.json'
        with open(parsed_question_train_file, 'r', encoding='utf-8') as f:
            parsed_dict = json.load(f)
            for item in questions_train:
                questions_parsed_train.append(json.dumps(parsed_dict[item[0]]["parsed_question"]))
        with open(parsed_question_test_file, 'r', encoding='utf-8') as f:
            parsed_dict = json.load(f)
            for item in questions_test:
                questions_parsed_test.append(json.dumps(parsed_dict[item[0]]["parsed_question"]))
        return questions_parsed_train, questions_parsed_test
    else:
        raise Exception("Unsupported dataset.")

def get_llm_evidence(dataset_dir, dataset_name, questions_train, questions_test):
    if dataset_name == 'bioasq':
        llm_evidence_train = []
        llm_evidence_test = []
        llm_evidence_train_file = dataset_dir + '/BioASQ/llm_evidence_train.json'
        llm_evidence_test_file = dataset_dir + '/BioASQ/test/llm_evidence_test_1.json'
        with open(llm_evidence_train_file, 'r', encoding='utf-8') as f:
            evidence_dict = json.load(f)
            for item in questions_train:
                llm_evidence_train.append(evidence_dict[item[0]]["llm_evidence"])
        with open(llm_evidence_test_file, 'r', encoding='utf-8') as f:
            evidence_dict = json.load(f)
            for item in questions_test:
                llm_evidence_test.append(evidence_dict[item[0]]["llm_evidence"])
        return llm_evidence_train, llm_evidence_test
    else:
        raise Exception("Unsupported dataset.")

def rouge_score_fn(candidate, reference):
    rouge_metric = evaluate.load("rouge")
    rouge_score = rouge_metric.compute(predictions=candidate, references=reference)
    return round(float(rouge_score['rougeL']), 2)

def bleu_score_fn(candidate, reference):
    bleu_metric = evaluate.load("bleu")
    bleu_score = bleu_metric.compute(predictions=candidate, references=reference)
    return round(bleu_score['bleu'] * 100, 2)

def bert_score_fn(candidate, reference):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=candidate, references=reference, lang="en", model_type="bert-base-uncased")
    return sum(results['f1']) / len(results['f1'])

def llm_score_fn(question, candidate, reference, api_key):
    sys_input = '''You are an AI assistant tasked with evaluating the quality of generated answers in terms of accuracy and fluency.'''
    user_input = '''The question is <question>. The generated answer is <answer_gen>. The reference answer is <answer_ref>.
Please score the generated answer based on accuracy and fluency separately. 
The scores should be between 1 and 10, where 1 indicates the generated answer is completely incorrect or unable to understand, 10 means the generated answer is very precise and fluent
Your output should be a dictionary such as {"accuracy": 6, "fluency": 9}.
Output: '''
    openai.api_key = api_key
    user_input = user_input.replace('<question>', question).replace('<answer_gen>', candidate).replace('<answer_ref>', reference)
    res = openai.chat.completions.create(
        model = 'gpt-4o-mini',
        messages=[
            {"role": "system", "content": sys_input},
            {"role": "user", "content": user_input}
        ]
    )
    output = res.choices[0].message.content
    try:
        score_dict = eval(output)
        acc = score_dict['accuracy']
        flu = score_dict['fluency']
        return acc, flu
    except Exception as e:
        print(e)
        return None

class QAData(Dataset):
    def __init__(self, question, answer, query_dict, llm_evidence):
        self.question = question
        self.answer = answer
        self.query_dict = query_dict
        self.llm_evidence = llm_evidence
  
    def __len__(self):
        return len(self.question)
  
    def __getitem__(self, index):
        return self.question[index], self.answer[index], self.query_dict[index], self.llm_evidence[index]
    
def evaluate_fn(test_loader, model, args):
    response_list = []
    answer_list = []
    llm_acc_list = []
    llm_flu_list = []
    for test_questions, test_answers, test_parsed_q, test_llm_evi in test_loader:
        model.eval()
        _, response = model(test_questions, test_answers, test_parsed_q, test_llm_evi, mode='test')
        response_list.append(response)
        answer_list.append(test_answers[0])
        llm_score = llm_score_fn(question[0], response, test_answers[0], args.api_key)
        if llm_score:
            llm_acc_list.append(llm_score[0])
            llm_flu_list.append(llm_score[1])
    rouge_score = rouge_score_fn(response_list, answer_list)
    bleu_score = bleu_score_fn(response_list, answer_list)
    bert_score = bert_score_fn(response_list, answer_list)
    llm_acc_score = sum(llm_acc_list) / len(llm_acc_list)
    llm_flu_score = sum(llm_flu_list) / len(llm_flu_list)
    print("Rouge-L: {}, BLEU-Score: {}, BERT-Score: {}, LLM-ACC-Score: {}, LLM-FLU-Score: {}".format(str(rouge_score), str(bleu_score), str(bert_score), str(llm_acc_score), str(llm_flu_score)))

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args):
    wandb.init(project=f"{args.project}", name=f"{args.dataset_name}_{args.framework}_seed{args.seed}", config=args)
    set_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    questions_train, answers_train, questions_test, answers_test = data_preprocess(args.dataset_dir, args.dataset_name)
    query_dict_train, query_dict_test = get_parsed_question(args.dataset_dir, args.dataset_name, questions_train, questions_test)
    llm_evidence_train, llm_evidence_test = get_llm_evidence(args.dataset_dir, args.dataset_name, questions_train, questions_test)

    questions_train = [item[1] for item in questions_train] # get only question text
    questions_test = [item[1] for item in questions_test] # get only question text

    train_dataset = QAData(questions_train, answers_train, query_dict_train, llm_evidence_train)
    test_dataset = QAData(questions_test, answers_test, query_dict_test, llm_evidence_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False)

    model = framework_selector[args.framework](args, device)
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr)
    model.to(device)

    num_training_steps = args.epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(args.epochs):
        epoch_loss, train_loss = 0., 0.
        model.train()
        print("epoch:{}".format(epoch))
        for i, batch in enumerate(train_loader):
            print("batch:{}/{}, epoch:{}".format(i, len(train_loader)-1, epoch))
            questions = batch[0]
            answers = batch[1]
            parsed_questions = batch[2]
            llm_evidences = batch[3]
            loss = model(questions, answers, parsed_questions, llm_evidences)
            print("current batch training loss:{}".format(loss))
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss, train_loss = epoch_loss + loss.item(), train_loss + loss.item()

            if (i + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr})
                wandb.log({'Accum Loss': train_loss / args.grad_steps})
                train_loss = 0.

            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        # Evaluation
        


if __name__ == "__main__":
    args = set_argument()
    main(args)
