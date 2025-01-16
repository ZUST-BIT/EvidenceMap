import os
import wandb
import random
import json
import openai
import evaluate
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from framework import framework_selector
from config import set_argument
from utils import adjust_learning_rate
from data_process import data_preprocess
import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    user_input = '''Please score the generated answer based on accuracy and fluency separately, comparing with the reference answer. 
The scores should be a float number between 0.0 and 1.0, where 0.0 indicates the generated answer is completely incorrect or unable to understand, 1.0 means the generated answer is totally precise and fluent.
Your output should be a dictionary such as {"accuracy": 0.8, "fluency": 0.9}, without any additional information.
The question, generated answer, and reference answer are give as below:
Question: <question>
Generated answer: <answer_gen>
Reference answer: <answer_ref>
Your output: '''
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
    def __init__(self, question, answer, question_neg, sample_id):
        self.question = question
        self.answer = answer
        self.question_neg = question_neg
        self.sample_id = sample_id
  
    def __len__(self):
        return len(self.question)
  
    def __getitem__(self, index):
        return self.question[index], self.answer[index], self.question_neg[index], self.sample_id[index]

def evaluate_fn2(question_list, answer_list, response_list):
    llm_acc_list = []
    llm_flu_list = []
    for question, answer, response in zip(question_list, answer_list, response_list):
        llm_score = llm_score_fn(question, response, answer, args.api_key)
        if llm_score:
            llm_acc_list.append(llm_score[0])
            llm_flu_list.append(llm_score[1])
    rouge_score = rouge_score_fn(response_list, answer_list)
    bleu_score = bleu_score_fn(response_list, answer_list)
    bert_score = bert_score_fn(response_list, answer_list)
    llm_acc_score = sum(llm_acc_list) / len(llm_acc_list)
    llm_flu_score = sum(llm_flu_list) / len(llm_flu_list)
    print("Rouge-L: {}, BLEU-Score: {}, BERT-Score: {}, LLM-ACC-Score: {}, LLM-FLU-Score: {}".format(str(rouge_score), str(bleu_score), str(bert_score), str(llm_acc_score), str(llm_flu_score)))
    
def evaluate_fn(test_loader, model, args, question_path, answer_path, response_path, mode='server'):
    print('Evaluating on test data...')
    response_list = []
    answer_list = []
    llm_acc_list = []
    llm_flu_list = []
    question_list = []
    progress_bar = tqdm(range(len(test_loader)))
    for test_questions, test_answers, test_questions_neg, test_sample_ids in test_loader:
        response = model.inference(test_questions, test_questions_neg, test_sample_ids)
        response_list.extend(response)
        answer_list.extend(test_answers)
        question_list.extend(test_questions)
        progress_bar.update(1)
    with open(question_path, 'w') as f:
        json.dump(question_list, f)
    with open(answer_path, 'w') as f:
        json.dump(answer_list, f)
    with open(response_path, 'w') as f:
        json.dump(response_list, f)
    if mode == 'local':
        for question, answer, response in zip(question_list, answer_list, response_list):
            llm_score = llm_score_fn(question, response, answer, args.api_key)
            if llm_score:
                llm_acc_list.append(llm_score[0])
                llm_flu_list.append(llm_score[1])
        rouge_score = rouge_score_fn(response_list, answer_list)
        bleu_score = bleu_score_fn(response_list, answer_list)
        bert_score = bert_score_fn(response_list, answer_list)
        llm_acc_score = sum(llm_acc_list) / len(llm_acc_list)
        llm_flu_score = sum(llm_flu_list) / len(llm_flu_list)
        print("Rouge-L: {}, BLEU-Score: {}, BERT-Score: {}, LLM-ACC-Score: {}, LLM-FLU-Score: {}".format(str(rouge_score), str(bleu_score), str(bert_score), str(llm_acc_score), str(llm_flu_score)))
    else:
        return

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

    device = torch.device("cuda:3" if args.use_cuda and torch.cuda.is_available() else "cpu")

    questions_train, answers_train, questions_neg_train, questions_test, answers_test, questions_neg_test = data_preprocess(args.dataset_dir, args.dataset_name)

    question_text_train = [item[1] for item in questions_train] # get question text
    question_text_test = [item[1] for item in questions_test] # get question text
    sample_ids_train = [item[0] for item in questions_train] # get sample id
    sample_ids_test = [item[0] for item in questions_test] # get sample id

    train_dataset = QAData(question_text_train, answers_train, questions_neg_train, sample_ids_train)
    test_dataset = QAData(question_text_test, answers_test, questions_neg_test, sample_ids_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False)
    
    plm_name = args.plm_model.split('/')[-1]
    llm_name = args.llm_model.split('/')[-1]
    test_q_path = './dataset/' + args.dataset_name + '/test/output/question_' + args.framework + '_' + plm_name + '_' + llm_name + '.json'
    test_a_path = './dataset/' + args.dataset_name + '/test/output/answer_' + args.framework + '_' + plm_name + '_' + llm_name + '.json'
    test_r_path = './dataset/' + args.dataset_name + '/test/output/response_' + args.framework + '_' + plm_name + '_' + llm_name + '.json'

    model = framework_selector[args.framework](args, device)

    if args.framework in ['rag', 'evimap_hard', 'llm_thought', 'rag_cot']:
        # Evaluation
        evaluate_fn(test_loader, model, args, test_q_path, test_a_path, test_r_path, mode='server')
    else:
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        trainable_params, all_param = model.print_trainable_params()
        print("Trainable parameters: "+str(trainable_params)+" All parameters: "+str(all_param))
        optimizer = optim.AdamW(params, lr=args.lr)
        model.to(device)

        for epoch in range(args.epochs):
            epoch_loss, train_loss = 0., 0.
            model.train()
            print("epoch:{}".format(epoch))
            progress_bar = tqdm(range(len(train_loader)))
            for i, batch in enumerate(train_loader):
                print("batch:{}/{}, epoch:{}".format(i, len(train_loader)-1, epoch))
                questions = batch[0]
                answers = batch[1]
                questions_neg = batch[2]
                sample_ids = batch[3]
                loss = model(questions, answers, questions_neg, sample_ids)
                print("current batch training loss:{}".format(loss))
                optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                if (i + 1) % args.grad_steps == 0:
                    adjust_learning_rate(optimizer.param_groups[0], args.lr, i / len(train_loader) + epoch, args.warmup_epochs, args.epochs)
                optimizer.step()
                epoch_loss, train_loss = epoch_loss + loss.item(), train_loss + loss.item()

                if (i + 1) % args.grad_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    wandb.log({'Lr': lr})
                    wandb.log({'Accum Loss': train_loss / args.grad_steps})
                    train_loss = 0.
                # evaluate_fn(test_loader, model, args, test_q_path, test_a_path, test_r_path, mode='server')
                progress_bar.update(1)
            print(f"Epoch: {epoch}|{args.epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
            wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

            # Evaluation
            model.eval()
            evaluate_fn(test_loader, model, args, test_q_path, test_a_path, test_r_path, mode='server')



if __name__ == "__main__":
    args = set_argument()
    main(args)
    # with open('./dataset/' + args.dataset_name + '/test/output/question.json', 'r') as f:
    #     question_list = json.load(f)
    # with open('./dataset/' + args.dataset_name + '/test/output/answer.json', 'r') as f:
    #     answer_list = json.load(f)
    # with open('./dataset/' + args.dataset_name + '/test/output/response.json', 'r') as f:
    #     response_list = json.load(f)
    # evaluate_fn2(question_list, answer_list, response_list)
