import json
import random

def data_preprocess(dataset_dir, dataset_name):
    if dataset_name == 'bioasq':
        train_data_file = dataset_dir + '/BioASQ/training.json'
        test_data_file = dataset_dir + '/BioASQ/test/12B1_golden.json'
        with open(train_data_file, 'r', encoding='utf-8') as f:
            train_dataset = json.load(f)
            item_list = train_dataset['questions']
            questions_train = [[item['id'], item['body']] for item in item_list]
            answers_train = [item['ideal_answer'][0] for item in item_list] # get the first ideal answer
            questions_neg_train = [item[1] for item in questions_train]
            random.shuffle(questions_neg_train)
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
            item_list = test_dataset['questions']
            questions_test = [[item['id'], item['body']] for item in item_list]
            answers_test = [item['ideal_answer'][0] for item in item_list] # get the first ideal answer
            questions_neg_test = [item[1] for item in questions_test]
            random.shuffle(questions_neg_test)
        return questions_train, answers_train, questions_neg_train, questions_test, answers_test, questions_neg_test
    else:
        raise Exception("Unsupported dataset.")

def get_parsed_question(dataset_dir, dataset_name, sample_ids, mode='train'):
    if dataset_name == 'bioasq':
        questions_parsed = []
        if mode == 'train':
            parsed_question_file = dataset_dir + '/BioASQ/parsed_question_train.json'
        else:
            parsed_question_file = dataset_dir + '/BioASQ/test/parsed_question_test_1.json'
        with open(parsed_question_file, 'r', encoding='utf-8') as f:
            parsed_dict = json.load(f)
            for item in sample_ids:
                questions_parsed.append(json.dumps(parsed_dict[item]["parsed_question"]))
        return questions_parsed
    else:
        raise Exception("Unsupported dataset.")

def get_evidence(dataset_dir, dataset_name, sample_ids, mode='train'):
    if dataset_name == 'bioasq':
        llm_evidence = []
        paper_evidence = []
        if mode == 'train':
            llm_evidence_file = dataset_dir + '/BioASQ/evidence_train.json'
        else:
            llm_evidence_file = dataset_dir + '/BioASQ/test/evidence_test_1.json'
        with open(llm_evidence_file, 'r', encoding='utf-8') as f:
            evidence_dict = json.load(f)
            for item in sample_ids:
                llm_evidence.append(evidence_dict[item]["llm_evidence"])
                paper_evidence.append(evidence_dict[item]["paper_evidence"])
        return llm_evidence, paper_evidence
    else:
        raise Exception("Unsupported dataset.")
