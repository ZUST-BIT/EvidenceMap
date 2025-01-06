# EvidenceMap: progressive and implicit knowledge representation with embeddings, follows retrieve-summary-analysis-reasoning paradigm
import json
import torch
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import itertools
from retrieval import EvidenceRetrieval
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from network.gnn import load_gnn_model
from network.mlp import MLP, Classifier
from data_process import get_parsed_question, get_evidence
import pdb

# llama 3 prompt template
BOS = '<|begin_of_text|>'
EOS_USER = '<|eot_id|>'
EOS = '<|end_of_text|>'

IGNORE_INDEX = -100


class EviMapEmb(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        print("Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        # Freeze LLM parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.word_embedding = self.model.model.get_input_embeddings().to(self.model.device)

        self.evi_map_builder = EviMapBuilder(self.args, device)

    def forward(self, questions, answers, questions_neg, sample_ids):
        parsed_questions = get_parsed_question(self.args.dataset_dir, self.args.dataset_name, sample_ids, mode='train')
        llm_evidences, paper_evidences = get_evidence(self.args.dataset_dir, self.args.dataset_name, sample_ids, mode='train')

        questions_token = self.tokenizer(questions, add_special_tokens=False)
        answers_token = self.tokenizer(answers, add_special_tokens=False)
        print("Building evidence map...")
        batch_evi_emb, batch_sup_emb, batch_rel_emb = self.evi_map_builder(questions, paper_evidences, llm_evidences, questions_neg) # batch_num * evidence_num * embedding_dim

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(self.args.batch_size):
            label_input_ids = answers_token.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = questions_token.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, batch_evi_emb[i], batch_sup_emb[i], batch_rel_emb[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(self.args.batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        print("Generating and calculating loss...")
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=label_input_ids,
        )
        loss = outputs.loss
        return loss

    def inference(self, questions, questions_neg, sample_ids):
        parsed_questions = get_parsed_question(self.args.dataset_dir, self.args.dataset_name, sample_ids, mode='test')
        llm_evidences, paper_evidences = get_evidence(self.args.dataset_dir, self.args.dataset_name, sample_ids, mode='test')

        questions_token = self.tokenizer(questions, add_special_tokens=False)
        print("Building evidence map...")
        batch_evi_emb, batch_sup_emb, batch_rel_emb = self.evi_map_builder(questions, paper_evidences, llm_evidences, questions_neg)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_inputs_embeds = []
        batch_attention_mask = []

        batch_size = len(questions_token.input_ids) # for number of samples less than a batch
        for i in range(batch_size):
            # input_ids = questions_token.input_ids[i] + eos_user_tokens.input_ids
            input_ids = questions_token.input_ids[i]
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, batch_evi_emb[i], batch_sup_emb[i], batch_rel_emb[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(pred)
        return pred

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param


class EviMapBuilder(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
        self.tokenizer.pad_token_id = 0

        self.model = AutoModelForMaskedLM.from_pretrained(
            args.plm_model,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # device_map="auto"
        ).to(self.device)
        # for name, param in self.model.named_parameters():
        #    param.requires_grad = False
        self.word_embedding = self.model.get_input_embeddings()

        self.projector = MLP(args.feature_dim, args.projector_hidden_dim, args.projector_output_dim).to(self.device)
        # self.projector = torch.nn.Linear(args.feature_dim, args.projector_output_dim).to(self.device)

    def cross_entropy_loss(self, output, label):
        predictions = output.view(output.shape[0] * output.shape[1], output.shape[2])
        label_tensor = torch.as_tensor(label).to(self.device) # batch_size
        labels = label_tensor.view(label_tensor.shape[0] * label_tensor.shape[1])
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(predictions, labels)
        return loss

    def build_map(self, question, evidence_set, question_neg):
        prompt_evidence = "Evidence: {evidence} \n This evidence means in one word:" # https://arxiv.org/pdf/2307.16645
        prompt_support = "Evidence: {evidence} \n Question: {question} \n How much the evidence supports the question:"
        prompt_relation = "Evidence1: {evidence1} \n Evidence2: {evidence2} \n The logical relationship between these two pieces of evidence:"

        evidence_prompt_list = [prompt_evidence.format(evidence=x) for x in evidence_set] # n
        support_prompt_list = [prompt_support.format(evidence=x, question=question) for x in evidence_set] # n
        relation_prompt_list = [prompt_relation.format(evidence1=pair[0], evidence2=pair[1]) for pair in itertools.combinations(evidence_set, 2)] # n*(n-1)/2

        evidence_input = self.tokenizer(evidence_prompt_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        support_input = self.tokenizer(support_prompt_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        relation_input = self.tokenizer(relation_prompt_list, padding=True, truncation=True, return_tensors="pt").to(self.device)

        evidence_last_hidden = self.model(**evidence_input).hidden_states[-1]
        evidence_idx_last_non_padding = evidence_input.attention_mask.bool().sum(1)-1
        evidence_embs = evidence_last_hidden[torch.arange(evidence_last_hidden.shape[0]), evidence_idx_last_non_padding]

        support_last_hidden = self.model(**support_input).hidden_states[-1]
        support_idx_last_non_padding = support_input.attention_mask.bool().sum(1)-1
        support_embs = support_last_hidden[torch.arange(support_last_hidden.shape[0]), support_idx_last_non_padding]

        relation_last_hidden = self.model(**relation_input).hidden_states[-1]
        relation_idx_last_non_padding = relation_input.attention_mask.bool().sum(1)-1
        relation_embs = relation_last_hidden[torch.arange(relation_last_hidden.shape[0]), relation_idx_last_non_padding]

        return evidence_embs, support_embs, relation_embs

    def forward(self, questions, paper_evis, llm_evis, questions_neg):
        article_num = 0
        for item in paper_evis:
            article_num += len(item)
        print("Found {} articles for {} questions.".format(article_num, self.args.batch_size))
        batch_evi_emb = []
        batch_sup_emb = []
        batch_rel_emb = []
        for question, paper_evi, llm_evi, question_neg in zip(questions, paper_evis, llm_evis, questions_neg):
            evidence_set = paper_evi + [llm_evi]
            evidence_embs, support_embs, relation_embs = self.build_map(question, evidence_set, question_neg)
            batch_evi_emb.append(evidence_embs)
            batch_sup_emb.append(support_embs)
            batch_rel_emb.append(relation_embs)

        batch_evi_emb = self.projector(torch.stack(batch_evi_emb)) # batch_num * evidence_num * embedding_dim
        batch_sup_emb = self.projector(torch.stack(batch_sup_emb))
        batch_rel_emb = self.projector(torch.stack(batch_rel_emb))
        return batch_evi_emb, batch_sup_emb, batch_rel_emb
