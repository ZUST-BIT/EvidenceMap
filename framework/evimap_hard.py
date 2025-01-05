# RAG hard: Classical RAG without specific prompt engineering and learning strategies
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import *
from retrieval import EvidenceRetrieval
import pdb

# llama 3 prompt template
BOS = '<|begin_of_text|>'
EOS_USER = '<|eot_id|>'
EOS = '<|end_of_text|>'

IGNORE_INDEX = -100

class EviMapHard(object):
    def __init__(self, args, device):
        self.args = args
        self.max_new_tokens = args.max_new_tokens
        print("Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, device_map="auto")
        # Freeze LLM parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.word_embedding = self.model.model.get_input_embeddings().to(self.model.device)
        self.model.eval()

        self.retriever = EvidenceRetrieval(self.args)
        self.summary = EvidenceSummary(self.args)
        self.analysis = EvidenceAnalysis(self.args)

    def llama_inference_batch(self, input_text_list):
        inputs_token = self.tokenizer(input_text_list, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_size = len(inputs_token.input_ids)
        for i in range(batch_size):
            input_ids = inputs_token.input_ids[i]
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
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
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def inference(self, questions, parsed_questions, llm_evidences, questions_neg):
        evidence_text_list = self.retriever.evidence_process(parsed_questions, questions, self.args)
        evidence_sum_list = self.summary.evidence_process(questions, evidence_text_list, llm_evidences)
        evidence_analysis_list = self.analysis.evidence_process(questions, evidence_sum_list)

        prompt_list = []
        for question, evidence_sum, evidence_analysis in zip(questions, evidence_sum_list, evidence_analysis_list):
            sys_input = integrated_answering_prompt['sys_input']
            user_input = integrated_answering_prompt['user_input']
            user_input = user_input.replace('<question>', question)
            user_input = user_input.replace('<evidence_dict>', json.dumps(evidence_sum, indent=4))
            user_input = user_input.replace('<evidence_analysis>', evidence_analysis)
            prompt_list.append(sys_input + '\n' + user_input)

        output_list = self.llama_inference_batch(prompt_list)
        print(output_list)
        return output_list


class EvidenceAnalysis(object):
    def __init__(self, args):
        self.args = args
        self.max_new_tokens = args.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        # Freeze LLM parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.word_embedding = self.model.model.get_input_embeddings().to(self.model.device)
        self.model.eval()

    def llama_inference_batch(self, input_text_list):
        inputs_token = self.tokenizer(input_text_list, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_size = len(inputs_token.input_ids)
        for i in range(batch_size):
            input_ids = inputs_token.input_ids[i]
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
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
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def evidence_process(self, questions, evidence_sum_list):
        prompt_list = []
        for question, evidence_sum in zip(questions, evidence_sum_list):
            sys_input = evidence_analysis_prompt['sys_input']
            user_input = evidence_analysis_prompt['user_input']
            user_input = user_input.replace('<evidence_dict>', json.dumps(evidence_sum, indent=4))
            user_input = user_input.replace('<question>', question)
            prompt_list.append(sys_input + '\n' + user_input)

        output_list = self.llama_inference_batch(prompt_list)
        return output_list


class EvidenceSummary(object):
    def __init__(self, args):
        self.args = args
        self.max_new_tokens = args.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, device_map="auto")
        # Freeze LLM parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.word_embedding = self.model.model.get_input_embeddings().to(self.model.device)
        self.model.eval()

    def llama_inference(self, input_text):
        input_token = self.tokenizer(input_text, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        input_ids = input_token.input_ids
        inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
        attention_mask = [1] * inputs_embeds.shape[0]

        inputs_embeds = inputs_embeds.unsqueeze(0).to(self.model.device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.model.device)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def subgraph_summary(self, question, evidence_raw):
        print('EvidenceMap: Summarizing subgraph evdience...\n')
        sys_input = subgraph_sum_prompt['sys_input']
        user_input = subgraph_sum_prompt['user_input']
        if evidence_raw['subgraph']:
            triple_text = ', '.join([str(triple) for triple in evidence_raw['subgraph']])
            user_input = user_input.replace('<triples>', triple_text)
            user_input = user_input.replace('<question>', question)
            output = self.llama_inference(sys_input + '\n' + user_input)[0]
        else:
            output = 'No subgraph.'
        return output

    def path_summary(self, question, evidence_raw):
        print('EvidenceMap: Summarizing path evdience...', end=' ', flush=True)
        sys_input = path_sum_prompt['sys_input']
        user_input_tmp = path_sum_prompt['user_input']
        user_input_tmp = user_input_tmp.replace('<question>', question)
        sum_list = []
        for path in evidence_raw['path']:
            user_input = user_input_tmp.replace('<path>', str(path))
            output = self.llama_inference(sys_input + '\n' + user_input)[0]
            sum_list.append(output)
            print('.', end=' ', flush=True)
        print('\n')
        if sum_list:
            return sum_list
        else:
            return ['No path.']

    def paper_summary(self, question, evidence_raw):
        print('EvidenceMap: Summarizing paper evdience...', end=' ', flush=True)
        sys_input = paper_sum_prompt['sys_input']
        user_input_tmp = paper_sum_prompt['user_input']
        user_input_tmp = user_input_tmp.replace('<question>', question)
        sum_list = []
        for paper in evidence_raw['paper']:
            user_input = user_input_tmp.replace('<paper>', paper['text'])
            output = self.llama_inference(sys_input + '\n' + user_input)[0]
            sum_list.append(output)
            print('.', end=' ', flush=True)
        print('\n')
        if sum_list:
            return sum_list
        else:
            return ['No paper.']

    def concept_summary(self, evidence_raw):
        print('EvidenceMap: Summarizing concept evidence...\n')
        concept_str_list = []
        if evidence_raw['concept']:
            for name, definition in evidence_raw['concept'].items():
                concept_str_list.append(name + ': ' + definition)
            output = '\n'.join(concept_str_list)
        else:
            output = "No concept."
        return output

    def evidence_process(self, questions, evidence_raw_list, llm_evidences):
        evidence_list = []
        for question, evidence_raw, llm_evi in zip(questions, evidence_raw_list, llm_evidences):
            evidence = {}
            if 'subgraph' in evidence_raw:
                evidence['subgraph'] = self.subgraph_summary(question, evidence_raw) # str
            if 'path' in evidence_raw:
                evidence['path'] = self.path_summary(question, evidence_raw) # list of str
            if 'paper' in evidence_raw:
                evidence['paper'] = self.paper_summary(question, evidence_raw) # list of str
            evidence['concept'] = self.concept_summary(evidence_raw) # str
            evidence['llm'] = llm_evi # str
            evidence_list.append(evidence)
        return evidence_list
