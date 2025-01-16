# EvidenceMap: Progressive and explicit knowledge representation with tokens, follows retrieve-summary-analysis-reasoning paradigm
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import *
from data_process import get_parsed_question, get_evidence
import pdb

# llama 3 prompt template
BOS = '<|begin_of_text|>'
EOS_USER = '<|eot_id|>'
EOS = '<|end_of_text|>'

IGNORE_INDEX = -100

class RAGCoT(object):
    def __init__(self, args, device):
        self.args = args
        self.max_new_tokens = args.max_new_tokens
        if not args.use_api:
            print("Loading LLM...")
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
            self.tokenizer.pad_token_id = 0
            self.tokenizer.padding_side = 'left'
            self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, low_cpu_mem_usage=True, device_map="cuda:3")
            # Freeze LLM parameters
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            self.word_embedding = self.model.model.get_input_embeddings().to(self.model.device)
            self.model.eval()

        self.summary = EvidenceProcess(self.args)

    def openai_inference(self, sys_input, user_input):
        openai.api_key = self.args.api_key
        res = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_input},
                {"role": "user", "content": user_input}
            ]
        )
        output = res.choices[0].message.content
        return output

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

    def get_initial_thought(self, question):
        print("Generate initial thought.")
        sys_input = initial_thought_prompt['sys_input']
        user_input = initial_thought_prompt['user_input']
        user_input = user_input.replace('<question>', question)
        initial_thought = self.llama_inference(sys_input + '\n' + user_input)[0]
        return initial_thought

    def get_optim_thought(self, question, evidence, initial_thought):
        print("Generate optimized thought.")
        sys_input = optimize_thought_prompt['sys_input']
        user_input = optimize_thought_prompt['user_input']
        user_input = user_input.replace('<question>', question)
        user_input = user_input.replace('<evidence>', evidence)
        user_input = user_input.replace('<thought>', initial_thought)
        optim_thought = self.llama_inference(sys_input + '\n' + user_input)[0]
        return optim_thought

    def get_answer(self, question, evidence, thought):
        print("Generate answer")
        sys_input = optimize_thought_prompt['sys_input']
        user_input = optimize_thought_prompt['user_input']
        user_input = user_input.replace('<question>', question)
        user_input = user_input.replace('<evidence>', evidence)
        user_input = user_input.replace('<thought>', thought)
        answer = self.llama_inference(sys_input + '\n' + user_input)[0]
        return answer

    def inference(self, questions, questions_neg, sample_ids):
        # parsed_questions = get_parsed_question(self.args.dataset_dir, self.args.dataset_name, sample_ids, mode='test')
        llm_evidences, paper_evidences = get_evidence(self.args.dataset_dir, self.args.dataset_name, sample_ids, mode='test')

        evidence_list = self.summary.evidence_process(paper_evidences, llm_evidences, self.args.paper_num)

        output_list = []
        for question, evidence in zip(questions, evidence_list):
            initial_thought = self.get_initial_thought(question)
            optim_thought = self.get_optim_thought(question, evidence, initial_thought)
            answer = self.get_answer(question, evidence, optim_thought)
            output_list.append(answer)

        print(output_list)
        return output_list
    

class EvidenceProcess(object):
    def __init__(self, args):
        self.args = args

    def paper_process(self, paper_evis):
        if paper_evis:
            output = '\n'.join([str(paper) for paper in paper_evis])
        else:
            output = 'No paper.'
        return output

    def evidence_process(self, paper_evidences, llm_evidences, max_paper_num):
        evidence_list = []
        for paper_evis, llm_evi in zip(paper_evidences, llm_evidences):
            evidence = []
            evidence.append(self.paper_process(paper_evis[:max_paper_num])) # str
            evidence.append(llm_evi) # str
            evidence_list.append('\n'.join(evidence))
        return evidence_list

