# RAG hard: Classical RAG without specific prompt engineering and learning strategies
from prompt import *

class RAG(object):
    def __init__(self, args):
        self.args = args
        print("Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, device_map="auto")

        self.retriever = EvidenceRetrieval(self.args)
        self.summary = EvidenceSummary(self.args)
        self.analysis = EvidenceAnalysis(self.args)

    def llama_inference_batch(self, input_text_list):
        encoding = self.tokenizer(input_text_list, return_tensors="pt").to(self.model.device)
        generation_output = self.model.generate(
            **encoding, 
            return_dict_in_generate=True, 
            output_logits=True, 
            max_new_tokens=self.args.max_new_tokens, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return output_text

    def inference(self, questions, parsed_questions, llm_evidences):
        evidence_text_list = self.retriever.evidence_process(parsed_questions, questions, self.args)
        evidence_sum_list = self.summary.evidence_process(questions, evidence_text_list, llm_evidences)
        evidence_analysis_list = self.analysis.evidence_process(questions, evidence_sum_list)

        prompt_list = []
        for evidence_sum, evidence_analysis in zip(evidence_sum_list, evidence_analysis_list):
            sys_input = integrated_answering_prompt['sys_input']
            user_input = integrated_answering_prompt['user_input']
            user_input = user_input.replace('<evidence_dict>', evidence_sum)
            user_input = user_input.replace('<evidence_analysis>', evidence_analysis)
            prompt_list.append(sys_input + '\n' + user_input)

        output_list = self.llama_inference_batch(prompt_list)
        return output_list


class EvidenceAnalysis(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, device_map="auto")

    def llama_inference_batch(self, input_text_list):
        encoding = self.tokenizer(input_text_list, return_tensors="pt").to(self.model.device)
        generation_output = self.model.generate(
            **encoding, 
            return_dict_in_generate=True, 
            output_logits=True, 
            max_new_tokens=self.args.max_new_tokens, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return output_text

    def evidence_process(self, questions, evidence_sum_list):
        prompt_list = []
        for question, evidence_sum in zip(questions, evidence_sum_list):
            sys_input = evidence_analysis_prompt['sys_input']
            user_input = evidence_analysis_prompt['user_input']
            user_input = user_input.replace('<evidence_dict>', str(evidence_sum))
            user_input = user_input.replace('<question>', question)
            prompt_list.append(sys_input + '\n' + user_input)

        output_list = self.llama_inference_batch(prompt_list)
        return output_list


class EvidenceSummary(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, device_map="auto")

    def llama_inference(self, input_text):
        encoding = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        generation_output = self.model.generate(
            **encoding, 
            return_dict_in_generate=True, 
            output_logits=True, 
            max_new_tokens=self.args.max_new_tokens, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.decode(generation_output, skip_special_tokens=True)
        return output_text

    def subgraph_summary(self, question, evidence_raw):
        print('EvidenceMap: Summarizing subgraph evdience...\n')
        sys_input = subgraph_sum_prompt['sys_input']
        user_input = subgraph_sum_prompt['user_input']
        triple_text = '\n'.join([str(triple) for triple in evidence_raw['subgraph']])
        user_input = user_input.replace('<triples>', triple_text)
        user_input = user_input.replace('<question>', question)
        output = self.llama_inference(sys_input + '\n' + user_input)
        return output

    def path_summary(self, question, evidence_raw):
        print('EvidenceMap: Summarizing path evdience...', end=' ', flush=True)
        sys_input = path_sum_prompt['sys_input']
        user_input_tmp = path_sum_prompt['user_input']
        user_input_tmp = user_input_tmp.replace('<question>', question)
        sum_list = []
        for path in evidence_raw['path']:
            user_input = user_input_tmp.replace('<path>', str(path))
            output = self.llama_inference(sys_input + '\n' + user_input)
            sum_list.append(output)
            print('.', end=' ', flush=True)
        print('\n')
        return sum_list

    def paper_summary(self, question, evidence_raw):
        print('EvidenceMap: Summarizing paper evdience...', end=' ', flush=True)
        sys_input = paper_sum_prompt['sys_input']
        user_input_tmp = paper_sum_prompt['user_input']
        user_input_tmp = user_input_tmp.replace('<question>', question)
        sum_list = []
        for paper in evidence_raw['paper']:
            user_input = user_input_tmp.replace('<paper>', paper['text'])
            output = self.llama_inference(sys_input + '\n' + user_input)
            sum_list.append(output)
            print('.', end=' ', flush=True)
        print('\n')
        return sum_list

    def concept_summary(self, evidence_raw):
        print('EvidenceMap: Summarizing concept evidence...\n')
        concept_str_list = []
        for name, definition in evidence_raw['concept']:
            concept_str_list.append(name + ': ' + definition)
        return '\n'.join(concept_str_list)

    def evidence_process(self, question, evidence_raw, llm_evidences):
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
        return evidence_list
