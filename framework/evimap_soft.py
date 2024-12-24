# EvidenceMap: progressive and implicit knowledge representation with embeddings, follows retrieve-summary-analysis-reasoning paradigm
import json
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from itertools import product
from retrieval import EvidenceRetrieval
from transformers import AutoTokenizer, AutoModelForCausalLM
from network.gnn import load_gnn_model
from network.mlp import MLP
import pdb

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class EviMapSoft(torch.nn.Module):
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

        self.retriever = EvidenceRetrieval(self.args)
        summarizer = EvidenceSummary(self.args, device)
        analyzer = EvidenceAnalysis(self.args, device)
        self.module_list = torch.nn.ModuleList([summarizer, analyzer])

    def forward(self, questions, answers, parsed_questions, llm_evidences):
        questions_token = self.tokenizer(questions, add_special_tokens=False)
        answers_token = self.tokenizer(answers, add_special_tokens=False)
        print("Retrieving knowledge...")
        evidence_text = self.retriever.evidence_process(parsed_questions, questions, self.args)
        print("Summarizing evidence...")
        evidence_sum_emb, evidence_sum_leap_emb = self.module_list[0](questions, evidence_text, llm_evidences) # batch_num * evidence_num * embedding_dim
        print("Analyzing evidence...")
        global_evi_emb, local_evi_emb = self.module_list[1](evidence_sum_emb) # tensor

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
            inputs_embeds = torch.cat([bos_embeds, global_evi_emb[i], local_evi_emb[i], evidence_sum_leap_emb[i], inputs_embeds], dim=0)
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

    def inference(self, samples, graph_embeds):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            attention_mask=attention_mask,
            # do_sample=True,
            use_cache=True
        )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {'id': samples['id'], 'question': samples['question'], 'pred': pred, 'answer': samples['answer']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param



class EvidenceAnalysis(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.graph_encoder = load_gnn_model[args.gnn_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.device)

        self.projector = MLP(args.gnn_hidden_dim, args.projector_hidden_dim, args.projector_output_dim).to(self.device)

        self.global_node = torch.nn.Parameter(torch.randn(args.sum_output_dim)).to(self.device)

    def encode_graphs(self, graphs):
        n_embeds = []
        for graph in graphs:
            n_embed = self.graph_encoder(graph.x.to(self.device), graph.edge_index.to(self.device))
            n_embeds.append(n_embed)
        graph_embeds = torch.stack(n_embeds)
        return graph_embeds

    def forward(self, samples, agg_mode='mean'):
        node_features_batch, global_embed, local_embeds = None, None, None
        # build graphs
        if agg_mode == 'virtual':
            global_node_batch = self.global_node.repeat(samples.shape[0],1,1)
            node_features_batch = torch.cat([samples, global_node_batch], dim=1)  # batch_size * (evidence_num+1) * input_dim
        elif agg_mode == 'mean':
            node_features_batch = samples
        node_ids = list(range(node_features_batch.shape[1]))
        tmp = list(product(node_ids, node_ids))
        full_connect = [item for item in tmp if item[0] != item[1]]
        edge_index = [[item[0] for item in full_connect], [item[1] for item in full_connect]]
        edge_index_batch = torch.tensor(edge_index, dtype=torch.long).repeat(samples.shape[0], 1, 1)
        graph_inputs = [Data(x=node_features_batch[i], edge_index=edge_index_batch[i]) for i in range(samples.shape[0])]

        # encode graphs
        graph_embeds = self.encode_graphs(graph_inputs)
        graph_embeds = self.projector(graph_embeds)

        if agg_mode == 'virtual':
            global_embed = graph_embeds[:, -1, :].unsqueeze(1)
            local_embeds = graph_embeds[:, :-1, :]
        elif agg_mode == 'mean':
            global_embed = torch.mean(graph_embeds, 1, keepdim=True) 
            local_embeds = graph_embeds
        return global_embed, local_embeds


class EvidenceSummary(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.slm_model)
        self.tokenizer.pad_token_id = 0

        self.model = AutoModelForCausalLM.from_pretrained(
            args.slm_model,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # device_map="auto"
        ).to(self.device)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.word_embedding = self.model.get_input_embeddings()

        self.projector_sum = MLP(args.feature_dim, args.sum_hidden_dim, args.sum_output_dim).to(self.device)
        self.projector_ana = MLP(args.sum_output_dim, args.projector_hidden_dim, args.projector_output_dim).to(self.device)
        # triple_sum = MLP(args)
        # text_sum = MLP(args)
        # list_sum = MLP(args)
        # self.module_list = torch.nn.ModuleList([triple_sum, text_sum, list_sum]) # modality summarizer (triple, text, list)


    def evidence_to_emb(self, question, evidence, llm_evidence): # evidence: {'subgraph': [], 'path': [], 'paper': [], 'concept': []}
        evidence_list = [question]
        evidence_list.append(llm_evidence)

        if 'subgraph' in evidence:
            if evidence['subgraph']: 
                triple_list = []
                for triple in evidence['subgraph']:
                    triple_str = triple[0] + ' has a relationship of ' + triple[1] + ' with ' + triple[2]
                    triple_list.append(triple_str)
                triples = ', '.join(triple_list)
            else:
                triples = "No subgraph."
            evidence_list.append(triples)

        if 'path' in evidence:
            path_list = []
            if len(evidence['path']) == self.args.path_num:
                for path in evidence['path']:
                    tmp = []
                    for triple in path:
                        triple_str = triple[0] + ' has a relationship of ' + triple[1] + ' with ' + triple[2]
                        tmp.append(triple_str)
                    path_list.append('Path: ' + ', '.join(tmp))
                paths = '\n'.join(path_list)
            else:
                paths = "No path."
            evidence_list.append(paths)

        if 'concept' in evidence:
            if evidence['concept']:
                concept_list = []
                for name, definition in evidence['concept'].items():
                    concept_list.append(name + ' has the meaning of ' + definition)
                concpets = '\n'.join(concept_list)
            else:
                concpets = "No concept."
            evidence_list.append(concpets)

        if 'paper' in evidence:
            paper_list = []
            if len(evidence['paper']) == self.args.paper_num:
                for paper in evidence['paper']:
                    paper_list.append(paper['text'])
            else:
                tmp = ["No paper." for i in range(self.args.paper_num - len(evidence['paper']))]
                paper_list.extend(tmp)
            evidence_list.extend(paper_list)

        prompt_template = "{text} This text means in one word:" # https://arxiv.org/pdf/2307.16645
        evidence_list = [prompt_template.format(text=x) for x in evidence_list]

        evidence_input = self.tokenizer(evidence_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        last_hidden_state = self.model(**evidence_input).hidden_states[-1]
        idx_last_non_padding_token = evidence_input.attention_mask.bool().sum(1)-1
        evidence_embs = last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx_last_non_padding_token]

        return evidence_embs

    def forward(self, questions, samples, llm_evis):
        modality_map = {}
        batch_sample_emb = []
        for question, sample, llm_evi in zip(questions, samples, llm_evis):
            tmp = []
            evidence = self.evidence_to_emb(question, sample, llm_evi)
            batch_sample_emb.append(evidence)
            # tmp.append(self.module_list[1](evidence[0].unsqueeze(0))) # question (text)
            # tmp.append(self.module_list[1](evidence[1].unsqueeze(0))) # llm_evidence (text)
            # tmp.append(self.module_list[0](evidence[2].unsqueeze(0))) # subgraph (triple)
            # tmp.append(self.module_list[0](evidence[3].unsqueeze(0))) # path (triple)
            # tmp.append(self.module_list[2](evidence[4].unsqueeze(0))) # concept (list)
            # tmp.append(self.module_list[1](evidence[5: ])) # papers (text) paper_num * embedding_dim
            # evidence_sum = torch.cat(tmp, dim=0)
            # batch_sample_emb.append(evidence_sum)
        batch_evidence_emb = torch.stack(batch_sample_emb) # batch_num * evidence_num * embedding_dim
        batch_evidence_emb = self.projector_sum(batch_evidence_emb)
        batch_evidence_leap_emb = self.projector_ana(batch_evidence_emb)
        return batch_evidence_emb, batch_evidence_leap_emb




