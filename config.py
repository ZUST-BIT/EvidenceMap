import argparse

def csv_list(string):
    return string.split(',')

def set_argument():
    parser = argparse.ArgumentParser(prog='EvidenceMap')
    parser.add_argument('--project', type=str, default='EvidenceMap')
    parser.add_argument('--framework', type=str, default='evimap_soft')
    parser.add_argument('--source', type=csv_list, default=['kg', 'paper', 'concept'])

    parser.add_argument('--dataset_name', type=str, default='bioasq')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--neo4j_usr', type=str, default='neo4j')
    parser.add_argument('--neo4j_pwd', type=str, default='zongc0725')
    parser.add_argument('--neo4j_url', type=str, default='bolt://localhost:7687')
    parser.add_argument('--mongodb_url', type=str, default='mongodb://localhost:27017/')
    parser.add_argument('--concept_path', type=str, default='./raw_data/umls_concept.json', help='File path of concept')
    parser.add_argument('--path_num', type=int, default=2)
    parser.add_argument('--paper_num', type=int, default=2)

    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--grad_steps", type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    parser.add_argument('--api_key', type=str, default='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')
    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--llm_type', type=str, default='local')
    parser.add_argument('--llm_model', type=str, default='/Users/zongchang/Desktop/models/llama3.2-1b-instruct') # replace with your own path
    parser.add_argument('--slm_model', type=str, default='/Users/zongchang/Desktop/models/bert-base-uncased') # replace with your own path
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument('--max_txt_len', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    
    parser.add_argument('--feature_dim', type=int, default=768) # depends on SLM's embedding dim
    parser.add_argument('--sum_hidden_dim', type=int, default=1024)
    parser.add_argument('--sum_output_dim', type=int, default=1024)
    parser.add_argument("--gnn_name", type=str, default='gt')
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)
    parser.add_argument("--projector_hidden_dim", type=int, default=1024)
    parser.add_argument("--projector_output_dim", type=int, default=2048) # depends on LLM's embedding dim
    args = parser.parse_args()
    return args