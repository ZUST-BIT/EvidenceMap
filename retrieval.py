import pymongo
import sqlite3
from py2neo import Graph
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import argparse
import openai
import json
from itertools import combinations
import pdb

# Step 1: Evidence Retrieval
class EvidenceRetrieval(object):
    def __init__(self, args, source_map=['kg', 'paper', 'concept']):
        print("EvidenceMap: Initializing... \n")
        self.api_key = args.api_key
        self.source_map = source_map
        self.source_env = {item: {} for item in source_map}
        if 'kg' in source_map:
            self.source_env['kg']['graph_db'] = Graph(args.neo4j_url, auth=(args.neo4j_usr, args.neo4j_pwd))
        if 'paper' in source_map:
            self.source_env['paper']['doc_db'] = pymongo.MongoClient(args.mongodb_url)
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.emb_model)
            client = chromadb.PersistentClient(path='./embedding/paper_emb')
            self.source_env['paper']['vec_db'] = client.get_or_create_collection(name='paper_emb', embedding_function=emb_fn)
        if 'concept' in source_map:
            with open(args.concept_path, 'r') as f:
                self.source_env['concept']['term_db'] = json.load(f)
        # if 'table' in source_map:
        #     conn = sqlite3.connect('omics.db')
        #     self.source_env['table']['num_db'] = conn.cursor()

    def openai_inference(self, sys_input, user_input):
        openai.api_key = self.api_key
        res = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_input},
                {"role": "user", "content": user_input},
            ],
        )
        output = res.choices[0].message.content
        return output
    
    def parse_query(self, query, model_type='openai'):
        print("EvidenceMap: Parsing question into named entities...\n")
        sys_prompt = "You are an AI assistant that helps to extract useful elements from a input sentence."
        user_prompt = '''Please extract all entities from the input question following instructions below: 
    1. You should extract entities including Disease, GeneProtein, Drug, Exposure, EffectPhenotype.
    2. You should output the extracted entities as a Python dictionary, with each type as a list, like: \{'Disease': ['disease_name1', 'disease_name2'], 'GeneProtein': ['gene_name1'], 'Drug': ['drug_name1']\}
    3. You should only output the dictionary without other information.
Question: %s
Output: ''' % (query)
        output = None
        try:
            if model_type == 'openai':
                res = self.openai_inference(sys_prompt, user_prompt)
            else:
                raise Exception("Unsupported LLM type.")
            output = eval(res)
        except Exception as e:
            print(e)
        return output
    
    def get_subgraph(self, query_dict, max_num=50):
        print("EvidenceMap: Retrieving subgraph...")
        node_name_id = {} # node_name: node_id
        subgraph_all = [] # list of triples
        for ent_type, ents in query_dict.items():
            for item in ents:
                cypher = "MATCH (n:%s)-[r]-(m) \
                    WHERE toLower(n.name)=toLower(\"%s\") \
                    RETURN r AS triple, rand() AS random \
                    ORDER BY random \
                    LIMIT %d" % (ent_type, item, max_num)
                rels = self.source_env['kg']['graph_db'].run(cypher).data()
                for t in rels:
                    label_s = str(t['triple'].start_node.labels).replace(':', '')
                    name_s = t['triple'].start_node.get('name')
                    label_e = str(t['triple'].end_node.labels).replace(':', '')
                    name_e = t['triple'].end_node.get('name')
                    link = list(t['triple'].types())[0]
                    triple = (label_s + ':' + name_s, link, label_e + ':' + name_e)
                    subgraph_all.append(triple)
                    node_name_id[name_s] = t['triple'].start_node.get('id')
                    node_name_id[name_e] = t['triple'].end_node.get('id')
        print("Found " + str(len(subgraph_all)) + " neighbor nodes.")
        return subgraph_all, node_name_id
    
    def get_path(self, query_dict, max_depth=10, max_num=1):
        print("EvidenceMap: Retrieving paths...")
        node_name_id = {} # node_name: node_id
        path_all = [] # list of tuples
        # inner paths between any two nodes
        ent_list = [] # list of dict
        for ent_type, ents in query_dict.items():
            ent_list.extend([(ent_type, ent) for ent in ents])
        combines = list(combinations(ent_list, 2))
        for pair in combines:
            cypher = "MATCH p=allShortestPaths((n:%s)-[r *..%d]-(m:%s)) \
                WHERE toLower(n.name)=toLower(\"%s\") AND toLower(m.name)=toLower(\"%s\") \
                RETURN p AS path \
                LIMIT %d" % (pair[0][0], max_depth, pair[1][0], pair[0][1], pair[1][1], max_num)
            paths = self.source_env['kg']['graph_db'].run(cypher).data()
            for p in paths:
                triples = []
                for rel in p['path'].relationships:
                    label_s = str(rel.start_node.labels).replace(':', '')
                    name_s = rel.start_node.get('name')
                    label_e = str(rel.end_node.labels).replace(':', '')
                    name_e = rel.end_node.get('name')
                    link = list(rel.types())[0]
                    triple = (label_s + ':' + name_s, link, label_e + ':' + name_e)
                    triples.append(triple)
                    node_name_id[name_s] = rel.start_node.get('id')
                    node_name_id[name_e] = rel.end_node.get('id')
                path_all.append(triples)
        # outer paths of each node
        for ent_type, ents in query_dict.items():
            for item in ents:
                cypher = "MATCH p=(n:%s)-[r *..%d]-() \
                    WHERE toLower(n.name)=toLower(\"%s\") \
                    RETURN p AS path \
                    LIMIT %d" % (ent_type, max_depth, item, max_num)
                paths = self.source_env['kg']['graph_db'].run(cypher).data()
                for p in paths:
                    triples = []
                    for rel in p['path'].relationships:
                        label_s = str(rel.start_node.labels).replace(':', '')
                        name_s = rel.start_node.get('name')
                        label_e = str(rel.end_node.labels).replace(':', '')
                        name_e = rel.end_node.get('name')
                        link = list(rel.types())[0]
                        triple = (label_s + ':' + name_s, link, label_e + ':' + name_e)
                        triples.append(triple)
                        node_name_id[name_s] = rel.start_node.get('id')
                        node_name_id[name_e] = rel.end_node.get('id')
                    if triples not in path_all:
                        path_all.append(triples)
        print("Found " + str(len(path_all)) + " paths.")
        return path_all, node_name_id

    def get_paper_from_db(self, condition, db_name="admin", col_name="pubmed_papers"):
        db = self.source_env['paper']['doc_db'][db_name]
        col = db[col_name]
        papers = col.find(condition)
        paper_list = []
        for item in papers:
            tmp = {}
            tmp['text'] = 'Title: ' + item['title'] + ' Abstract: ' + item['abstract']
            tmp['id'] = item['id']
            tmp['doi'] = item['doi'] if 'doi' in item else ''
            if tmp['doi'] is None:
                tmp['doi'] = ''
            paper_list.append(tmp)
        return paper_list
        
    def index_paper(self, paper_list):
        documents = []
        ids = []
        metadatas = []
        for paper_dict in paper_list:
            ids.append(paper_dict['id'])
            documents.append(paper_dict['text'])
            metadatas.append({'doi': paper_dict['doi']})
        max_limit = 10000
        doc_batches = [documents[i:i+max_limit] for i in range(0, len(documents), max_limit)]
        id_batches = [ids[i:i+max_limit] for i in range(0, len(ids), max_limit)]
        metadata_batches = [metadatas[i:i+max_limit] for i in range(0, len(metadatas), max_limit)]
        for doc_b, id_b, metadata_b in zip(doc_batches, id_batches, metadata_batches):
            self.source_env['paper']['vec_db'].add(documents=doc_b, ids=id_b, metadatas=metadata_b)
            print('Added ' + str(len(doc_b)) + ' document embeddings')

    def similarity_retrieval(self, query, top_k=1):
        print('EvidenceMap: Retrieving paper...')
        res = self.source_env['paper']['vec_db'].query(query_texts=[query], n_results=top_k)
        docs = []
        for i in range(top_k):
            tmp = {}
            tmp['text'] = res['documents'][0][i]
            tmp['id'] = res['ids'][0][i]
            tmp['metadata'] = res['metadatas'][0][i]
            docs.append(tmp)
        return docs

    def kg_evidence_retrieval(self, query_dict, subgraph=True, path=True, path_num=5):
        kg_evidence = {'subgraph': None, 'path': None}
        concept_evidence = {} # node_name: definition
        if subgraph:
            if query_dict:
                kg_evidence['subgraph'], node_name_id1 = self.get_subgraph(query_dict)
                for name, idx in node_name_id1.items():
                    if idx in self.source_env['concept']['term_db']:
                        concept_evidence[name] = self.source_env['concept']['term_db'][idx]
                    else:
                        concept_evidence[name] = name
            else:
                kg_evidence['subgraph'] = []
        if path:
            if query_dict:
                kg_evidence['path'], node_name_id2 = self.get_path(query_dict, max_num=path_num)
                for name, idx in node_name_id2.items():
                    if idx in self.source_env['concept']['term_db']:
                        concept_evidence[name] = self.source_env['concept']['term_db'][idx]
                    else:
                        concept_evidence[name] = name
            else:
                kg_evidence['path'] = []
        return kg_evidence, concept_evidence

    def paper_evidence_retrieval(self, query, paper_num=5):
        paper_evidence = self.similarity_retrieval(query, top_k=paper_num)
        print("Found " + str(len(paper_evidence)) + " papers.")
        return paper_evidence

    def evidence_process(self, query_dict, query, args):
        evidence = {}
        if 'kg' in self.source_map:
            kg_evidence, concept_evidence = self.kg_evidence_retrieval(query_dict, subgraph=True, path=True, path_num=args.path_num)
            evidence['subgraph'] = kg_evidence['subgraph'] # list of triples
            evidence['path'] = kg_evidence['path'] # list of list of triples
            evidence['concept'] = concept_evidence # dict of name: definition
        if 'paper' in self.source_map:
            if self.source_env['paper']['vec_db'].count() == 0:
                paper_list = self.get_paper_from_db({})
                self.index_paper(paper_list)
            paper_evidence = self.paper_evidence_retrieval(query, paper_num=args.paper_num)
            evidence['paper'] = paper_evidence
        return evidence

