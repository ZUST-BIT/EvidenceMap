# LLM question parsing prompt
question_parsing_prompt = {'sys_prompt': "You are an AI assistant that helps to extract useful elements from a input sentence.",
    'user_prompt': '''Please extract all entities from the input question following instructions below: 
1. You should extract entities including Disease, GeneProtein, Drug, Exposure, EffectPhenotype.
2. You should output the extracted entities as a Python dictionary, with each type as a list, like: {'Disease': ['disease_name1', 'disease_name2'], 'GeneProtein': ['gene_name1'], 'Drug': ['drug_name1']}
3. You should only output the dictionary without other information.

Question: <question>
Output: '''}

# LLM evidence summary prompt
llm_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst to discover evidence that supports the question.''',
    'user_input': '''Goal: Provide a concise summary that supports answering the question with your own knowledge. The summary should be an evidence including key insights that can explain your answer to the question.

Question: <question>
Summary: '''}

# subgraph summary prompt
subgraph_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst to discover evidence that supports the question.''',
    'user_input': '''Goal: Write a concise summary of a subgraph, given a list of triplets that belong to the subgraph. The summary will be used to inform analyst about key information associated with the question. The content of this summary includes some key insights about the relationships in the subgraph.
Note: Each triplet is represented as a tuple: ('head entity', 'relationship', 'tail entity'), in a separate line.

Question:
<question>

Triplets:
<triples>

Summary: '''}

# path summary prompt
path_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst to discover evidence that supports the question.''',
    'user_input': '''Goal: Write a concise summary given a path from a graph. A path is composed of multiple triplets. The summary will be used to inform analyst about key information associated with the question. The content of this summary includes some key insights about the multi-hop relationships in the path.
Note: A path is represented as a list of tuples in a line. Each tuple is a triplets as ('head entity', 'relationship', 'tail entity').

Question:
<question>

Path:
<path>

Summary: '''}

# paper summary prompt
paper_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst to discover evidence that supports the question.''',
    'user_input': '''Goal: Write a concise summary given the title and abstract of a paper. The summary will be used to inform analyst about key information associated with the question. The content of the summary should includes some key insights about the findings in the paper.

Question:
<question>

Title and Abstract:
<paper>

Summary: '''}

# evidence support prompt
evidence_useful_prompt={
    'sys_input': '''You are an AI assistant that helps a human analyst to discover whether a piece of evidence is helpful to answer a question.''',
    'user_input': '''Goal: Decide whether an evidence in a dictionary is helpful to the question according to the following requirements:
1. Label each evidence in the dictionary with 'yes' or 'no', where 'yes' means the evidence is helpful to answer the question, 'no' means the evidence is not helpful.
2. Output the label of each piece of evidence as a list, including the evidence ID and the label, such as: ["evidence1", "yes"].
3. Output each list in a separate line, without any additional information, such as: ["evidence1", "yes"]\\n["evidence2", "no"]...

Evidence dictionary: 
<evidence_dict>

Question: 
<question>

Output:
'''}

# evidence logic prompt
evidence_logic_prompt={
    'sys_input': '''You are an AI assistant that helps a human analyst to discover logical relationships between any two pieces of evidence for answering a question.''',
    'user_input': '''Goal: Provide logical relationships between any two pieces of evidence from an evidence dictionary, according to the following requirements: 
1. Please determine whether there is a logical relationship between any two pieces of evidence. You can refer to the question related to the evidence.
2. A logical relationship is in one of: consistency, contradiction, condition, causality, etc.
3. If there is a logical relationship between two pieces of evidence, output the relationship as a triplet list including evidence IDs and their relationship, such as: ["evidence1", "consistency", "evidence2"].
4. Output each triplet on a separate line, without any additional information, such as: ["evidence1", "contradiction", "evidence2"]\\n["evidence2", "causality", "evidence4"]...

Evidence dictionary: 
<evidence_dict>

Question: 
<question>

Output: 
'''}

# integrated answering prompt with tokens
integrated_answering_prompt_token={
    'sys_input': '''You are an AI assistant that helps a human analyst to answer a question using the provided evidence and the relationships among the pieces of evidence''',
    'user_input': '''Goal: Provide a concise answer to the question according to a dictionary of evidence that supports the question and the logical relationships between any two pieces of evidence. You should reasoning over the logical relationships to obtain the final answer.
Note: Evidence are given as a dictionary with IDs as keys and content as values. Logical relationships between evidence are given as triplets in a list format per line such as: ['evidence1', "against", "evidence2"]\\n["evidence2", "compliment", "evidence4"]... You are going to answer a <question_type> question.

Evidence dictionary: 
<evidence_dict>

Evidence relationships:
<evidence_rel>

Question: <question>

Answer: '''}

# integrated answering prompt
integrated_answering_prompt_emb={
    'sys_input': '''You are an AI assistant that helps a human analyst to answer a question''',
    'user_input': '''Question: <question>
    
Local Evidence:
<local_evi_emb>

Global Evidence: 
<global_evi_emb>

Answer: '''}
