# LLM question parsing prompt
question_parsing_prompt = {'sys_prompt': "You are an AI assistant that helps extract useful elements from a input sentence.",
    'user_prompt': '''Please extract all entities from the input question following instructions below: 
1. You should extract entities including Disease, GeneProtein, Drug, Exposure, EffectPhenotype.
2. You should output the extracted entities as a Python dictionary, with each type as a list, like: {'Disease': ['disease_name1', 'disease_name2'], 'GeneProtein': ['gene_name1'], 'Drug': ['drug_name1']}
3. You should only output the dictionary without other information.

Question: <question>
Output: '''}

# LLM evidence summary prompt
llm_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst discover evidence that supports the question.''',
    'user_input': '''Goal: Provide a concise summary that supports answering the question with your own knowledge. The summary should be evidence including key insights that can explain your answer to the question.

Question: <question>
Summary: '''}

# subgraph summary prompt
subgraph_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst discover evidence that supports the question.''',
    'user_input': '''Goal: Write a concise summary of a subgraph, given a list of triplets that belong to the subgraph. The summary will be used to inform analyst about key information associated with the question. The content of this summary includes some key insights about the relationships in the subgraph.
Note: Each triplet is represented as a tuple: ('head entity', 'relationship', 'tail entity'), in a separate line.

Question:
<question>

Triplets:
<triples>

Summary: '''}

# path summary prompt
path_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst discover evidence that supports the question.''',
    'user_input': '''Goal: Write a concise summary given a path from a graph. A path is composed of multiple triplets. The summary will be used to inform analyst about key information associated with the question. The content of this summary includes some key insights about the multi-hop relationships in the path.
Note: A path is represented as a list of tuples in a line. Each tuple is a triplets as ('head entity', 'relationship', 'tail entity').

Question:
<question>

Path:
<path>

Summary: '''}

# paper summary prompt
paper_sum_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst discover evidence that supports the question.''',
    'user_input': '''Goal: Write a concise summary given the title and abstract of a paper. The summary will be used to inform analyst about key information associated with the question. The content of the summary should includes some key insights about the findings in the paper.

Question:
<question>

Title and Abstract:
<paper>

Summary: '''}

evidence_analysis_prompt={
    'sys_input': '''You are an AI assistant that helps a human analyst discover logical relationships between any two pieces of evidence for answering a question.''',
    'user_input': '''Goal: Analysis logical relationships between any two pieces of evidence. Evidence is given as a dictionary where keys are types and values are evidence items.

Evidence dictionary: 
<evidence_dict>

Question: 
<question>

Output: '''
}


# integrated answering prompt
integrated_answering_prompt={
    'sys_input': '''You are an AI assistant that helps a human analyst answer a question with the give evidence and the logical relationships between any two pieces of evidence.''',
    'user_input': '''Goal: Answer the question with the given evidence and the relationships betwen any two pieces of evidence. Evidence is given as a dictionary where keys are types and values are evidence items.
    
Evidence dictionary: 
<evidence_dict>

Evidence relationship analysis: 
<evidence_analysis>

Question: <question>

Answer: '''}

# RAG answering prompt
integrated_answering_prompt={
    'sys_input': '''You are an AI assistant that helps a human analyst answer a question with the give evidence and the logical analysis among those evidences.''',
    'user_input': '''Question: <question>
    
Evidence: 
<evidence>

Answer: '''}


# RAG with CoT answering prompt
initial_thought_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst answer a question by giving several steps of thinking process.''',
    'user_input': '''Question: <question>

Thinking steps: '''
}

optimize_thought_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst answer a question by optimizing a thinking process with the given evidence.''',
    'user_input': '''Question: <question>

Thinking process:
<thought>

Evidence:
<evidence>

Optimized thinking process:'''
}

answering_with_thinking_prompt = {
    'sys_input': '''You are an AI assistant that helps a human analyst give a short answer to a question with the provided thinking process and evidence.''',
    'user_input': '''Question: <question>

Thinking process:
<thought>

Evidence:
<evidence>

Short answer:'''
}

