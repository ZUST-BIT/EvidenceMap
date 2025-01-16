from framework.evimap_emb import EviMapEmb, EviMapBuilder
from framework.evimap_soft import EviMapSoft, EvidenceAnalysis, EvidenceSummary
from framework.evimap_hard import EviMapHard
from framework.graph_llm import GraphLLM
from framework.llm_thought import LLMThought
from framework.prompt_soft import PromptSoft
from framework.rag import RAG
from framework.rag_cot import RAGCoT


framework_selector = {
    "evimap_emb": EviMapEmb,
    "evimap_soft": EviMapSoft,
    "evimap_hard": EviMapHard,
    "graph_llm": GraphLLM,
    "prompt_soft": PromptSoft,
    "llm_thought": LLMThought,
    "rag": RAG,
    "rag_cot": RAGCoT
}
