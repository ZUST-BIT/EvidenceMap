from framework.evimap_soft import EviMapSoft, EvidenceAnalysis, EvidenceSummary
from framework.evimap_hard import EviMapHard
from framework.graph_llm import GraphLLM
from framework.llm_thought import LLMThought
from framework.prompt_soft import PromptSoft
from framework.rag import RAG


framework_selector = {
    "evimap_soft": EviMapSoft,
    "evimap_hard": EviMapHard,
    "graph_llm": GraphLLM,
    "prompt_soft": PromptSoft,
    "llm_thought": LLMThought,
    "rag": RAG
}