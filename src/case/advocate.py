import json
import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import dspy
except ImportError:
    print("Warning: dspy-ai not installed. Please install it to use AdvocateAgent.")
    dspy = None

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    LuceneSearcher = None
    # print("Warning: pyserini not installed or not found. BM25 will be disabled.")

from retrieval.indexing.neural_rag import RAGIndexer
from models.openrouter_client import OpenRouterClient

# Define DSPy Signature
if dspy:
    class FindSupportingEvidence(dspy.Signature):
        """
        You are a legal advocate. Your goal is to find evidence to SUPPORT a given hypothesis.
        You have access to retrieval tools to find relevant legal passages.
        Use the tools to find passages that support the hypothesis.
        """
        hypothesis = dspy.InputField(desc="The legal hypothesis to support.")
        evidence = dspy.OutputField(desc="A list of passage texts that support the hypothesis.")
else:
    FindSupportingEvidence = None


class AdvocateAgent:
    def __init__(
        self,
        llm_client: OpenRouterClient,
        bm25_index_path: str = None,
        faiss_index_dir: str = None,
        faiss_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.llm_client = llm_client
        
        # Configure DSPy
        api_key = os.getenv("OPENROUTER_API_KEY")
        if dspy and api_key:
            # Configure DSPy with OpenRouter
            lm = dspy.LM(
                model='openai/meta-llama/llama-3.3-70b-instruct:free',
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                max_tokens=2048
            )
            dspy.settings.configure(lm=lm)
        
        # Initialize BM25
        self.bm25_searcher = None
        if LuceneSearcher and bm25_index_path and os.path.exists(bm25_index_path):
            try:
                self.bm25_searcher = LuceneSearcher(bm25_index_path)
                print(f"BM25 index loaded from {bm25_index_path}")
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")
        elif bm25_index_path:
             print(f"BM25 index path not found: {bm25_index_path}")

        # Initialize FAISS
        self.rag_indexer = None
        if faiss_index_dir and os.path.exists(faiss_index_dir):
            try:
                self.rag_indexer = RAGIndexer(model_name=faiss_model_name)
                # Assuming metadata.pkl is in the same directory
                metadata_path = os.path.join(faiss_index_dir, "metadata.pkl")
                index_path = os.path.join(faiss_index_dir, "faiss_index.bin")
                if os.path.exists(index_path) and os.path.exists(metadata_path):
                    self.rag_indexer.load_index(index_path, metadata_path)
                    print(f"FAISS index loaded from {faiss_index_dir}")
                else:
                    print(f"FAISS index files not found in {faiss_index_dir}")
            except Exception as e:
                print(f"Failed to load FAISS index: {e}")
        else:
             print(f"FAISS index directory not found: {faiss_index_dir}")

        # Define Tools
        self.tools = []
        if self.bm25_searcher:
            self.tools.append(self._bm25_tool)
        if self.rag_indexer:
            self.tools.append(self._dense_tool)
            
        # Initialize ReAct Agent
        if dspy and FindSupportingEvidence:
            self.react_agent = dspy.ReAct(FindSupportingEvidence, tools=self.tools)
        else:
            self.react_agent = None

    def _bm25_tool(self, query: str) -> str:
        """
        Search for passages using BM25 keyword search.
        Args:
            query: Keywords to search for.
        Returns:
            Top 3 passages found.
        """
        print(f"\n[Tool: BM25] Searching for: '{query}'")
        if not self.bm25_searcher:
            print("[Tool: BM25] Searcher not available.")
            return "BM25 search not available."
        try:
            hits = self.bm25_searcher.search(query, k=3)
            print(f"[Tool: BM25] Found {len(hits)} hits.")
            results = []
            for hit in hits:
                results.append(f"[BM25] {hit.raw[:300]}...")
            return "\n\n".join(results)
        except Exception as e:
            print(f"[Tool: BM25] Error: {e}")
            return f"Error in BM25 search: {e}"

    def _dense_tool(self, query: str) -> str:
        """
        Search for passages using dense vector semantic search.
        Args:
            query: Natural language query.
        Returns:
            Top 3 passages found.
        """
        print(f"\n[Tool: Dense] Searching for: '{query}'")
        if not self.rag_indexer:
            print("[Tool: Dense] Indexer not available.")
            return "Dense search not available."
        try:
            results = self.rag_indexer.search(query, k=3)
            print(f"[Tool: Dense] Found {len(results)} results.")
            formatted = []
            for res in results:
                formatted.append(f"[Dense] {res['passage'][:300]}...")
            return "\n\n".join(formatted)
        except Exception as e:
            print(f"[Tool: Dense] Error: {e}")
            return f"Error in Dense search: {e}"

    def retrieve(self, hypothesis: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve evidence for a given hypothesis using a DSPy ReAct agent.
        """
        if not self.react_agent:
            print("DSPy ReAct agent not initialized.")
            return []
            
        print(f"Advocate retrieving for hypothesis: {hypothesis[:50]}...")
        
        try:
            # Run ReAct agent
            prediction = self.react_agent(hypothesis=hypothesis)
            
            evidence_text = prediction.evidence
            
            # Simple parsing if it's a string
            if isinstance(evidence_text, str):
                passages = [p for p in evidence_text.split('\n') if p.strip()]
            elif isinstance(evidence_text, list):
                passages = evidence_text
            else:
                passages = [str(evidence_text)]
                
            return [{"text": p, "source": "dspy_react", "score": 1.0} for p in passages[:k]]
            
        except Exception as e:
            print(f"Error in ReAct retrieval: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    client = OpenRouterClient()
    
    # Update these paths
    faiss_path = "src/data/processed/faiss_index_all-mini-llm"
    bm25_path = "path/to/bm25/index" 
    
    advocate = AdvocateAgent(client, bm25_index_path=bm25_path, faiss_index_dir=faiss_path)
    
    hypothesis = "A landlord cannot evict a tenant for complaining about code violations."
    evidence = advocate.retrieve(hypothesis, k=3)
    
    print("\nSelected Evidence:")
    for e in evidence:
        print(f"- {e['text'][:100]}... (Source: {e['source']})")
