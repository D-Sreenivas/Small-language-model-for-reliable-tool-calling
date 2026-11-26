"""Mock Search Tool simulating search engines or document stores."""
from typing import Any, Dict, List
from backend.tools.base import BaseTool


class MockSearchTool(BaseTool):
    name = "mock_search"
    description = "Simulates web search or enterprise document retrieval for real-time information and documentation."

    KB_ARTICLES = [
        {"title": "Python 3.12 Release Notes", "category": "documentation", "snippet": "Introduces isolated subinterpreters, comprehension inlining, and improved error tracebacks."},
        {"title": "Kubernetes Pod Scaling", "category": "documentation", "snippet": "Horizontal Pod Autoscaling (HPA) automates pod replica scaling based on CPU, memory or custom metrics."},
        {"title": "Solar Energy Breakthroughs 2026", "category": "news", "snippet": "Perovskite-silicon tandem cells surpass 34% lab efficiency records in mass manufacture tests."},
        {"title": "Ultra-lightweight Carbon Bicycles", "category": "products", "snippet": "Aerodynamic carbon fiber road bikes starting at $2,499 with wireless electronic shifting."},
        {"title": "Small Language Models for Edge AI", "category": "general", "snippet": "1-3B parameter SLMs achieve frontier function calling parity through constrained grammar decoding."}
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query phrase."},
                    "category": {
                        "type": "string",
                        "enum": ["general", "news", "documentation", "products"],
                        "default": "general",
                        "description": "Category domain of search."
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3}
                },
                "required": ["query"]
            }
        }

    def execute(self, query: str, category: str = "general", max_results: int = 3, **kwargs) -> Dict[str, Any]:
        q_tokens = query.lower().split()
        cat = category.lower().strip()
        hits = []

        for art in self.KB_ARTICLES:
            text = (art["title"] + " " + art["snippet"]).lower()
            score = sum(1 for tok in q_tokens if tok in text)
            if cat != "general" and art["category"] != cat:
                continue
            if score > 0 or cat == art["category"]:
                hits.append({**art, "relevance_score": score + (1 if art["category"] == cat else 0)})

        hits.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = hits[:int(max_results)]
        
        # Fallback if empty hits
        if not results:
            results = [{
                "title": f"Search results for: {query}",
                "category": category,
                "snippet": f"No exact enterprise articles matched '{query}'. Verified tool invocation successful."
            }]

        return {"query": query, "category": category, "count": len(results), "results": results}

