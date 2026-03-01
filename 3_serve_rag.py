from ray import serve
from fastapi import FastAPI, Request
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# 1. Initialize FastAPI app for routing
app = FastAPI()


# 2. Bind FastAPI to the Ray deployment using Ingress
@serve.deployment(num_replicas=1)
@serve.ingress(app)
class RAGService:
    def __init__(self):
        # Initialize the Non-Parametric Memory (Retriever)
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
        self.vector_db = Chroma(
            persist_directory="./chroma_db", embedding_function=embeddings
        )
        # Increased k=5 for better context coverage
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

        # Initialize the Parametric Memory (Generator)
        self.llm = Ollama(
            model="mistral",
            base_url="http://localhost:11434",
            temperature=0.2,
        )

        self.prompt_template = PromptTemplate.from_template(
            "You are a helpful research assistant. Use the following context from research papers "
            "to answer the question. If the context discusses 'Selective SSMs', 'Selective State Space Models', "
            "or 'simplified architectures', assume these refer to the Mamba architecture.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    # 3. Define the explicit POST route using FastAPI
    @app.post("/rag")
    async def handle_request(self, request: Request):
        payload = await request.json()
        query = payload.get("query", "")
        use_rag = payload.get("use_rag", True)

        if not query:
            return {"error": "No query provided"}

        if use_rag:
            docs = self.retriever.invoke(query)
            context = "\n---\n".join([doc.page_content for doc in docs])
            prompt = self.prompt_template.format(context=context, question=query)
            response = self.llm.invoke(prompt)
            sources = [doc.metadata.get("page") for doc in docs]
        else:
            prompt = f"Question: {query}\nAnswer:"
            response = self.llm.invoke(prompt)
            sources = ["None (Direct LLM Generation)"]

        return {
            "query": query,
            "response": response,
            "sources": sources,
            "used_rag": use_rag,
        }


# Bind the deployment
rag_app = RAGService.bind()
