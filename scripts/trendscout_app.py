"""
TrendScout AI - Complete Application (v2)
RAG + Knowledge Graph + Web Interface

Updated for v2 schema with:
- Post titles
- AI Model providers (is_used vs is_released)
- Partner categories

Features:
1. ChromaDB for semantic search (RAG) - uses raw post text
2. Neo4j for structured queries (Knowledge Graph) - uses extracted entities
3. Hybrid retrieval combining both
4. Simple Flask web interface

Requirements:
pip install flask langchain langchain-openai langchain-community chromadb neo4j python-dotenv
"""

import os
import json
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Neo4j
from neo4j import GraphDatabase

load_dotenv()

app = Flask(__name__)

# =============================================================================
# Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
CHROMA_PERSIST_DIR = "./chroma_db"

# =============================================================================
# RAG Component (ChromaDB) - Uses raw post text
# =============================================================================

class RAGRetriever:
    """Vector-based semantic search using ChromaDB"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vectorstore = None
        # Increased chunk size to avoid splitting short posts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased from 500
            chunk_overlap=100  # Increased from 50
        )
    
    def load_documents(self, json_file: str):
        """Load documents from raw posts JSON file into ChromaDB"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Handle different structures
        if isinstance(data, dict):
            # Combined posts file structure
            if 'posts' in data:
                posts = data['posts']
            elif 'extracted_data' in data:
                posts = data['extracted_data']
            else:
                posts = [data]
        else:
            posts = data
        
        for item in posts:
            if isinstance(item, dict):
                # Build content from post fields
                content_parts = []
                
                # Add company info
                if 'company' in item:
                    content_parts.append(f"Company: {item['company']}")
                
                # Add date
                if 'date' in item:
                    content_parts.append(f"Date: {item['date']}")
                
                # Add title if available (v2)
                if 'title' in item:
                    content_parts.append(f"Title: {item['title']}")
                
                # Add summary if available (v2)
                if 'summary' in item:
                    content_parts.append(f"Summary: {item['summary']}")
                
                # Add original content/text if available
                if 'content' in item:
                    content_parts.append(f"Content: {item['content']}")
                elif 'text' in item:
                    content_parts.append(f"Content: {item['text']}")
                
                # Add post type
                if 'post_type' in item:
                    content_parts.append(f"Type: {item['post_type']}")
                
                content = "\n".join(content_parts)
                
                # Metadata for filtering
                metadata = {
                    'company': item.get('company', ''),
                    'post_id': item.get('post_id', item.get('global_id', '')),
                    'date': item.get('date', ''),
                    'post_type': item.get('post_type', '')
                }
                
                if content.strip():
                    documents.append(Document(page_content=content, metadata=metadata))
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        print(f"✅ Loaded {len(split_docs)} chunks into ChromaDB")
        return len(split_docs)
    
    def load_existing(self):
        """Load existing ChromaDB"""
        if os.path.exists(CHROMA_PERSIST_DIR):
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings
            )
            print("✅ Loaded existing ChromaDB")
            return True
        return False
    
    def _estimate_sources_needed(self, query: str) -> int:
        """Estimate how many sources are needed based on query type"""
        query_lower = query.lower()
        
        # Simple factual questions → 1-2 sources
        if any(w in query_lower for w in ['what is', 'who is', 'when did', 'which company', 'what platform', 'what product']):
            return 2
        
        # Specific single-item questions → 1 source
        if any(w in query_lower for w in ['specific', 'exactly', 'precisely']):
            return 1
        
        # Comparison questions → 3-4 sources
        if any(w in query_lower for w in ['compare', 'difference', 'vs', 'versus', 'between', 'better']):
            return 4
        
        # List/overview questions → 4-5 sources
        if any(w in query_lower for w in ['list', 'all', 'every', 'overview', 'summarize', 'how many', 'what are']):
            return 5
        
        # Company-specific questions → 2-3 sources
        if any(w in query_lower for w in ['perplexity', 'openai', 'anthropic', 'mistral', 'deepseek']):
            return 3
        
        # Default
        return 3

    def search(self, query: str, k: int = None, min_score: float = 1.0) -> str:
        """Search for relevant documents with smart source selection
        
        Args:
            query: Search query
            k: Number of results (if None, auto-estimate based on query)
            min_score: Maximum distance threshold (lower = more similar, typically 0.0-2.0)
        """
        if not self.vectorstore:
            return "RAG not initialized. Please load documents first."
        
        # Auto-estimate k if not provided
        if k is None:
            k = self._estimate_sources_needed(query)
        
        # Fetch more results to allow for deduplication and filtering
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*3)
        
        if not results_with_scores:
            return "No relevant documents found."
        
        # Deduplicate by post_id or content
        seen_posts = set()
        unique_results = []
        
        for doc, score in results_with_scores:
            # Use post_id if available, otherwise use content hash
            post_id = doc.metadata.get('post_id', '')
            if not post_id:
                # Fallback to content-based dedup (first 200 chars)
                post_id = doc.page_content[:200]
            
            if post_id not in seen_posts:
                # Only keep results below score threshold (lower score = more relevant)
                if score <= min_score:
                    seen_posts.add(post_id)
                    unique_results.append((doc, score))
                    
                    # Stop once we have enough unique results
                    if len(unique_results) >= k:
                        break
        
        # If no results passed threshold, keep at least the best one
        if not unique_results and results_with_scores:
            best_doc, best_score = results_with_scores[0]
            unique_results = [(best_doc, best_score)]
        
        # Format results
        context_parts = []
        for i, (doc, score) in enumerate(unique_results, 1):
            # Include metadata for context
            company = doc.metadata.get('company', 'Unknown')
            date = doc.metadata.get('date', '')
            
            # Format header with relevance indicator
            relevance = "🟢" if score < 0.5 else "🟡" if score < 0.8 else "🟠"
            header = f"{relevance} [{company}]" + (f" ({date})" if date else "")
            context_parts.append(f"[Doc {i}] {header}:\n{doc.page_content}")
        
        # Add source count info
        source_info = f"(Showing {len(unique_results)} relevant source{'s' if len(unique_results) != 1 else ''} based on query type)"
        
        return source_info + "\n\n" + "\n\n".join(context_parts)


# =============================================================================
# Knowledge Graph Component (Neo4j) - v2 Schema
# =============================================================================

class KnowledgeGraphRetriever:
    """Structured queries using Neo4j Knowledge Graph (v2 schema)"""
    
    def __init__(self):
        self.driver = None
        self.connected = False
    
    def connect(self):
        """Connect to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            self.connected = True
            print("✅ Connected to Neo4j")
            return True
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
    
    def query(self, user_query: str) -> str:
        """Query the knowledge graph based on user intent"""
        if not self.connected:
            return "Knowledge Graph not connected."
        
        query_lower = user_query.lower()
        cypher = self._get_cypher(query_lower)
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = [dict(record) for record in result]
            
            return self._format_results(records)
        except Exception as e:
            return f"KG Query Error: {str(e)}"
    
    def _get_cypher(self, query: str) -> str:
        """Map user intent to Cypher query - Updated for v2 schema"""
        
        # Products
        if any(w in query for w in ['product', 'develop', 'build', 'offer', 'create', 'tool', 'app']):
            # Check for specific company
            if 'perplexity' in query:
                return """
                MATCH (c:Company {id: 'perplexity-ai'})-[:DEVELOPS]->(p:Product)
                RETURN c.name as company, p.name as product, p.type as type, p.description as description
                """
            elif 'openai' in query:
                return """
                MATCH (c:Company {id: 'openai'})-[:DEVELOPS]->(p:Product)
                RETURN c.name as company, p.name as product, p.type as type, p.description as description
                """
            elif 'anthropic' in query or 'claude' in query:
                return """
                MATCH (c:Company {id: 'anthropicresearch'})-[:DEVELOPS]->(p:Product)
                RETURN c.name as company, p.name as product, p.type as type, p.description as description
                """
            elif 'mistral' in query:
                return """
                MATCH (c:Company {id: 'mistralai'})-[:DEVELOPS]->(p:Product)
                RETURN c.name as company, p.name as product, p.type as type, p.description as description
                """
            elif 'deepseek' in query:
                return """
                MATCH (c:Company {id: 'deepseek-ai'})-[:DEVELOPS]->(p:Product)
                RETURN c.name as company, p.name as product, p.type as type, p.description as description
                """
            return """
            MATCH (c:Company)-[:DEVELOPS]->(p:Product)
            RETURN c.name as company, p.name as product, p.type as type, p.description as description
            ORDER BY c.name
            """
        
        # AI Models - Updated for v2 with provider and is_used/is_released
        if any(w in query for w in ['model', 'gpt', 'llm', 'ai model', 'claude', 'deepseek', 'mistral', 'sora']):
            # Who uses which models
            if 'use' in query or 'using' in query:
                return """
                MATCH (c:Company)-[:USES_MODEL]->(m:AIModel)
                RETURN c.name as company, collect(m.name) as models_used, collect(m.provider) as providers
                """
            # Who released which models
            if 'release' in query or 'created' in query or 'developed' in query:
                return """
                MATCH (c:Company)-[:RELEASED_MODEL]->(m:AIModel)
                RETURN c.name as company, collect(m.name) as models_released
                """
            # Models by provider
            if 'provider' in query or 'openai' in query or 'anthropic' in query:
                return """
                MATCH (m:AIModel)
                RETURN m.provider as provider, collect(m.name) as models
                ORDER BY m.provider
                """
            # Default: all models
            return """
            MATCH (m:AIModel)
            OPTIONAL MATCH (c:Company)-[:RELEASED_MODEL]->(m)
            RETURN m.name as model, m.provider as provider, c.name as released_by
            ORDER BY m.provider
            """
        
        # Partnerships - Updated for v2 with categories
        if any(w in query for w in ['partner', 'partnership', 'collaborate', 'work with']):
            # By category
            if 'government' in query:
                return """
                MATCH (c:Company)-[:PARTNERS_WITH]->(p:Partner {category: 'government'})
                RETURN c.name as company, p.name as partner, p.details as details
                """
            if 'tech' in query:
                return """
                MATCH (c:Company)-[:PARTNERS_WITH]->(p:Partner)
                WHERE p.category IN ['tech_giant', 'tech_company']
                RETURN c.name as company, p.name as partner, p.category as category, p.details as details
                """
            if 'enterprise' in query or 'business' in query:
                return """
                MATCH (c:Company)-[:PARTNERS_WITH]->(p:Partner {category: 'enterprise'})
                RETURN c.name as company, p.name as partner, p.details as details
                """
            if 'healthcare' in query or 'health' in query:
                return """
                MATCH (c:Company)-[:PARTNERS_WITH]->(p:Partner {category: 'healthcare'})
                RETURN c.name as company, p.name as partner, p.details as details
                """
            if 'financial' in query or 'finance' in query or 'bank' in query:
                return """
                MATCH (c:Company)-[:PARTNERS_WITH]->(p:Partner {category: 'financial'})
                RETURN c.name as company, p.name as partner, p.details as details
                """
            # All partnerships with categories
            return """
            MATCH (c:Company)-[:PARTNERS_WITH]->(p:Partner)
            RETURN c.name as company, p.name as partner, p.category as category, p.type as type
            ORDER BY p.category, c.name
            """
        
        # Posts - New for v2 with titles
        if any(w in query for w in ['post', 'announcement', 'news', 'recent', 'latest', 'update']):
            if 'perplexity' in query:
                return """
                MATCH (c:Company {id: 'perplexity-ai'})-[:PUBLISHED]->(p:Post)
                RETURN p.title as title, p.date as date, p.post_type as type, p.summary as summary
                ORDER BY p.global_id DESC LIMIT 10
                """
            if 'anthropic' in query:
                return """
                MATCH (c:Company {id: 'anthropicresearch'})-[:PUBLISHED]->(p:Post)
                RETURN p.title as title, p.date as date, p.post_type as type, p.summary as summary
                ORDER BY p.global_id DESC LIMIT 10
                """
            if 'openai' in query:
                return """
                MATCH (c:Company {id: 'openai'})-[:PUBLISHED]->(p:Post)
                RETURN p.title as title, p.date as date, p.post_type as type, p.summary as summary
                ORDER BY p.global_id DESC LIMIT 10
                """
            return """
            MATCH (c:Company)-[:PUBLISHED]->(p:Post)
            RETURN c.name as company, p.title as title, p.date as date, p.post_type as type
            ORDER BY p.global_id DESC LIMIT 15
            """
        
        # Features
        if any(w in query for w in ['feature', 'capability', 'function']):
            return """
            MATCH (p:Post)-[:HAS_FEATURE]->(f:Feature)
            RETURN p.title as post, f.name as feature, f.description as description, f.availability as availability
            LIMIT 20
            """
        
        # Topics/Tags
        if any(w in query for w in ['topic', 'tag', 'about', 'related to']):
            topic = None
            topics = ['coding', 'enterprise', 'government', 'security', 'agents', 'models', 
                     'partnerships', 'funding', 'mobile', 'research', 'education', 'healthcare']
            for t in topics:
                if t in query:
                    topic = t
                    break
            
            if topic:
                return f"""
                MATCH (p:Post)-[:TAGGED]->(t:Topic {{name: '{topic}'}})
                RETURN p.title as post, p.company as company, p.date as date
                LIMIT 10
                """
            return """
            MATCH (t:Topic)<-[:TAGGED]-(p:Post)
            RETURN t.name as topic, count(p) as post_count
            ORDER BY post_count DESC LIMIT 15
            """
        
        # Company-specific overview
        if 'perplexity' in query:
            return """
            MATCH (c:Company {id: 'perplexity-ai'})
            OPTIONAL MATCH (c)-[:DEVELOPS]->(pr:Product)
            OPTIONAL MATCH (c)-[:PARTNERS_WITH]->(pa:Partner)
            OPTIONAL MATCH (c)-[:USES_MODEL]->(m:AIModel)
            RETURN c.name as company, 
                   collect(DISTINCT pr.name) as products,
                   collect(DISTINCT pa.name) as partners,
                   collect(DISTINCT m.name) as models_used
            """
        
        if 'anthropic' in query:
            return """
            MATCH (c:Company {id: 'anthropicresearch'})
            OPTIONAL MATCH (c)-[:DEVELOPS]->(pr:Product)
            OPTIONAL MATCH (c)-[:PARTNERS_WITH]->(pa:Partner)
            OPTIONAL MATCH (c)-[:RELEASED_MODEL]->(m:AIModel)
            RETURN c.name as company, 
                   collect(DISTINCT pr.name) as products,
                   collect(DISTINCT pa.name) as partners,
                   collect(DISTINCT m.name) as models_released
            """
        
        if 'openai' in query:
            return """
            MATCH (c:Company {id: 'openai'})
            OPTIONAL MATCH (c)-[:DEVELOPS]->(pr:Product)
            OPTIONAL MATCH (c)-[:RELEASED_MODEL]->(m:AIModel)
            RETURN c.name as company, 
                   collect(DISTINCT pr.name) as products,
                   collect(DISTINCT m.name) as models_released
            """
        
        if 'mistral' in query:
            return """
            MATCH (c:Company {id: 'mistralai'})
            OPTIONAL MATCH (c)-[:DEVELOPS]->(pr:Product)
            OPTIONAL MATCH (c)-[:PARTNERS_WITH]->(pa:Partner)
            OPTIONAL MATCH (c)-[:RELEASED_MODEL]->(m:AIModel)
            RETURN c.name as company, 
                   collect(DISTINCT pr.name) as products,
                   collect(DISTINCT pa.name) as partners,
                   collect(DISTINCT m.name) as models_released
            """
        
        if 'deepseek' in query:
            return """
            MATCH (c:Company {id: 'deepseek-ai'})
            OPTIONAL MATCH (c)-[:DEVELOPS]->(pr:Product)
            OPTIONAL MATCH (c)-[:RELEASED_MODEL]->(m:AIModel)
            RETURN c.name as company, 
                   collect(DISTINCT pr.name) as products,
                   collect(DISTINCT m.name) as models_released
            """
        
        # Default overview - all companies
        return """
        MATCH (c:Company)
        OPTIONAL MATCH (c)-[:DEVELOPS]->(p:Product)
        OPTIONAL MATCH (c)-[:PARTNERS_WITH]->(pa:Partner)
        RETURN c.name as company, 
               count(DISTINCT p) as product_count,
               count(DISTINCT pa) as partner_count
        ORDER BY c.name
        """
    
    def _format_results(self, records: list) -> str:
        """Format Neo4j results as context"""
        if not records:
            return "No results found in Knowledge Graph."
        
        formatted = []
        for record in records:
            parts = []
            for key, value in record.items():
                if value is not None:
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value if v)
                    if value:
                        parts.append(f"{key}: {value}")
            if parts:
                formatted.append(" | ".join(parts))
        
        return "\n".join(formatted)


# =============================================================================
# Hybrid TrendScout AI
# =============================================================================

class TrendScoutAI:
    """Main AI system combining RAG and Knowledge Graph"""
    
    def __init__(self):
        self.rag = RAGRetriever()
        self.kg = KnowledgeGraphRetriever()
        self.llm = None
        self.chain = None
    
    def initialize(self, load_new_docs: str = None):
        """Initialize all components"""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Load RAG
        if load_new_docs:
            self.rag.load_documents(load_new_docs)
        else:
            self.rag.load_existing()
        
        # Connect KG
        self.kg.connect()
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["kg_context", "rag_context", "question"],
            template="""You are TrendScout AI, an expert assistant for AI startup market intelligence.

Use the following information to answer the question. Prioritize Knowledge Graph facts for structured data (companies, products, partnerships, models), and use RAG context for additional details and explanations.

=== KNOWLEDGE GRAPH FACTS ===
{kg_context}

=== RAG CONTEXT ===
{rag_context}

=== QUESTION ===
{question}

Provide a clear, comprehensive answer. If you cite specific facts, mention whether they come from the structured data (KG) or document search (RAG).
If you don't have enough information, say so honestly.

Answer:"""
        )
        
        # Create chain using LCEL
        self.chain = prompt_template | self.llm | StrOutputParser()
        
        print("✅ TrendScout AI initialized")
    
    def ask(self, question: str, use_kg: bool = True, use_rag: bool = True) -> dict:
        """Ask a question using hybrid retrieval"""
        kg_context = ""
        rag_context = ""
        
        if use_kg and self.kg.connected:
            kg_context = self.kg.query(question)
        
        if use_rag and self.rag.vectorstore:
            rag_context = self.rag.search(question)
        
        # Generate response
        response = self.chain.invoke({
            "kg_context": kg_context or "No Knowledge Graph data available.",
            "rag_context": rag_context or "No RAG context available.",
            "question": question
        })
        
        return {
            "answer": response,
            "kg_context": kg_context,
            "rag_context": rag_context,
            "sources": {
                "kg_used": use_kg and bool(kg_context),
                "rag_used": use_rag and bool(rag_context)
            }
        }
    
    def close(self):
        """Clean up resources"""
        self.kg.close()


# =============================================================================
# Flask Web Interface
# =============================================================================

# Global instance
trendscout = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrendScout AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e7;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }
        header {
            text-align: center;
            margin-bottom: 2rem;
        }
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle { color: #9ca3af; font-size: 1.1rem; }
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .status-dot.green { background: #22c55e; }
        .status-dot.yellow { background: #eab308; }
        .status-dot.red { background: #ef4444; }
        .chat-container {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
        }
        .message {
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 12px;
        }
        .user-message {
            background: rgba(96, 165, 250, 0.2);
            margin-left: 2rem;
        }
        .ai-message {
            background: rgba(167, 139, 250, 0.2);
            margin-right: 2rem;
        }
        .message-label {
            font-size: 0.8rem;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }
        .sources {
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }
        .source-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            margin-right: 0.5rem;
            font-size: 0.75rem;
        }
        .source-kg { background: rgba(34, 197, 94, 0.3); color: #86efac; }
        .source-rag { background: rgba(59, 130, 246, 0.3); color: #93c5fd; }
        .input-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        input[type="text"] {
            flex: 1;
            padding: 1rem;
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            background: rgba(255,255,255,0.05);
            color: #e4e4e7;
            font-size: 1rem;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #60a5fa;
        }
        button {
            padding: 1rem 2rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.02); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .toggles {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 1rem;
        }
        .toggle {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .toggle input { width: 18px; height: 18px; }
        .context-section {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            font-size: 0.85rem;
            max-height: 150px;
            overflow-y: auto;
        }
        .context-section h4 {
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }
        .context-section pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #d1d5db;
        }
        .hidden { display: none; }
        #loading {
            text-align: center;
            padding: 2rem;
            color: #9ca3af;
        }
        .show-sources-btn {
            background: transparent;
            border: 1px solid rgba(255,255,255,0.2);
            padding: 0.3rem 0.8rem;
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 TrendScout AI</h1>
            <p class="subtitle">AI Startup Market Intelligence - RAG + Knowledge Graph</p>
        </header>
        
        <div class="status-bar">
            <div class="status">
                <div class="status-dot {{ 'green' if kg_status else 'red' }}"></div>
                <span>KG {{ 'Connected' if kg_status else 'Disconnected' }}</span>
            </div>
            <div class="status">
                <div class="status-dot {{ 'green' if rag_status else 'yellow' }}"></div>
                <span>RAG {{ 'Ready' if rag_status else 'Not Loaded' }}</span>
            </div>
        </div>
        
        <div class="chat-container" id="chat">
            <div class="message ai-message">
                <div class="message-label">TrendScout AI</div>
                <p>Hello! I'm TrendScout AI, your AI startup market intelligence assistant. I can answer questions about AI companies like Perplexity, OpenAI, Anthropic, Mistral, and DeepSeek.</p>
                <p style="margin-top: 0.5rem;">Try asking about:</p>
                <ul style="margin-left: 1.5rem; margin-top: 0.5rem;">
                    <li>Products: "What products does Anthropic offer?"</li>
                    <li>Models: "Which AI models does Perplexity use?"</li>
                    <li>Partnerships: "Who are Mistral's government partners?"</li>
                    <li>Posts: "What are OpenAI's recent announcements?"</li>
                </ul>
            </div>
        </div>
        
        <div class="toggles">
            <label class="toggle">
                <input type="checkbox" id="useKG" checked>
                <span>🔗 Knowledge Graph</span>
            </label>
            <label class="toggle">
                <input type="checkbox" id="useRAG" checked>
                <span>📚 RAG Search</span>
            </label>
            <label class="toggle">
                <input type="checkbox" id="showSources">
                <span>👁️ Show Sources</span>
            </label>
        </div>
        
        <div class="input-container">
            <input type="text" id="question" placeholder="Ask about AI startups..." onkeypress="if(event.key==='Enter')askQuestion()">
            <button onclick="askQuestion()" id="askBtn">Ask</button>
        </div>
    </div>
    
    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) return;
            
            const useKG = document.getElementById('useKG').checked;
            const useRAG = document.getElementById('useRAG').checked;
            const showSources = document.getElementById('showSources').checked;
            const chat = document.getElementById('chat');
            const btn = document.getElementById('askBtn');
            
            // Add user message
            chat.innerHTML += `
                <div class="message user-message">
                    <div class="message-label">You</div>
                    <p>${question}</p>
                </div>
            `;
            
            // Show loading
            btn.disabled = true;
            btn.textContent = 'Thinking...';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, use_kg: useKG, use_rag: useRAG })
                });
                
                const data = await response.json();
                
                // Format answer - convert markdown-like formatting
                let formattedAnswer = data.answer
                    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                    .replace(/\\n/g, '<br>');
                
                // Build sources badges
                let sourcesBadges = '';
                if (data.sources.kg_used) sourcesBadges += '<span class="source-badge source-kg">🔗 KG</span>';
                if (data.sources.rag_used) sourcesBadges += '<span class="source-badge source-rag">📚 RAG</span>';
                
                // Build context sections
                let contextHtml = '';
                if (showSources) {
                    if (data.kg_context) {
                        contextHtml += `
                            <div class="context-section">
                                <h4>🔗 Knowledge Graph Context</h4>
                                <pre>${data.kg_context}</pre>
                            </div>
                        `;
                    }
                    if (data.rag_context) {
                        contextHtml += `
                            <div class="context-section">
                                <h4>📚 RAG Context</h4>
                                <pre>${data.rag_context}</pre>
                            </div>
                        `;
                    }
                }
                
                chat.innerHTML += `
                    <div class="message ai-message">
                        <div class="message-label">TrendScout AI</div>
                        <p>${formattedAnswer}</p>
                        <div class="sources">${sourcesBadges}</div>
                        ${contextHtml}
                    </div>
                `;
            } catch (error) {
                chat.innerHTML += `
                    <div class="message ai-message">
                        <div class="message-label">TrendScout AI</div>
                        <p style="color: #f87171;">Error: ${error.message}</p>
                    </div>
                `;
            }
            
            btn.disabled = false;
            btn.textContent = 'Ask';
            document.getElementById('question').value = '';
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    kg_status = trendscout.kg.connected if trendscout else False
    rag_status = trendscout.rag.vectorstore is not None if trendscout else False
    return render_template_string(HTML_TEMPLATE, kg_status=kg_status, rag_status=rag_status)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    use_kg = data.get('use_kg', True)
    use_rag = data.get('use_rag', True)
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    result = trendscout.ask(question, use_kg=use_kg, use_rag=use_rag)
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "kg_connected": trendscout.kg.connected if trendscout else False,
        "rag_loaded": trendscout.rag.vectorstore is not None if trendscout else False
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TrendScout AI Server")
    parser.add_argument("--load-docs", type=str, help="Load documents from JSON file into RAG")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on")
    args = parser.parse_args()
    
    print("🚀 Starting TrendScout AI...")
    
    trendscout = TrendScoutAI()
    trendscout.initialize(load_new_docs=args.load_docs)
    
    print(f"\n🌐 Server running at http://localhost:{args.port}")
    print("   Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=True)