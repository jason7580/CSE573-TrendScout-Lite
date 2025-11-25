"""
TrendScout AI - Neo4j Knowledge Graph Loader v2
================================================
Loads extracted LinkedIn post data from 5 AI companies into Neo4j

Version 2 Features:
- Post titles (displayed in Neo4j visualization)
- AI Model providers (who created vs who uses)
- Partner categories (government, tech_giant, tech_company, etc.)

Companies: Perplexity AI, OpenAI, Mistral AI, Anthropic, DeepSeek

Node Types:
- Company: AI companies (5 total)
- Post: LinkedIn posts with titles (174 total)
- Product: Products and platforms
- AIModel: AI models with provider info
- Partner: Partnership organizations with categories
- Feature: Product features
- Topic: Topic tags

Relationships:
- (Company)-[:PUBLISHED]->(Post)
- (Company)-[:DEVELOPS]->(Product)
- (Company)-[:USES_MODEL]->(AIModel)
- (Company)-[:RELEASED_MODEL]->(AIModel)
- (Company)-[:PARTNERS_WITH]->(Partner)
- (Post)-[:ANNOUNCES]->(Product)
- (Post)-[:MENTIONS_MODEL]->(AIModel)
- (Post)-[:MENTIONS_PARTNER]->(Partner)
- (Post)-[:HAS_FEATURE]->(Feature)
- (Post)-[:TAGGED]->(Topic)
- (Product)-[:HAS_FEATURE]->(Feature)

Usage:
    python load_kg_v2.py --input all_companies_KG_v2.json --clear

Requirements:
    pip install neo4j python-dotenv
"""

import json
import os
import argparse
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Company display names
COMPANY_NAMES = {
    "perplexity-ai": "Perplexity AI",
    "openai": "OpenAI",
    "mistralai": "Mistral AI",
    "anthropicresearch": "Anthropic",
    "deepseek-ai": "DeepSeek"
}

# =============================================================================
# Neo4j Loader Class
# =============================================================================

class TrendScoutKGLoaderV2:
    """Load AI company data into Neo4j Knowledge Graph (v2 with titles)"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.stats = {
            "companies": 0,
            "posts": 0,
            "products": 0,
            "ai_models": 0,
            "partners": 0,
            "features": 0,
            "topics": 0,
            "relationships": 0
        }
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️  Cleared existing data")
    
    def create_constraints(self):
        """Create uniqueness constraints for better performance"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Product) REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:AIModel) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pa:Partner) REQUIRE pa.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    pass  # Constraint may already exist
        
        print("✅ Created constraints")
    
    def create_companies(self, companies: list):
        """Create Company nodes"""
        with self.driver.session(database=self.database) as session:
            for company_id in companies:
                display_name = COMPANY_NAMES.get(company_id, company_id)
                session.run("""
                    MERGE (c:Company {id: $id})
                    SET c.name = $display_name
                """, id=company_id, display_name=display_name)
                self.stats["companies"] += 1
        
        print(f"✅ Created {self.stats['companies']} companies")
    
    def load_posts(self, posts: list):
        """Load all posts and their related entities"""
        with self.driver.session(database=self.database) as session:
            for post in posts:
                self._load_single_post(session, post)
        
        print(f"✅ Loaded {self.stats['posts']} posts")
        print(f"   📦 Products: {self.stats['products']}")
        print(f"   🤖 AI Models: {self.stats['ai_models']}")
        print(f"   🤝 Partners: {self.stats['partners']}")
        print(f"   ⚡ Features: {self.stats['features']}")
        print(f"   🏷️  Topics: {self.stats['topics']}")
    
    def _load_single_post(self, session, post: dict):
        """Load a single post with all its relationships"""
        post_id = post.get("post_id")
        company_id = post.get("company")
        title = post.get("title", "Untitled")
        
        # Create Post node with TITLE as the main display property
        session.run("""
            MERGE (p:Post {post_id: $post_id})
            SET p.title = $title,
                p.summary = $summary,
                p.company = $company,
                p.date = $date,
                p.post_type = $post_type,
                p.global_id = $global_id
        """, 
            post_id=post_id,
            title=title,
            summary=post.get("summary"),
            company=company_id,
            date=post.get("date"),
            post_type=post.get("post_type"),
            global_id=post.get("global_id")
        )
        self.stats["posts"] += 1
        
        # Link Post to Company
        session.run("""
            MATCH (c:Company {id: $company_id})
            MATCH (p:Post {post_id: $post_id})
            MERGE (c)-[:PUBLISHED]->(p)
        """, company_id=company_id, post_id=post_id)
        self.stats["relationships"] += 1
        
        # Process Products
        for product in post.get("products", []):
            if product.get("name"):
                self._create_product(session, product, post_id, company_id)
        
        # Process AI Models (with provider info)
        for model in post.get("ai_models", []):
            if isinstance(model, dict) and model.get("name"):
                self._create_ai_model(session, model, post_id, company_id)
            elif isinstance(model, str) and model:
                # Handle legacy string format
                self._create_ai_model(session, {"name": model, "provider": "Unknown"}, post_id, company_id)
        
        # Process Partnerships (with category)
        for partnership in post.get("partnerships", []):
            if partnership.get("partner"):
                self._create_partnership(session, partnership, post_id, company_id)
        
        # Process Features
        for feature in post.get("features", []):
            if feature.get("name"):
                self._create_feature(session, feature, post_id)
        
        # Process Topics
        for topic in post.get("topics", []):
            if topic:
                self._create_topic(session, topic, post_id)
    
    def _create_product(self, session, product: dict, post_id: str, company_id: str):
        """Create Product node and relationships"""
        name = product.get("name")
        
        session.run("""
            MERGE (pr:Product {name: $name})
            SET pr.type = $type,
                pr.description = $description
        """,
            name=name,
            type=product.get("type"),
            description=product.get("description")
        )
        self.stats["products"] += 1
        
        # Link Product to Post
        session.run("""
            MATCH (p:Post {post_id: $post_id})
            MATCH (pr:Product {name: $name})
            MERGE (p)-[:ANNOUNCES]->(pr)
        """, post_id=post_id, name=name)
        self.stats["relationships"] += 1
        
        # Link Product to Company
        session.run("""
            MATCH (c:Company {id: $company_id})
            MATCH (pr:Product {name: $name})
            MERGE (c)-[:DEVELOPS]->(pr)
        """, company_id=company_id, name=name)
        self.stats["relationships"] += 1
    
    def _create_ai_model(self, session, model: dict, post_id: str, company_id: str):
        """Create AIModel node with provider and relationships"""
        model_name = model.get("name")
        provider = model.get("provider", "Unknown")
        is_used = model.get("is_used", False)
        is_released = model.get("is_released", False)
        
        # Create or update AIModel node with provider
        session.run("""
            MERGE (m:AIModel {name: $name})
            SET m.provider = $provider
        """, name=model_name, provider=provider)
        self.stats["ai_models"] += 1
        
        # Link to Post
        session.run("""
            MATCH (p:Post {post_id: $post_id})
            MATCH (m:AIModel {name: $name})
            MERGE (p)-[:MENTIONS_MODEL]->(m)
        """, post_id=post_id, name=model_name)
        self.stats["relationships"] += 1
        
        # Link to Company based on is_used/is_released
        if is_used:
            session.run("""
                MATCH (c:Company {id: $company_id})
                MATCH (m:AIModel {name: $name})
                MERGE (c)-[:USES_MODEL]->(m)
            """, company_id=company_id, name=model_name)
            self.stats["relationships"] += 1
        
        if is_released:
            session.run("""
                MATCH (c:Company {id: $company_id})
                MATCH (m:AIModel {name: $name})
                MERGE (c)-[:RELEASED_MODEL]->(m)
            """, company_id=company_id, name=model_name)
            self.stats["relationships"] += 1
    
    def _create_partnership(self, session, partnership: dict, post_id: str, company_id: str):
        """Create Partner node with category and relationships"""
        partner_name = partnership.get("partner")
        category = partnership.get("category", "other")
        
        session.run("""
            MERGE (pa:Partner {name: $name})
            SET pa.category = $category,
                pa.type = $type,
                pa.details = $details
        """,
            name=partner_name,
            category=category,
            type=partnership.get("type"),
            details=partnership.get("details")
        )
        self.stats["partners"] += 1
        
        # Link to Post
        session.run("""
            MATCH (p:Post {post_id: $post_id})
            MATCH (pa:Partner {name: $name})
            MERGE (p)-[:MENTIONS_PARTNER]->(pa)
        """, post_id=post_id, name=partner_name)
        self.stats["relationships"] += 1
        
        # Link to Company
        session.run("""
            MATCH (c:Company {id: $company_id})
            MATCH (pa:Partner {name: $name})
            MERGE (c)-[:PARTNERS_WITH]->(pa)
        """, company_id=company_id, name=partner_name)
        self.stats["relationships"] += 1
    
    def _create_feature(self, session, feature: dict, post_id: str):
        """Create Feature node and relationship"""
        feature_name = feature.get("name")
        
        # Use name + availability as unique identifier
        feature_id = f"{feature_name}_{feature.get('availability', 'all')}"
        
        session.run("""
            MERGE (f:Feature {id: $id})
            SET f.name = $name,
                f.description = $description,
                f.availability = $availability
        """,
            id=feature_id,
            name=feature_name,
            description=feature.get("description"),
            availability=feature.get("availability")
        )
        self.stats["features"] += 1
        
        session.run("""
            MATCH (p:Post {post_id: $post_id})
            MATCH (f:Feature {id: $id})
            MERGE (p)-[:HAS_FEATURE]->(f)
        """, post_id=post_id, id=feature_id)
        self.stats["relationships"] += 1
    
    def _create_topic(self, session, topic_name: str, post_id: str):
        """Create Topic node and relationship"""
        session.run("""
            MERGE (t:Topic {name: $name})
        """, name=topic_name)
        self.stats["topics"] += 1
        
        session.run("""
            MATCH (p:Post {post_id: $post_id})
            MATCH (t:Topic {name: $name})
            MERGE (p)-[:TAGGED]->(t)
        """, post_id=post_id, name=topic_name)
        self.stats["relationships"] += 1
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("📊 KNOWLEDGE GRAPH SUMMARY")
        print("="*60)
        
        with self.driver.session(database=self.database) as session:
            # Count nodes by type
            labels = ["Company", "Post", "Product", "AIModel", "Partner", "Feature", "Topic"]
            
            print("\n📦 Nodes by Type:")
            total_nodes = 0
            for label in labels:
                try:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    count = result.single()["count"]
                    if count > 0:
                        print(f"   {label}: {count}")
                        total_nodes += count
                except:
                    pass
            
            print(f"\n   Total Nodes: {total_nodes}")
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            print(f"\n🔗 Total Relationships: {rel_count}")
            
            # Show relationship types
            print("\n🔗 Relationships by Type:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   {record['type']}: {record['count']}")
            
            # Show AI Models by provider
            print("\n🤖 AI Models by Provider:")
            result = session.run("""
                MATCH (m:AIModel)
                RETURN m.provider as provider, collect(m.name) as models
                ORDER BY size(collect(m.name)) DESC
            """)
            for record in result:
                provider = record['provider'] or 'Unknown'
                models = record['models'][:5]  # Show first 5
                more = len(record['models']) - 5
                models_str = ", ".join(models)
                if more > 0:
                    models_str += f" (+{more} more)"
                print(f"   {provider}: {models_str}")
            
            # Show Partners by category
            print("\n🤝 Partners by Category:")
            result = session.run("""
                MATCH (pa:Partner)
                RETURN pa.category as category, count(pa) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   {record['category']}: {record['count']}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Load AI company data into Neo4j (v2)")
    parser.add_argument("--input", default="../data/all_companies_KG_v2.json", help="Input JSON file")
    parser.add_argument("--clear", action="store_true", help="Clear database before loading")
    parser.add_argument("--uri", default=NEO4J_URI, help="Neo4j URI")
    parser.add_argument("--user", default=NEO4J_USER, help="Neo4j username")
    parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j password")
    parser.add_argument("--database", default=NEO4J_DATABASE, help="Neo4j database")
    
    args = parser.parse_args()
    
    print("🚀 TrendScout AI - Knowledge Graph Loader v2")
    print("="*60)
    print(f"📁 Input file: {args.input}")
    print(f"🔗 Neo4j URI: {args.uri}")
    print(f"📊 Database: {args.database}")
    print("="*60)
    
    # Load JSON data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.input}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON: {e}")
        return
    
    metadata = data.get("metadata", {})
    posts = data.get("extracted_data", [])
    companies = metadata.get("companies", [])
    
    print(f"\n📊 Data Summary:")
    print(f"   Version: {metadata.get('version', '1.0')}")
    print(f"   Companies: {len(companies)}")
    print(f"   Posts: {len(posts)}")
    print(f"   Features: {', '.join(metadata.get('features', []))}")
    
    # Initialize loader
    try:
        loader = TrendScoutKGLoaderV2(args.uri, args.user, args.password, args.database)
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        print("\n💡 Make sure Neo4j is running!")
        return
    
    try:
        # Clear database if requested
        if args.clear:
            loader.clear_database()
        
        # Create constraints
        loader.create_constraints()
        
        # Load data
        print("\n📥 Loading data...")
        loader.create_companies(companies)
        loader.load_posts(posts)
        
        # Print summary
        loader.print_summary()
        
        print("\n✅ Knowledge Graph v2 loaded successfully!")
        print("\n💡 Try these queries in Neo4j Browser (http://localhost:7474):")
        print("")
        print("   // View all posts with their titles")
        print("   MATCH (p:Post) RETURN p.title, p.company, p.date LIMIT 20")
        print("")
        print("   // View companies and their products")
        print("   MATCH (c:Company)-[:DEVELOPS]->(pr:Product) RETURN c.name, collect(pr.name)")
        print("")
        print("   // View AI models by provider")
        print("   MATCH (m:AIModel) RETURN m.provider, collect(m.name)")
        print("")
        print("   // View partnerships by category")
        print("   MATCH (c:Company)-[:PARTNERS_WITH]->(pa:Partner)")
        print("   RETURN c.name, pa.category, collect(pa.name)")
        print("")
        print("   // View which companies USE vs RELEASED models")
        print("   MATCH (c:Company)-[r:USES_MODEL|RELEASED_MODEL]->(m:AIModel)")
        print("   RETURN c.name, type(r), m.name, m.provider")
        print("")
        print("   // Full graph visualization (limited)")
        print("   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
        print("")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loader.close()


if __name__ == "__main__":
    main()
