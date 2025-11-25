"""
TrendScout AI - Neo4j Knowledge Graph Loader
=============================================
Loads extracted LinkedIn post data from 5 AI companies into Neo4j

Companies: Perplexity AI, OpenAI, Mistral AI, Anthropic, DeepSeek

Node Types:
- Company: AI companies (5 total)
- Post: LinkedIn posts (174 total)
- Product: Products and platforms
- AIModel: AI models mentioned
- Partner: Partnership organizations
- Feature: Product features
- Metric: Performance metrics
- Topic: Topic tags
- Platform: Availability platforms (web, iOS, API, etc.)
- Tier: Subscription tiers (Free, Pro, Max, Enterprise, etc.)

Relationships:
- (Company)-[:PUBLISHED]->(Post)
- (Company)-[:DEVELOPS]->(Product)
- (Company)-[:RELEASED]->(AIModel)
- (Company)-[:PARTNERS_WITH]->(Partner)
- (Post)-[:ANNOUNCES]->(Product)
- (Post)-[:MENTIONS_MODEL]->(AIModel)
- (Post)-[:MENTIONS_PARTNER]->(Partner)
- (Post)-[:HAS_FEATURE]->(Feature)
- (Post)-[:REPORTS_METRIC]->(Metric)
- (Post)-[:HAS_TOPIC]->(Topic)
- (Product)-[:AVAILABLE_ON]->(Platform)
- (Product)-[:AVAILABLE_FOR]->(Tier)
- (Product)-[:HAS_FEATURE]->(Feature)
- (AIModel)-[:AVAILABLE_ON]->(Platform)

Usage:
    python load_all_companies_kg.py --input all_companies_KG.json

Requirements:
    pip install neo4j python-dotenv
"""

import json
import os
import argparse
from datetime import datetime
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

class TrendScoutKGLoader:
    """Load AI company data into Neo4j Knowledge Graph"""
    
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
            "metrics": 0,
            "topics": 0,
            "platforms": 0,
            "tiers": 0,
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
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Post) REQUIRE p.global_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Product) REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:AIModel) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pa:Partner) REQUIRE pa.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pl:Platform) REQUIRE pl.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ti:Tier) REQUIRE ti.name IS UNIQUE",
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
                    SET c.name = $name,
                        c.display_name = $display_name
                """, id=company_id, name=company_id, display_name=display_name)
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
        print(f"   📊 Metrics: {self.stats['metrics']}")
        print(f"   🏷️  Topics: {self.stats['topics']}")
        print(f"   📱 Platforms: {self.stats['platforms']}")
        print(f"   💎 Tiers: {self.stats['tiers']}")
    
    def _load_single_post(self, session, post: dict):
        """Load a single post with all its relationships"""
        global_id = post.get("global_id")
        company_id = post.get("company")
        
        # Create Post node
        session.run("""
            MERGE (p:Post {global_id: $global_id})
            SET p.company = $company,
                p.date = $date,
                p.post_type = $post_type
        """, 
            global_id=global_id,
            company=company_id,
            date=post.get("date"),
            post_type=post.get("post_type")
        )
        self.stats["posts"] += 1
        
        # Link Post to Company
        session.run("""
            MATCH (c:Company {id: $company_id})
            MATCH (p:Post {global_id: $global_id})
            MERGE (c)-[:PUBLISHED]->(p)
        """, company_id=company_id, global_id=global_id)
        self.stats["relationships"] += 1
        
        # Process Products
        for product in post.get("products", []):
            if product.get("name"):
                self._create_product(session, product, global_id, company_id)
        
        # Process AI Models
        for model_name in post.get("ai_models", []):
            if model_name:
                self._create_ai_model(session, model_name, global_id, company_id)
        
        # Process Partnerships
        for partnership in post.get("partnerships", []):
            if partnership.get("partner"):
                self._create_partnership(session, partnership, global_id, company_id)
        
        # Process Features
        for feature in post.get("features", []):
            if feature.get("name"):
                self._create_feature(session, feature, global_id)
        
        # Process Metrics
        for metric in post.get("metrics", []):
            if metric.get("metric"):
                self._create_metric(session, metric, global_id)
        
        # Process Availability
        availability = post.get("availability", {})
        for platform in availability.get("platforms", []):
            if platform:
                self._create_platform(session, platform, global_id)
        
        for tier in availability.get("tiers", []):
            if tier:
                self._create_tier(session, tier, global_id)
        
        # Process Topics
        for topic in post.get("topics", []):
            if topic:
                self._create_topic(session, topic, global_id)
        
        # Process Funding (if present and is a dict)
        funding = post.get("funding")
        if funding and isinstance(funding, dict) and funding.get("amount"):
            self._create_funding(session, funding, global_id, company_id)
    
    def _create_product(self, session, product: dict, post_id: int, company_id: str):
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
            MATCH (p:Post {global_id: $post_id})
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
    
    def _create_ai_model(self, session, model_name: str, post_id: int, company_id: str):
        """Create AIModel node and relationships"""
        session.run("""
            MERGE (m:AIModel {name: $name})
        """, name=model_name)
        self.stats["ai_models"] += 1
        
        # Link to Post
        session.run("""
            MATCH (p:Post {global_id: $post_id})
            MATCH (m:AIModel {name: $name})
            MERGE (p)-[:MENTIONS_MODEL]->(m)
        """, post_id=post_id, name=model_name)
        self.stats["relationships"] += 1
        
        # Link to Company
        session.run("""
            MATCH (c:Company {id: $company_id})
            MATCH (m:AIModel {name: $name})
            MERGE (c)-[:RELEASED]->(m)
        """, company_id=company_id, name=model_name)
        self.stats["relationships"] += 1
    
    def _create_partnership(self, session, partnership: dict, post_id: int, company_id: str):
        """Create Partner node and relationships"""
        partner_name = partnership.get("partner")
        
        session.run("""
            MERGE (pa:Partner {name: $name})
            SET pa.type = $type,
                pa.details = $details
        """,
            name=partner_name,
            type=partnership.get("type"),
            details=partnership.get("details")
        )
        self.stats["partners"] += 1
        
        # Link to Post
        session.run("""
            MATCH (p:Post {global_id: $post_id})
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
    
    def _create_feature(self, session, feature: dict, post_id: int):
        """Create Feature node and relationship"""
        feature_name = feature.get("name")
        
        # Use a composite key for features since they can have same name but different availability
        feature_id = f"{feature_name}_{feature.get('availability', 'all')}"
        
        session.run("""
            MERGE (f:Feature {id: $id})
            SET f.name = $name,
                f.availability = $availability
        """,
            id=feature_id,
            name=feature_name,
            availability=feature.get("availability")
        )
        self.stats["features"] += 1
        
        session.run("""
            MATCH (p:Post {global_id: $post_id})
            MATCH (f:Feature {id: $id})
            MERGE (p)-[:HAS_FEATURE]->(f)
        """, post_id=post_id, id=feature_id)
        self.stats["relationships"] += 1
    
    def _create_metric(self, session, metric: dict, post_id: int):
        """Create Metric node and relationship"""
        metric_id = f"{post_id}_{metric.get('metric')}_{metric.get('value')}"
        
        session.run("""
            MERGE (m:Metric {id: $id})
            SET m.metric = $metric,
                m.value = $value,
                m.context = $context
        """,
            id=metric_id,
            metric=metric.get("metric"),
            value=metric.get("value"),
            context=metric.get("context")
        )
        self.stats["metrics"] += 1
        
        session.run("""
            MATCH (p:Post {global_id: $post_id})
            MATCH (m:Metric {id: $id})
            MERGE (p)-[:REPORTS_METRIC]->(m)
        """, post_id=post_id, id=metric_id)
        self.stats["relationships"] += 1
    
    def _create_platform(self, session, platform_name: str, post_id: int):
        """Create Platform node and relationship"""
        session.run("""
            MERGE (pl:Platform {name: $name})
        """, name=platform_name)
        self.stats["platforms"] += 1
        
        session.run("""
            MATCH (p:Post {global_id: $post_id})
            MATCH (pl:Platform {name: $name})
            MERGE (p)-[:AVAILABLE_ON]->(pl)
        """, post_id=post_id, name=platform_name)
        self.stats["relationships"] += 1
    
    def _create_tier(self, session, tier_name: str, post_id: int):
        """Create Tier node and relationship"""
        session.run("""
            MERGE (t:Tier {name: $name})
        """, name=tier_name)
        self.stats["tiers"] += 1
        
        session.run("""
            MATCH (p:Post {global_id: $post_id})
            MATCH (t:Tier {name: $name})
            MERGE (p)-[:AVAILABLE_FOR]->(t)
        """, post_id=post_id, name=tier_name)
        self.stats["relationships"] += 1
    
    def _create_topic(self, session, topic_name: str, post_id: int):
        """Create Topic node and relationship"""
        session.run("""
            MERGE (t:Topic {name: $name})
        """, name=topic_name)
        self.stats["topics"] += 1
        
        session.run("""
            MATCH (p:Post {global_id: $post_id})
            MATCH (t:Topic {name: $name})
            MERGE (p)-[:HAS_TOPIC]->(t)
        """, post_id=post_id, name=topic_name)
        self.stats["relationships"] += 1
    
    def _create_funding(self, session, funding: dict, post_id: int, company_id: str):
        """Create Funding node and relationship"""
        funding_id = f"{company_id}_{funding.get('amount')}_{funding.get('round_type', 'unknown')}"
        
        session.run("""
            MERGE (f:Funding {id: $id})
            SET f.amount = $amount,
                f.currency = $currency,
                f.round_type = $round_type
        """,
            id=funding_id,
            amount=funding.get("amount"),
            currency=funding.get("currency"),
            round_type=funding.get("round_type")
        )
        
        session.run("""
            MATCH (c:Company {id: $company_id})
            MATCH (f:Funding {id: $id})
            MERGE (c)-[:RAISED]->(f)
        """, company_id=company_id, id=funding_id)
        
        # Create investor relationships
        for investor in funding.get("investors", []):
            if investor:
                session.run("""
                    MERGE (i:Investor {name: $name})
                """, name=investor)
                
                session.run("""
                    MATCH (i:Investor {name: $name})
                    MATCH (f:Funding {id: $id})
                    MERGE (i)-[:INVESTED_IN]->(f)
                """, name=investor, id=funding_id)
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("📊 KNOWLEDGE GRAPH SUMMARY")
        print("="*60)
        
        with self.driver.session(database=self.database) as session:
            # Count nodes by type
            result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
                RETURN label, value.count as count
            """)
            
            # Fallback without APOC
            labels = ["Company", "Post", "Product", "AIModel", "Partner", "Feature", 
                     "Metric", "Topic", "Platform", "Tier", "Funding", "Investor"]
            
            print("\n📦 Nodes by Type:")
            for label in labels:
                try:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    count = result.single()["count"]
                    if count > 0:
                        print(f"   {label}: {count}")
                except:
                    pass
            
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


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Load AI company data into Neo4j")
    parser.add_argument("--input", default="../data/all_companies_KG.json", help="Input JSON file")
    parser.add_argument("--clear", action="store_true", help="Clear database before loading")
    parser.add_argument("--uri", default=NEO4J_URI, help="Neo4j URI")
    parser.add_argument("--user", default=NEO4J_USER, help="Neo4j username")
    parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j password")
    parser.add_argument("--database", default=NEO4J_DATABASE, help="Neo4j database")
    
    args = parser.parse_args()
    
    print("🚀 TrendScout AI - Knowledge Graph Loader")
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
    print(f"   Companies: {len(companies)}")
    print(f"   Posts: {len(posts)}")
    
    # Initialize loader
    try:
        loader = TrendScoutKGLoader(args.uri, args.user, args.password, args.database)
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        print("\n💡 Make sure Neo4j is running!")
        print("   Check Neo4j Desktop or run: neo4j start")
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
        
        print("\n✅ Knowledge Graph loaded successfully!")
        print("\n💡 Try these queries in Neo4j Browser (http://localhost:7474):")
        print("")
        print("   // View all companies and their products")
        print("   MATCH (c:Company)-[:DEVELOPS]->(p:Product) RETURN c, p")
        print("")
        print("   // View partnerships")
        print("   MATCH (c:Company)-[:PARTNERS_WITH]->(pa:Partner) RETURN c, pa")
        print("")
        print("   // View AI models by company")
        print("   MATCH (c:Company)-[:RELEASED]->(m:AIModel) RETURN c.display_name, collect(m.name)")
        print("")
        print("   // View metrics and performance claims")
        print("   MATCH (p:Post)-[:REPORTS_METRIC]->(m:Metric) RETURN p.company, m.metric, m.value, m.context")
        print("")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loader.close()


if __name__ == "__main__":
    main()