"""
Load LinkedIn Company Posts into Neo4j Knowledge Graph
Handles: Products, Features, Partnerships, AI Models, Topics
"""
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import json

class LinkedInPostsKG:
    """
    Build Knowledge Graph from LinkedIn company posts
    Entities: Company, Product, Feature, Partnership, AIModel, Topic, Post
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✅ Connected to Neo4j at {uri}")
    
    def close(self):
        self.driver.close()
        print("✅ Closed Neo4j connection")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("⚠️  Database cleared")
    
    def create_indexes(self):
        """Create indexes for better performance"""
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (f:Feature) ON (f.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:AIModel) ON (m.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (partner:Partner) ON (partner.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (post:Post) ON (post.id)")
        print("✅ Created indexes")
    
    def build_graph_from_posts(self, json_file: str):
        """
        Build knowledge graph from LinkedIn posts JSON
        
        Args:
            json_file: Path to the JSON file with extracted post data
        """
        print(f"\n📊 Building Knowledge Graph from {json_file}...")
        
        # Load data
        with open(json_file, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        
        print(f"📄 Loaded {len(posts)} posts\n")
        
        with self.driver.session() as session:
            for post in posts:
                post_id = post['id']
                company_name = post['company']
                post_type = post.get('post_type', 'general_update')
                
                print(f"Processing post {post_id}: {post_type}")
                
                # 1. Create Company node (MERGE to avoid duplicates)
                session.run("""
                    MERGE (c:Company {name: $name})
                """, name=company_name)
                
                # 2. Create Post node
                session.run("""
                    MATCH (c:Company {name: $company})
                    CREATE (p:Post {
                        id: $post_id,
                        type: $post_type,
                        date: $date,
                        target_audience: $target_audience
                    })
                    MERGE (c)-[:PUBLISHED]->(p)
                """,
                    company=company_name,
                    post_id=post_id,
                    post_type=post_type,
                    date=post.get('date'),
                    target_audience=post.get('target_audience')
                )
                
                # 3. Create Product nodes
                for product in post.get('products_mentioned', []):
                    if not product.get('name'):
                        continue
                    product_name = product['name']
                    product_type = product.get('type', 'product')
                    description = product.get('description') or ''
                    
                    session.run("""
                        MERGE (prod:Product {name: $name})
                        ON CREATE SET prod.type = $type, prod.description = $description
                        ON MATCH SET prod.description = COALESCE($description, prod.description)
                        
                        WITH prod
                        MATCH (c:Company {name: $company})
                        MERGE (c)-[:DEVELOPS]->(prod)
                        
                        WITH prod
                        MATCH (p:Post {id: $post_id})
                        MERGE (p)-[:MENTIONS]->(prod)
                    """,
                        name=product_name,
                        type=product_type,
                        description=description,
                        company=company_name,
                        post_id=post_id
                    )
                    print(f"  ✓ Product: {product_name} ({product_type})")
                
                # 4. Create Partnership nodes and relationships
                for partnership in post.get('partnerships', []):
                    partner_name = partnership['partner']
                    partnership_type = partnership.get('partnership_type', 'partnership')
                    details = partnership.get('details')
                    
                    session.run("""
                        MERGE (partner:Partner {name: $partner_name})
                        
                        WITH partner
                        MATCH (c:Company {name: $company})
                        MERGE (c)-[r:PARTNERS_WITH]->(partner)
                        SET r.type = $partnership_type,
                            r.details = $details
                        
                        WITH partner
                        MATCH (p:Post {id: $post_id})
                        MERGE (p)-[:ANNOUNCES]->(partner)
                    """,
                        partner_name=partner_name,
                        company=company_name,
                        partnership_type=partnership_type,
                        details=details,
                        post_id=post_id
                    )
                    print(f"  ✓ Partnership: {partner_name} ({partnership_type})")
                
                # 5. Create Feature nodes
                for feature in post.get('features_announced', []):
                    if not feature.get('feature_name'):
                        continue
                    feature_name = feature['feature_name']
                    description = feature.get('description') or ''
                    availability = feature.get('availability') or ''
                    
                    session.run("""
                        MERGE (f:Feature {name: $name})
                        ON CREATE SET f.description = $description, f.availability = $availability
                        
                        WITH f
                        MATCH (c:Company {name: $company})
                        MERGE (c)-[:OFFERS]->(f)
                        
                        WITH f
                        MATCH (p:Post {id: $post_id})
                        MERGE (p)-[:ANNOUNCES]->(f)
                    """,
                        name=feature_name,
                        description=description,
                        availability=availability,
                        company=company_name,
                        post_id=post_id
                    )
                    print(f"  ✓ Feature: {feature_name}")
                
                # 6. Create AI Model nodes
                for model_name in post.get('ai_models', []):
                    session.run("""
                        MERGE (m:AIModel {name: $name})
                        
                        WITH m
                        MATCH (c:Company {name: $company})
                        MERGE (c)-[:SUPPORTS]->(m)
                        
                        WITH m
                        MATCH (p:Post {id: $post_id})
                        MERGE (p)-[:MENTIONS]->(m)
                    """,
                        name=model_name,
                        company=company_name,
                        post_id=post_id
                    )
                    print(f"  ✓ AI Model: {model_name}")
                
                # 7. Create Topic nodes
                for topic in post.get('topics', []):
                    session.run("""
                        MERGE (t:Topic {name: $topic})
                        
                        WITH t
                        MATCH (p:Post {id: $post_id})
                        MERGE (p)-[:TOPIC]->(t)
                    """,
                        topic=topic,
                        post_id=post_id
                    )
                
                if post.get('topics'):
                    print(f"  ✓ Topics: {', '.join(post['topics'])}")
                
                # 8. Store key claims as properties on Post
                if post.get('key_claims'):
                    session.run("""
                        MATCH (p:Post {id: $post_id})
                        SET p.key_claims = $claims
                    """,
                        post_id=post_id,
                        claims=post['key_claims']
                    )
                    print(f"  ✓ Key Claims: {len(post['key_claims'])} claims")
                
                print()  # Blank line between posts
        
        print("✅ Knowledge Graph built successfully!")
        self.print_stats()
    
    def print_stats(self):
        """Print knowledge graph statistics"""
        with self.driver.session() as session:
            print("\n📊 Knowledge Graph Statistics:")
            print("="*60)
            
            # Count nodes by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"  {record['label']}: {record['count']}")
            
            # Count relationships
            print("\n🔗 Relationships:")
            print("="*60)
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship, count(r) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"  {record['relationship']}: {record['count']}")
    
    def run_sample_queries(self):
        """Run sample queries to demonstrate the graph"""
        print("\n🔍 Sample Insights:")
        print("="*60)
        
        with self.driver.session() as session:
            # Query 1: All Products
            print("\n1. Products Developed:")
            result = session.run("""
                MATCH (c:Company)-[:DEVELOPS]->(p:Product)
                RETURN c.name as company, p.name as product, p.type as type
                ORDER BY c.name, p.name
            """)
            for record in result:
                print(f"   • {record['company']} → {record['product']} ({record['type']})")
            
            # Query 2: Partnerships
            print("\n2. Partnerships:")
            result = session.run("""
                MATCH (c:Company)-[r:PARTNERS_WITH]->(p:Partner)
                RETURN c.name as company, p.name as partner, r.type as type
            """)
            for record in result:
                print(f"   • {record['company']} ↔ {record['partner']} ({record['type']})")
            
            # Query 3: AI Models Supported
            print("\n3. AI Models:")
            result = session.run("""
                MATCH (c:Company)-[:SUPPORTS]->(m:AIModel)
                RETURN c.name as company, collect(m.name) as models
            """)
            for record in result:
                print(f"   • {record['company']}: {', '.join(record['models'])}")
            
            # Query 4: Features by Company
            print("\n4. Features Offered:")
            result = session.run("""
                MATCH (c:Company)-[:OFFERS]->(f:Feature)
                RETURN c.name as company, f.name as feature, f.availability as availability
                LIMIT 10
            """)
            for record in result:
                avail = record['availability'] or 'all users'
                print(f"   • {record['feature']} ({avail})")
            
            # Query 5: Post Types Distribution
            print("\n5. Post Types:")
            result = session.run("""
                MATCH (p:Post)
                RETURN p.type as post_type, count(*) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   • {record['post_type']}: {record['count']} posts")
            
            # Query 6: Hot Topics
            print("\n6. Most Discussed Topics:")
            result = session.run("""
                MATCH (p:Post)-[:TOPIC]->(t:Topic)
                RETURN t.name as topic, count(p) as mentions
                ORDER BY mentions DESC
                LIMIT 10
            """)
            for record in result:
                print(f"   • {record['topic']}: {record['mentions']} mentions")


def main():
    """Main execution function"""
    
    load_dotenv()

    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")   
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    JSON_FILE = "../data/perplexityAI_KG.json"
    
    print("🚀 LinkedIn Posts → Knowledge Graph")
    print("="*60)
    
    # Connect to Neo4j
    kg = LinkedInPostsKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Optional: Clear existing data (uncomment if you want fresh start)
        # kg.clear_database()
        
        # Create indexes
        kg.create_indexes()
        
        # Build the graph
        kg.build_graph_from_posts(JSON_FILE)
        
        # Run sample queries
        kg.run_sample_queries()
        
    finally:
        kg.close()
    
    print("\n✅ Done! Open Neo4j Browser to explore: http://localhost:7474")
    print("\n💡 Try these queries:")
    print("   MATCH (n) RETURN n LIMIT 100")
    print("   MATCH p=(c:Company)-[:DEVELOPS]->(prod:Product) RETURN p")
    print("   MATCH p=(c:Company)-[:PARTNERS_WITH]->(partner) RETURN p")


if __name__ == "__main__":
    main()
