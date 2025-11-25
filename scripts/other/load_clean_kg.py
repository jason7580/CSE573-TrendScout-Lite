"""
Load LLM-extracted TechFundingNews data into Neo4j Knowledge Graph
This script uses clean, structured JSON from Gemini/Claude extraction
"""

from neo4j import GraphDatabase
import json

class CleanTechFundingKG:
    """
    Build Knowledge Graph from LLM-extracted structured data
    Much cleaner than regex extraction!
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
            session.run("CREATE INDEX IF NOT EXISTS FOR (i:Investor) ON (i.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (f:Funding) ON (f.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (l:Location) ON (l.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (pr:Product) ON (pr.name)")
        print("✅ Created indexes")
    
    def build_graph_from_clean_json(self, json_file: str):
        """
        Build knowledge graph from LLM-extracted structured JSON
        
        Args:
            json_file: Path to the cleaned JSON file from Gemini/Claude
        """
        print(f"\n📊 Building Knowledge Graph from {json_file}...")
        
        # Load data
        with open(json_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"📄 Loaded {len(articles)} articles\n")
        
        with self.driver.session() as session:
            for i, article in enumerate(articles, 1):
                company_name = article['company']['name']
                print(f"Processing article {i}/{len(articles)}: {company_name}")
                
                # 1. Create Company node
                session.run("""
                    MERGE (c:Company {name: $name})
                    SET c.location = $location,
                        c.founded_date = $founded_date,
                        c.description = $description
                """,
                    name=company_name,
                    location=article['company'].get('location'),
                    founded_date=article['company'].get('founded_date'),
                    description=article['company'].get('description')
                )
                print(f"  ✓ Company: {company_name}")
                
                # 2. Create Location node
                if article['company'].get('location'):
                    location = article['company']['location']
                    session.run("""
                        MERGE (l:Location {name: $location})
                        WITH l
                        MATCH (c:Company {name: $company})
                        MERGE (c)-[:LOCATED_IN]->(l)
                    """,
                        location=location,
                        company=company_name
                    )
                    print(f"  ✓ Location: {location}")
                
                # 3. Create Funding node
                if article.get('funding') and article['funding'].get('amount'):
                    funding = article['funding']
                    funding_id = f"{company_name}_{funding.get('date', 'unknown')}_{funding.get('round_type', 'funding')}"
                    
                    session.run("""
                        MATCH (c:Company {name: $company})
                        CREATE (f:Funding {
                            id: $funding_id,
                            amount: $amount,
                            currency: $currency,
                            round_type: $round_type,
                            date: $date
                        })
                        MERGE (c)-[:RAISED]->(f)
                    """,
                        company=company_name,
                        funding_id=funding_id,
                        amount=funding.get('amount'),
                        currency=funding.get('currency'),
                        round_type=funding.get('round_type'),
                        date=funding.get('date')
                    )
                    
                    round_type = funding.get('round_type', 'round')
                    amount = funding.get('amount', 'N/A')
                    currency = funding.get('currency', '')
                    print(f"  ✓ Funding: {currency} {amount}M ({round_type})")
                    
                    # 4. Create Investor nodes and relationships
                    for investor in article.get('investors', []):
                        investor_name = investor['name']
                        is_lead = (investor.get('role') == 'lead')
                        
                        session.run("""
                            MERGE (i:Investor {name: $investor_name})
                            WITH i
                            MATCH (f:Funding {id: $funding_id})
                            MERGE (i)-[r:INVESTED_IN]->(f)
                            SET r.is_lead = $is_lead
                        """,
                            investor_name=investor_name,
                            funding_id=funding_id,
                            is_lead=is_lead
                        )
                        
                        role_str = "lead" if is_lead else "participant"
                        print(f"    • Investor: {investor_name} ({role_str})")
                
                # 5. Create Founder nodes
                for founder in article.get('founders', []):
                    founder_name = founder['name']
                    founder_role = founder.get('role', 'Founder')
                    
                    session.run("""
                        MERGE (p:Person {name: $founder_name})
                        SET p.role = $role
                        WITH p
                        MATCH (c:Company {name: $company})
                        MERGE (p)-[:FOUNDED]->(c)
                    """,
                        founder_name=founder_name,
                        role=founder_role,
                        company=company_name
                    )
                    print(f"  ✓ Founder: {founder_name} ({founder_role})")
                
                # 6. Create Product nodes
                for product in article.get('products', []):
                    session.run("""
                        MERGE (pr:Product {name: $product_name})
                        WITH pr
                        MATCH (c:Company {name: $company})
                        MERGE (c)-[:DEVELOPS]->(pr)
                    """,
                        product_name=product,
                        company=company_name
                    )
                    print(f"  ✓ Product: {product}")
                
                # 7. Create Key People nodes (executives, partners, etc.)
                for person in article.get('key_people', []):
                    person_name = person['name']
                    person_role = person.get('role', 'Executive')
                    person_company = person.get('company', 'Unknown')
                    
                    session.run("""
                        MERGE (p:Person {name: $person_name})
                        SET p.role = $role
                        WITH p
                        
                        // Connect to their company if it exists in graph
                        OPTIONAL MATCH (org:Company {name: $person_company})
                        FOREACH (o IN CASE WHEN org IS NOT NULL THEN [org] ELSE [] END |
                            MERGE (p)-[:WORKS_AT]->(o)
                        )
                        
                        // Also link to the company mentioned in this article
                        WITH p
                        MATCH (c:Company {name: $article_company})
                        MERGE (p)-[:MENTIONED_IN_CONTEXT]->(c)
                    """,
                        person_name=person_name,
                        role=person_role,
                        person_company=person_company,
                        article_company=company_name
                    )
                    print(f"  ✓ Key Person: {person_name} ({person_role} at {person_company})")
                
                print()  # Blank line between articles
        
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
            # Query 1: Top funded companies
            print("\n1. Top 5 Funded Companies:")
            result = session.run("""
                MATCH (c:Company)-[:RAISED]->(f:Funding)
                WHERE f.amount IS NOT NULL
                RETURN c.name as company, 
                       f.amount as amount,
                       f.currency as currency,
                       f.round_type as round
                ORDER BY f.amount DESC
                LIMIT 5
            """)
            for record in result:
                round_type = record['round'] or 'funding round'
                print(f"   • {record['company']}: {record['currency']} {record['amount']}M ({round_type})")
            
            # Query 2: Most active investors
            print("\n2. Most Active Investors:")
            result = session.run("""
                MATCH (i:Investor)-[:INVESTED_IN]->()
                RETURN i.name as investor, count(*) as investments
                ORDER BY investments DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   • {record['investor']}: {record['investments']} investments")
            
            # Query 3: Companies by location
            print("\n3. Top Startup Locations:")
            result = session.run("""
                MATCH (c:Company)-[:LOCATED_IN]->(l:Location)
                RETURN l.name as location, collect(c.name) as companies, count(c) as count
                ORDER BY count DESC
                LIMIT 5
            """)
            for record in result:
                companies_str = ", ".join(record['companies'][:3])
                if len(record['companies']) > 3:
                    companies_str += f" (+{len(record['companies'])-3} more)"
                print(f"   • {record['location']}: {record['count']} companies")
                print(f"     {companies_str}")
            
            # Query 4: Lead investors
            print("\n4. Lead Investors:")
            result = session.run("""
                MATCH (i:Investor)-[r:INVESTED_IN]->(f:Funding)
                WHERE r.is_lead = true
                MATCH (f)<-[:RAISED]-(c:Company)
                RETURN i.name as investor, collect(c.name) as companies
                ORDER BY size(companies) DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   • {record['investor']}: led rounds in {', '.join(record['companies'])}")
            
            # Query 5: Founders and their companies
            print("\n5. Notable Founders:")
            result = session.run("""
                MATCH (p:Person)-[:FOUNDED]->(c:Company)
                RETURN p.name as founder, c.name as company, p.role as role
                LIMIT 10
            """)
            for record in result:
                role = record['role'] or 'Founder'
                print(f"   • {record['founder']} ({role}) → {record['company']}")


def main():
    """Main execution function"""
    
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"  # Updated for Neo4j Desktop 2025
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neo4jneo4j"  # CHANGE THIS!
    JSON_FILE = "../data/techfundingnews_KG.json"
    
    print("🚀 TechFundingNews (LLM-Extracted) → Knowledge Graph")
    print("="*60)
    
    # Connect to Neo4j
    kg = CleanTechFundingKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Optional: Clear existing data
        # WARNING: Uncomment to delete everything!
        kg.clear_database()
        
        # Create indexes
        kg.create_indexes()
        
        # Build the graph from clean LLM-extracted data
        kg.build_graph_from_clean_json(JSON_FILE)
        
        # Run sample queries
        kg.run_sample_queries()
        
    finally:
        kg.close()
    
    print("\n✅ Done! Open Neo4j Browser to explore: http://localhost:7474")
    print("\n💡 Try these queries:")
    print("   MATCH (n) RETURN n LIMIT 100")
    print("   MATCH p=(c:Company)-[:RAISED]->(f:Funding)<-[:INVESTED_IN]-(i:Investor)")
    print("   RETURN p LIMIT 25")


if __name__ == "__main__":
    main()
