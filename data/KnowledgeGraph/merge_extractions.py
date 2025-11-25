"""
Merge 5 Company Extractions into One Knowledge Graph File
==========================================================

After running Gemini on each company separately, use this script to merge them.

Usage:
    python merge_extractions.py

Input files (place in same directory):
    - perplexity_extracted.json
    - openai_extracted.json
    - mistral_extracted.json
    - anthropic_extracted.json
    - deepseek_extracted.json

Output:
    - all_companies_KG_v2.json
"""

import json
import os
from datetime import datetime

def load_json(filepath):
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON error in {filepath}: {e}")
        return None

def merge_extractions():
    """Merge all company extractions into one file"""
    
    # Define input files
    files = [
        ("perplexity_extracted.json", "perplexity-ai", "Perplexity AI", 1),
        ("openai_extracted.json", "openai", "OpenAI", 41),
        ("mistral_extracted.json", "mistralai", "Mistral AI", 81),
        ("anthropic_extracted.json", "anthropicresearch", "Anthropic", 121),
        ("deepseek_extracted.json", "deepseek-ai", "DeepSeek", 161),
    ]
    
    all_posts = []
    companies = []
    total_loaded = 0
    
    print("🔄 Merging company extractions...")
    print("="*60)
    
    for filename, company_id, display_name, expected_start in files:
        data = load_json(filename)
        
        if data is None:
            print(f"   ⏭️  Skipping {display_name}")
            continue
        
        # Handle different JSON structures
        if "posts" in data:
            posts = data["posts"]
        elif "extracted_data" in data:
            posts = data["extracted_data"]
        elif isinstance(data, list):
            posts = data
        else:
            print(f"   ❌ Unknown structure in {filename}")
            continue
        
        # Ensure global_id is correct
        for i, post in enumerate(posts):
            post["global_id"] = expected_start + i
            post["company"] = company_id
            if "post_id" not in post:
                post["post_id"] = f"{company_id.replace('-ai', '').replace('research', '')}-{i+1}"
        
        all_posts.extend(posts)
        companies.append(company_id)
        total_loaded += len(posts)
        
        print(f"   ✅ {display_name}: {len(posts)} posts (global_id {expected_start}-{expected_start + len(posts) - 1})")
    
    print("="*60)
    print(f"📊 Total posts: {total_loaded}")
    
    # Create merged output
    output = {
        "metadata": {
            "extraction_date": datetime.now().strftime("%Y-%m-%d"),
            "total_posts": len(all_posts),
            "companies": companies,
            "version": "2.0",
            "features": ["post_titles", "ai_model_providers", "partner_categories"]
        },
        "extracted_data": all_posts
    }
    
    # Save merged file
    output_file = "all_companies_KG_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to {output_file}")
    
    # Print summary
    print("\n📊 Summary by Company:")
    for company_id in companies:
        count = len([p for p in all_posts if p.get("company") == company_id])
        print(f"   {company_id}: {count} posts")
    
    return output

def validate_extraction(filepath):
    """Validate a single extraction file"""
    data = load_json(filepath)
    if data is None:
        return False
    
    posts = data.get("posts", data.get("extracted_data", data if isinstance(data, list) else []))
    
    print(f"\n🔍 Validating {filepath}...")
    print(f"   Posts found: {len(posts)}")
    
    issues = []
    for i, post in enumerate(posts):
        if not post.get("title"):
            issues.append(f"Post {i+1}: missing title")
        if not post.get("summary"):
            issues.append(f"Post {i+1}: missing summary")
        if not post.get("post_type"):
            issues.append(f"Post {i+1}: missing post_type")
    
    if issues:
        print(f"   ⚠️  Issues found: {len(issues)}")
        for issue in issues[:5]:
            print(f"      - {issue}")
        if len(issues) > 5:
            print(f"      ... and {len(issues) - 5} more")
    else:
        print("   ✅ All posts valid")
    
    return len(issues) == 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Validate mode
        files = [
            "perplexity_extracted.json",
            "openai_extracted.json", 
            "mistral_extracted.json",
            "anthropic_extracted.json",
            "deepseek_extracted.json"
        ]
        for f in files:
            if os.path.exists(f):
                validate_extraction(f)
    else:
        # Merge mode
        merge_extractions()
