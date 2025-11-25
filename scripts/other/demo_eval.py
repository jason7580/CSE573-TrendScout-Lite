"""
TrendScout AI - Simple Demo Evaluation
Tests ONE question on both systems (RAG-only vs RAG+KG)

Requirements:
    pip install google-genai

Usage:
    python demo_eval.py
    python demo_eval.py --question "What products does Anthropic offer?"
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =============================================================================
# Simple Test
# =============================================================================

def test_single_question(question: str, expected_answer: str = None):
    """Test a single question on both systems"""
    
    print("=" * 70)
    print("🧪 TrendScout AI - Single Question Demo")
    print("=" * 70)
    print(f"\n📝 Question: {question}")
    if expected_answer:
        print(f"✅ Expected: {expected_answer}")
    print()
    
    # Import TrendScout
    try:
        from trendscout_app import TrendScoutAI
        print("✅ Imported trendscout_app")
    except ImportError:
        try:
            from trendscout_app import TrendScoutAI
            print("✅ Imported trendscout_app (original)")
        except ImportError:
            print("❌ Could not import TrendScoutAI. Make sure the file is in the same directory.")
            return
    
    # Initialize
    print("\n🔧 Initializing TrendScout AI...")
    trendscout = TrendScoutAI()
    trendscout.initialize()
    
    # ==========================================================================
    # Test 1: RAG-only
    # ==========================================================================
    print("\n" + "-" * 70)
    print("🔍 SYSTEM A: RAG-ONLY")
    print("-" * 70)
    
    rag_result = trendscout.ask(
        question=question,
        use_kg=False,
        use_rag=True
    )
    
    print(f"\n📄 RAG Context Used:")
    rag_ctx = rag_result.get('rag_context', 'None')
    print(rag_ctx[:500] + "..." if len(rag_ctx) > 500 else rag_ctx)
    
    print(f"\n💬 RAG-only Answer:")
    print(rag_result['answer'])
    
    # ==========================================================================
    # Test 2: RAG + KG
    # ==========================================================================
    print("\n" + "-" * 70)
    print("🔗 SYSTEM B: RAG + KNOWLEDGE GRAPH")
    print("-" * 70)
    
    kg_result = trendscout.ask(
        question=question,
        use_kg=True,
        use_rag=True
    )
    
    print(f"\n📊 KG Context Used:")
    print(kg_result.get('kg_context', 'None'))
    
    print(f"\n📄 RAG Context Used:")
    rag_ctx = kg_result.get('rag_context', 'None')
    print(rag_ctx[:500] + "..." if len(rag_ctx) > 500 else rag_ctx)
    
    print(f"\n💬 RAG+KG Answer:")
    print(kg_result['answer'])
    
    # ==========================================================================
    # Score with Gemini (if API key available)
    # ==========================================================================
    score = None  # Initialize score variable
    
    if GEMINI_API_KEY:
        print("\n" + "-" * 70)
        print("📊 GEMINI SCORING")
        print("-" * 70)
        
        score = score_with_gemini(
            question=question,
            expected=expected_answer or "Not provided",
            rag_answer=rag_result['answer'],
            kg_answer=kg_result['answer']
        )
        
        if score:
            print(f"\n🏆 WINNER: {score.get('winner', 'Unknown')}")
            print(f"\n📝 Explanation: {score.get('explanation', 'N/A')}")
            print(f"\n📈 Scores:")
            print(f"   RAG-only:  {score.get('system_a_scores', {}).get('total', 'N/A')} / 15")
            print(f"   RAG+KG:    {score.get('system_b_scores', {}).get('total', 'N/A')} / 15")
    else:
        print("\n⚠️  GEMINI_API_KEY not set - skipping auto-scoring")
        print("   Add to .env: GEMINI_API_KEY=your-key-here")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    results = {
        "question": question,
        "expected_answer": expected_answer,
        "rag_only": {
            "answer": rag_result['answer'],
            "context_length": len(rag_result.get('rag_context', '')),
            "context": rag_result.get('rag_context', '')
        },
        "rag_kg": {
            "answer": kg_result['answer'],
            "kg_context_length": len(kg_result.get('kg_context', '')),
            "rag_context_length": len(kg_result.get('rag_context', '')),
            "kg_context": kg_result.get('kg_context', ''),
            "rag_context": kg_result.get('rag_context', '')
        },
        "scoring": None,
        "summary": None
    }
    
    # Add scoring results if available
    if GEMINI_API_KEY and score:
        results["scoring"] = {
            "winner": score.get('winner', 'Unknown'),
            "explanation": score.get('explanation', 'N/A'),
            "rag_only_scores": score.get('system_a_scores', {}),
            "rag_kg_scores": score.get('system_b_scores', {})
        }
        
        # Create summary
        rag_total = score.get('system_a_scores', {}).get('total', 0)
        kg_total = score.get('system_b_scores', {}).get('total', 0)
        
        results["summary"] = {
            "winner": score.get('winner', 'Unknown'),
            "winner_system": "RAG+KG" if score.get('winner') == 'B' else "RAG-only" if score.get('winner') == 'A' else "Tie",
            "rag_only_total_score": rag_total,
            "rag_kg_total_score": kg_total,
            "score_difference": kg_total - rag_total,
            "rag_only_breakdown": {
                "accuracy": score.get('system_a_scores', {}).get('accuracy', 0),
                "completeness": score.get('system_a_scores', {}).get('completeness', 0),
                "relevance": score.get('system_a_scores', {}).get('relevance', 0)
            },
            "rag_kg_breakdown": {
                "accuracy": score.get('system_b_scores', {}).get('accuracy', 0),
                "completeness": score.get('system_b_scores', {}).get('completeness', 0),
                "relevance": score.get('system_b_scores', {}).get('relevance', 0)
            },
            "explanation": score.get('explanation', 'N/A')
        }
        
        # Print detailed summary
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        SCORING BREAKDOWN                            │
├─────────────────────────────────────────────────────────────────────┤
│  Metric          │  RAG-only  │  RAG+KG   │  Difference             │
├─────────────────────────────────────────────────────────────────────┤
│  Accuracy        │     {results['summary']['rag_only_breakdown']['accuracy']}      │     {results['summary']['rag_kg_breakdown']['accuracy']}     │     {results['summary']['rag_kg_breakdown']['accuracy'] - results['summary']['rag_only_breakdown']['accuracy']:+d}                  │
│  Completeness    │     {results['summary']['rag_only_breakdown']['completeness']}      │     {results['summary']['rag_kg_breakdown']['completeness']}     │     {results['summary']['rag_kg_breakdown']['completeness'] - results['summary']['rag_only_breakdown']['completeness']:+d}                  │
│  Relevance       │     {results['summary']['rag_only_breakdown']['relevance']}      │     {results['summary']['rag_kg_breakdown']['relevance']}     │     {results['summary']['rag_kg_breakdown']['relevance'] - results['summary']['rag_only_breakdown']['relevance']:+d}                  │
├─────────────────────────────────────────────────────────────────────┤
│  TOTAL           │    {rag_total:>2}/15   │   {kg_total:>2}/15   │     {kg_total - rag_total:+d}                  │
├─────────────────────────────────────────────────────────────────────┤
│  🏆 WINNER: {results['summary']['winner_system']:<20}                              │
└─────────────────────────────────────────────────────────────────────┘
        """)
    
    # Save to file
    output_file = "demo_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Cleanup
    trendscout.close()
    
    return results


def score_with_gemini(question: str, expected: str, rag_answer: str, kg_answer: str) -> dict:
    """Score answers using NEW Gemini API (google-genai SDK)"""
    
    prompt = f"""You are evaluating two AI system responses for the same question.

QUESTION: {question}

EXPECTED ANSWER: {expected}

SYSTEM A (RAG-only) ANSWER:
{rag_answer}

SYSTEM B (RAG+KG) ANSWER:
{kg_answer}

Score each answer from 1-5 on these criteria:
1. **Accuracy**: Does the answer match the expected answer? (1=wrong, 5=exactly right)
2. **Completeness**: Does it cover all relevant information? (1=missing most, 5=comprehensive)
3. **Relevance**: Is the answer focused on the question? (1=off-topic, 5=perfectly relevant)

Respond in this exact JSON format:
{{
    "system_a_scores": {{
        "accuracy": <1-5>,
        "completeness": <1-5>,
        "relevance": <1-5>,
        "total": <3-15>
    }},
    "system_b_scores": {{
        "accuracy": <1-5>,
        "completeness": <1-5>,
        "relevance": <1-5>,
        "total": <3-15>
    }},
    "winner": "<A|B|tie>",
    "explanation": "<brief explanation of why one is better>"
}}

Only output the JSON, nothing else."""

    try:
        # NEW: Use google-genai SDK
        from google import genai
        
        # Create client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Generate content using latest model
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Latest model as of 2025
            contents=prompt
        )
        
        print(f"   ✅ Gemini API call successful")
        
        # Get response text
        text = response.text
        
        # Clean JSON response
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        return json.loads(text.strip())
            
    except ImportError:
        print("   ❌ google-genai not installed. Run: pip install google-genai")
        return None
    except Exception as e:
        print(f"   ❌ Gemini Error: {e}")
        return None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test single question on both systems")
    parser.add_argument("--question", type=str, 
                        default="What products does Anthropic offer?",
                        help="Question to test")
    parser.add_argument("--expected", type=str,
                        default="Claude, Claude Code, Claude for Financial Services, Claude for Life Sciences, Claude for Education, Claude Pro, Artifacts",
                        help="Expected answer")
    args = parser.parse_args()
    
    test_single_question(args.question, args.expected)