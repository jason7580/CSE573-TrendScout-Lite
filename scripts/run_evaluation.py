"""
TrendScout AI Evaluation Pipeline

This script:
1. Loads 50 evaluation questions
2. Tests RAG-only system
3. Tests RAG+KG system
4. Sends answers to Gemini for scoring
5. Generates comparison report

Requirements:
    pip install google-genai python-dotenv

Usage:
    python run_evaluation.py --questions eval_questions.json --output ./eval_results
"""

import os
import json
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

import sys
sys.path.append('.')

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =============================================================================
# Gemini Scorer using NEW SDK
# =============================================================================

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
        from google import genai
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        text = response.text.strip()
        
        # Clean JSON response
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
# Evaluation Runner
# =============================================================================

class EvaluationRunner:
    """Run evaluation comparing RAG-only vs RAG+KG"""
    
    def __init__(self, questions_file: str):
        self.questions = self._load_questions(questions_file)
    
    def _load_questions(self, filepath: str) -> list:
        """Load evaluation questions from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'evaluation_questions' in data:
            return data['evaluation_questions']
        return data
    
    def run_rag_only(self, trendscout) -> list:
        """Run all questions through RAG-only system"""
        print("\n" + "="*60)
        print("🔍 Running RAG-ONLY evaluation...")
        print("="*60)
        
        answers = []
        for i, q in enumerate(self.questions, 1):
            print(f"  [{i}/{len(self.questions)}] {q['question'][:50]}...")
            
            result = trendscout.ask(
                question=q['question'],
                use_kg=False,
                use_rag=True
            )
            
            answers.append({
                "question_id": q['id'],
                "answer": result['answer'],
                "rag_context": result.get('rag_context', '')
            })
            
            time.sleep(0.3)
        
        print(f"✅ Completed {len(answers)} RAG-only answers")
        return answers
    
    def run_rag_kg(self, trendscout) -> list:
        """Run all questions through RAG+KG system"""
        print("\n" + "="*60)
        print("🔗 Running RAG+KG evaluation...")
        print("="*60)
        
        answers = []
        for i, q in enumerate(self.questions, 1):
            print(f"  [{i}/{len(self.questions)}] {q['question'][:50]}...")
            
            result = trendscout.ask(
                question=q['question'],
                use_kg=True,
                use_rag=True
            )
            
            answers.append({
                "question_id": q['id'],
                "answer": result['answer'],
                "kg_context": result.get('kg_context', ''),
                "rag_context": result.get('rag_context', '')
            })
            
            time.sleep(0.3)
        
        print(f"✅ Completed {len(answers)} RAG+KG answers")
        return answers
    
    def score_all(self, rag_answers: list, kg_answers: list) -> list:
        """Score all answers with Gemini"""
        print("\n" + "="*60)
        print("📊 Scoring with Gemini...")
        print("="*60)
        
        scores = []
        
        for i, q in enumerate(self.questions):
            rag_answer = rag_answers[i]['answer']
            kg_answer = kg_answers[i]['answer']
            expected = q.get('expected_answer', 'N/A')
            
            print(f"  [{i+1}/{len(self.questions)}] Scoring: {q['question'][:40]}...")
            
            score = score_with_gemini(
                question=q['question'],
                expected=expected,
                rag_answer=rag_answer,
                kg_answer=kg_answer
            )
            
            if score:
                score['question_id'] = q['id']
                score['question'] = q['question']
                score['category'] = q.get('category', 'unknown')
                scores.append(score)
                print(f"       Winner: {score.get('winner', '?')}")
            else:
                scores.append({
                    "question_id": q['id'],
                    "question": q['question'],
                    "category": q.get('category', 'unknown'),
                    "error": "Scoring failed"
                })
                print(f"       ❌ Scoring failed")
            
            time.sleep(1)  # Rate limiting
        
        return scores
    
    def generate_report(self, rag_answers: list, kg_answers: list, scores: list) -> dict:
        """Generate evaluation report with detailed scoring breakdown"""
        print("\n" + "="*60)
        print("📈 Generating Report...")
        print("="*60)
        
        valid_scores = [s for s in scores if 'error' not in s]
        rag_wins = sum(1 for s in valid_scores if s.get('winner') == 'A')
        kg_wins = sum(1 for s in valid_scores if s.get('winner') == 'B')
        ties = sum(1 for s in valid_scores if s.get('winner') == 'tie')
        errors = sum(1 for s in scores if 'error' in s)
        
        # Calculate totals
        avg_rag_total = sum(s['system_a_scores']['total'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        avg_kg_total = sum(s['system_b_scores']['total'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        
        # Calculate average for each metric
        avg_rag_accuracy = sum(s['system_a_scores']['accuracy'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        avg_rag_completeness = sum(s['system_a_scores']['completeness'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        avg_rag_relevance = sum(s['system_a_scores']['relevance'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        
        avg_kg_accuracy = sum(s['system_b_scores']['accuracy'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        avg_kg_completeness = sum(s['system_b_scores']['completeness'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        avg_kg_relevance = sum(s['system_b_scores']['relevance'] for s in valid_scores) / len(valid_scores) if valid_scores else 0
        
        # Scores by category
        category_results = {}
        for s in valid_scores:
            cat = s.get('category', 'unknown')
            if cat not in category_results:
                category_results[cat] = {
                    'rag_wins': 0, 'kg_wins': 0, 'ties': 0,
                    'rag_scores': [], 'kg_scores': [],
                    'rag_accuracy': [], 'rag_completeness': [], 'rag_relevance': [],
                    'kg_accuracy': [], 'kg_completeness': [], 'kg_relevance': []
                }
            
            r = category_results[cat]
            r['rag_scores'].append(s['system_a_scores']['total'])
            r['kg_scores'].append(s['system_b_scores']['total'])
            r['rag_accuracy'].append(s['system_a_scores']['accuracy'])
            r['rag_completeness'].append(s['system_a_scores']['completeness'])
            r['rag_relevance'].append(s['system_a_scores']['relevance'])
            r['kg_accuracy'].append(s['system_b_scores']['accuracy'])
            r['kg_completeness'].append(s['system_b_scores']['completeness'])
            r['kg_relevance'].append(s['system_b_scores']['relevance'])
            
            if s['winner'] == 'A':
                r['rag_wins'] += 1
            elif s['winner'] == 'B':
                r['kg_wins'] += 1
            else:
                r['ties'] += 1
        
        # Calculate category averages
        for cat in category_results:
            r = category_results[cat]
            r['rag_avg'] = round(sum(r['rag_scores']) / len(r['rag_scores']), 2) if r['rag_scores'] else 0
            r['kg_avg'] = round(sum(r['kg_scores']) / len(r['kg_scores']), 2) if r['kg_scores'] else 0
            r['rag_accuracy_avg'] = round(sum(r['rag_accuracy']) / len(r['rag_accuracy']), 2) if r['rag_accuracy'] else 0
            r['rag_completeness_avg'] = round(sum(r['rag_completeness']) / len(r['rag_completeness']), 2) if r['rag_completeness'] else 0
            r['rag_relevance_avg'] = round(sum(r['rag_relevance']) / len(r['rag_relevance']), 2) if r['rag_relevance'] else 0
            r['kg_accuracy_avg'] = round(sum(r['kg_accuracy']) / len(r['kg_accuracy']), 2) if r['kg_accuracy'] else 0
            r['kg_completeness_avg'] = round(sum(r['kg_completeness']) / len(r['kg_completeness']), 2) if r['kg_completeness'] else 0
            r['kg_relevance_avg'] = round(sum(r['kg_relevance']) / len(r['kg_relevance']), 2) if r['kg_relevance'] else 0
            # Clean up raw lists
            del r['rag_scores'], r['kg_scores']
            del r['rag_accuracy'], r['rag_completeness'], r['rag_relevance']
            del r['kg_accuracy'], r['kg_completeness'], r['kg_relevance']
        
        total = len(self.questions)
        report = {
            "summary": {
                "total_questions": total,
                "scored_questions": len(valid_scores),
                "rag_only_wins": rag_wins,
                "rag_kg_wins": kg_wins,
                "ties": ties,
                "errors": errors,
                "rag_win_pct": round(rag_wins / len(valid_scores) * 100, 1) if valid_scores else 0,
                "kg_win_pct": round(kg_wins / len(valid_scores) * 100, 1) if valid_scores else 0,
                "avg_rag_score": round(avg_rag_total, 2),
                "avg_kg_score": round(avg_kg_total, 2),
                "score_difference": round(avg_kg_total - avg_rag_total, 2),
                "winner": "RAG+KG" if kg_wins > rag_wins else "RAG-only" if rag_wins > kg_wins else "Tie"
            },
            "scoring_breakdown": {
                "rag_only": {
                    "avg_total": round(avg_rag_total, 2),
                    "avg_accuracy": round(avg_rag_accuracy, 2),
                    "avg_completeness": round(avg_rag_completeness, 2),
                    "avg_relevance": round(avg_rag_relevance, 2)
                },
                "rag_kg": {
                    "avg_total": round(avg_kg_total, 2),
                    "avg_accuracy": round(avg_kg_accuracy, 2),
                    "avg_completeness": round(avg_kg_completeness, 2),
                    "avg_relevance": round(avg_kg_relevance, 2)
                },
                "difference": {
                    "total": round(avg_kg_total - avg_rag_total, 2),
                    "accuracy": round(avg_kg_accuracy - avg_rag_accuracy, 2),
                    "completeness": round(avg_kg_completeness - avg_rag_completeness, 2),
                    "relevance": round(avg_kg_relevance - avg_rag_relevance, 2)
                }
            },
            "category_results": category_results,
            "detailed_scores": scores,
            "rag_answers": rag_answers,
            "kg_answers": kg_answers
        }
        
        # Print summary
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║               EVALUATION RESULTS SUMMARY                      ║
╠══════════════════════════════════════════════════════════════╣
║  Total Questions:     {total:>4}                                    ║
║  Successfully Scored: {len(valid_scores):>4}                                    ║
║  ───────────────────────────────────────────────────────────  ║
║  RAG-only Wins:       {rag_wins:>4}  ({report['summary']['rag_win_pct']:>5.1f}%)                       ║
║  RAG+KG Wins:         {kg_wins:>4}  ({report['summary']['kg_win_pct']:>5.1f}%)                       ║
║  Ties:                {ties:>4}                                    ║
║  ───────────────────────────────────────────────────────────  ║
║  Avg RAG-only Score:  {avg_rag_total:>5.2f} / 15                           ║
║  Avg RAG+KG Score:    {avg_kg_total:>5.2f} / 15                           ║
║  ───────────────────────────────────────────────────────────  ║
║  🏆 OVERALL WINNER:   {report['summary']['winner']:<15}                    ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Print scoring breakdown
        print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     SCORING BREAKDOWN (Averages)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Metric          │  RAG-only  │  RAG+KG   │  Difference             │
├─────────────────────────────────────────────────────────────────────┤""")
        print(f"│  Accuracy        │    {avg_rag_accuracy:>5.2f}   │   {avg_kg_accuracy:>5.2f}   │    {avg_kg_accuracy - avg_rag_accuracy:>+5.2f}                 │")
        print(f"│  Completeness    │    {avg_rag_completeness:>5.2f}   │   {avg_kg_completeness:>5.2f}   │    {avg_kg_completeness - avg_rag_completeness:>+5.2f}                 │")
        print(f"│  Relevance       │    {avg_rag_relevance:>5.2f}   │   {avg_kg_relevance:>5.2f}   │    {avg_kg_relevance - avg_rag_relevance:>+5.2f}                 │")
        print(f"├─────────────────────────────────────────────────────────────────────┤")
        print(f"│  TOTAL           │   {avg_rag_total:>5.2f}   │  {avg_kg_total:>5.2f}   │    {avg_kg_total - avg_rag_total:>+5.2f}                 │")
        print(f"└─────────────────────────────────────────────────────────────────────┘")
        
        # Print category breakdown
        print("\n📊 Results by Category:")
        print("-" * 85)
        print(f"{'Category':<15} | {'RAG Wins':<9} | {'KG Wins':<9} | {'Ties':<6} | {'RAG Avg':<8} | {'KG Avg':<8} | {'Diff':<6}")
        print("-" * 85)
        for cat, r in sorted(category_results.items()):
            diff = r['kg_avg'] - r['rag_avg']
            print(f"{cat:<15} | {r['rag_wins']:<9} | {r['kg_wins']:<9} | {r['ties']:<6} | {r['rag_avg']:<8} | {r['kg_avg']:<8} | {diff:>+5.2f}")
        print("-" * 85)
        
        return report
    
    def save_results(self, report: dict, output_dir: str):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full report
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full report saved: {report_file}")
        
        # Summary only (for slides)
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": report['summary'],
                "scoring_breakdown": report.get('scoring_breakdown', {}),
                "category_results": report['category_results']
            }, f, indent=2)
        print(f"💾 Summary saved: {summary_file}")
        
        # CSV for easy viewing (with all metrics)
        csv_file = os.path.join(output_dir, f"scores_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("question_id,category,winner,rag_total,kg_total,rag_accuracy,kg_accuracy,rag_completeness,kg_completeness,rag_relevance,kg_relevance,explanation\n")
            for s in report['detailed_scores']:
                if 'error' not in s:
                    rag = s['system_a_scores']
                    kg = s['system_b_scores']
                    expl = s.get('explanation', '').replace('"', "'").replace('\n', ' ')
                    f.write(f"{s['question_id']},{s['category']},{s['winner']},{rag['total']},{kg['total']},{rag['accuracy']},{kg['accuracy']},{rag['completeness']},{kg['completeness']},{rag['relevance']},{kg['relevance']},\"{expl}\"\n")
        print(f"💾 CSV saved: {csv_file}")
        
        # Markdown summary (for presentation)
        md_file = os.path.join(output_dir, f"summary_{timestamp}.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            s = report['summary']
            b = report.get('scoring_breakdown', {})
            
            f.write("# TrendScout AI Evaluation Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Questions | {s['total_questions']} |\n")
            f.write(f"| RAG-only Wins | {s['rag_only_wins']} ({s['rag_win_pct']}%) |\n")
            f.write(f"| RAG+KG Wins | {s['rag_kg_wins']} ({s['kg_win_pct']}%) |\n")
            f.write(f"| Ties | {s['ties']} |\n")
            f.write(f"| **Winner** | **{s['winner']}** |\n\n")
            
            if b:
                f.write("## Scoring Breakdown (Averages)\n\n")
                f.write(f"| Metric | RAG-only | RAG+KG | Difference |\n")
                f.write(f"|--------|----------|--------|------------|\n")
                f.write(f"| Accuracy | {b['rag_only']['avg_accuracy']} | {b['rag_kg']['avg_accuracy']} | {b['difference']['accuracy']:+.2f} |\n")
                f.write(f"| Completeness | {b['rag_only']['avg_completeness']} | {b['rag_kg']['avg_completeness']} | {b['difference']['completeness']:+.2f} |\n")
                f.write(f"| Relevance | {b['rag_only']['avg_relevance']} | {b['rag_kg']['avg_relevance']} | {b['difference']['relevance']:+.2f} |\n")
                f.write(f"| **Total** | **{b['rag_only']['avg_total']}** | **{b['rag_kg']['avg_total']}** | **{b['difference']['total']:+.2f}** |\n\n")
            
            f.write("## Results by Category\n\n")
            f.write(f"| Category | RAG Wins | KG Wins | Ties | RAG Avg | KG Avg |\n")
            f.write(f"|----------|----------|---------|------|---------|--------|\n")
            for cat, r in sorted(report['category_results'].items()):
                f.write(f"| {cat} | {r['rag_wins']} | {r['kg_wins']} | {r['ties']} | {r['rag_avg']} | {r['kg_avg']} |\n")
            
            f.write("\n## Key Findings\n\n")
            if s['winner'] == 'RAG+KG':
                f.write(f"1. **RAG+KG outperforms RAG-only** with {s['kg_win_pct']}% win rate\n")
                if b:
                    f.write(f"2. **Average score improvement:** {b['difference']['total']:+.2f} points (out of 15)\n")
                    # Find biggest improvement
                    diffs = {'Accuracy': b['difference']['accuracy'], 'Completeness': b['difference']['completeness'], 'Relevance': b['difference']['relevance']}
                    best = max(diffs, key=diffs.get)
                    f.write(f"3. **Biggest improvement in:** {best} ({diffs[best]:+.2f})\n")
            else:
                f.write(f"1. **{s['winner']}** performed better overall\n")
        
        print(f"💾 Markdown saved: {md_file}")
        
        return report_file


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TrendScout AI Evaluation Pipeline")
    parser.add_argument("--questions", type=str, required=True, help="Path to evaluation questions JSON")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--rag-data", type=str, help="Path to RAG data JSON (for loading fresh)")
    args = parser.parse_args()
    
    print("🚀 TrendScout AI Evaluation Pipeline")
    print("="*60)
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found!")
        print("   Add to .env: GEMINI_API_KEY=your-key-here")
        return
    
    # Import TrendScout
    try:
        from trendscout_app import TrendScoutAI
        print("✅ Using trendscout_app")
    except ImportError:
        try:
            from trendscout_app import TrendScoutAI
            print("✅ Using trendscout_app")
        except ImportError:
            print("❌ Could not import TrendScoutAI!")
            return
    
    trendscout = TrendScoutAI()
    trendscout.initialize(load_new_docs=args.rag_data)
    
    runner = EvaluationRunner(args.questions)
    print(f"✅ Loaded {len(runner.questions)} questions")
    
    # Run evaluations
    rag_answers = runner.run_rag_only(trendscout)
    kg_answers = runner.run_rag_kg(trendscout)
    
    # Score with Gemini
    scores = runner.score_all(rag_answers, kg_answers)
    
    # Generate report
    report = runner.generate_report(rag_answers, kg_answers, scores)
    
    # Save results
    runner.save_results(report, args.output)
    
    trendscout.close()
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()