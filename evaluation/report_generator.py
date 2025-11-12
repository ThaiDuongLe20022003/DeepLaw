"""
Evaluation report generation and formatting.
"""

from datetime import datetime
from typing import Dict, Any, List

class ReportGenerator:
    """Generates comprehensive evaluation reports"""
    
    @staticmethod
    def generate_report(summary: Dict[str, Any], model_name: str = "N/A") -> str:
        """Generate a comprehensive evaluation report"""
        if not summary:
            return "No metrics collected yet."
        
        report_lines = [
            "=== MULTI-JUDGE EVALUATION REPORT ===",
            f"Session ID: {summary.get('session_id', 'N/A')}",
            f"Total Interactions: {summary['total_interactions']}",
            f"Total Evaluations: {summary['total_evaluations']}",
            f"Unique Judge Models: {summary['unique_judges']}",
            f"Average Response Time: {summary['avg_response_time']}s",
            f"Total Tokens Generated: {summary['total_tokens_generated']}",
            f"Average Throughput: {summary['avg_tokens_per_second']} tokens/s",
        ]
        
        if 'avg_overall_score' in summary:
            report_lines.extend(ReportGenerator._generate_quality_section(summary))
        
        report_lines.extend(ReportGenerator._generate_footer(model_name))
        
        return "\n".join(report_lines)
    
    @staticmethod
    def _generate_quality_section(summary: Dict[str, Any]) -> List[str]:
        """Generate the quality evaluation section of the report"""
        section = [
            "",
            "=== OVERALL QUALITY EVALUATION (0.0-10.0 scale) ===",
            f"Overall Quality: {summary['avg_overall_score']}/10.0",
            f"Faithfulness: {summary['avg_faithfulness']}/10.0 (reliance on context)",
            f"Groundedness: {summary['avg_groundedness']}/10.0 (traceability to context)",
            f"Factual Consistency: {summary['avg_factual_consistency']}/10.0 (accuracy vs context)",
            f"Relevance: {summary['avg_relevance']}/10.0 (addresses query)",
            f"Completeness: {summary['avg_completeness']}/10.0 (covers all aspects)",
            f"Fluency: {summary['avg_fluency']}/10.0 (natural language)",
        ]
        
        if 'rating_distribution' in summary:
            section.extend(ReportGenerator._generate_rating_section(summary['rating_distribution']))
        
        if 'judge_models' in summary:
            section.extend(ReportGenerator._generate_judge_section(summary['judge_models']))
        
        return section
    
    @staticmethod
    def _generate_rating_section(rating_distribution: Dict[str, int]) -> List[str]:
        """Generate rating distribution section"""
        return [
            "",
            "=== RATING DISTRIBUTION ===",
            f"Excellent (9.0-10.0): {rating_distribution['Excellent']} evaluation(s)",
            f"Good (8.0-8.9): {rating_distribution['Good']} evaluation(s)",
            f"Fair (6.5-7.9): {rating_distribution['Fair']} evaluation(s)",
            f"Average (5.0-6.4): {rating_distribution['Average']} evaluation(s)",
            f"Poor / Weak (<5.0): {rating_distribution['Poor / Weak']} evaluation(s)",
        ]
    
    @staticmethod
    def _generate_judge_section(judge_models: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate judge model evaluation section"""
        section = [
            "",
            "=== EVALUATION BY JUDGE MODEL ===",
        ]
        for judge, stats in judge_models.items():
            section.append(f"{judge}: {stats['avg_score']}/10.0 ({stats['count']} evaluation(s))")
        return section
    
    @staticmethod
    def _generate_footer(model_name: str) -> List[str]:
        """Generate report footer"""
        return [
            "",
            "RATING SCALE:",
            "9.0 – 10.0: Excellent",
            "8.0 – 8.9: Good", 
            "6.5 – 7.9: Fair",
            "5.0 – 6.4: Average",
            "< 5.0: Poor / Weak",
            "",
            f"Response Model: {model_name}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "========================================="
        ]