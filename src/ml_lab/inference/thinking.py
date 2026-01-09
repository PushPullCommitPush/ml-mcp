"""
Deep thinking model analysis for ML Lab.

Uses Ollama reasoning models (DeepSeek R1, QwQ, etc.) to provide
thorough analysis of training runs, experiments, and activity.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .ollama import OllamaClient


class AnalysisType(str, Enum):
    """Types of analysis available."""
    TRAINING = "training"  # Analyze training run (loss, convergence, etc.)
    EXPERIMENT = "experiment"  # Compare experiments, suggest improvements
    ACTIVITY = "activity"  # Review audit log for patterns/anomalies
    COST = "cost"  # Cost efficiency analysis
    DATASET = "dataset"  # Dataset quality analysis
    GENERAL = "general"  # General query with context


class ScheduleFrequency(str, Enum):
    """Schedule frequencies for automated analysis."""
    AFTER_TRAINING = "after_training"  # After each training run
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class AnalysisReport:
    """Result from a thinking analysis."""
    report_id: str
    analysis_type: AnalysisType
    model_used: str
    timestamp: datetime
    input_summary: str
    analysis: str
    recommendations: list[str]
    thinking_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "analysis_type": self.analysis_type.value,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "input_summary": self.input_summary,
            "analysis": self.analysis,
            "recommendations": self.recommendations,
            "thinking_time_seconds": self.thinking_time_seconds,
        }


@dataclass
class ScheduledAnalysis:
    """A scheduled analysis configuration."""
    schedule_id: str
    analysis_type: AnalysisType
    frequency: ScheduleFrequency
    model: str
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "analysis_type": self.analysis_type.value,
            "frequency": self.frequency.value,
            "model": self.model,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "config": self.config,
        }


class ThinkingAnalyzer:
    """
    Deep thinking analysis using Ollama reasoning models.

    Provides thorough analysis of ML training, experiments, and activity
    using models optimized for reasoning (DeepSeek R1, QwQ, etc.).
    """

    DEFAULT_MODEL = "deepseek-r1:latest"
    REPORTS_DIR = Path.home() / ".cache" / "ml-lab" / "reports"
    SCHEDULES_FILE = Path.home() / ".config" / "ml-lab" / "thinking_schedules.json"

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        default_model: str | None = None,
    ):
        self._ollama = ollama_client or OllamaClient()
        self._default_model = default_model or self.DEFAULT_MODEL
        self._schedules: dict[str, ScheduledAnalysis] = {}
        self._scheduler_task: asyncio.Task | None = None

        # Ensure directories exist
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.SCHEDULES_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing schedules
        self._load_schedules()

    def _load_schedules(self) -> None:
        """Load scheduled analyses from disk."""
        if self.SCHEDULES_FILE.exists():
            try:
                data = json.loads(self.SCHEDULES_FILE.read_text())
                for item in data:
                    schedule = ScheduledAnalysis(
                        schedule_id=item["schedule_id"],
                        analysis_type=AnalysisType(item["analysis_type"]),
                        frequency=ScheduleFrequency(item["frequency"]),
                        model=item["model"],
                        enabled=item.get("enabled", True),
                        last_run=datetime.fromisoformat(item["last_run"]) if item.get("last_run") else None,
                        next_run=datetime.fromisoformat(item["next_run"]) if item.get("next_run") else None,
                        config=item.get("config", {}),
                    )
                    self._schedules[schedule.schedule_id] = schedule
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_schedules(self) -> None:
        """Save scheduled analyses to disk."""
        data = [s.to_dict() for s in self._schedules.values()]
        self.SCHEDULES_FILE.write_text(json.dumps(data, indent=2))

    def _generate_report_id(self) -> str:
        """Generate a unique report ID."""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:12]

    def _generate_schedule_id(self) -> str:
        """Generate a unique schedule ID."""
        import hashlib
        timestamp = datetime.now().isoformat()
        return f"sched_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"

    def _save_report(self, report: AnalysisReport) -> Path:
        """Save a report to disk."""
        report_path = self.REPORTS_DIR / f"{report.report_id}.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        return report_path

    def _build_training_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for training analysis."""
        return f"""Analyze this ML training run in detail. Think through each aspect carefully.

Training Configuration:
{json.dumps(context.get('config', {}), indent=2)}

Training Logs (recent):
```
{context.get('logs', 'No logs available')}
```

Metrics:
{json.dumps(context.get('metrics', {}), indent=2)}

Status: {context.get('status', 'unknown')}
Error (if any): {context.get('error', 'None')}

Provide a thorough analysis covering:
1. Training Progress - Is it converging? Any concerning patterns?
2. Loss Analysis - Is the loss curve healthy? Signs of overfitting/underfitting?
3. Hyperparameter Assessment - Are the settings appropriate for this model/data?
4. Resource Efficiency - Memory usage, training speed observations
5. Recommendations - Specific actionable improvements

End with a bullet list of concrete recommendations."""

    def _build_experiment_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for experiment comparison."""
        experiments = context.get('experiments', [])
        exp_text = "\n\n".join([
            f"Experiment: {exp.get('name', 'unnamed')}\n"
            f"Config: {json.dumps(exp.get('config', {}), indent=2)}\n"
            f"Metrics: {json.dumps(exp.get('metrics', {}), indent=2)}\n"
            f"Status: {exp.get('status', 'unknown')}"
            for exp in experiments
        ])

        return f"""Compare and analyze these ML experiments. Think through the differences carefully.

{exp_text}

Provide a thorough analysis covering:
1. Performance Comparison - Which experiment performed best and why?
2. Configuration Differences - What settings led to different outcomes?
3. Trade-offs - Speed vs quality, memory vs performance
4. Statistical Significance - Are the differences meaningful?
5. Recommendations - What configuration would you recommend and why?

End with a bullet list of concrete recommendations for the next experiment."""

    def _build_activity_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for activity/audit analysis."""
        return f"""Analyze this ML Lab activity log for patterns, anomalies, and insights.

Recent Activity (Audit Log):
```
{context.get('audit_log', 'No audit log available')}
```

Time Period: {context.get('time_period', 'unknown')}

Provide a thorough analysis covering:
1. Activity Patterns - What are the main activities? Any unusual patterns?
2. Security Review - Any concerning access patterns or failures?
3. Efficiency Analysis - Are operations being done efficiently?
4. Error Patterns - Recurring issues or failures?
5. Recommendations - Security improvements, workflow optimizations

End with a bullet list of concrete recommendations."""

    def _build_cost_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for cost analysis."""
        return f"""Analyze the cost efficiency of this ML training activity.

Training Runs:
{json.dumps(context.get('runs', []), indent=2)}

Provider Usage:
{json.dumps(context.get('provider_usage', {}), indent=2)}

Total Spend: {context.get('total_spend', 'unknown')}
Time Period: {context.get('time_period', 'unknown')}

Provide a thorough analysis covering:
1. Cost Breakdown - Where is money being spent?
2. Efficiency Analysis - Cost per training run, cost per epoch
3. Provider Comparison - Could cheaper providers have been used?
4. Optimization Opportunities - Spot instances, shorter runs, better configs
5. ROI Assessment - Are the results worth the cost?

End with a bullet list of concrete cost-saving recommendations."""

    def _build_dataset_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for dataset analysis."""
        return f"""Analyze this ML dataset for quality and training suitability.

Dataset Info:
{json.dumps(context.get('dataset_info', {}), indent=2)}

Sample Data:
```
{context.get('samples', 'No samples available')}
```

Statistics:
{json.dumps(context.get('statistics', {}), indent=2)}

Training Issues (if any):
{context.get('training_issues', 'None reported')}

Provide a thorough analysis covering:
1. Data Quality - Completeness, consistency, formatting issues
2. Distribution Analysis - Class balance, length distributions
3. Content Review - Any problematic patterns in the samples?
4. Training Suitability - Is this data appropriate for the intended model?
5. Preprocessing Recommendations - Cleaning, augmentation, splitting

End with a bullet list of concrete data improvement recommendations."""

    def _build_general_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for general analysis."""
        query = context.get('query', 'Provide general analysis')
        additional_context = context.get('context', '')

        return f"""Analyze the following and provide thorough insights.

Query: {query}

Context:
{additional_context}

Think through this carefully and provide:
1. Direct answer to the query
2. Supporting analysis and reasoning
3. Relevant considerations or caveats
4. Recommendations based on your analysis

End with a bullet list of concrete recommendations."""

    async def analyze(
        self,
        analysis_type: AnalysisType | str,
        context: dict[str, Any],
        model: str | None = None,
        store_report: bool = True,
    ) -> AnalysisReport:
        """
        Run a deep thinking analysis.

        Args:
            analysis_type: Type of analysis to perform.
            context: Context data for the analysis.
            model: Ollama model to use (default: deepseek-r1:latest).
            store_report: Whether to save the report to disk.

        Returns:
            AnalysisReport with the analysis results.
        """
        if isinstance(analysis_type, str):
            analysis_type = AnalysisType(analysis_type)

        model = model or self._default_model

        # Build appropriate prompt
        prompt_builders = {
            AnalysisType.TRAINING: self._build_training_prompt,
            AnalysisType.EXPERIMENT: self._build_experiment_prompt,
            AnalysisType.ACTIVITY: self._build_activity_prompt,
            AnalysisType.COST: self._build_cost_prompt,
            AnalysisType.DATASET: self._build_dataset_prompt,
            AnalysisType.GENERAL: self._build_general_prompt,
        }

        prompt = prompt_builders[analysis_type](context)

        # Run the analysis
        start_time = datetime.now()

        response = await self._ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        thinking_time = (datetime.now() - start_time).total_seconds()

        # Parse response
        analysis_text = response.get("message", {}).get("content", "")

        # Extract recommendations (look for bullet points at the end)
        recommendations = []
        lines = analysis_text.split("\n")
        in_recommendations = False
        for line in lines:
            line = line.strip()
            if "recommendation" in line.lower() and ":" in line:
                in_recommendations = True
                continue
            if in_recommendations and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                recommendations.append(line.lstrip("-•* "))

        # Create report
        report = AnalysisReport(
            report_id=self._generate_report_id(),
            analysis_type=analysis_type,
            model_used=model,
            timestamp=datetime.now(),
            input_summary=self._summarize_context(context),
            analysis=analysis_text,
            recommendations=recommendations,
            thinking_time_seconds=thinking_time,
        )

        if store_report:
            self._save_report(report)

        return report

    def _summarize_context(self, context: dict[str, Any]) -> str:
        """Create a brief summary of the input context."""
        parts = []
        if "config" in context:
            parts.append(f"config with {len(context['config'])} keys")
        if "logs" in context:
            parts.append(f"logs ({len(context['logs'])} chars)")
        if "experiments" in context:
            parts.append(f"{len(context['experiments'])} experiments")
        if "query" in context:
            parts.append(f"query: {context['query'][:50]}...")
        return ", ".join(parts) if parts else "minimal context"

    def schedule(
        self,
        analysis_type: AnalysisType | str,
        frequency: ScheduleFrequency | str,
        model: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> ScheduledAnalysis:
        """
        Schedule an automated analysis.

        Args:
            analysis_type: Type of analysis to schedule.
            frequency: How often to run.
            model: Ollama model to use.
            config: Additional configuration.

        Returns:
            ScheduledAnalysis configuration.
        """
        if isinstance(analysis_type, str):
            analysis_type = AnalysisType(analysis_type)
        if isinstance(frequency, str):
            frequency = ScheduleFrequency(frequency)

        schedule_id = self._generate_schedule_id()

        # Calculate next run time
        next_run = self._calculate_next_run(frequency)

        schedule = ScheduledAnalysis(
            schedule_id=schedule_id,
            analysis_type=analysis_type,
            frequency=frequency,
            model=model or self._default_model,
            enabled=True,
            next_run=next_run,
            config=config or {},
        )

        self._schedules[schedule_id] = schedule
        self._save_schedules()

        return schedule

    def _calculate_next_run(self, frequency: ScheduleFrequency) -> datetime:
        """Calculate the next run time for a frequency."""
        now = datetime.now()

        if frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == ScheduleFrequency.DAILY:
            # Next day at 6 AM
            next_day = now + timedelta(days=1)
            return next_day.replace(hour=6, minute=0, second=0, microsecond=0)
        elif frequency == ScheduleFrequency.WEEKLY:
            # Next Monday at 6 AM
            days_until_monday = (7 - now.weekday()) % 7 or 7
            next_monday = now + timedelta(days=days_until_monday)
            return next_monday.replace(hour=6, minute=0, second=0, microsecond=0)
        else:
            # AFTER_TRAINING - no scheduled time, triggered manually
            return now

    def unschedule(self, schedule_id: str) -> bool:
        """Remove a scheduled analysis."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            self._save_schedules()
            return True
        return False

    def list_schedules(self) -> list[ScheduledAnalysis]:
        """List all scheduled analyses."""
        return list(self._schedules.values())

    def get_schedule(self, schedule_id: str) -> ScheduledAnalysis | None:
        """Get a specific schedule."""
        return self._schedules.get(schedule_id)

    def toggle_schedule(self, schedule_id: str, enabled: bool) -> bool:
        """Enable or disable a schedule."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id].enabled = enabled
            self._save_schedules()
            return True
        return False

    def list_reports(
        self,
        analysis_type: AnalysisType | str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List stored reports."""
        reports = []

        for report_path in sorted(self.REPORTS_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(report_path.read_text())
                if analysis_type:
                    type_str = analysis_type.value if isinstance(analysis_type, AnalysisType) else analysis_type
                    if data.get("analysis_type") != type_str:
                        continue
                reports.append(data)
                if len(reports) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        return reports

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Get a specific report."""
        report_path = self.REPORTS_DIR / f"{report_id}.json"
        if report_path.exists():
            try:
                return json.loads(report_path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    async def trigger_after_training(self, run_id: str, context: dict[str, Any]) -> AnalysisReport | None:
        """
        Trigger after-training analysis if scheduled.

        Called by train_status when a run completes.
        """
        for schedule in self._schedules.values():
            if (
                schedule.enabled
                and schedule.frequency == ScheduleFrequency.AFTER_TRAINING
                and schedule.analysis_type == AnalysisType.TRAINING
            ):
                context["run_id"] = run_id
                report = await self.analyze(
                    AnalysisType.TRAINING,
                    context,
                    model=schedule.model,
                )

                # Update schedule
                schedule.last_run = datetime.now()
                self._save_schedules()

                return report

        return None


# Singleton instance
_analyzer: ThinkingAnalyzer | None = None


def get_thinking_analyzer() -> ThinkingAnalyzer:
    """Get or create the global thinking analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ThinkingAnalyzer()
    return _analyzer
