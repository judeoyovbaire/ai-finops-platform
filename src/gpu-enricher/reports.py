"""
Chargeback Report Generation Module for AI FinOps Platform

Generates cost allocation reports in multiple formats:
- CSV for data analysis and import
- PDF for executive summaries and distribution

Reports include:
- Team cost breakdown
- GPU utilization metrics
- Optimization recommendations
- Budget vs actual comparison
- Trend analysis
"""

import csv
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Optional reportlab import for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        PageBreak,
    )

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("reportlab not available - PDF report generation disabled")


class ReportFormat(Enum):
    """Supported report formats."""

    CSV = "csv"
    PDF = "pdf"
    JSON = "json"


class ReportPeriod(Enum):
    """Report time periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class ReportMetadata:
    """Report metadata."""

    title: str
    period: ReportPeriod
    start_date: datetime
    end_date: datetime
    generated_at: datetime
    generated_by: str
    version: str = "1.0"


@dataclass
class TeamCostReport:
    """Cost report data for a team."""

    team: str
    gpu_cost: float
    k8s_cost: float
    total_cost: float
    gpu_hours: float
    gpu_count: int
    avg_utilization: float
    idle_hours: float
    spot_savings_potential: float
    budget: float
    budget_remaining: float
    cost_trend_pct: float  # vs previous period


class ChargebackReportGenerator:
    """Generates chargeback reports in various formats."""

    def __init__(self):
        self.styles = None
        if PDF_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for PDF."""
        if not self.styles:
            return

        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor("#1a1a2e"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.HexColor("#16213e"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SubHeader",
                parent=self.styles["Heading3"],
                fontSize=12,
                spaceBefore=10,
                spaceAfter=5,
            )
        )

    def generate_csv(
        self,
        team_reports: list[TeamCostReport],
        metadata: ReportMetadata,
    ) -> str:
        """
        Generate CSV report.

        Args:
            team_reports: List of team cost reports
            metadata: Report metadata

        Returns:
            CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header with metadata
        writer.writerow(["# AI FinOps Chargeback Report"])
        writer.writerow([f"# Period: {metadata.period.value}"])
        writer.writerow(
            [f"# Date Range: {metadata.start_date.date()} to {metadata.end_date.date()}"]
        )
        writer.writerow([f"# Generated: {metadata.generated_at.isoformat()}"])
        writer.writerow([])

        # Summary section
        writer.writerow(["## Summary"])
        total_cost = sum(r.total_cost for r in team_reports)
        total_gpu_cost = sum(r.gpu_cost for r in team_reports)
        total_k8s_cost = sum(r.k8s_cost for r in team_reports)
        total_savings_potential = sum(r.spot_savings_potential for r in team_reports)

        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Cost", f"${total_cost:.2f}"])
        writer.writerow(["GPU Cost", f"${total_gpu_cost:.2f}"])
        writer.writerow(["K8s Cost", f"${total_k8s_cost:.2f}"])
        writer.writerow(["Potential Spot Savings", f"${total_savings_potential:.2f}"])
        writer.writerow(["Teams", len(team_reports)])
        writer.writerow([])

        # Team breakdown
        writer.writerow(["## Team Cost Breakdown"])
        writer.writerow([
            "Team",
            "GPU Cost ($)",
            "K8s Cost ($)",
            "Total Cost ($)",
            "GPU Hours",
            "GPU Count",
            "Avg Utilization (%)",
            "Idle Hours",
            "Spot Savings Potential ($)",
            "Budget ($)",
            "Budget Remaining ($)",
            "Cost Trend (%)",
        ])

        for report in team_reports:
            writer.writerow([
                report.team,
                f"{report.gpu_cost:.2f}",
                f"{report.k8s_cost:.2f}",
                f"{report.total_cost:.2f}",
                f"{report.gpu_hours:.1f}",
                report.gpu_count,
                f"{report.avg_utilization:.1f}",
                f"{report.idle_hours:.1f}",
                f"{report.spot_savings_potential:.2f}",
                f"{report.budget:.2f}",
                f"{report.budget_remaining:.2f}",
                f"{report.cost_trend_pct:+.1f}",
            ])

        return output.getvalue()

    def generate_pdf(
        self,
        team_reports: list[TeamCostReport],
        metadata: ReportMetadata,
        recommendations: Optional[list[dict]] = None,
        anomalies: Optional[list[dict]] = None,
    ) -> bytes:
        """
        Generate PDF report.

        Args:
            team_reports: List of team cost reports
            metadata: Report metadata
            recommendations: Optional list of recommendations
            anomalies: Optional list of detected anomalies

        Returns:
            PDF content as bytes
        """
        if not PDF_AVAILABLE:
            raise RuntimeError("PDF generation requires reportlab package")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
        )

        elements = []

        # Title
        elements.append(
            Paragraph(metadata.title, self.styles["ReportTitle"])
        )
        elements.append(
            Paragraph(
                f"Period: {metadata.start_date.date()} to {metadata.end_date.date()}",
                self.styles["Normal"],
            )
        )
        elements.append(
            Paragraph(
                f"Generated: {metadata.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
                self.styles["Normal"],
            )
        )
        elements.append(Spacer(1, 20))

        # Executive Summary
        elements.append(
            Paragraph("Executive Summary", self.styles["SectionHeader"])
        )

        total_cost = sum(r.total_cost for r in team_reports)
        total_gpu_cost = sum(r.gpu_cost for r in team_reports)
        total_savings = sum(r.spot_savings_potential for r in team_reports)
        avg_utilization = (
            sum(r.avg_utilization * r.gpu_count for r in team_reports)
            / sum(r.gpu_count for r in team_reports)
            if team_reports
            else 0
        )

        summary_data = [
            ["Metric", "Value"],
            ["Total Cost", f"${total_cost:,.2f}"],
            ["GPU Cost", f"${total_gpu_cost:,.2f}"],
            ["Teams", str(len(team_reports))],
            ["Avg GPU Utilization", f"{avg_utilization:.1f}%"],
            ["Potential Savings", f"${total_savings:,.2f}"],
        ]

        summary_table = Table(summary_data, colWidths=[2.5 * inch, 2 * inch])
        summary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("TOPPADDING", (0, 1), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ])
        )
        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Team Cost Breakdown
        elements.append(
            Paragraph("Team Cost Breakdown", self.styles["SectionHeader"])
        )

        team_data = [
            ["Team", "Total Cost", "GPU Cost", "Utilization", "Budget Status"],
        ]
        for report in sorted(team_reports, key=lambda x: x.total_cost, reverse=True):
            budget_status = (
                "On Track"
                if report.budget_remaining > 0
                else f"Over (${abs(report.budget_remaining):.0f})"
            )
            team_data.append([
                report.team,
                f"${report.total_cost:,.2f}",
                f"${report.gpu_cost:,.2f}",
                f"{report.avg_utilization:.1f}%",
                budget_status,
            ])

        team_table = Table(
            team_data,
            colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 1 * inch, 1.2 * inch],
        )
        team_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ])
        )
        elements.append(team_table)
        elements.append(Spacer(1, 20))

        # Recommendations (if provided)
        if recommendations:
            elements.append(
                Paragraph("Optimization Recommendations", self.styles["SectionHeader"])
            )

            rec_data = [["Priority", "Type", "Team", "Potential Savings"]]
            for rec in recommendations[:10]:  # Top 10
                rec_data.append([
                    rec.get("severity", "medium").title(),
                    rec.get("type", "").replace("_", " ").title(),
                    rec.get("team", ""),
                    f"${rec.get('potential_savings_daily', 0) * 30:,.0f}/mo",
                ])

            rec_table = Table(
                rec_data,
                colWidths=[1 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch],
            )
            rec_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                ])
            )
            elements.append(rec_table)
            elements.append(Spacer(1, 20))

        # Anomalies (if provided)
        if anomalies:
            elements.append(
                Paragraph("Detected Anomalies", self.styles["SectionHeader"])
            )

            for anomaly in anomalies[:5]:  # Top 5
                severity_color = {
                    "critical": colors.red,
                    "warning": colors.orange,
                    "info": colors.blue,
                }.get(anomaly.get("severity", "info"), colors.grey)

                elements.append(
                    Paragraph(
                        f"<font color='{severity_color}'>[{anomaly.get('severity', 'info').upper()}]</font> "
                        f"{anomaly.get('description', '')}",
                        self.styles["Normal"],
                    )
                )
            elements.append(Spacer(1, 20))

        # Footer
        elements.append(
            Paragraph(
                f"Report Version: {metadata.version} | "
                f"Generated by: {metadata.generated_by}",
                self.styles["Normal"],
            )
        )

        doc.build(elements)
        return buffer.getvalue()

    def generate_report(
        self,
        team_reports: list[TeamCostReport],
        metadata: ReportMetadata,
        format: ReportFormat = ReportFormat.CSV,
        recommendations: Optional[list[dict]] = None,
        anomalies: Optional[list[dict]] = None,
    ) -> tuple[bytes, str]:
        """
        Generate report in specified format.

        Args:
            team_reports: List of team cost reports
            metadata: Report metadata
            format: Output format
            recommendations: Optional recommendations
            anomalies: Optional anomalies

        Returns:
            Tuple of (content_bytes, content_type)
        """
        if format == ReportFormat.CSV:
            content = self.generate_csv(team_reports, metadata)
            return content.encode("utf-8"), "text/csv"

        elif format == ReportFormat.PDF:
            content = self.generate_pdf(
                team_reports, metadata, recommendations, anomalies
            )
            return content, "application/pdf"

        elif format == ReportFormat.JSON:
            import json

            data = {
                "metadata": {
                    "title": metadata.title,
                    "period": metadata.period.value,
                    "start_date": metadata.start_date.isoformat(),
                    "end_date": metadata.end_date.isoformat(),
                    "generated_at": metadata.generated_at.isoformat(),
                },
                "summary": {
                    "total_cost": sum(r.total_cost for r in team_reports),
                    "total_gpu_cost": sum(r.gpu_cost for r in team_reports),
                    "total_k8s_cost": sum(r.k8s_cost for r in team_reports),
                    "team_count": len(team_reports),
                },
                "teams": [
                    {
                        "team": r.team,
                        "gpu_cost": r.gpu_cost,
                        "k8s_cost": r.k8s_cost,
                        "total_cost": r.total_cost,
                        "gpu_hours": r.gpu_hours,
                        "gpu_count": r.gpu_count,
                        "avg_utilization": r.avg_utilization,
                        "idle_hours": r.idle_hours,
                        "spot_savings_potential": r.spot_savings_potential,
                        "budget": r.budget,
                        "budget_remaining": r.budget_remaining,
                        "cost_trend_pct": r.cost_trend_pct,
                    }
                    for r in team_reports
                ],
                "recommendations": recommendations or [],
                "anomalies": anomalies or [],
            }
            return json.dumps(data, indent=2).encode("utf-8"), "application/json"

        raise ValueError(f"Unsupported format: {format}")


# Global report generator instance
report_generator = ChargebackReportGenerator()


def generate_monthly_chargeback_report(
    team_data: dict,
    recommendations: list[dict] = None,
    anomalies: list[dict] = None,
    format: str = "csv",
) -> tuple[bytes, str, str]:
    """
    Convenience function to generate monthly chargeback report.

    Args:
        team_data: Dictionary of team cost data
        recommendations: Optional recommendations
        anomalies: Optional anomalies
        format: Output format (csv, pdf, json)

    Returns:
        Tuple of (content_bytes, content_type, filename)
    """
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    metadata = ReportMetadata(
        title="AI Infrastructure Chargeback Report",
        period=ReportPeriod.MONTHLY,
        start_date=start_of_month,
        end_date=now,
        generated_at=now,
        generated_by="AI FinOps Platform",
    )

    team_reports = []
    for team, data in team_data.items():
        report = TeamCostReport(
            team=team,
            gpu_cost=data.get("gpu_cost_daily", 0) * now.day,
            k8s_cost=data.get("k8s_cost_daily", 0) * now.day,
            total_cost=data.get("total_cost_daily", 0) * now.day,
            gpu_hours=data.get("gpu_count", 0) * 24 * now.day,
            gpu_count=data.get("gpu_count", 0),
            avg_utilization=data.get("avg_utilization", 0),
            idle_hours=data.get("idle_gpu_hours", 0) * now.day,
            spot_savings_potential=data.get("spot_savings_potential", 0) * now.day,
            budget=data.get("budget", 5000),
            budget_remaining=data.get("budget", 5000)
            - data.get("total_cost_daily", 0) * now.day,
            cost_trend_pct=data.get("cost_trend_pct", 0),
        )
        team_reports.append(report)

    report_format = ReportFormat(format.lower())
    content, content_type = report_generator.generate_report(
        team_reports=team_reports,
        metadata=metadata,
        format=report_format,
        recommendations=recommendations,
        anomalies=anomalies,
    )

    filename = f"chargeback-report-{now.strftime('%Y-%m')}.{format.lower()}"

    return content, content_type, filename
