"""
src/alerts — District mapping, CAP 1.2 XML generation, and Forecaster Approval workflow.
"""

from src.alerts.cap_xml import (
    CAP_NAMESPACE,
    generate_cap_xml,
    get_cap_severity,
    get_cap_urgency,
    validate_cap_xml,
)
from src.alerts.district_join import DistrictImpact, DistrictJoinEngine
from src.alerts.workflow import AlertRecord, AlertWorkflowManager

__all__ = [
    "CAP_NAMESPACE",
    "AlertRecord",
    "AlertWorkflowManager",
    "DistrictImpact",
    "DistrictJoinEngine",
    "generate_cap_xml",
    "get_cap_severity",
    "get_cap_urgency",
    "validate_cap_xml",
]
