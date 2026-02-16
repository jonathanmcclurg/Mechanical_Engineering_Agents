"""Failure case schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CaseStatus(str, Enum):
    """Status of an RCA case."""
    
    INTAKE = "intake"
    RESEARCHING = "researching"
    HYPOTHESIZING = "hypothesizing"
    TESTING = "testing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    AWAITING_FEEDBACK = "awaiting_feedback"
    CLOSED = "closed"


class FailureCase(BaseModel):
    """A manufacturing failure case to investigate."""
    
    case_id: str = Field(..., description="Unique identifier for this case")
    
    # Failure identification
    failure_type: str = Field(..., description="Category of failure (e.g., leak_test_fail, dimensional_oot)")
    failure_description: str = Field(..., description="Inspector/tester description of the failure")
    failure_datetime: datetime = Field(..., description="When the failure was detected")
    
    # Part/product info
    part_number: str = Field(..., description="Part number of the failed item")
    serial_number: Optional[str] = Field(None, description="Serial number if available")
    lot_number: Optional[str] = Field(None, description="Manufacturing lot number")
    
    # Location/context
    station_id: Optional[str] = Field(None, description="Test/inspection station ID")
    line_id: Optional[str] = Field(None, description="Production line ID")
    shift: Optional[str] = Field(None, description="Shift during which failure occurred")
    operator_id: Optional[str] = Field(None, description="Operator ID (anonymized)")
    
    # Test data
    test_name: Optional[str] = Field(None, description="Name of the test that failed")
    test_value: Optional[float] = Field(None, description="Measured value that caused failure")
    spec_lower: Optional[float] = Field(None, description="Lower specification limit")
    spec_upper: Optional[float] = Field(None, description="Upper specification limit")
    
    # Component traceability
    component_lots: Optional[dict[str, str]] = Field(
        None, 
        description="Map of component name to lot number for traceability"
    )
    
    # Metadata
    status: CaseStatus = Field(default=CaseStatus.INTAKE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Links to outputs
    report_id: Optional[str] = Field(None, description="ID of generated RCA report")
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "CASE-2026-00142",
                "failure_type": "leak_test_fail",
                "failure_description": "Unit failed helium leak test at station 4. Leak rate 2.3e-6 mbar*l/s, spec max 1e-6.",
                "failure_datetime": "2026-02-03T14:30:00Z",
                "part_number": "HYD-VALVE-200",
                "serial_number": "SN-789456",
                "lot_number": "LOT-2026-W05-001",
                "station_id": "LEAK-STATION-04",
                "line_id": "LINE-A",
                "shift": "DAY",
                "test_name": "Helium_Leak_Test",
                "test_value": 2.3e-6,
                "spec_lower": 0,
                "spec_upper": 1e-6,
                "component_lots": {
                    "o_ring_main": "ORING-LOT-2026-003",
                    "valve_body": "BODY-LOT-2026-012"
                }
            }
        }
