"""Data Fetch Tool for querying internal manufacturing data API.

Provides an interface for agents to request test data, ROA parameters,
operator buyoffs, and other manufacturing data in tabular format
suitable for statistical analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
import hashlib

import pandas as pd
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.tools.data_catalog import (
    DataCatalog,
    CatalogField as DataField,
    DataCategory,
    DataFieldType,
)


class DataFetchRequest(BaseModel):
    """Request specification for fetching tabular data."""
    
    # Required: at least one serial number or filter
    serial_numbers: Optional[list[str]] = Field(
        None, 
        description="List of unit serial numbers to fetch data for"
    )
    lot_numbers: Optional[list[str]] = Field(
        None,
        description="List of lot numbers to filter by"
    )
    
    # Columns to include in the result
    test_ids: list[str] = Field(
        default_factory=list,
        description="Test IDs to include as columns"
    )
    roa_parameters: list[str] = Field(
        default_factory=list,
        description="ROA parameter IDs to include as columns"
    )
    operator_buyoffs: list[str] = Field(
        default_factory=list,
        description="Operator buyoff IDs to include as columns"
    )
    process_parameters: list[str] = Field(
        default_factory=list,
        description="Process parameter IDs to include as columns"
    )
    
    # Time filtering
    start_date: Optional[datetime] = Field(None, description="Start of date range")
    end_date: Optional[datetime] = Field(None, description="End of date range")
    time_window: Optional[str] = Field(
        None, 
        description="Alternative to start/end date: e.g., '30d', '90d'"
    )
    
    # Limits
    limit: int = Field(default=10000, description="Maximum rows to return")
    
    # Aggregation options
    aggregate_by_serial: bool = Field(
        default=False,
        description="If True, aggregate multiple values per serial (use latest)"
    )


@dataclass
class DataFetchResult:
    """Result of a data fetch operation."""
    
    data: pd.DataFrame
    request_id: str
    fetched_at: datetime
    row_count: int
    columns: list[str]
    
    # Metadata about what was fetched
    test_ids_found: list[str]
    roa_parameters_found: list[str]
    operator_buyoffs_found: list[str]
    
    # Data quality info
    missing_data_summary: dict[str, int]  # column -> count of missing values
    warnings: list[str]
    
    def to_citation_dict(self) -> dict:
        """Convert to citation dictionary format."""
        return {
            "source_type": "internal_data_api",
            "source_id": self.request_id,
            "source_name": "Manufacturing Data API",
            "timestamp": self.fetched_at.isoformat(),
            "excerpt": f"Fetched {self.row_count} units with {len(self.columns)} data fields",
            "details": {
                "test_ids": self.test_ids_found,
                "roa_parameters": self.roa_parameters_found,
                "operator_buyoffs": self.operator_buyoffs_found,
            }
        }


class DataFetchTool:
    """Tool for fetching manufacturing data from the internal API.
    
    This tool interfaces with the company's internal data API to fetch
    test data, ROA parameters, operator buyoffs, and other manufacturing
    data in tabular format for statistical analysis.
    """
    
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        mock_mode: bool = True,
        data_catalog: Optional[DataCatalog] = None,
    ):
        """Initialize the data fetch tool.
        
        Args:
            api_base_url: Base URL for the internal data API
            api_key: API key for authentication
            mock_mode: If True, generate mock data for development
            data_catalog: Shared data catalog for field validation/discovery
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.mock_mode = mock_mode
        if data_catalog is not None:
            self.data_catalog = data_catalog
        else:
            settings = get_settings()
            self.data_catalog = DataCatalog(
                catalog_dir=settings.catalog_dir,
                db_url=settings.catalog_db_url,
            )
        self._request_log: list[dict] = []
    
    def list_available_fields(
        self, 
        category: Optional[DataCategory] = None
    ) -> list[DataField]:
        """List available data fields, optionally filtered by category.
        
        Args:
            category: If provided, filter to only this category
            
        Returns:
            List of available DataField objects
        """
        return self.data_catalog.list_fields(category=category)
    
    def list_test_ids(self) -> list[str]:
        """List available test IDs that can be requested."""
        return [f.field_id for f in self.list_available_fields(DataCategory.TEST_DATA)]
    
    def list_roa_parameters(self) -> list[str]:
        """List available ROA parameters that can be requested."""
        return [f.field_id for f in self.list_available_fields(DataCategory.ROA)]
    
    def list_operator_buyoffs(self) -> list[str]:
        """List available operator buyoff fields that can be requested."""
        return [f.field_id for f in self.list_available_fields(DataCategory.OPERATOR_BUYOFF)]
    
    def list_process_parameters(self) -> list[str]:
        """List available process parameters that can be requested."""
        return [f.field_id for f in self.list_available_fields(DataCategory.PROCESS_PARAMETER)]
    
    def _parse_time_window(self, window: str) -> timedelta:
        """Parse time window string (e.g., '30d', '90d') to timedelta."""
        unit = window[-1].lower()
        value = int(window[:-1])
        
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(days=value * 30)  # Approximate months
        else:
            raise ValueError(f"Unknown time unit: {unit}")
    
    def fetch_data(self, request: DataFetchRequest) -> DataFetchResult:
        """Fetch data from the internal API based on request specification.
        
        This is the main method for retrieving tabular data. The returned
        DataFrame will have serial_number as the first column, followed by
        the requested test IDs, ROA parameters, operator buyoffs, etc.
        
        Args:
            request: DataFetchRequest specifying what data to fetch
            
        Returns:
            DataFetchResult with the data and metadata
        """
        # Validate requested fields
        all_requested = (
            request.test_ids + 
            request.roa_parameters + 
            request.operator_buyoffs +
            request.process_parameters
        )
        
        for field_id in all_requested:
            if not self.data_catalog.has_field(field_id):
                raise ValueError(f"Unknown field ID: {field_id}")
        
        # Generate request ID
        request_hash = hashlib.md5(
            str(request.model_dump()).encode()
        ).hexdigest()[:8]
        request_id = f"DF-{datetime.now().strftime('%Y%m%d%H%M%S')}-{request_hash}"
        
        if self.mock_mode:
            df = self._generate_mock_data(request)
        else:
            df = self._fetch_from_api(request)
        
        # Calculate missing data summary
        missing_summary = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_summary[col] = int(missing_count)
        
        # Collect warnings
        warnings = []
        if len(df) == 0:
            warnings.append("No data found matching the request criteria")
        elif len(df) < 30:
            warnings.append(f"Limited data: only {len(df)} units found. Statistical significance may be affected.")
        
        if len(missing_summary) > 0:
            high_missing = [col for col, count in missing_summary.items() if count > len(df) * 0.2]
            if high_missing:
                warnings.append(f"High missing data (>20%) in columns: {', '.join(high_missing)}")
        
        result = DataFetchResult(
            data=df,
            request_id=request_id,
            fetched_at=datetime.now(),
            row_count=len(df),
            columns=list(df.columns),
            test_ids_found=[f for f in request.test_ids if f in df.columns],
            roa_parameters_found=[f for f in request.roa_parameters if f in df.columns],
            operator_buyoffs_found=[f for f in request.operator_buyoffs if f in df.columns],
            missing_data_summary=missing_summary,
            warnings=warnings,
        )
        
        # Log request for audit
        self._request_log.append({
            "request_id": request_id,
            "fetched_at": result.fetched_at.isoformat(),
            "row_count": result.row_count,
            "columns_requested": all_requested,
            "columns_returned": result.columns,
        })
        
        return result
    
    def fetch_for_analysis(
        self,
        serial_numbers: Optional[list[str]] = None,
        lot_numbers: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
        roa_parameters: Optional[list[str]] = None,
        operator_buyoffs: Optional[list[str]] = None,
        process_parameters: Optional[list[str]] = None,
        time_window: str = "90d",
        limit: int = 10000,
    ) -> DataFetchResult:
        """Simplified interface for fetching data for statistical analysis.
        
        This is a convenience method that wraps fetch_data with commonly
        used defaults.
        
        Args:
            serial_numbers: List of serial numbers to fetch (optional)
            lot_numbers: List of lot numbers to filter by (optional)
            test_ids: List of test IDs to include
            roa_parameters: List of ROA parameters to include
            operator_buyoffs: List of operator buyoffs to include
            process_parameters: List of process parameters to include
            time_window: Time window for data (e.g., '30d', '90d')
            limit: Maximum rows to return
            
        Returns:
            DataFetchResult with the data
        """
        request = DataFetchRequest(
            serial_numbers=serial_numbers,
            lot_numbers=lot_numbers,
            test_ids=test_ids or [],
            roa_parameters=roa_parameters or [],
            operator_buyoffs=operator_buyoffs or [],
            process_parameters=process_parameters or [],
            time_window=time_window,
            limit=limit,
            aggregate_by_serial=True,
        )
        
        return self.fetch_data(request)
    
    def _fetch_from_api(self, request: DataFetchRequest) -> pd.DataFrame:
        """Fetch data from the actual internal API.
        
        In production, this would make HTTP requests to the internal
        data API. The implementation would depend on the specific API
        contract.
        """
        raise NotImplementedError(
            "Real API integration not yet implemented. "
            "Set mock_mode=True for development."
        )
    
    def _generate_mock_data(self, request: DataFetchRequest) -> pd.DataFrame:
        """Generate mock data for development/testing."""
        import numpy as np
        
        # Determine number of rows
        if request.serial_numbers:
            n_rows = len(request.serial_numbers)
        elif request.lot_numbers:
            n_rows = len(request.lot_numbers) * 20  # ~20 units per lot
        else:
            n_rows = min(100, request.limit)
        
        data = {}
        
        # Always include serial number as first column
        if request.serial_numbers:
            data["serial_number"] = request.serial_numbers
        else:
            data["serial_number"] = [f"SN-{10000 + i}" for i in range(n_rows)]
        
        # Add lot number if requested or filtering
        if request.lot_numbers:
            # Distribute serial numbers across lots
            lots = request.lot_numbers * (n_rows // len(request.lot_numbers) + 1)
            data["lot_number"] = lots[:n_rows]
        else:
            data["lot_number"] = [f"LOT-{100 + i // 10}" for i in range(n_rows)]
        
        # Add test data columns
        for test_id in request.test_ids:
            data[test_id] = self._generate_mock_column(test_id, n_rows)
        
        # Add ROA parameter columns
        for param in request.roa_parameters:
            data[param] = self._generate_mock_column(param, n_rows)
        
        # Add operator buyoff columns
        for buyoff in request.operator_buyoffs:
            data[buyoff] = self._generate_mock_column(buyoff, n_rows)
        
        # Add process parameter columns
        for param in request.process_parameters:
            data[param] = self._generate_mock_column(param, n_rows)
        
        df = pd.DataFrame(data)
        
        # Apply limit
        if len(df) > request.limit:
            df = df.head(request.limit)
        
        return df
    
    def _generate_mock_column(self, field_id: str, n: int) -> list:
        """Generate mock data for a single column based on field definition."""
        import numpy as np
        
        field = self.data_catalog.get_field(field_id)
        
        if field is None:
            return [f"{field_id}_{i}" for i in range(n)]
        
        if field.field_type == DataFieldType.NUMERIC:
            # Generate realistic numeric data based on field
            if "TORQUE" in field_id:
                return list(np.random.normal(25, 2, n))  # Torque values
            elif "TEMP" in field_id:
                return list(np.random.normal(22, 3, n))  # Temperature
            elif "HUMIDITY" in field_id:
                return list(np.random.normal(45, 10, n))  # Humidity
            elif "RATE" in field_id:
                # Some failures (bimodal distribution)
                good = np.random.exponential(1e-6, int(n * 0.9))
                bad = np.random.exponential(1e-4, int(n * 0.1))
                values = list(good) + list(bad)
                np.random.shuffle(values)
                return values[:n]
            elif "TIME" in field_id:
                return list(np.random.normal(60, 10, n))  # Time in minutes
            elif "DROP" in field_id:
                return list(np.random.normal(0.5, 0.1, n))  # Pressure drop
            else:
                return list(np.random.normal(100, 10, n))
        
        elif field.field_type == DataFieldType.CATEGORICAL:
            if "RESULT" in field_id or "BUYOFF" in field_id:
                # Mostly pass, some fail
                choices = ["PASS"] * 90 + ["FAIL"] * 10
                return [np.random.choice(choices) for _ in range(n)]
            elif "SHIFT" in field_id:
                return [np.random.choice(["1", "2", "3"]) for _ in range(n)]
            else:
                return [np.random.choice(["A", "B", "C"]) for _ in range(n)]
        
        elif field.field_type == DataFieldType.IDENTIFIER:
            if "OPERATOR" in field_id:
                return [f"OP-{np.random.randint(1, 20)}" for _ in range(n)]
            elif "STATION" in field_id:
                return [f"STN-{np.random.randint(1, 10)}" for _ in range(n)]
            elif "LOT" in field_id or "BATCH" in field_id:
                return [f"LOT-{200 + i // 15}" for i in range(n)]
            elif "FIXTURE" in field_id:
                return [f"FIX-{np.random.randint(1, 5)}" for _ in range(n)]
            elif "LINE" in field_id:
                return [f"LINE-{np.random.choice(['A', 'B', 'C'])}" for _ in range(n)]
            else:
                return [f"{field_id[:3]}-{i}" for i in range(n)]
        
        else:
            return [f"{field_id}_{i}" for i in range(n)]
    
    def get_audit_log(self) -> list[dict]:
        """Get the request audit log."""
        return self._request_log.copy()
