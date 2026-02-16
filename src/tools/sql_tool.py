"""SQL tool for querying internal manufacturing databases.

Provides a secure interface for agents to query structured data
with proper access controls and audit logging.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
import hashlib
import json

import pandas as pd
from pydantic import BaseModel, Field


class QueryConfig(BaseModel):
    """Configuration for a SQL query."""
    
    source_name: str = Field(..., description="Name of the data source")
    query_type: str = Field(..., description="Type of query: select, aggregate")
    
    # Table/view specification
    table_name: str = Field(..., description="Table or view to query")
    
    # Columns
    select_columns: list[str] = Field(..., description="Columns to select")
    
    # Filters
    where_conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Column: value conditions for WHERE clause"
    )
    time_column: Optional[str] = Field(None, description="Column for time filtering")
    time_window: Optional[str] = Field(None, description="Time window (e.g., '30d', '90d')")
    
    # Grouping
    group_by: Optional[list[str]] = Field(None, description="Columns for GROUP BY")
    
    # Limits
    limit: int = Field(default=10000, description="Maximum rows to return")


@dataclass
class QueryResult:
    """Result of a SQL query with metadata."""
    
    data: pd.DataFrame
    query_id: str
    source_name: str
    executed_at: datetime
    row_count: int
    execution_time_ms: float
    
    # For citation
    query_hash: str
    columns_returned: list[str]
    
    # Audit
    filters_applied: dict[str, Any]
    
    def to_citation_dict(self) -> dict:
        """Convert to citation dictionary format."""
        # Keep source_type aligned to report schema enums used across environments.
        source_type = "historical_case" if self.source_name == "historical_cases" else "test_data"
        return {
            "source_type": source_type,
            "source_id": self.query_id,
            "source_name": self.source_name,
            "timestamp": self.executed_at.isoformat(),
            "excerpt": f"Query returned {self.row_count} rows from {self.source_name}",
            "excerpt_hash": self.query_hash,
        }


class SQLTool:
    """SQL tool for secure database queries with audit logging."""
    
    # Define allowed tables and columns for each data source
    # This enforces access control at the tool level
    ALLOWED_SOURCES = {
        "leak_test_history": {
            "table": "leak_tests",
            "allowed_columns": [
                "serial_number", "test_datetime", "leak_rate", "lot_number",
                "station_id", "operator_id", "ambient_temp", "ambient_humidity",
                "test_result", "spec_limit"
            ],
            "time_column": "test_datetime",
        },
        "component_lots": {
            "table": "component_traceability",
            "allowed_columns": [
                "serial_number", "component_name", "component_lot", 
                "supplier_id", "receive_date"
            ],
        },
        "assembly_parameters": {
            "table": "assembly_data",
            "allowed_columns": [
                "serial_number", "torque_value", "assembly_datetime",
                "fixture_id", "operator_id", "line_id"
            ],
            "time_column": "assembly_datetime",
        },
        "dimensional_history": {
            "table": "dimensional_inspections",
            "allowed_columns": [
                "serial_number", "measurement_datetime", "dimension_name",
                "measured_value", "lot_number", "machine_id", "operator_id",
                "fixture_id", "tool_id", "ambient_temp", "nominal", "tolerance"
            ],
            "time_column": "measurement_datetime",
        },
        "machine_parameters": {
            "table": "machine_data",
            "allowed_columns": [
                "machine_id", "parameter_datetime", "spindle_speed",
                "feed_rate", "coolant_temp", "tool_wear", "vibration_level"
            ],
            "time_column": "parameter_datetime",
        },
        "tool_changes": {
            "table": "tool_change_log",
            "allowed_columns": [
                "machine_id", "tool_id", "change_datetime", "tool_lot",
                "parts_since_change", "reason"
            ],
            "time_column": "change_datetime",
        },
        "material_lots": {
            "table": "material_receiving",
            "allowed_columns": [
                "lot_number", "material_lot", "supplier_id",
                "material_cert_hardness", "receive_date", "material_type"
            ],
        },
        "historical_cases": {
            "table": "rca_cases",
            "allowed_columns": [
                "case_id", "failure_type", "failure_datetime", "part_number",
                "lot_number", "root_cause", "resolution", "status"
            ],
            "time_column": "failure_datetime",
        },
    }
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        mock_mode: bool = True,  # Use mock data for development
    ):
        self.connection_string = connection_string
        self.mock_mode = mock_mode
        self._connection = None
        self._query_log: list[dict] = []
    
    def _validate_query(self, config: QueryConfig) -> None:
        """Validate that the query is allowed."""
        if config.source_name not in self.ALLOWED_SOURCES:
            raise ValueError(f"Unknown data source: {config.source_name}")
        
        allowed = self.ALLOWED_SOURCES[config.source_name]
        
        # Check columns
        for col in config.select_columns:
            if col not in allowed["allowed_columns"] and col != "*":
                raise ValueError(
                    f"Column '{col}' not allowed for source '{config.source_name}'"
                )
        
        # Check filter columns
        for col in config.where_conditions.keys():
            if col not in allowed["allowed_columns"]:
                raise ValueError(
                    f"Filter column '{col}' not allowed for source '{config.source_name}'"
                )
    
    def _parse_time_window(self, window: str) -> timedelta:
        """Parse time window string (e.g., '30d', '90d') to timedelta."""
        match = {
            'd': lambda x: timedelta(days=x),
            'w': lambda x: timedelta(weeks=x),
            'h': lambda x: timedelta(hours=x),
        }
        
        unit = window[-1].lower()
        value = int(window[:-1])
        
        if unit in match:
            return match[unit](value)
        else:
            raise ValueError(f"Unknown time unit: {unit}")
    
    def _build_query(self, config: QueryConfig) -> str:
        """Build SQL query string from config (for logging/audit only)."""
        source = self.ALLOWED_SOURCES[config.source_name]
        
        columns = ", ".join(config.select_columns) if config.select_columns else "*"
        query = f"SELECT {columns} FROM {source['table']}"
        
        conditions = []
        for col, val in config.where_conditions.items():
            if isinstance(val, str):
                conditions.append(f"{col} = '{val}'")
            elif isinstance(val, (list, tuple)):
                values = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in val)
                conditions.append(f"{col} IN ({values})")
            else:
                conditions.append(f"{col} = {val}")
        
        if config.time_column and config.time_window:
            # This would be parameterized in real query
            conditions.append(f"{config.time_column} >= NOW() - INTERVAL '{config.time_window}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if config.group_by:
            query += " GROUP BY " + ", ".join(config.group_by)
        
        query += f" LIMIT {config.limit}"
        
        return query
    
    def _generate_mock_data(self, config: QueryConfig) -> pd.DataFrame:
        """Generate mock data for development/testing."""
        import numpy as np
        
        n_rows = min(100, config.limit)
        data = {}
        
        for col in config.select_columns:
            if col == "*":
                # Generate all allowed columns
                for allowed_col in self.ALLOWED_SOURCES[config.source_name]["allowed_columns"]:
                    data[allowed_col] = self._generate_mock_column(allowed_col, n_rows)
            else:
                data[col] = self._generate_mock_column(col, n_rows)
        
        return pd.DataFrame(data)
    
    def _generate_mock_column(self, col_name: str, n: int) -> list:
        """Generate mock data for a single column."""
        import numpy as np
        
        # Pattern-based generation
        if "datetime" in col_name or "date" in col_name:
            base = datetime.now() - timedelta(days=90)
            return [base + timedelta(hours=i*2) for i in range(n)]
        elif "rate" in col_name or "value" in col_name or "temp" in col_name:
            return list(np.random.normal(100, 10, n))
        elif "id" in col_name:
            if "serial" in col_name:
                return [f"SN-{1000+i}" for i in range(n)]
            elif "lot" in col_name:
                return [f"LOT-{i//10}" for i in range(n)]  # ~10 per lot
            elif "station" in col_name:
                return [f"STATION-{i%5+1}" for i in range(n)]
            elif "machine" in col_name:
                return [f"MACH-{i%3+1}" for i in range(n)]
            elif "operator" in col_name:
                return [f"OP-{i%10+1}" for i in range(n)]
            else:
                return [f"ID-{i}" for i in range(n)]
        elif "number" in col_name:
            if "lot" in col_name:
                return [f"LOT-{i//10}" for i in range(n)]
            elif "part" in col_name:
                return ["PART-001"] * n
            elif "serial" in col_name:
                return [f"SN-{1000+i}" for i in range(n)]
            else:
                return list(range(n))
        elif "name" in col_name:
            return [f"Item_{i%5}" for i in range(n)]
        elif "result" in col_name or "status" in col_name:
            import random
            return [random.choice(["PASS", "FAIL"]) for _ in range(n)]
        else:
            return [f"{col_name}_{i}" for i in range(n)]
    
    def query(self, config: QueryConfig) -> QueryResult:
        """Execute a query and return results.
        
        Args:
            config: Query configuration
            
        Returns:
            QueryResult with data and metadata
        """
        # Validate the query
        self._validate_query(config)
        
        start_time = datetime.now()
        
        if self.mock_mode:
            # Generate mock data for development
            df = self._generate_mock_data(config)
        else:
            # Execute real query
            # In production, this would use parameterized queries
            query_str = self._build_query(config)
            # df = pd.read_sql(query_str, self._connection)
            raise NotImplementedError("Real database queries not yet implemented")
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Generate query ID and hash for audit
        query_str = self._build_query(config)
        query_hash = hashlib.md5(query_str.encode()).hexdigest()[:12]
        query_id = f"Q-{datetime.now().strftime('%Y%m%d%H%M%S')}-{query_hash}"
        
        result = QueryResult(
            data=df,
            query_id=query_id,
            source_name=config.source_name,
            executed_at=start_time,
            row_count=len(df),
            execution_time_ms=execution_time,
            query_hash=query_hash,
            columns_returned=list(df.columns),
            filters_applied=config.where_conditions,
        )
        
        # Log query for audit
        self._query_log.append({
            "query_id": query_id,
            "source_name": config.source_name,
            "executed_at": start_time.isoformat(),
            "row_count": len(df),
            "filters": config.where_conditions,
        })
        
        return result
    
    def query_for_analysis(
        self,
        source_name: str,
        columns: list[str],
        filters: dict[str, Any] = None,
        time_window: str = None,
        limit: int = 10000,
    ) -> QueryResult:
        """Simplified query interface for statistical analysis.
        
        Args:
            source_name: Name of the data source
            columns: Columns to select
            filters: Filter conditions
            time_window: Time window (e.g., '30d')
            limit: Max rows
            
        Returns:
            QueryResult with data
        """
        source = self.ALLOWED_SOURCES.get(source_name, {})
        
        config = QueryConfig(
            source_name=source_name,
            query_type="select",
            table_name=source.get("table", source_name),
            select_columns=columns,
            where_conditions=filters or {},
            time_column=source.get("time_column"),
            time_window=time_window,
            limit=limit,
        )
        
        return self.query(config)
    
    def get_similar_cases(
        self,
        failure_type: str,
        part_number: str,
        time_window: str = "365d",
        limit: int = 50,
    ) -> QueryResult:
        """Get similar historical RCA cases.
        
        Args:
            failure_type: Type of failure to match
            part_number: Part number to match
            time_window: How far back to look
            limit: Max cases to return
            
        Returns:
            QueryResult with similar cases
        """
        return self.query_for_analysis(
            source_name="historical_cases",
            columns=["*"],
            filters={
                "failure_type": failure_type,
                "part_number": part_number,
            },
            time_window=time_window,
            limit=limit,
        )
    
    def get_audit_log(self) -> list[dict]:
        """Get the query audit log."""
        return self._query_log.copy()
