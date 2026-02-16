"""Analysis recipe schemas - statistical analysis configurations per failure type."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class StatisticalMethod(str, Enum):
    """Available statistical methods."""
    
    # Control charts
    XBAR_S_CHART = "xbar_s_chart"
    XBAR_R_CHART = "xbar_r_chart"
    INDIVIDUALS_MR = "individuals_mr"
    P_CHART = "p_chart"
    C_CHART = "c_chart"
    
    # Hypothesis tests
    TWO_SAMPLE_TTEST = "two_sample_ttest"
    PAIRED_TTEST = "paired_ttest"
    WELCH_TTEST = "welch_ttest"
    MANN_WHITNEY_U = "mann_whitney_u"
    ONE_WAY_ANOVA = "one_way_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"
    
    # Correlation/regression
    CORRELATION = "correlation"
    LINEAR_REGRESSION = "linear_regression"
    
    # Distribution tests
    NORMALITY_TEST = "normality_test"
    LEVENE_TEST = "levene_test"
    
    # Capability
    CAPABILITY_STUDY = "capability_study"


class MethodConfig(BaseModel):
    """Configuration for a statistical method."""
    
    method: StatisticalMethod
    description: str = Field(..., description="When to use this method")
    
    # Required inputs
    target_variable: str = Field(..., description="Column name of the measurement to analyze")
    grouping_key: Optional[str] = Field(
        None, 
        description="Column to use for subgrouping (e.g., 'component_lot', 'date')"
    )
    
    # Method-specific parameters
    subgroup_size: Optional[int] = Field(None, description="For control charts")
    alpha: float = Field(default=0.05, description="Significance level")
    practical_threshold: Optional[float] = Field(
        None, 
        description="Minimum effect size for practical significance"
    )
    baseline_period: Optional[str] = Field(
        None, 
        description="Time period for baseline (e.g., '30d', '90d')"
    )
    
    # Control chart rules
    control_rules: list[str] = Field(
        default=["beyond_3sigma", "run_of_8", "trend_of_6"],
        description="Western Electric rules to apply"
    )
    
    # Fallback
    nonparametric_fallback: bool = Field(
        default=True, 
        description="Use nonparametric test if assumptions fail"
    )


class DataRequirement(BaseModel):
    """A data source requirement for an analysis recipe."""
    
    source_name: str = Field(..., description="Name of the data source")
    source_type: str = Field(..., description="Type: sql, api, rag")
    required_columns: list[str] = Field(..., description="Columns that must be present")
    optional_columns: list[str] = Field(default_factory=list)
    min_rows: int = Field(default=30, description="Minimum rows for valid analysis")
    time_window: Optional[str] = Field(None, description="How far back to pull data")


class AnalysisRecipe(BaseModel):
    """A complete analysis recipe for a failure type."""
    
    recipe_id: str = Field(..., description="Unique identifier")
    failure_type: str = Field(..., description="Failure type this recipe applies to")
    version: str = Field(default="1.0.0")
    
    # Description
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="When to use this recipe")
    
    # Data requirements
    required_case_fields: list[str] = Field(
        ..., 
        description="Fields that must be present in the failure case"
    )
    data_requirements: list[DataRequirement] = Field(
        ..., 
        description="Data sources needed"
    )
    
    # Analysis methods (in recommended order)
    primary_methods: list[MethodConfig] = Field(
        ..., 
        description="Main statistical methods to run"
    )
    secondary_methods: list[MethodConfig] = Field(
        default_factory=list, 
        description="Additional methods if primary inconclusive"
    )
    
    # Hypothesis templates
    common_hypotheses: list[str] = Field(
        ..., 
        description="Common root causes to consider for this failure type"
    )
    
    # Product guide sections to always retrieve
    relevant_guide_sections: list[str] = Field(
        default_factory=list,
        description="Product guide sections relevant to this failure type"
    )
    must_include_fields: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Fields that must always be fetched for this failure type, keyed by category"
    )
    
    # Thresholds
    min_effect_size: float = Field(
        default=0.5, 
        description="Cohen's d threshold for practical significance"
    )
    min_confidence: float = Field(
        default=0.7, 
        description="Minimum confidence to report a hypothesis as supported"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "recipe_id": "leak-test-001",
                "failure_type": "leak_test_fail",
                "name": "Helium Leak Test Failure Analysis",
                "description": "Recipe for analyzing units that fail helium leak testing",
                "required_case_fields": [
                    "part_number", "lot_number", "test_value", 
                    "spec_upper", "component_lots"
                ],
                "data_requirements": [
                    {
                        "source_name": "leak_test_history",
                        "source_type": "sql",
                        "required_columns": ["serial_number", "test_datetime", "leak_rate", "lot_number"],
                        "min_rows": 50,
                        "time_window": "90d"
                    }
                ],
                "primary_methods": [
                    {
                        "method": "xbar_s_chart",
                        "description": "Check for special cause variation by component lot",
                        "target_variable": "leak_rate",
                        "grouping_key": "component_lot_oring",
                        "subgroup_size": 5,
                        "control_rules": ["beyond_3sigma", "run_of_8"]
                    }
                ],
                "common_hypotheses": [
                    "O-ring lot variation",
                    "Surface finish degradation",
                    "Assembly torque drift",
                    "Contamination"
                ],
                "must_include_fields": {
                    "test_ids": ["LEAK_RATE", "LEAK_RESULT", "PRESSURE_DROP"],
                    "roa_parameters": ["TORQUE_SEAL_RING", "GASKET_LOT", "SEAL_LOT"],
                    "operator_buyoffs": ["BUYOFF_SEAL_PLACEMENT", "BUYOFF_VISUAL_INSPECTION"]
                }
            }
        }
