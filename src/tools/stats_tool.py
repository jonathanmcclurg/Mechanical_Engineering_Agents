"""Statistical analysis tool for hypothesis testing.

This tool provides a consistent interface for statistical analysis
including control charts, hypothesis tests, and capability studies.
All outputs include assumption checks, effect sizes, and artifacts.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt


class AssumptionStatus(str, Enum):
    """Status of a statistical assumption check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_CHECKED = "not_checked"


@dataclass
class AssumptionCheck:
    """Result of checking a statistical assumption."""
    assumption: str
    status: AssumptionStatus
    test_used: Optional[str] = None
    p_value: Optional[float] = None
    details: Optional[str] = None


@dataclass 
class ControlChartSignal:
    """A signal/violation detected on a control chart."""
    rule: str
    subgroups: list[str]
    description: str


@dataclass
class StatsResult:
    """Structured result from a statistical analysis."""
    
    # Identification
    test_name: str
    method: str
    timestamp: datetime
    
    # Hypothesis
    null_hypothesis: str
    alternative_hypothesis: str
    
    # Core results
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    confidence_interval: Optional[tuple[float, float]] = None
    confidence_level: float = 0.95
    
    # Practical significance
    practical_threshold: Optional[float] = None
    is_statistically_significant: Optional[bool] = None
    is_practically_significant: Optional[bool] = None
    
    # Control chart specific
    center_line: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None
    signals: list[ControlChartSignal] = None
    
    # Capability specific
    cp: Optional[float] = None
    cpk: Optional[float] = None
    
    # Assumption checks
    assumptions_checked: list[AssumptionCheck] = None
    assumptions_satisfied: bool = True
    
    # Warnings
    warnings: list[str] = None
    
    # Sample info
    n_total: int = 0
    n_groups: Optional[int] = None
    group_sizes: Optional[dict[str, int]] = None
    
    # Artifacts
    chart_path: Optional[str] = None
    summary_path: Optional[str] = None
    
    # Raw data for audit
    raw_summary: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        if self.signals is None:
            self.signals = []
        if self.assumptions_checked is None:
            self.assumptions_checked = []
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "method": self.method,
            "timestamp": self.timestamp.isoformat(),
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "confidence_interval": self.confidence_interval,
            "is_statistically_significant": self.is_statistically_significant,
            "is_practically_significant": self.is_practically_significant,
            "center_line": self.center_line,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "signals": [{"rule": s.rule, "subgroups": s.subgroups, "description": s.description} 
                       for s in self.signals],
            "cp": self.cp,
            "cpk": self.cpk,
            "assumptions_satisfied": self.assumptions_satisfied,
            "warnings": self.warnings,
            "n_total": self.n_total,
            "chart_path": self.chart_path,
        }


class StatsTool:
    """Statistical analysis tool with guardrails and consistent outputs."""
    
    def __init__(
        self,
        artifacts_dir: str = "./data/artifacts",
        min_sample_size: int = 30,
        default_alpha: float = 0.05,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.min_sample_size = min_sample_size
        self.default_alpha = default_alpha
    
    def _generate_artifact_id(self, data: pd.DataFrame, method: str) -> str:
        """Generate a unique ID for artifacts based on data hash."""
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data).values.tobytes()
        ).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{method}_{timestamp}_{data_hash}"
    
    def _check_normality(
        self, 
        data: np.ndarray, 
        alpha: float = 0.05
    ) -> AssumptionCheck:
        """Check normality using Shapiro-Wilk test."""
        if len(data) < 3:
            return AssumptionCheck(
                assumption="normality",
                status=AssumptionStatus.NOT_CHECKED,
                details="Insufficient data for normality test"
            )
        
        # Use Shapiro-Wilk for small samples, D'Agostino-Pearson for larger
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_value = stats.normaltest(data)
            test_name = "D'Agostino-Pearson"
        
        status = AssumptionStatus.PASSED if p_value > alpha else AssumptionStatus.FAILED
        
        return AssumptionCheck(
            assumption="normality",
            status=status,
            test_used=test_name,
            p_value=p_value,
            details=f"p={p_value:.4f}, {'normal' if status == AssumptionStatus.PASSED else 'non-normal'} distribution"
        )
    
    def _check_equal_variance(
        self,
        groups: list[np.ndarray],
        alpha: float = 0.05
    ) -> AssumptionCheck:
        """Check homogeneity of variance using Levene's test."""
        if len(groups) < 2:
            return AssumptionCheck(
                assumption="equal_variance",
                status=AssumptionStatus.NOT_CHECKED,
                details="Need at least 2 groups"
            )
        
        stat, p_value = stats.levene(*groups)
        status = AssumptionStatus.PASSED if p_value > alpha else AssumptionStatus.FAILED
        
        return AssumptionCheck(
            assumption="equal_variance",
            status=status,
            test_used="Levene's test",
            p_value=p_value,
            details=f"p={p_value:.4f}, variances are {'equal' if status == AssumptionStatus.PASSED else 'unequal'}"
        )
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def xbar_s_chart(
        self,
        data: pd.DataFrame,
        value_column: str,
        subgroup_column: str,
        subgroup_size: Optional[int] = None,
        control_rules: list[str] = None,
        alpha: float = 0.05,
        title: str = None,
    ) -> StatsResult:
        """
        Create X-bar S control chart with rule-based signal detection.
        
        Args:
            data: DataFrame with measurements
            value_column: Column containing the measurement values
            subgroup_column: Column to use for subgrouping (e.g., 'component_lot')
            subgroup_size: Expected subgroup size (for validation)
            control_rules: List of Western Electric rules to apply
            alpha: Significance level for tests
            title: Chart title
            
        Returns:
            StatsResult with control limits, signals, and chart path
        """
        if control_rules is None:
            control_rules = ["beyond_3sigma", "run_of_8", "trend_of_6"]
        
        # Validate inputs
        if value_column not in data.columns:
            raise ValueError(f"Column '{value_column}' not found in data")
        if subgroup_column not in data.columns:
            raise ValueError(f"Column '{subgroup_column}' not found in data")
        
        # Calculate subgroup statistics
        subgroup_stats = data.groupby(subgroup_column)[value_column].agg(['mean', 'std', 'count'])
        subgroup_stats.columns = ['xbar', 's', 'n']
        subgroup_stats = subgroup_stats.reset_index()
        
        # Filter to subgroups with enough data
        valid_subgroups = subgroup_stats[subgroup_stats['n'] >= 2]
        
        if len(valid_subgroups) < 5:
            return StatsResult(
                test_name="X-bar S Chart",
                method="xbar_s_chart",
                timestamp=datetime.now(),
                null_hypothesis="Process is in statistical control",
                alternative_hypothesis="Process exhibits special cause variation",
                warnings=["Insufficient subgroups for reliable control chart (need >= 5)"],
                n_total=len(data),
                n_groups=len(valid_subgroups),
            )
        
        # Calculate overall statistics
        grand_mean = valid_subgroups['xbar'].mean()
        avg_n = valid_subgroups['n'].mean()
        
        # Calculate S-bar (average within-subgroup std dev)
        s_bar = valid_subgroups['s'].mean()
        
        # Control chart constants (approximations for varying n)
        # Using A3 and B3/B4 constants
        c4 = np.sqrt(2 / (avg_n - 1)) * (
            np.math.gamma(avg_n / 2) / np.math.gamma((avg_n - 1) / 2)
        )
        
        # Control limits for X-bar chart
        A3 = 3 / (c4 * np.sqrt(avg_n))
        ucl_xbar = grand_mean + A3 * s_bar
        lcl_xbar = grand_mean - A3 * s_bar
        
        # Detect signals
        signals = []
        xbar_values = valid_subgroups['xbar'].values
        subgroup_names = valid_subgroups[subgroup_column].values
        
        # Rule 1: Beyond 3 sigma
        if "beyond_3sigma" in control_rules:
            beyond_ucl = np.where(xbar_values > ucl_xbar)[0]
            beyond_lcl = np.where(xbar_values < lcl_xbar)[0]
            if len(beyond_ucl) > 0 or len(beyond_lcl) > 0:
                all_beyond = list(beyond_ucl) + list(beyond_lcl)
                signals.append(ControlChartSignal(
                    rule="beyond_3sigma",
                    subgroups=[str(subgroup_names[i]) for i in all_beyond],
                    description=f"{len(all_beyond)} point(s) beyond 3-sigma limits"
                ))
        
        # Rule 2: Run of 8 on same side of center
        if "run_of_8" in control_rules:
            above = xbar_values > grand_mean
            for i in range(len(above) - 7):
                if all(above[i:i+8]) or not any(above[i:i+8]):
                    signals.append(ControlChartSignal(
                        rule="run_of_8",
                        subgroups=[str(subgroup_names[j]) for j in range(i, i+8)],
                        description="8 consecutive points on same side of center line"
                    ))
                    break
        
        # Rule 3: Trend of 6
        if "trend_of_6" in control_rules:
            for i in range(len(xbar_values) - 5):
                segment = xbar_values[i:i+6]
                if all(np.diff(segment) > 0) or all(np.diff(segment) < 0):
                    signals.append(ControlChartSignal(
                        rule="trend_of_6",
                        subgroups=[str(subgroup_names[j]) for j in range(i, i+6)],
                        description="6 consecutive points trending in same direction"
                    ))
                    break
        
        # Generate chart
        artifact_id = self._generate_artifact_id(data, "xbar_s")
        chart_path = self.artifacts_dir / f"{artifact_id}_xbar.png"
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # X-bar chart
        ax1.plot(range(len(xbar_values)), xbar_values, 'b-o', markersize=4)
        ax1.axhline(y=grand_mean, color='g', linestyle='-', label=f'CL={grand_mean:.4f}')
        ax1.axhline(y=ucl_xbar, color='r', linestyle='--', label=f'UCL={ucl_xbar:.4f}')
        ax1.axhline(y=lcl_xbar, color='r', linestyle='--', label=f'LCL={lcl_xbar:.4f}')
        
        # Mark signals
        for signal in signals:
            if signal.rule == "beyond_3sigma":
                for sg in signal.subgroups:
                    idx = np.where(subgroup_names == sg)[0]
                    if len(idx) > 0:
                        ax1.plot(idx[0], xbar_values[idx[0]], 'ro', markersize=10)
        
        ax1.set_xlabel(f'Subgroup ({subgroup_column})')
        ax1.set_ylabel(f'Mean {value_column}')
        ax1.set_title(title or f'X-bar Chart: {value_column} by {subgroup_column}')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # S chart
        s_values = valid_subgroups['s'].values
        B3 = max(0, 1 - 3 * np.sqrt(1 - c4**2) / c4)
        B4 = 1 + 3 * np.sqrt(1 - c4**2) / c4
        ucl_s = B4 * s_bar
        lcl_s = B3 * s_bar
        
        ax2.plot(range(len(s_values)), s_values, 'b-o', markersize=4)
        ax2.axhline(y=s_bar, color='g', linestyle='-', label=f'CL={s_bar:.4f}')
        ax2.axhline(y=ucl_s, color='r', linestyle='--', label=f'UCL={ucl_s:.4f}')
        ax2.axhline(y=lcl_s, color='r', linestyle='--', label=f'LCL={lcl_s:.4f}')
        ax2.set_xlabel(f'Subgroup ({subgroup_column})')
        ax2.set_ylabel('Std Dev')
        ax2.set_title('S Chart')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return StatsResult(
            test_name="X-bar S Chart",
            method="xbar_s_chart",
            timestamp=datetime.now(),
            null_hypothesis="Process is in statistical control",
            alternative_hypothesis="Process exhibits special cause variation",
            center_line=grand_mean,
            ucl=ucl_xbar,
            lcl=lcl_xbar,
            signals=signals,
            is_statistically_significant=len(signals) > 0,
            n_total=len(data),
            n_groups=len(valid_subgroups),
            group_sizes={str(row[subgroup_column]): int(row['n']) 
                        for _, row in valid_subgroups.iterrows()},
            chart_path=str(chart_path),
            raw_summary={
                "grand_mean": grand_mean,
                "s_bar": s_bar,
                "ucl_xbar": ucl_xbar,
                "lcl_xbar": lcl_xbar,
                "ucl_s": ucl_s,
                "lcl_s": lcl_s,
            }
        )
    
    def two_sample_ttest(
        self,
        data: pd.DataFrame,
        value_column: str,
        group_column: str,
        group1_value: Any = None,
        group2_value: Any = None,
        alpha: float = 0.05,
        practical_threshold: float = 0.5,
        use_nonparametric_fallback: bool = True,
    ) -> StatsResult:
        """
        Perform two-sample t-test with assumption checks.
        
        Args:
            data: DataFrame with measurements
            value_column: Column containing the measurement values
            group_column: Column containing group labels
            group1_value: Value identifying first group (if None, uses first unique value)
            group2_value: Value identifying second group (if None, uses second unique value)
            alpha: Significance level
            practical_threshold: Cohen's d threshold for practical significance
            use_nonparametric_fallback: If True, use Mann-Whitney U when assumptions fail
            
        Returns:
            StatsResult with test statistics, p-value, effect size, and CI
        """
        # Validate inputs
        if value_column not in data.columns:
            raise ValueError(f"Column '{value_column}' not found in data")
        if group_column not in data.columns:
            raise ValueError(f"Column '{group_column}' not found in data")
        
        # Get groups
        unique_groups = data[group_column].dropna().unique()
        if len(unique_groups) < 2:
            raise ValueError(f"Need at least 2 groups, found {len(unique_groups)}")
        
        if group1_value is None:
            group1_value = unique_groups[0]
        if group2_value is None:
            group2_value = unique_groups[1]
        
        group1_data = data[data[group_column] == group1_value][value_column].dropna().values
        group2_data = data[data[group_column] == group2_value][value_column].dropna().values
        
        warnings = []
        
        # Check sample sizes
        if len(group1_data) < self.min_sample_size or len(group2_data) < self.min_sample_size:
            warnings.append(
                f"Small sample sizes (n1={len(group1_data)}, n2={len(group2_data)}). "
                f"Results may have low power."
            )
        
        # Check assumptions
        normality1 = self._check_normality(group1_data, alpha)
        normality2 = self._check_normality(group2_data, alpha)
        equal_var = self._check_equal_variance([group1_data, group2_data], alpha)
        
        assumptions = [normality1, normality2, equal_var]
        assumptions_satisfied = all(
            a.status in [AssumptionStatus.PASSED, AssumptionStatus.NOT_CHECKED] 
            for a in assumptions
        )
        
        # Decide which test to use
        use_welch = equal_var.status == AssumptionStatus.FAILED
        use_nonparametric = (
            use_nonparametric_fallback and 
            (normality1.status == AssumptionStatus.FAILED or 
             normality2.status == AssumptionStatus.FAILED)
        )
        
        if use_nonparametric:
            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(
                group1_data, group2_data, alternative='two-sided'
            )
            test_name = "Mann-Whitney U Test"
            method = "mann_whitney_u"
            
            # Effect size: rank-biserial correlation
            n1, n2 = len(group1_data), len(group2_data)
            effect_size = 1 - (2 * stat) / (n1 * n2)
            effect_size_name = "rank-biserial correlation"
            
            warnings.append("Used nonparametric test due to non-normal distribution")
            
        elif use_welch:
            # Welch's t-test (unequal variances)
            stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            test_name = "Welch's t-test"
            method = "welch_ttest"
            effect_size = self._cohens_d(group1_data, group2_data)
            effect_size_name = "Cohen's d"
            warnings.append("Used Welch's t-test due to unequal variances")
            
        else:
            # Standard t-test
            stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)
            test_name = "Two-Sample t-test"
            method = "two_sample_ttest"
            effect_size = self._cohens_d(group1_data, group2_data)
            effect_size_name = "Cohen's d"
        
        # Calculate confidence interval for difference in means
        mean_diff = np.mean(group1_data) - np.mean(group2_data)
        se_diff = np.sqrt(
            np.var(group1_data, ddof=1) / len(group1_data) + 
            np.var(group2_data, ddof=1) / len(group2_data)
        )
        df = len(group1_data) + len(group2_data) - 2
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)
        
        # Determine significance
        is_stat_sig = p_value < alpha
        is_pract_sig = abs(effect_size) >= practical_threshold
        
        return StatsResult(
            test_name=test_name,
            method=method,
            timestamp=datetime.now(),
            null_hypothesis=f"Mean of {group1_value} equals mean of {group2_value}",
            alternative_hypothesis=f"Mean of {group1_value} differs from mean of {group2_value}",
            test_statistic=float(stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name=effect_size_name,
            confidence_interval=ci,
            confidence_level=1 - alpha,
            practical_threshold=practical_threshold,
            is_statistically_significant=is_stat_sig,
            is_practically_significant=is_pract_sig,
            assumptions_checked=assumptions,
            assumptions_satisfied=assumptions_satisfied,
            warnings=warnings,
            n_total=len(group1_data) + len(group2_data),
            n_groups=2,
            group_sizes={str(group1_value): len(group1_data), str(group2_value): len(group2_data)},
            raw_summary={
                "group1_mean": float(np.mean(group1_data)),
                "group2_mean": float(np.mean(group2_data)),
                "group1_std": float(np.std(group1_data, ddof=1)),
                "group2_std": float(np.std(group2_data, ddof=1)),
                "mean_difference": float(mean_diff),
            }
        )
    
    def capability_study(
        self,
        data: pd.DataFrame,
        value_column: str,
        lsl: float,
        usl: float,
        target: Optional[float] = None,
    ) -> StatsResult:
        """
        Perform process capability study (Cp, Cpk).
        
        Args:
            data: DataFrame with measurements
            value_column: Column containing the measurement values
            lsl: Lower specification limit
            usl: Upper specification limit
            target: Target value (if None, uses midpoint of specs)
            
        Returns:
            StatsResult with Cp, Cpk, and distribution info
        """
        values = data[value_column].dropna().values
        
        if len(values) < self.min_sample_size:
            return StatsResult(
                test_name="Process Capability Study",
                method="capability_study",
                timestamp=datetime.now(),
                null_hypothesis="Process is capable (Cpk >= 1.33)",
                alternative_hypothesis="Process is not capable",
                warnings=[f"Insufficient data (n={len(values)}), need at least {self.min_sample_size}"],
                n_total=len(values),
            )
        
        # Check normality
        normality = self._check_normality(values)
        warnings = []
        if normality.status == AssumptionStatus.FAILED:
            warnings.append(
                "Data is non-normal. Capability indices may not be reliable. "
                "Consider data transformation or non-normal capability analysis."
            )
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if target is None:
            target = (lsl + usl) / 2
        
        # Cp: potential capability (if centered)
        cp = (usl - lsl) / (6 * std) if std > 0 else float('inf')
        
        # Cpk: actual capability (accounts for centering)
        cpu = (usl - mean) / (3 * std) if std > 0 else float('inf')
        cpl = (mean - lsl) / (3 * std) if std > 0 else float('inf')
        cpk = min(cpu, cpl)
        
        # Generate histogram with spec limits
        artifact_id = self._generate_artifact_id(data, "capability")
        chart_path = self.artifacts_dir / f"{artifact_id}_capability.png"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(values, bins=30, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(x=lsl, color='r', linestyle='--', linewidth=2, label=f'LSL={lsl}')
        ax.axvline(x=usl, color='r', linestyle='--', linewidth=2, label=f'USL={usl}')
        ax.axvline(x=mean, color='g', linestyle='-', linewidth=2, label=f'Mean={mean:.4f}')
        ax.axvline(x=target, color='b', linestyle=':', linewidth=2, label=f'Target={target}')
        
        # Add normal curve
        x = np.linspace(min(values) - std, max(values) + std, 100)
        ax.plot(x, stats.norm.pdf(x, mean, std), 'k-', linewidth=2, label='Normal fit')
        
        ax.set_xlabel(value_column)
        ax.set_ylabel('Density')
        ax.set_title(f'Process Capability: Cp={cp:.3f}, Cpk={cpk:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Interpret capability
        if cpk >= 1.67:
            interpretation = "Excellent capability"
        elif cpk >= 1.33:
            interpretation = "Good capability"
        elif cpk >= 1.0:
            interpretation = "Marginal capability"
        else:
            interpretation = "Poor capability - process improvement needed"
        
        return StatsResult(
            test_name="Process Capability Study",
            method="capability_study",
            timestamp=datetime.now(),
            null_hypothesis="Process is capable (Cpk >= 1.33)",
            alternative_hypothesis="Process is not capable",
            cp=cp,
            cpk=cpk,
            is_statistically_significant=cpk < 1.33,
            assumptions_checked=[normality],
            assumptions_satisfied=normality.status == AssumptionStatus.PASSED,
            warnings=warnings,
            n_total=len(values),
            chart_path=str(chart_path),
            raw_summary={
                "mean": float(mean),
                "std": float(std),
                "lsl": lsl,
                "usl": usl,
                "target": target,
                "cp": cp,
                "cpk": cpk,
                "cpu": cpu,
                "cpl": cpl,
                "interpretation": interpretation,
                "pct_below_lsl": float(np.mean(values < lsl) * 100),
                "pct_above_usl": float(np.mean(values > usl) * 100),
            }
        )
    
    def correlation(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        alpha: float = 0.05,
        title: str = None,
    ) -> StatsResult:
        """
        Compute Pearson correlation between two numeric columns.

        Args:
            data: DataFrame with measurements
            x_column: First variable column
            y_column: Second variable column
            alpha: Significance level
            title: Chart title

        Returns:
            StatsResult with correlation coefficient, p-value, and scatter plot
        """
        for col in (x_column, y_column):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        clean = data[[x_column, y_column]].dropna()
        x = clean[x_column].values.astype(float)
        y = clean[y_column].values.astype(float)

        if len(x) < 5:
            return StatsResult(
                test_name="Pearson Correlation",
                method="correlation",
                timestamp=datetime.now(),
                null_hypothesis=f"No linear relationship between {x_column} and {y_column}",
                alternative_hypothesis=f"Linear relationship exists between {x_column} and {y_column}",
                warnings=[f"Insufficient data (n={len(x)}), need at least 5"],
                n_total=len(x),
            )

        r, p_value = stats.pearsonr(x, y)

        # Assumption check: normality of both variables
        norm_x = self._check_normality(x, alpha)
        norm_y = self._check_normality(y, alpha)
        assumptions = [norm_x, norm_y]
        assumptions_satisfied = all(
            a.status in [AssumptionStatus.PASSED, AssumptionStatus.NOT_CHECKED]
            for a in assumptions
        )

        warnings: list[str] = []
        if not assumptions_satisfied:
            warnings.append(
                "Non-normal data detected; Pearson r may be unreliable. "
                "Consider Spearman rank correlation."
            )

        # Generate scatter plot
        artifact_id = self._generate_artifact_id(data, "corr")
        chart_path = self.artifacts_dir / f"{artifact_id}_scatter.png"

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.5, edgecolors="k", linewidths=0.5)
        # Trend line
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, p_line(x_sorted), "r--", linewidth=2,
                label=f"r={r:.3f}, p={p_value:.4f}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title or f"Correlation: {x_column} vs {y_column}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        is_stat_sig = p_value < alpha
        # Practical significance: |r| >= 0.3 (moderate effect)
        is_pract_sig = abs(r) >= 0.3

        return StatsResult(
            test_name="Pearson Correlation",
            method="correlation",
            timestamp=datetime.now(),
            null_hypothesis=f"No linear relationship between {x_column} and {y_column}",
            alternative_hypothesis=f"Linear relationship exists between {x_column} and {y_column}",
            test_statistic=float(r),
            p_value=float(p_value),
            effect_size=float(r),
            effect_size_name="Pearson r",
            practical_threshold=0.3,
            is_statistically_significant=is_stat_sig,
            is_practically_significant=is_pract_sig,
            assumptions_checked=assumptions,
            assumptions_satisfied=assumptions_satisfied,
            warnings=warnings,
            n_total=len(x),
            chart_path=str(chart_path),
            raw_summary={
                "r": float(r),
                "r_squared": float(r ** 2),
                "p_value": float(p_value),
                "slope": float(z[0]),
                "intercept": float(z[1]),
            },
        )

    def individuals_mr_chart(
        self,
        data: pd.DataFrame,
        value_column: str,
        order_column: str = None,
        control_rules: list[str] = None,
        title: str = None,
    ) -> StatsResult:
        """
        Create an Individuals and Moving Range (I-MR) control chart.

        Used when subgroup size is 1 (individual measurements over time).

        Args:
            data: DataFrame with measurements
            value_column: Column containing the measurement values
            order_column: Column for ordering observations (e.g. datetime).
                          If None, row order is used.
            control_rules: Western Electric rules to apply
            title: Chart title

        Returns:
            StatsResult with control limits, signals, and chart path
        """
        if control_rules is None:
            control_rules = ["beyond_3sigma", "run_of_8", "trend_of_6"]

        if value_column not in data.columns:
            raise ValueError(f"Column '{value_column}' not found in data")

        df_sorted = data.copy()
        if order_column and order_column in data.columns:
            df_sorted = df_sorted.sort_values(order_column)

        values = df_sorted[value_column].dropna().values.astype(float)

        if len(values) < 5:
            return StatsResult(
                test_name="I-MR Chart",
                method="individuals_mr",
                timestamp=datetime.now(),
                null_hypothesis="Process is in statistical control",
                alternative_hypothesis="Process exhibits special cause variation",
                warnings=[f"Insufficient data (n={len(values)}), need at least 5"],
                n_total=len(values),
            )

        # Moving range (consecutive differences)
        mr = np.abs(np.diff(values))
        mr_bar = np.mean(mr)
        x_bar = np.mean(values)

        # d2 constant for n=2 (moving range of 2 consecutive observations)
        d2 = 1.128
        sigma_hat = mr_bar / d2

        # Control limits for individuals chart
        ucl_i = x_bar + 3 * sigma_hat
        lcl_i = x_bar - 3 * sigma_hat

        # Control limits for MR chart (D3=0, D4=3.267 for n=2)
        ucl_mr = 3.267 * mr_bar
        lcl_mr = 0.0

        # Detect signals on individuals chart
        signals: list[ControlChartSignal] = []

        if "beyond_3sigma" in control_rules:
            beyond = np.where((values > ucl_i) | (values < lcl_i))[0]
            if len(beyond) > 0:
                signals.append(ControlChartSignal(
                    rule="beyond_3sigma",
                    subgroups=[str(i) for i in beyond],
                    description=f"{len(beyond)} point(s) beyond 3-sigma limits",
                ))

        if "run_of_8" in control_rules:
            above = values > x_bar
            for i in range(len(above) - 7):
                if all(above[i:i + 8]) or not any(above[i:i + 8]):
                    signals.append(ControlChartSignal(
                        rule="run_of_8",
                        subgroups=[str(j) for j in range(i, i + 8)],
                        description="8 consecutive points on same side of center line",
                    ))
                    break

        if "trend_of_6" in control_rules:
            for i in range(len(values) - 5):
                seg = values[i:i + 6]
                if all(np.diff(seg) > 0) or all(np.diff(seg) < 0):
                    signals.append(ControlChartSignal(
                        rule="trend_of_6",
                        subgroups=[str(j) for j in range(i, i + 6)],
                        description="6 consecutive points trending in same direction",
                    ))
                    break

        # Generate chart
        artifact_id = self._generate_artifact_id(data, "imr")
        chart_path = self.artifacts_dir / f"{artifact_id}_imr.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Individuals chart
        ax1.plot(range(len(values)), values, "b-o", markersize=3)
        ax1.axhline(y=x_bar, color="g", linestyle="-", label=f"CL={x_bar:.4f}")
        ax1.axhline(y=ucl_i, color="r", linestyle="--", label=f"UCL={ucl_i:.4f}")
        ax1.axhline(y=lcl_i, color="r", linestyle="--", label=f"LCL={lcl_i:.4f}")
        for sig in signals:
            if sig.rule == "beyond_3sigma":
                for idx_str in sig.subgroups:
                    idx = int(idx_str)
                    ax1.plot(idx, values[idx], "ro", markersize=10)
        ax1.set_xlabel("Observation")
        ax1.set_ylabel(value_column)
        ax1.set_title(title or f"Individuals Chart: {value_column}")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Moving Range chart
        ax2.plot(range(len(mr)), mr, "b-o", markersize=3)
        ax2.axhline(y=mr_bar, color="g", linestyle="-", label=f"CL={mr_bar:.4f}")
        ax2.axhline(y=ucl_mr, color="r", linestyle="--", label=f"UCL={ucl_mr:.4f}")
        ax2.axhline(y=lcl_mr, color="r", linestyle="--", label=f"LCL={lcl_mr:.4f}")
        ax2.set_xlabel("Observation")
        ax2.set_ylabel("Moving Range")
        ax2.set_title("Moving Range Chart")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        return StatsResult(
            test_name="I-MR Chart",
            method="individuals_mr",
            timestamp=datetime.now(),
            null_hypothesis="Process is in statistical control",
            alternative_hypothesis="Process exhibits special cause variation",
            center_line=float(x_bar),
            ucl=float(ucl_i),
            lcl=float(lcl_i),
            signals=signals,
            is_statistically_significant=len(signals) > 0,
            is_practically_significant=len(signals) > 0,
            n_total=len(values),
            chart_path=str(chart_path),
            raw_summary={
                "x_bar": float(x_bar),
                "mr_bar": float(mr_bar),
                "sigma_hat": float(sigma_hat),
                "ucl_individuals": float(ucl_i),
                "lcl_individuals": float(lcl_i),
                "ucl_mr": float(ucl_mr),
            },
        )

    def one_way_anova(
        self,
        data: pd.DataFrame,
        value_column: str,
        group_column: str,
        alpha: float = 0.05,
        use_nonparametric_fallback: bool = True,
    ) -> StatsResult:
        """
        Perform one-way ANOVA with assumption checks.
        
        Args:
            data: DataFrame with measurements
            value_column: Column containing the measurement values
            group_column: Column containing group labels
            alpha: Significance level
            use_nonparametric_fallback: If True, use Kruskal-Wallis when assumptions fail
            
        Returns:
            StatsResult with F-statistic, p-value, and effect size
        """
        # Get groups
        groups = []
        group_names = []
        for name, group_data in data.groupby(group_column):
            values = group_data[value_column].dropna().values
            if len(values) >= 2:
                groups.append(values)
                group_names.append(name)
        
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups with sufficient data")
        
        warnings = []
        
        # Check assumptions
        normality_checks = [self._check_normality(g, alpha) for g in groups]
        equal_var = self._check_equal_variance(groups, alpha)
        
        all_normal = all(
            nc.status in [AssumptionStatus.PASSED, AssumptionStatus.NOT_CHECKED]
            for nc in normality_checks
        )
        
        assumptions = normality_checks + [equal_var]
        
        use_nonparametric = (
            use_nonparametric_fallback and 
            (not all_normal or equal_var.status == AssumptionStatus.FAILED)
        )
        
        if use_nonparametric:
            # Kruskal-Wallis test
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis H Test"
            method = "kruskal_wallis"
            
            # Effect size: eta-squared approximation
            n_total = sum(len(g) for g in groups)
            effect_size = (stat - len(groups) + 1) / (n_total - len(groups))
            effect_size_name = "eta-squared (H)"
            
            warnings.append("Used nonparametric test due to assumption violations")
        else:
            # Standard ANOVA
            stat, p_value = stats.f_oneway(*groups)
            test_name = "One-Way ANOVA"
            method = "one_way_anova"
            
            # Effect size: eta-squared
            all_values = np.concatenate(groups)
            grand_mean = np.mean(all_values)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = np.sum((all_values - grand_mean)**2)
            effect_size = ss_between / ss_total if ss_total > 0 else 0
            effect_size_name = "eta-squared"
        
        return StatsResult(
            test_name=test_name,
            method=method,
            timestamp=datetime.now(),
            null_hypothesis="All group means are equal",
            alternative_hypothesis="At least one group mean differs",
            test_statistic=float(stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name=effect_size_name,
            is_statistically_significant=p_value < alpha,
            assumptions_checked=assumptions,
            assumptions_satisfied=all_normal and equal_var.status == AssumptionStatus.PASSED,
            warnings=warnings,
            n_total=sum(len(g) for g in groups),
            n_groups=len(groups),
            group_sizes={str(name): len(g) for name, g in zip(group_names, groups)},
            raw_summary={
                "group_means": {str(name): float(np.mean(g)) for name, g in zip(group_names, groups)},
                "group_stds": {str(name): float(np.std(g, ddof=1)) for name, g in zip(group_names, groups)},
            }
        )
