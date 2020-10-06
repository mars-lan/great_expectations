from typing import Iterable

from great_expectations.execution_engine import (
    PandasExecutionEngine,
    SparkDFExecutionEngine,
)
from great_expectations.execution_engine.sqlalchemy_execution_engine import (
    SqlAlchemyExecutionEngine,
)
from great_expectations.expectations.metrics.metric import (
    ColumnAggregateMetric,
    ColumnMapMetric,
    Metric,
    metric,
)

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None
    np = None

try:
    import sqlalchemy as sa
except ImportError:
    sa = None
"""
    @PandasExecutionEngine.metric(
        metric_name="snippet",
        metric_domain_keys=domain_keys,
        metric_value_keys=tuple(),
        metric_dependencies=tuple(),
        bundle_computation=True,
    )
    def _snippet(
        self,
        batches: Dict[str, Batch],
        execution_engine: PandasExecutionEngine,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        metrics: dict,
        runtime_configuration: dict = None,
        filter_column_isnull: bool = True,
    ):
        df = execution_engine.get_domain_dataframe(
            domain_kwargs=metric_domain_kwargs, batches=batches
        )
        return df
"""


class ColumnRowCount(Metric):
    metric_name = "column.row_count"

    filter_column_isnull: bool = False

    @metric(engine=PandasExecutionEngine)
    def _pandas(
        engine, domain: pd.DataFrame, **kwargs,
    ):
        return domain.shape[0]

    @metric(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy(engine, domain: sa.table, **kwargs):
        return sa.func.count(), domain

    @metric(engine=SparkDFExecutionEngine)
    def _spark(engine, domain: "pyspark.sql.DataFrame", **kwargs):
        return domain.count()


class ColumnMax(ColumnAggregateMetric):
    metric_name = "column.aggregate.max"

    """Return the maximum value in a given column"""

    @metric(engine=PandasExecutionEngine)
    def _pandas(engine, domain: pd.Series, **kwargs):
        """Max Metric Function"""
        return domain.max()

    @metric(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy(
        engine, domain: sa.column,
    ):
        return sa.func.max(domain)


class ColumnMean(ColumnAggregateMetric):
    metric_name = "column.aggregate.mean"

    @metric(engine=PandasExecutionEngine)
    def _pandas(engine, domain: pd.Series, **kwargs):
        """Mean Metric Function"""
        return domain.mean()

    @metric(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy(engine, domain: sa.column, **kwargs):
        return sa.func.avg(domain)


class ColumnValuesInSet(ColumnMapMetric):
    metric_name = "column_values.in_set"

    value_keys = ("value_set",)

    @metric(engine=PandasExecutionEngine)
    def _pandas(engine, domain: pd.Series, value_set: Iterable = None, **kwargs):
        if value_set is None:
            # Vacuously true
            return np.ones(len(domain), dtype=np.bool_)
        if pd.api.types.is_datetime64_any_dtype(domain):
            parsed_value_set = engine.parse_value_set(value_set=value_set)
        else:
            parsed_value_set = value_set

        return domain.isin(parsed_value_set)

    @metric(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy(engine, domain: sa.column, value_set: Iterable = None, **kwargs):
        if value_set is None:
            # Vacuously true
            return True
        # TODO: check type of column
        if False:
            parsed_value_set = engine.parse_value_set(value_set=value_set)
        else:
            parsed_value_set = value_set

        return column.in_(tuple(parsed_value_set))


class ColumnValuesNotNull(ColumnMapMetric):
    metric_name = "column_values.nonnull"

    @metric(engine=PandasExecutionEngine)
    def _pandas(engine, domain: pd.Series, **kwargs):
        return ~domain.isnull()

    @metric(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy_nonnull_map_metric(engine, domain: sa.column, **kwargs):
        import sqlalchemy as sa

        return sa.not_(domain.is_(None))

    @metric(engine=SparkDFExecutionEngine)
    def _spark_null_map_metric(engine, domain: "pyspark.sql.Column", **kwargs):
        return domain.isNotNull()
