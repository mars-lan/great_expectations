from typing import Optional

import pandas as pd

from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.execution_engine import ExecutionEngine, PandasExecutionEngine

from ...data_asset.util import parse_result_format
from ...exceptions import InvalidExpectationConfigurationError
from ...execution_engine.sqlalchemy_execution_engine import SqlAlchemyExecutionEngine
from ..expectation import ColumnMapDatasetExpectation, Expectation, _format_map_output
from ..registry import extract_metrics

try:
    import sqlalchemy as sa
except ImportError:
    pass


class ExpectColumnValuesToBeOfType(ColumnMapDatasetExpectation):
    map_metric = "column_values.value_length_between"
    metric_dependencies = (
        "column_values.value_length_between.count",
        "column_values.nonnull.count",
    )
    success_keys = (
        "min_value",
        "max_value",
        "strict_min",
        "strict_max",
        "mostly",
    )

    default_kwarg_values = {
        "row_condition": None,
        "condition_parser": None,
        "min_value": None,
        "max_value": None,
        "strict_min": None,
        "strict_max": None,
        "mostly": 1,
        "result_format": "BASIC",
        "include_config": True,
        "catch_exceptions": False,
    }

    def validate_configuration(self, configuration: Optional[ExpectationConfiguration]):
        super().validate_configuration(configuration)

        if configuration is None:
            configuration = self.configuration

        try:
            assert configuration.kwargs.get("min_value") or configuration.kwargs.get(
                "max_value"
            ), "min_value and max_value cannot both be None"
            if configuration.kwargs.get("min_value"):
                assert float(
                    configuration.kwargs.get("min_value")
                ).is_integer(), "min_value and max_value must be integers"
            if configuration.kwargs.get("max_value"):
                assert float(
                    configuration.kwargs.get("max_value")
                ).is_integer(), "min_value and max_value must be integers"
        except AssertionError as e:
            raise InvalidExpectationConfigurationError(str(e))
        return True

    # @PandasExecutionEngine.column_map_metric(
    #     metric_name="column_values.value_length_between",
    #     metric_domain_keys=ColumnMapDatasetExpectation.domain_keys,
    #     metric_value_keys=("min_value", "max_value", "strict_min", "strict_max"),
    #     metric_dependencies=tuple(),
    #     filter_column_isnull=True,
    # )
    def _pandas_value_length_between(
        self,
        series: pd.Series,
        metrics: dict,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        runtime_configuration: dict = None,
        filter_column_isnull: bool = True,
    ):
        min_value = metric_value_kwargs["min_value"]
        max_value = metric_value_kwargs["max_value"]
        strict_min = metric_value_kwargs["strict_min"]
        strict_max = metric_value_kwargs["strict_max"]

        column_lengths = series.astype(str).str.len()

        if min_value is not None and max_value is not None:
            if strict_min and strict_max:
                metric_series = column_lengths.between(
                    min_value, max_value, inclusive=False
                )
            elif strict_min and not strict_max:
                metric_series = (column_lengths > min_value) & (
                    column_lengths <= max_value
                )
            elif not strict_min and strict_max:
                metric_series = (column_lengths >= min_value) & (
                    column_lengths < max_value
                )
            elif not strict_min and not strict_max:
                metric_series = column_lengths.between(
                    min_value, max_value, inclusive=True
                )
        elif min_value is None and max_value is not None:
            if strict_max:
                metric_series = column_lengths < max_value
            else:
                metric_series = column_lengths <= max_value
        elif min_value is not None and max_value is None:
            if strict_min:
                metric_series = column_lengths > min_value
            else:
                metric_series = column_lengths >= min_value

        return pd.DataFrame({"column_values.value_length_between": metric_series})

    # @SqlAlchemyExecutionEngine.column_map_metric(
    #     metric_name="column_values.value_length_between",
    #     metric_domain_keys=ColumnMapDatasetExpectation.domain_keys,
    #     metric_value_keys=("min_value", "max_value", "strict_min", "strict_max"),
    #     metric_dependencies=tuple(),
    #     filter_column_isnull=True,
    # )
    def _sqlalchemy_value_length_between(
        self,
        column: sa.column,
        metrics: dict,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        runtime_configuration: dict = None,
        filter_column_isnull: bool = True,
    ):
        min_value = metric_value_kwargs["min_value"]
        max_value = metric_value_kwargs["max_value"]
        strict_min = metric_value_kwargs["strict_min"]
        strict_max = metric_value_kwargs["strict_max"]

        if min_value is not None and max_value is not None:
            if strict_min and strict_max:
                return sa.and_(
                    sa.func.length(column) > min_value,
                    sa.func.length(column) < max_value,
                )
            elif strict_min and not strict_max:
                return sa.and_(
                    sa.func.length(column) > min_value,
                    sa.func.length(column) <= max_value,
                )
            elif not strict_min and strict_max:
                return sa.and_(
                    sa.func.length(column) >= min_value,
                    sa.func.length(column) < max_value,
                )
            elif not strict_min and not strict_max:
                return sa.and_(
                    sa.func.length(column) >= min_value,
                    sa.func.length(column) <= max_value,
                )
        elif min_value is None and max_value is not None:
            if strict_max:
                return sa.func.length(column) < max_value
            else:
                return sa.func.length(column) <= max_value
        elif min_value is not None and max_value is None:
            if strict_min:
                return sa.func.length(column) > min_value
            else:
                return sa.func.length(column) >= min_value

    @Expectation.validates(metric_dependencies=metric_dependencies)
    def _validates(
        self,
        configuration: ExpectationConfiguration,
        metrics: dict,
        runtime_configuration: dict = None,
        execution_engine: ExecutionEngine = None,
    ):
        metric_dependencies = self.get_validation_dependencies(
            configuration=configuration,
            execution_engine=execution_engine,
            runtime_configuration=runtime_configuration,
        )["metrics"]
        metric_vals = extract_metrics(
            metric_dependencies, metrics, configuration, runtime_configuration
        )
        mostly = self.get_success_kwargs().get(
            "mostly", self.default_kwarg_values.get("mostly")
        )
        if runtime_configuration:
            result_format = runtime_configuration.get(
                "result_format",
                configuration.kwargs.get(
                    "result_format", self.default_kwarg_values.get("result_format")
                ),
            )
        else:
            result_format = configuration.kwargs.get(
                "result_format", self.default_kwarg_values.get("result_format")
            )
        return _format_map_output(
            result_format=parse_result_format(result_format),
            success=(
                metric_vals.get("column_values.value_length_between.count")
                / metric_vals.get("column_values.nonnull.count")
            )
            >= mostly,
            element_count=metric_vals.get("column_values.count"),
            nonnull_count=metric_vals.get("column_values.nonnull.count"),
            unexpected_count=metric_vals.get("column_values.nonnull.count")
            - metric_vals.get("column_values.value_length_between.count"),
            unexpected_list=metric_vals.get(
                "column_values.value_length_between.unexpected_values"
            ),
            unexpected_index_list=metric_vals.get(
                "column_values.value_length_between.unexpected_index"
            ),
        )
