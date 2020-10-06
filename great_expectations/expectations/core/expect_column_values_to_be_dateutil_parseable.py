from datetime import datetime
from typing import Dict, List, Optional, Union

import dateutil
import numpy as np
import pandas as pd
from dateutil.parser import parse

from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.execution_engine import (
    ExecutionEngine,
    PandasExecutionEngine,
    SparkDFExecutionEngine,
)

from ...core.batch import Batch
from ...data_asset.util import parse_result_format
from ...execution_engine.sqlalchemy_execution_engine import SqlAlchemyExecutionEngine
from ..expectation import (
    ColumnMapDatasetExpectation,
    Expectation,
    InvalidExpectationConfigurationError,
    _format_map_output,
)
from ..registry import extract_metrics, get_metric_kwargs

try:
    import sqlalchemy as sa
except ImportError:
    pass


class ExpectColumnValuesToBeDateutilParseable(ColumnMapDatasetExpectation):
    """Expect column entries to be parsable using dateutil.

    expect_column_values_to_be_dateutil_parseable is a \
    :func:`column_map_expectation <great_expectations.execution_engine.execution_engine.MetaExecutionEngine
    .column_map_expectation>`.

    Args:
        column (str): \
            The column name.

    Keyword Args:
        mostly (None or a float between 0 and 1): \
            Return `"success": True` if at least mostly fraction of values match the expectation. \
            For more detail, see :ref:`mostly`.

    Other Parameters:
        result_format (str or None): \
            Which output mode to use: `BOOLEAN_ONLY`, `BASIC`, `COMPLETE`, or `SUMMARY`.
            For more detail, see :ref:`result_format <result_format>`.
        include_config (boolean): \
            If True, then include the expectation config as part of the result object. \
            For more detail, see :ref:`include_config`.
        catch_exceptions (boolean or None): \
            If True, then catch exceptions and include them as part of the result object. \
            For more detail, see :ref:`catch_exceptions`.
        meta (dict or None): \
            A JSON-serializable dictionary (nesting allowed) that will be included in the output without \
            modification. For more detail, see :ref:`meta`.

    Returns:
        An ExpectationSuiteValidationResult

        Exact fields vary depending on the values passed to :ref:`result_format <result_format>` and
        :ref:`include_config`, :ref:`catch_exceptions`, and :ref:`meta`.

    """

    map_metric = "column_values.dateutil_parsable"
    metric_dependencies = (
        "column_values.dateutil_parsable.count",
        "column_values.nonnull.count",
    )
    success_keys = ("mostly",)

    default_kwarg_values = {
        "row_condition": None,
        "condition_parser": None,  # we expect this to be explicitly set whenever a row_condition is passed
        "mostly": 1,
        "result_format": "BASIC",
        "include_config": True,
        "catch_exceptions": True,
    }

    def validate_configuration(self, configuration: Optional[ExpectationConfiguration]):
        super().validate_configuration(configuration)
        if configuration is None:
            configuration = self.configuration
        return True

    # @PandasExecutionEngine.column_map_metric(
    #     metric_name="column_values.dateutil_parsable",
    #     metric_domain_keys=ColumnMapDatasetExpectation.domain_keys,
    #     metric_value_keys=tuple(),
    #     metric_dependencies=tuple(),
    #     filter_column_isnull=True,
    # )
    def _pandas_column_values_dateutil_parsable(
        self,
        series: pd.Series,
        metrics: dict,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        runtime_configuration: dict = None,
        filter_column_isnull: bool = True,
    ):
        def is_parseable(val):
            try:
                if type(val) != str:
                    raise TypeError(
                        "Values passed to expect_column_values_to_be_dateutil_parseable must be of type string.\nIf you want to validate a column of dates or timestamps, please call the expectation before converting from string format."
                    )

                parse(val)
                return True

            except (ValueError, OverflowError):
                return False

        return pd.DataFrame(
            {"column_values.dateutil_parsable": series.map(is_parseable)}
        )

    # @SqlAlchemyExecutionEngine.column_map_metric(
    #     metric_name="column_values.dateutil_parsable",
    #     metric_domain_keys=ColumnMapDatasetExpectation.domain_keys,
    #     metric_value_keys=("regex",),
    #     metric_dependencies=tuple(),
    # )
    # def _sqlalchemy_dateutil_parsable(
    #     self,
    #     column: sa.column,
    #     regex: str,
    #     runtime_configuration: dict = None,
    #     filter_column_isnull: bool = True,
    # ):
    #     regex_expression = execution_engine._get_dialect_regex_expression(column, regex)
    #     if regex_expression is None:
    #         logger.warning(
    #             "Regex is not supported for dialect %s" % str(self.sql_engine_dialect)
    #         )
    #         raise NotImplementedError
    #
    #     return regex_expression
    #     if regex is None:
    #         # vacuously true
    #         return True
    #
    #     return column.in_(tuple(regex))
    #
    # @SparkDFExecutionEngine.column_map_metric(
    #     metric_name="column_values.dateutil_parsable",
    #     metric_domain_keys=ColumnMapDatasetExpectation.domain_keys,
    #     metric_value_keys=("regex",),
    #     metric_dependencies=tuple(),
    # )
    # def _spark_dateutil_parsable(
    #     self,
    #     data: "pyspark.sql.DataFrame",
    #     column: str,
    #     regex: str,
    #     runtime_configuration: dict = None,
    #     filter_column_isnull: bool = True,
    # ):
    #     import pyspark.sql.functions as F
    #
    #     if regex is None:
    #         # vacuously true
    #         return data.withColumn(column + "__success", F.lit(True))
    #
    #     return data.withColumn(column + "__success", F.col(column).isin(regex))

    # @Expectation.validates(metric_dependencies=metric_dependencies)
    def _validates(
        self,
        configuration: ExpectationConfiguration,
        metrics: dict,
        runtime_configuration: dict = None,
        execution_engine: ExecutionEngine = None,
    ):
        metric_dependencies = self.get_validation_dependencies(
            configuration, execution_engine, runtime_configuration
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

        if metric_vals.get("column_values.nonnull.count") > 0:
            success = metric_vals.get(
                "column_values.dateutil_parsable.count"
            ) / metric_vals.get("column_values.nonnull.count")
        else:
            # TODO: Setting this to 1 based on the notion that tests on empty columns should be vacuously true. Confirm.
            success = 1
        return _format_map_output(
            result_format=parse_result_format(result_format),
            success=success >= mostly,
            element_count=metric_vals.get("column_values.count"),
            nonnull_count=metric_vals.get("column_values.nonnull.count"),
            unexpected_count=metric_vals.get("column_values.nonnull.count")
            - metric_vals.get("column_values.dateutil_parsable.count"),
            unexpected_list=metric_vals.get(
                "column_values.dateutil_parsable.unexpected_values"
            ),
            unexpected_index_list=metric_vals.get(
                "column_values.dateutil_parsable.unexpected_index_list"
            ),
        )
