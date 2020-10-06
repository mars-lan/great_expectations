from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

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


class ExpectColumnValueLengthsToEqual(ColumnMapDatasetExpectation):
    """Expect column entries to be strings with length equal to the provided value.

    This expectation only works for string-type values. Invoking it on ints or floats will raise a TypeError.

    expect_column_values_to_be_between is a \
    :func:`column_map_expectation <great_expectations.execution_engine.execution_engine.MetaExecutionEngine
    .column_map_expectation>`.

    Args:
        column (str): \
            The column name.
        value (int or None): \
            The expected value for a column entry length.

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

    See Also:
        :func:`expect_column_value_lengths_to_be_between \
        <great_expectations.execution_engine.execution_engine.ExecutionEngine
        .expect_column_value_lengths_to_be_between>`

    """

    map_metric = "column_values.length_equals"
    metric_dependencies = (
        "column_values.length_equals.count",
        "column_values.nonnull.count",
        "column.value_lengths",
    )
    success_keys = ("value", "mostly", "parse_strings_as_datetimes")

    default_kwarg_values = {
        "row_condition": None,
        "condition_parser": None,  # we expect this to be explicitly set whenever a row_condition is passed
        "mostly": 1,
        "parse_strings_as_datetimes": None,
        "result_format": "BASIC",
        "include_config": True,
        "catch_exceptions": False,
    }

    def validate_configuration(self, configuration: Optional[ExpectationConfiguration]):
        super().validate_configuration(configuration)
        if configuration is None:
            configuration = self.configuration
        try:
            assert (
                "value" in configuration.kwargs
            ), "The length parameter 'value' is required"
            assert isinstance(
                configuration.kwargs["value"], (float, int)
            ), "given value must be numerical"
        except AssertionError as e:
            raise InvalidExpectationConfigurationError(str(e))
        return True

    # @PandasExecutionEngine.column_map_metric(
    #     metric_name="column_values.length_equals",
    #     metric_domain_keys=ColumnMapDatasetExpectation.domain_keys,
    #     metric_value_keys=("value",),
    #     metric_dependencies=("column.value_lengths",),
    #     filter_column_isnull=True,
    # )
    def _pandas_column_values_length_equals(
        self,
        series: pd.Series,
        metrics: dict,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        runtime_configuration: dict = None,
        filter_column_isnull: bool = True,
    ):
        """Tests whether or not value lengths equal threshold"""
        value = metric_value_kwargs["value"]
        length_equals = series.str.len() == value
        return pd.DataFrame({"column_values.length_equals": length_equals})

    """ A metric decorator for individual value lengths"""

    # @PandasExecutionEngine.metric(
    #     metric_name="column.value_lengths",
    #     metric_domain_keys=ColumnMapDatasetExpectation.domain_keys,
    #     metric_value_keys=(),
    #     metric_dependencies=tuple(),
    # )
    def _pandas_value_lengths(
        self,
        batches: Dict[str, Batch],
        execution_engine: PandasExecutionEngine,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        metrics: dict,
        runtime_configuration: dict = None,
    ):
        """Extracts lengths of individual entries"""
        series = execution_engine.get_domain_dataframe(
            domain_kwargs=metric_domain_kwargs, batches=batches
        )

        return series.str.len()

    @Expectation.validates(metric_dependencies=metric_dependencies)
    def _validates(
        self,
        configuration: ExpectationConfiguration,
        metrics: dict,
        runtime_configuration: dict = None,
        execution_engine: ExecutionEngine = None,
    ):
        validation_dependencies = self.get_validation_dependencies(
            configuration, execution_engine, runtime_configuration
        )["metrics"]
        # Extracting metrics
        metric_vals = extract_metrics(
            validation_dependencies, metrics, configuration, runtime_configuration
        )

        # Runtime configuration has preference
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
        mostly = self.get_success_kwargs().get(
            "mostly", self.default_kwarg_values.get("mostly")
        )
        if runtime_configuration:
            result_format = runtime_configuration.get(
                "result_format", self.default_kwarg_values.get("result_format")
            )
        else:
            result_format = self.default_kwarg_values.get("result_format")
        return _format_map_output(
            result_format=parse_result_format(result_format),
            success=(
                metric_vals.get("column_values.length_equals.count")
                / metric_vals.get("column_values.nonnull.count")
            )
            >= mostly,
            element_count=metric_vals.get("column_values.count"),
            nonnull_count=metric_vals.get("column_values.nonnull.count"),
            unexpected_count=metric_vals.get("column_values.nonnull.count")
            - metric_vals.get("column_values.length_equals.count"),
            unexpected_list=metric_vals.get(
                "column_values.length_equals.unexpected_values"
            ),
            unexpected_index_list=metric_vals.get(
                "column_values.length_equals.unexpected_index_list"
            ),
        )

    #
    # @renders(StringTemplate, modes=())
    # def lkjdsf(self, mode={prescriptive}, {descriptive}, {valiation}):
    #     return "I'm a thing"
