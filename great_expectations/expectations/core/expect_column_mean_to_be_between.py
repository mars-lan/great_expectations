from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from great_expectations.core.batch import Batch
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.execution_engine import ExecutionEngine, PandasExecutionEngine

from ..expectation import (
    ColumnMapDatasetExpectation,
    DatasetExpectation,
    Expectation,
    InvalidExpectationConfigurationError,
    _format_map_output,
)
from ..registry import extract_metrics


class ExpectColumnMeanToBeBetween(DatasetExpectation):
    """Expect the column mean to be between a minimum value and a maximum value (inclusive).

            expect_column_mean_to_be_between is a \
            :func:`column_aggregate_expectation \
            <great_expectations.execution_engine.execution_engine.MetaExecutionEngine.column_aggregate_expectation>`.

            Args:
                column (str): \
                    The column name.
                min_value (float or None): \
                    The minimum value for the column mean.
                max_value (float or None): \
                    The maximum value for the column mean.
                strict_min (boolean):
                    If True, the column mean must be strictly larger than min_value, default=False
                strict_max (boolean):
                    If True, the column mean must be strictly smaller than max_value, default=False

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

            Notes:
                These fields in the result object are customized for this expectation:
                ::

                    {
                        "observed_value": (float) The true mean for the column
                    }

                * min_value and max_value are both inclusive unless strict_min or strict_max are set to True.
                * If min_value is None, then max_value is treated as an upper bound.
                * If max_value is None, then min_value is treated as a lower bound.

            See Also:
                :func:`expect_column_median_to_be_between \
                <great_expectations.execution_engine.execution_engine.ExecutionEngine.expect_column_median_to_be_between>`

                :func:`expect_column_stdev_to_be_between \
                <great_expectations.execution_engine.execution_engine.ExecutionEngine.expect_column_stdev_to_be_between>`

            """

    # Setting necessary computation metric dependencies and defining kwargs, as well as assigning kwargs default values\
    metric_dependencies = ("column.aggregate.mean",)
    success_keys = ("min_value", "strict_min", "max_value", "strict_max")

    # Default values
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

    """ A Column Aggregate Metric Decorator for the Mean"""

    def validate_configuration(self, configuration: Optional[ExpectationConfiguration]):
        """
        Validates that a configuration has been set, and sets a configuration if it has yet to be set. Ensures that
        neccessary configuration arguments have been provided for the validation of the expectation.

        Args:
            configuration (OPTIONAL[ExpectationConfiguration]): \
                An optional Expectation Configuration entry that will be used to configure the expectation
        Returns:
            True if the configuration has been validated successfully. Otherwise, raises an exception
        """
        min_val = None
        max_val = None

        # Setting up a configuration
        super().validate_configuration(configuration)
        if configuration is None:
            configuration = self.configuration

        # Ensuring basic configuration parameters are properly set
        try:
            assert (
                "column" in configuration.kwargs
            ), "'column' parameter is required for column map expectations"
        except AssertionError as e:
            raise InvalidExpectationConfigurationError(str(e))

        # Validating that Minimum and Maximum values are of the proper format and type
        if "min_value" in configuration.kwargs:
            min_val = configuration.kwargs["min_value"]

        if "max_value" in configuration.kwargs:
            max_val = configuration.kwargs["max_value"]

        try:
            # Ensuring Proper interval has been provided
            assert (
                min_val is not None or max_val is not None
            ), "min_value and max_value cannot both be none"
            assert min_val is None or isinstance(
                min_val, (float, int)
            ), "Provided min threshold must be a number"
            assert max_val is None or isinstance(
                max_val, (float, int)
            ), "Provided max threshold must be a number"

        except AssertionError as e:
            raise InvalidExpectationConfigurationError(str(e))

        if min_val is not None and max_val is not None and min_val > max_val:
            raise InvalidExpectationConfigurationError(
                "Minimum Threshold cannot be larger than Maximum Threshold"
            )

        return True

    @Expectation.validates(metric_dependencies=metric_dependencies)
    def _validates(
        self,
        configuration: ExpectationConfiguration,
        metrics: dict,
        runtime_configuration: dict = None,
        execution_engine: ExecutionEngine = None,
    ):
        """Validates the given data against the set boundaries for mean to ensure it lies within proper range"""
        # Obtaining dependencies used to validate the expectation
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
        column_mean = metric_vals.get("column.aggregate.mean")

        # Obtaining components needed for validation
        min_value = self.get_success_kwargs(configuration).get("min_value")
        strict_min = self.get_success_kwargs(configuration).get("strict_min")
        max_value = self.get_success_kwargs(configuration).get("max_value")
        strict_max = self.get_success_kwargs(configuration).get("strict_max")

        # Checking if mean lies between thresholds
        if min_value is not None:
            if strict_min:
                above_min = column_mean > min_value
            else:
                above_min = column_mean >= min_value
        else:
            above_min = True

        if max_value is not None:
            if strict_max:
                below_max = column_mean < max_value
            else:
                below_max = column_mean <= max_value
        else:
            below_max = True

        success = above_min and below_max

        return {"success": success, "result": {"observed_value": column_mean}}
