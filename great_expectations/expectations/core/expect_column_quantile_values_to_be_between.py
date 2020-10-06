from typing import Dict, Optional

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationConfiguration
from great_expectations.core.batch import Batch
from great_expectations.exceptions import InvalidExpectationConfigurationError
from great_expectations.execution_engine import ExecutionEngine, PandasExecutionEngine
from great_expectations.expectations.expectation import DatasetExpectation, Expectation
from great_expectations.expectations.registry import extract_metrics


class ExpectColumnQuantileValuesToBeBetween(DatasetExpectation):
    """Expect specific provided column quantiles to be between provided minimum and maximum values.

           ``quantile_ranges`` must be a dictionary with two keys:

               * ``quantiles``: (list of float) increasing ordered list of desired quantile values

               * ``value_ranges``: (list of lists): Each element in this list consists of a list with two values, a lower \
                 and upper bound (inclusive) for the corresponding quantile.


           For each provided range:

               * min_value and max_value are both inclusive.
               * If min_value is None, then max_value is treated as an upper bound only
               * If max_value is None, then min_value is treated as a lower bound only

           The length of the quantiles list and quantile_values list must be equal.

           For example:
           ::

               # my_df.my_col = [1,2,2,3,3,3,4]
               >>> my_df.expect_column_quantile_values_to_be_between(
                   "my_col",
                   {
                       "quantiles": [0., 0.333, 0.6667, 1.],
                       "value_ranges": [[0,1], [2,3], [3,4], [4,5]]
                   }
               )
               {
                 "success": True,
                   "result": {
                     "observed_value": {
                       "quantiles: [0., 0.333, 0.6667, 1.],
                       "values": [1, 2, 3, 4],
                     }
                     "element_count": 7,
                     "missing_count": 0,
                     "missing_percent": 0.0,
                     "details": {
                       "success_details": [true, true, true, true]
                     }
                   }
                 }
               }

           `expect_column_quantile_values_to_be_between` can be computationally intensive for large datasets.

           expect_column_quantile_values_to_be_between is a \
           :func:`column_aggregate_expectation
           <great_expectations.execution_engine.MetaExecutionEngine.column_aggregate_expectation>`.

           Args:
               column (str): \
                   The column name.
               quantile_ranges (dictionary): \
                   Quantiles and associated value ranges for the column. See above for details.
               allow_relative_error (boolean): \
                   Whether to allow relative error in quantile communications on backends that support or require it.

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
               details.success_details

           See Also:
               :func:`expect_column_min_to_be_between \
               <great_expectations.execution_engine.execution_engine.ExecutionEngine.expect_column_min_to_be_between>`

               :func:`expect_column_max_to_be_between \
               <great_expectations.execution_engine.execution_engine.ExecutionEngine.expect_column_max_to_be_between>`

               :func:`expect_column_median_to_be_between \
               <great_expectations.execution_engine.execution_engine.ExecutionEngine.expect_column_median_to_be_between>`

           """

    metric_dependencies = ("column.aggregate.quantiles",)
    success_keys = (
        "quantile_ranges",
        "allow_relative_error",
    )
    default_kwarg_values = {
        "row_condition": None,
        "allow_relative_eror": None,
        "condition_parser": None,
        "quantile_ranges": None,
        "result_format": "BASIC",
        "allow_relative_error": False,
        "include_config": True,
        "catch_exceptions": False,
    }

    def validate_configuration(self, configuration: Optional[ExpectationConfiguration]):
        super().validate_configuration(configuration)

        # Ensuring necessary parameters are present and of the proper type
        min_val = None
        max_val = None

        # Testing that proper thresholds are in place
        if "min_value" in configuration.kwargs:
            min_val = configuration.kwargs["min_value"]

        if "max_value" in configuration.kwargs:
            max_val = configuration.kwargs["max_value"]

        try:
            assert (
                "column" in configuration.kwargs
            ), "'column' parameter is required for metric"
            assert (
                min_val is not None or max_val is not None
            ), "min_value and max_value cannot both be none"
            assert (
                "quantile_ranges" in configuration.kwargs
            ), "quantile ranges must be provided"
            assert (
                type(configuration.kwargs["quantile_ranges"]) == dict
            ), "quantile_ranges should be a dictionary"

        except AssertionError as e:
            raise InvalidExpectationConfigurationError(str(e))

        # Ensuring actual quantiles and their value ranges match up
        quantile_ranges = configuration.kwargs["quantile_ranges"]
        quantiles = quantile_ranges["quantiles"]
        quantile_value_ranges = quantile_ranges["value_ranges"]
        if "allow_relative_error" in configuration.kwargs:
            allow_relative_error = configuration.kwargs["allow_relative_error"]
        else:
            allow_relative_error = False

        if allow_relative_error is not False:
            raise ValueError(
                "PandasExecutionEngine does not support relative error in column quantiles."
            )

        if len(quantiles) != len(quantile_value_ranges):
            raise ValueError(
                "quntile_values and quantiles must have the same number of elements"
            )
        return True

    # @PandasExecutionEngine.metric(
    #        metric_name="column.aggregate.quantiles",
    #        metric_domain_keys=DatasetExpectation.domain_keys,
    #        metric_value_keys=("quantile_ranges",),
    #        metric_dependencies=tuple(),
    #        filter_column_isnull=False,
    #    )
    def _pandas_quantiles(
        self,
        batches: Dict[str, Batch],
        execution_engine: PandasExecutionEngine,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        metrics: dict,
        runtime_configuration: dict = None,
    ):

        """Quantile Function"""
        series = execution_engine.get_domain_dataframe(
            domain_kwargs=metric_domain_kwargs, batches=batches
        )
        quantile_ranges = metric_value_kwargs["quantile_ranges"]
        return series.quantile(
            tuple(quantile_ranges["quantiles"],), interpolation="nearest"
        ).tolist()

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

        quantile_vals = metric_vals.get("column.aggregate.quantiles")
        quantile_ranges = self.get_success_kwargs(configuration).get("quantile_ranges")
        quantiles = quantile_ranges["quantiles"]
        quantile_value_ranges = quantile_ranges["value_ranges"]

        # We explicitly allow "None" to be interpreted as +/- infinity
        comparison_quantile_ranges = [
            [
                -np.inf if lower_bound is None else lower_bound,
                np.inf if upper_bound is None else upper_bound,
            ]
            for (lower_bound, upper_bound) in quantile_value_ranges
        ]
        success_details = [
            range_[0] <= quantile_vals[idx] <= range_[1]
            for idx, range_ in enumerate(comparison_quantile_ranges)
        ]

        return {
            "success": np.all(success_details),
            "result": {
                "observed_value": {"quantiles": quantiles, "values": quantile_vals},
                "details": {"success_details": success_details},
            },
        }
