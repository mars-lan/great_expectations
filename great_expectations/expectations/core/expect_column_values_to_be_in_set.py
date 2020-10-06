from typing import Optional

from great_expectations.core.expectation_configuration import ExpectationConfiguration

from ..expectation import (
    ColumnMapDatasetExpectation,
    InvalidExpectationConfigurationError,
)

try:
    import sqlalchemy as sa
except ImportError:
    pass


class ExpectColumnValuesToBeInSet(ColumnMapDatasetExpectation):
    """Expect each column value to be in a given set.

    For example:
    ::

        # my_df.my_col = [1,2,2,3,3,3]
        >>> my_df.expect_column_values_to_be_in_set(
            "my_col",
            [2,3]
        )
        {
          "success": false
          "result": {
            "unexpected_count": 1
            "unexpected_percent": 16.66666666666666666,
            "unexpected_percent_nonmissing": 16.66666666666666666,
            "partial_unexpected_list": [
              1
            ],
          },
        }

    expect_column_values_to_be_in_set is a \
    :func:`column_map_expectation <great_expectations.execution_engine.execution_engine.MetaExecutionEngine
    .column_map_expectation>`.

    Args:
        column (str): \
            The column name.
        value_set (set-like): \
            A set of objects used for comparison.

    Keyword Args:
        mostly (None or a float between 0 and 1): \
            Return `"success": True` if at least mostly fraction of values match the expectation. \
            For more detail, see :ref:`mostly`.
        parse_strings_as_datetimes (boolean or None) : If True values provided in value_set will be parsed as \
            datetimes before making comparisons.

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
        :func:`expect_column_values_to_not_be_in_set \
        <great_expectations.execution_engine.execution_engine.ExecutionEngine
        .expect_column_values_to_not_be_in_set>`

    """

    map_metric = "column_values.in_set"
    metric_dependencies = (
        "column_values.in_set.count",
        "column_values.nonnull.count",
    )
    success_keys = (
        "value_set",
        "mostly",
        "parse_strings_as_datetimes",
    )

    default_kwarg_values = {"value_set": None, "parse_strings_as_datetimes": False}

    def validate_configuration(self, configuration: Optional[ExpectationConfiguration]):
        super().validate_configuration(configuration)
        if configuration is None:
            configuration = self.configuration
        try:
            assert "value_set" in configuration.kwargs, "value_set is required"
            assert isinstance(
                configuration.kwargs["value_set"], (list, set)
            ), "value_set must be a list or a set"
        except AssertionError as e:
            raise InvalidExpectationConfigurationError(str(e))
        return True
