from functools import wraps
from typing import Callable, Optional, Type, Union

from great_expectations.core import ExpectationConfiguration
from great_expectations.execution_engine import ExecutionEngine
from great_expectations.validator.validation_graph import MetricEdgeKey

try:
    import sqlalchemy as sa
except ImportError:
    sa = None


def metric(engine: Union[Callable, Type[ExecutionEngine]], **kwargs):
    """The metric decorator annotates a method """

    def wrapper(metric_fn: Callable):
        @wraps(metric_fn)
        def inner_func(*args, **kwargs):
            return metric_fn(*args, **kwargs)

        inner_func._engine = engine
        inner_func._metric_fn_kwargs = kwargs
        return inner_func

    return wrapper


class MetaMetric(type):
    """MetaMetric registers metrics as they are defined."""

    def __new__(cls, clsname, bases, attrs):
        newclass = super().__new__(cls, clsname, bases, attrs)
        for attr in attrs.keys():
            engine = getattr(attrs[attr], "_engine", None)
            if engine:
                if issubclass(engine, ExecutionEngine):
                    decorator_name = attrs.get("engine_decorator")
                    for base in bases:
                        if decorator_name:
                            break
                        decorator_name = getattr(base, "engine_decorator", None)
                    if not decorator_name:
                        decorator_name = "metric"
                    decorator = getattr(engine, decorator_name, None)
                elif isinstance(engine, Callable):
                    decorator = engine
                else:
                    decorator = None
                if decorator:
                    setattr(newclass, attr, decorator(newclass)(attrs[attr]))
        return newclass


class Metric(metaclass=MetaMetric):
    metric_name = "_abstract"
    domain_keys = tuple()
    value_keys = tuple()
    bundle_computation = False

    def get_evaluation_dependencies(
        self,
        metric: MetricEdgeKey,
        configuration: Optional[ExpectationConfiguration] = None,
        execution_engine: Optional[ExecutionEngine] = None,
        runtime_configuration: Optional[dict] = None,
    ):
        """This should return a dictionary:

        {
          "dependency_name": MetricEdgeKey,
          ...
        }
        """
        return dict()


class ColumnMapMetric(Metric):
    domain_keys = (
        "batch_id",
        "table",
        "column",
        "row_condition",
        "condition_parser",
    )
    filter_column_isnull = True
    engine_decorator = "column_map_metric"

    @classmethod
    def get_evaluation_dependencies(
        cls,
        metric: MetricEdgeKey,
        configuration: Optional[ExpectationConfiguration] = None,
        execution_engine: Optional[ExecutionEngine] = None,
        runtime_configuration: Optional[dict] = None,
    ):
        """This should return a dictionary:

        {
          "dependency_name": MetricEdgeKey,
          ...
        }
        """
        metric_name = metric.metric_name
        base_metric_value_kwargs = {
            k: v for k, v in metric.metric_value_kwargs.items() if k != "result_format"
        }
        if metric_name.endswith(".count"):
            return {
                metric_name: MetricEdgeKey(
                    metric_name[: -len(".count")],
                    metric.metric_domain_kwargs,
                    base_metric_value_kwargs,
                )
            }

        if metric_name.endswith(".unexpected_values"):
            return {
                metric_name: MetricEdgeKey(
                    metric_name[: -len(".unexpected_values")],
                    metric.metric_domain_kwargs,
                    base_metric_value_kwargs,
                )
            }

        if metric_name.endswith(".unexpected_index_list"):
            return {
                metric_name: MetricEdgeKey(
                    metric_name[: -len(".unexpected_index_list")],
                    metric.metric_domain_kwargs,
                    base_metric_value_kwargs,
                )
            }

        if metric_name.endswith(".unexpected_value_counts"):
            return {
                metric_name: MetricEdgeKey(
                    metric_name[: -len(".unexpected_value_counts")],
                    metric.metric_domain_kwargs,
                    base_metric_value_kwargs,
                )
            }

        if metric_name.endswith(".unexpected_rows"):
            return {
                metric_name: MetricEdgeKey(
                    metric_name[: -len(".unexpected_rows")],
                    metric.metric_domain_kwargs,
                    base_metric_value_kwargs,
                )
            }

        return dict()


class ColumnAggregateMetric(Metric):
    domain_keys = (
        "batch_id",
        "table",
        "column",
    )
    bundle_computation = True
    filter_column_isnull = True
