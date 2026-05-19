"""Cross-condition experiment definitions."""

from __future__ import annotations

from .types import ExperimentSpec


BASE_TEST_DOMAINS = ("F130", "F130A", "F130C")
TRAIN_DOMAIN_GROUPS = (
    ("F130",),
    ("F130A",),
    ("F130C",),
    ("F130", "F130A"),
    ("F130", "F130C"),
    ("F130A", "F130C"),
    ("F130", "F130A", "F130C"),
)


def build_experiment_specs() -> list[ExperimentSpec]:
    """Return the seven training-domain combinations required by the protocol."""
    specs: list[ExperimentSpec] = []
    for index, train_domains in enumerate(TRAIN_DOMAIN_GROUPS, start=1):
        combo = "_".join(train_domains)
        specs.append(
            ExperimentSpec(
                name=f"exp_{index:02d}_train_{combo}",
                train_domains=tuple(train_domains),
                test_domains=BASE_TEST_DOMAINS,
            )
        )
    return specs
