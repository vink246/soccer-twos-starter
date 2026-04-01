"""
Backward-compatible shim: re-exports ``soccer_rl.common.training_utils``.

Prefer importing from ``soccer_rl.common.training_utils`` in new code.
"""

from soccer_rl.common.training_utils import (  # noqa: F401
    PlotCallback,
    ProgressPrintCallback,
    create_rllib_env,
    get_num_gpus,
    has_matplotlib,
    load_config,
    print_gpu_status,
)
