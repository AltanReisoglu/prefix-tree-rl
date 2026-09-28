"""prefix_tree_rl: critic'siz, token başına RL kredisi.

Çekirdek (saf Python):   PrefixTree, token_advantages, build_tree, render_tree
Torch (HF/TRL düzeni):  completion_advantages, grpo_advantages, dr_grpo_advantages, rloo_advantages
"""

from .tree import PrefixTree, build_tree, render_tree, token_advantages

__all__ = ["PrefixTree", "build_tree", "render_tree", "token_advantages"]

try:  # torch opsiyonel
    from .torch_ops import (
        broadcast,
        completion_advantages,
        dr_grpo_advantages,
        grpo_advantages,
        rloo_advantages,
    )

    __all__ += ["broadcast", "completion_advantages", "dr_grpo_advantages", "grpo_advantages", "rloo_advantages"]
except ImportError:
    pass

__version__ = "0.1.0"
