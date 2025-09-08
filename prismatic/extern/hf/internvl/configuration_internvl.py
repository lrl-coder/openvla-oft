from typing import Any, Dict, List, Optional

from transformers.models.internvl import InternVLConfig


class OpenVLAInternVLConfig(InternVLConfig):
    model_type = "openvla_internvl"

    def __init__(
        self,
        # pad_to_multiple_of: int = 64,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        **kwargs: Any,
    ) -> None:
        # self.pad_to_multiple_of = pad_to_multiple_of

        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        super().__init__(**kwargs)
