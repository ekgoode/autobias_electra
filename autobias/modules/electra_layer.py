from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from transformers import ElectraConfig, ElectraModel

from autobias import config
from autobias.config import TRANSFORMER_CACHE_DIR
from autobias.modules.layers import Layer
from autobias.utils.ops import load_and_log


@dataclass
class ElectraOutput:
    embeddings: torch.Tensor
    hidden_states: List[torch.Tensor]
    pooled: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    other_features: Optional[Dict] = None


class ElectraLayer(Layer):  # Renamed for ELECTRA functionality but keeping class name to avoid breaking references
    """Wraps transformer's ELECTRA modules as a `Layer`"""

    def __init__(self, electra_model: str, max_layer=None, pool=True, freeze_embeddings=False):
        super().__init__()
        self.freeze_embeddings = freeze_embeddings
        self.config = ElectraConfig.from_pretrained(electra_model, cache_dir=TRANSFORMER_CACHE_DIR)
        self.config.output_hidden_states = True  # Enable hidden states for compatibility
        self.model = ElectraModel.from_pretrained(electra_model, config=self.config, cache_dir=TRANSFORMER_CACHE_DIR)
        self.pool = pool
        self.max_layer = max_layer
        self.electra_model = electra_model

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Adjust attention mask for compatibility
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError()

        dtype = self.model.embeddings.word_embeddings.weight.dtype
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Forward pass through ELECTRA
        with torch.no_grad() if self.freeze_embeddings else torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=self.config.output_hidden_states,
            )

        last_hidden_state = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        pooled_output = last_hidden_state[:, 0, :] if self.pool else None  # Use CLS token as pooled output

        return ElectraOutput(
            embeddings=last_hidden_state,
            hidden_states=hidden_states,
            pooled=pooled_output,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

    def reset_parameters(self):
        # Load pretrained weights for ELECTRA
        state_dict = self.model.state_dict()
        load_and_log(self.model, state_dict)
