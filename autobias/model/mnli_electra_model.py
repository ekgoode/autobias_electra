from dataclasses import dataclass
from os import makedirs
from os.path import join, exists
from typing import List, Tuple, Any, Dict, Iterable, Union, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from autobias import config
from autobias.datasets.dataset import Dataset
from autobias.datasets.entailment_datasets import load_hypothesis_bias, TextPairExample
from autobias.model.model import Model, Predictor
from autobias.modules.electra_layer import ElectraLayer, ElectraOutput
from autobias.modules.layers import Mapper, Layer
from autobias.modules.word_and_char_encoder import WordAndCharEncoder
from autobias.utils import ops, py_utils
from autobias.utils.ops import Mask, collate_flat_dict
from autobias.utils.process_par import Processor, process_par
from autobias.utils.tokenizer import NltkAndPunctTokenizer


@dataclass
class TextPairTensors:
    a_embed: Union[List[torch.Tensor], torch.Tensor]
    b_embed: Union[List[torch.Tensor], torch.Tensor]
    a_mask: Mask
    b_mask: Mask
    other_features: Optional[Dict]

    def to(self, device):
        return TextPairTensors(
            [x.to(device) for x in self.a_embed],
            [x.to(device) for x in self.b_embed],
            self.a_mask.to(device),
            self.b_mask.to(device),
            None if self.other_features is None else {
                k: v.to(device) for k, v in self.other_features.items()
            }
        )

    def pin_memory(self):
        return TextPairTensors(
            [x.pin_memory() for x in self.a_embed],
            [x.pin_memory() for x in self.b_embed],
            self.a_mask.pin_memory(),
            self.b_mask.pin_memory(),
            None if self.other_features is None else {
                k: v.pin_memory() for k, v in self.other_features.items()
            }
        )


@dataclass
class DualTokenizedExample:
    """
    Text-pair example encoded both as ELECTRA ids and as a pair of token sequences
    """
    example_id: str

    # ELECTRA data
    electra_input_ids: np.ndarray
    segment2_start: int

    # Token data
    a_tokens: Union[List[str], Tuple]
    b_tokens: Union[List[str], Tuple]

    label: Any
    other_features: Dict

    def get_len(self):
        return len(self.electra_input_ids)


@dataclass
class ElectraAndEmbedOutput:
    """
    `DualTokenizedExample` that has been embedded both by ELECTRA and by word vectors
    """
    electra_embed: torch.Tensor
    electra_hidden_states: List[torch.Tensor]
    electra_pooled: torch.Tensor
    electra_token_type_ids: torch.Tensor
    electra_attention_mask: torch.Tensor

    mask_a: Mask
    mask_b: Mask
    embed_a: torch.Tensor
    embed_b: torch.Tensor

    other_features: Dict

    def get_electra_output(self):
        return ElectraOutput(
            self.electra_embed, self.electra_hidden_states, self.electra_pooled,
            self.electra_token_type_ids, self.electra_attention_mask, None
        )

    def get_text_pair_tensors(self):
        return TextPairTensors(
            self.embed_a, self.embed_b, self.mask_a, self.mask_b, None
        )


class ElectraAndEmbedModel(Model):
    """Encodes text pairs as an `ElectraAndEmbedOutput`, which is passed to `self.predictor`"""

    def __init__(self, electra_model, max_seq_len, tokenizer, encoder: WordAndCharEncoder,
                 predictor, needs_pooled=True):
        super().__init__()
        if predictor is None:
            raise ValueError()
        self.electra_model = electra_model
        self.max_seq_len = max_seq_len
        self._electra_tok = None
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.electra = ElectraLayer(electra_model, pool=needs_pooled)
        self.predictor = predictor
        self._tok = None
        self.needs_pooled = needs_pooled

    def get_electra_tokenizer(self):
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.electra_model)
        return self._tok

    def preprocess_datasets(self, datasets: List[Dataset], n_processes=None):
        tokenized_datasets: List[List[DualTokenizedExample]] = []
        for ds in datasets:
            do_cache = self.electra_model == "google/electra-small-discriminator" and self.tokenizer is not None
            if do_cache:
                cache = join(config.DUAL_TOKENIZED_CACHE, ds.fullname + ".pkl")
                if exists(cache):
                    tokenized_datasets.append(py_utils.load_pickle(cache))
                    continue

            tokenized_datasets.append(dual_tokenize_dataset(
                ds, self.get_electra_tokenizer(), self.tokenizer, self.max_seq_len, n_processes))

            if do_cache:
                makedirs(config.DUAL_TOKENIZED_CACHE, exist_ok=True)
                py_utils.write_pickle(tokenized_datasets[-1], cache)

        if self.tokenizer is not None:
            voc = set()
            for ds in tokenized_datasets:
                for x in ds:
                    voc.update(x.a_tokens)
                    voc.update(x.b_tokens)

            self.encoder.set_vocab(voc)

            for ds in tokenized_datasets:
                for ex in ds:
                    ex.a_tokens = self.encoder.tensorize(ex.a_tokens)
                    ex.b_tokens = self.encoder.tensorize(ex.b_tokens)

        for tokenized, ds in zip(tokenized_datasets, datasets):
            self.predictor.preprocess_dataset(False, tokenized, ds)

        return tokenized_datasets

    def forward(self, features, label=None, **kwargs):
        if self.predictor.has_batch_loss() and kwargs.get("mode") == "loss":
            return self.predictor(features, label, **kwargs)

        input_ids, segment_ids, mask, a_embed, a_mask, b_embed, b_mask, other_features = features
        electra_out: ElectraOutput = self.electra(input_ids, segment_ids, mask)

        if self.encoder is not None:
            a_embed = self.encoder(a_embed)
            b_embed = self.encoder(b_embed)
        else:
            a_embed, b_embed = None, None

        features = ElectraAndEmbedOutput(
            electra_out.embeddings, electra_out.hidden_states, electra_out.pooled,
            electra_out.token_type_ids, electra_out.attention_mask,
            a_mask, b_mask, a_embed, b_embed, other_features
        )
        return self.predictor(features, label, **kwargs)

    def has_batch_loss(self):
        return self.predictor.has_batch_loss()

    def get_state(self):
        return (None if self.encoder is None else self.encoder.get_state()), \
               self.electra.state_dict(), \
               self.predictor.get_state()

    def load_state(self, state):
        if self.encoder is not None:
            self.encoder.load_state(state[0])
        self.electra.load_state_dict(state[1])
        self.predictor.load_state(state[2])
