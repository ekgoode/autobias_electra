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

    a_mask: Mask
    b_mask: Mask
    a_embed: torch.Tensor
    b_embed: torch.Tensor

    other_features: Dict

    def get_electra_output(self):
        return ElectraOutput(
            self.electra_embed, self.electra_hidden_states, self.electra_pooled,
            self.electra_token_type_ids, self.electra_attention_mask, None
        )

    def get_text_pair_tensors(self):
        return TextPairTensors(
            self.a_embed, self.b_embed, self.a_mask, self.b_mask, None
        )


class FromPooled(Predictor):
    def __init__(self, mapper):
        super().__init__()
        self.mapper = mapper

    def reset_parameters(self):
        self.mapper.reset_parameters()

    def forward(self, electra_out: ElectraOutput, labels=None, **kwargs):
        return self.mapper(electra_out.electra_pooled)
    
class FromEmbeddingPredictor(Predictor):
    def __init__(self,predictor):
        super().__init__()
        self.predictor = predictor

    def reset_parameters(self):
        self.predictor.reset_parameters()

    def has_batch_loss(self):
        return self.predictor.has_batch_loss()
    
    def forward(self, features: TextPairTensors, label=None, **kwargs):
        return self.predictor(features, label, **kwargs)

class BifusePredictor(Predictor):
    def __init__(self, pre_mapper: Mapper, bifuse_layer, post_mapper: Mapper, pooler, pooled_mapper):
        super().__init__()
        self.pre_mapper = pre_mapper
        self.bifuse_layer = bifuse_layer
        self.post_mapper = post_mapper
        self.pooler = pooler
        self.pooled_mapper = pooled_mapper

    def reset_parameters(self):
        if self.pre_mapper:
            self.pre_mapper.reset_parameters()
        self.bifuse_layer.reset_parameters()
        if self.post_mapper:
            self.post_mapper.reset_parameters()
        self.pooler.reset_parameters()
        self.pooled_mapper.reset_parameters()

    def forward(self, features: TextPairTensors, label=None, **kwargs):
        a_embed, a_mask = features.a_embed, features.a_mask
        b_embed, b_mask = features.b_embed, features.b_mask

        if self.pre_mapper:
            a_embed = self.pre_mapper(a_embed, a_mask)
            b_embed = self.pre_mapper(b_embed, b_mask)

        a_embed, b_embed = self.bifuse_layer(a_embed, b_embed, a_mask, b_mask)

        if self.post_mapper:
            a_embed = self.post_mapper(a_embed, a_mask)
            b_embed = self.post_mapper(b_embed, b_mask)

        pooled = torch.cat([self.pooler(a_embed, a_mask), self.pooler(b_embed, b_mask)], 1)
        pooled = self.pooled_mapper(pooled)

        return pooled
    
class DualTokenizer(Processor):
    """Applies tokenizers to build `DualTokenizedExample` from `TextPairExample`"""

    def __init__(self, electra_tokenizer, tokenizer, max_seq_len):
        self.electra_tokenizer = electra_tokenizer
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def process(self, data: Iterable[TextPairExample]):
        out = []
        for x in data:
            # Token-level embeddings
            if self.tokenizer is None:
                a_tokens, b_tokens = None, None
            else:
                a_tokens = self.tokenizer.tokenize(x.text_a)
                b_tokens = self.tokenizer.tokenize(x.text_b)

            # ELECTRA tokenization
            encoded = self.electra_tokenizer(
                x.text_a,
                x.text_b,
                truncation=True,
                max_length=self.max_seq_len,
                return_token_type_ids=True,
                return_attention_mask=True
            )

            electra_input_ids = encoded["input_ids"]
            segment2_start = len(encoded["input_ids"]) - len(encoded["token_type_ids"])

            out.append(DualTokenizedExample(
                x.example_id,
                np.array(electra_input_ids, dtype=np.int64),
                segment2_start,
                a_tokens,
                b_tokens,
                x.label,
                x.other_features or {}
            ))
        return out


def dual_tokenize_dataset(
    dataset, electra_tokenizer: AutoTokenizer,
    tokenizer: NltkAndPunctTokenizer, max_seq_len, n_processes
):
    """
    Tokenizes a dataset for ELECTRA and word embeddings simultaneously.

    Args:
        dataset (Dataset): The dataset to tokenize.
        electra_tokenizer (AutoTokenizer): ELECTRA tokenizer.
        tokenizer (NltkAndPunctTokenizer): Tokenizer for word-level embeddings.
        max_seq_len (int): Maximum sequence length for tokenization.
        n_processes (Optional[int]): Number of processes for parallel processing.

    Returns:
        List[DualTokenizedExample]: A list of tokenized examples.
    """
    tok = DualTokenizer(electra_tokenizer, tokenizer, max_seq_len)
    return process_par(dataset.load(), tok, n_processes, name="tokenizing")

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

    def get_collate_fn(self):
        """
        Define a collate function for batching examples.
        """
        def collate(batch: List[DualTokenizedExample]):
            max_seq_len = max(len(x.electra_input_ids) for x in batch)
            sz = len(batch)

            # Initialize tensors
            input_ids = np.zeros((sz, max_seq_len), dtype=np.int64)
            segment_ids = torch.zeros(sz, max_seq_len, dtype=torch.int64)
            mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)

            for i, ex in enumerate(batch):
                input_ids[i, :len(ex.electra_input_ids)] = ex.electra_input_ids
                segment_ids[i, ex.segment2_start:len(ex.electra_input_ids)] = 1
                mask[i, :len(ex.electra_input_ids)] = 1

            input_ids = torch.as_tensor(input_ids)
            label_ids = torch.as_tensor([x.label for x in batch], dtype=torch.int64)

            if self.encoder is not None:
                a_embed = ops.collate_list_of_tuples([x.a_tokens for x in batch])
                a_mask = ops.build_masks([len(x.a_tokens[0]) for x in batch])

                b_embed = ops.collate_list_of_tuples([x.b_tokens for x in batch])
                b_mask = ops.build_masks([len(x.b_tokens[0]) for x in batch])
            else:
                a_embed, a_mask, b_embed, b_mask = [None] * 4

            if batch[0].other_features is not None and len(batch[0].other_features) > 0:
                other_features = collate_flat_dict([x.other_features for x in batch])
            else:
                other_features = {}

            return (input_ids, segment_ids, mask, a_embed, a_mask, b_embed, b_mask, other_features), label_ids

        return collate
    
    def preprocess_datasets(self, datasets: List[Dataset], n_processes=None):
        tokenized_datasets: List[List[DualTokenizedExample]] = []
        for ds in datasets:
            do_cache = self.electra_model == "google/electra-small-discriminator" and self.tokenizer is not None
            if do_cache:
                cache = join(config.DUAL_TOKENIZED_CACHE, ds.fullname + ".pkl")
                if exists(cache):
                    tokenized_datasets.append(py_utils.load_pickle(cache))
                    continue

            # Tokenize datasets
            tokenized_datasets.append(dual_tokenize_dataset(
                ds, self.get_electra_tokenizer(), self.tokenizer, self.max_seq_len, n_processes))

            # Cache datasets if needed
            if do_cache:
                makedirs(config.DUAL_TOKENIZED_CACHE, exist_ok=True)
                py_utils.write_pickle(tokenized_datasets[-1], cache)

        # Set vocabulary and tensorize tokens if encoder is used
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

        # Preprocess datasets with the predictor
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
