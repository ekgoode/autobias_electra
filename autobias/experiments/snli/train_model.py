import argparse
# Ignore tensor future works, thank you transformers for importing that
import json
import warnings
import sys
import importlib

sys.path.append('/content')
from autobias.training import optimizer
from autobias.model import snli_model
importlib.reload(snli_model)
from autobias.experiments.train_args import add_train_args
from autobias.model.snli_model import ElectraAndEmbedModel, FromPooled, \
  FromEmbeddingPredictor, BifusePredictor
from autobias.modules import layers
from autobias.modules.attention_layers import AttentionBiFuse, DotAttention
from autobias.modules.classifier_ensembles import ClfHead, ClfArgminEnsemble
from autobias.modules.word_and_char_encoder import WordAndCharEncoder
from autobias.training import train_utils
from autobias.utils.tokenizer import NltkAndPunctTokenizer

warnings.simplefilter(action='ignore', category=FutureWarning)

from autobias.argmin_modules.affine_nll import AffineNLL
from autobias.argmin_modules.argmin_function import NumpyOptimizer
from autobias.argmin_modules.argmin_transform import ArgminTransformFunction
from autobias.argmin_modules.l2_norm import L2NormPenalty

from autobias.datasets.entailment_datasets import MnliTrain, MnliDevMatched
from autobias.training.hard_easy_evaluator import ClfHardEasyEvaluator

from autobias.training.data_iterator import TorchDataIterator
from autobias.training.post_fit_rescaling_ensemble import FitRescaleParameters

from autobias.modules.layers import FullyConnected, Dropout, seq, MaxPooling
from autobias.training.data_batcher import SortedBatchSampler, \
  SubsetSampler, StratifiedSampler
from autobias.training.optimizer import Adam, SGD, LinearTriangle, ParameterSet, ConstantLearningRate
from autobias.training.trainer import Trainer, EvalDataset
from autobias.utils import py_utils


def decatt_bias(embed_dim, out_dim, n_out=3, drop=0.1):
  return FromEmbeddingPredictor(
    BifusePredictor(
      seq(
        Dropout(drop),
        FullyConnected(embed_dim, out_dim),
      ),
      AttentionBiFuse(DotAttention(out_dim)),
      seq(
        Dropout(drop),
        FullyConnected(out_dim*3, out_dim),
      ),
      MaxPooling(),
      seq(
        Dropout(drop),
        FullyConnected(out_dim*2, n_out, None)
      )
    )
  )


MODES = [
  "none", "mce", "mce-same-model",
  "oracle", "nobp", "adv", "noci"
]


def main():
  parser = argparse.ArgumentParser()
  add_train_args(parser, default_entropy_penalty=0.1, default_adv_penalty=0.3,
                 default_batch_size=16, default_epochs=3, lc_weight_default=None)

  args = parser.parse_args()
  lc_weight = args.lc_weight
  if lc_weight is None:
    # Default depends on the mode
    if args.mode == "adv":
      lc_weight = 0.3
    else:
      lc_weight = 0.2

  dbg = args.debug
  py_utils.add_stdout_logger()

  main_model = FromPooled(FullyConnected(256, 3, None))

  if args.mode in {"mce", "noci", "nobp", "adv"}:
    lc_model = decatt_bias(150 if dbg else 400, 200)
  else:
    raise NotImplementedError(args.mode)
  
  if args.mode in {"mce", "nobp"}:
    rescaler = lambda: ArgminTransformFunction(AffineNLL(
        3, 3, NumpyOptimizer(),
        residual=True, penalty=L2NormPenalty(0.002),
        fix_last_bias_to_zero=True,
        ), backprop_argmin=args.mode == "mce")
    
    predictor = ClfArgminEnsemble(
        [
        ClfHead(
            lc_model, head_name="bias",
            rescaler=rescaler(),
            nll_penalty=lc_weight,
        ),
        ClfHead(
            main_model, head_name="debiased",
            rescaler=rescaler(),
        )
        ],
        n_classes=3,
        add_prior=False,  # prior is uniform
        no_rescale_on_first_step=True
    )

  bias_set = [ParameterSet(
    "predictor", "(encoder\..*)|(predictor\.heads\.0\..*)|(predictor\.(bias|main_to_bias).*)",
    dict(lr=1e-3, e=1e-8, weight_decay=0.0),
    ConstantLearningRate())
  ]
  enc = WordAndCharEncoder(
    "random" if dbg else "crawl-300d-2M",
    None,
    24,
    layers.Conv1D(24, 100, 5),
    MaxPooling(),
  )
  model = ElectraAndEmbedModel(
    "google/electra-small-discriminator", 128,
    NltkAndPunctTokenizer(), enc, predictor)

  opt = SGD(
      lr=0.01,  # Learning rate
      momentum=0.9,  # Momentum factor
      weight_decay=0.01,  # Weight decay for regularization
      clip_grad_norm=1.0,  # Gradient clipping (optional)
      schedule=LinearTriangle(0.1)  # Learning rate schedule
  )

  n_final_eval = 512 if dbg else 4096
  dev = MnliDevMatched(512 if dbg else None)
  train = MnliTrain(4096 if dbg else None)
  evaluator = ClfHardEasyEvaluator(prefix_format="{output}-{metric}/{split}")
  batch_size = args.batch_size
  trainer = Trainer(
    opt,
    train,
    eval_sets=None,
    pre_eval_hook=FitRescaleParameters(256, n_final_eval),
    train_iterator=TorchDataIterator(StratifiedSampler(batch_size)),
    num_train_epochs=args.epochs,
    tb_factor=batch_size/256
  )

  if args.init_only:
    train_utils.init_model_dir(args.output_dir, trainer, model)
  else:
    trainer.train(model, args.output_dir, args.seed, args.n_processes,
                  fp16=args.fp16, no_cuda=args.nocuda)


if __name__ == "__main__":
  main()