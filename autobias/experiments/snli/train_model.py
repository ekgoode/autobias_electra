import argparse
import warnings
import torch

from autobias.model.snli_model import ElectraAndEmbedModel, FromPooled, FromEmbeddingPredictor, BifusePredictor
from autobias.modules.layers import FullyConnected, Dropout, seq, MaxPooling
from autobias.modules.attention_layers import AttentionBiFuse, DotAttention
from autobias.training.optimizer import Adam, LinearTriangle
from autobias.training.trainer import Trainer, EvalDataset
from autobias.datasets.entailment_datasets import SnliTrain, SnliDev
from autobias.training.data_iterator import TorchDataIterator
from autobias.utils.tokenizer import NltkAndPunctTokenizer
from autobias.modules.word_and_char_encoder import WordAndCharEncoder

warnings.simplefilter(action="ignore", category=FutureWarning)


def decatt_bias(embed_dim, out_dim, n_out=3, drop=0.1):
    """Decomposable Attention Bias Model."""
    return FromEmbeddingPredictor(
        BifusePredictor(
            seq(
                Dropout(drop),
                FullyConnected(embed_dim, out_dim),
            ),
            AttentionBiFuse(DotAttention(out_dim)),
            seq(
                Dropout(drop),
                FullyConnected(out_dim * 3, out_dim),
            ),
            MaxPooling(),
            seq(
                Dropout(drop),
                FullyConnected(out_dim * 2, n_out, None)
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Train ELECTRA Ensemble on SNLI Dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    args = parser.parse_args()

    # Define main and bias models
    main_model = FromPooled(FullyConnected(256, 3, None))  # ELECTRA hidden size is 256
    lc_model = decatt_bias(400, 200)  # Bias model

    # Ensemble predictor
    predictor = ClfArgminEnsemble(
        [
            ClfHead(lc_model, head_name="bias", rescaler=None, nll_penalty=0.2),
            ClfHead(main_model, head_name="debiased", rescaler=None),
        ],
        n_classes=3,
        add_prior=False,
        no_rescale_on_first_step=True
    )

    # Initialize encoder and full model
    enc = WordAndCharEncoder(
        "crawl-300d-2M",
        None,
        24,
        seq(
            FullyConnected(24, 100),
            MaxPooling()
        )
    )
    model = ElectraAndEmbedModel(
        "google/electra-small-discriminator", 128,
        NltkAndPunctTokenizer(), enc, predictor
    )

    # Optimizer
    opt = Adam(
        lr=args.lr, e=1e-6, weight_decay=0.01, max_grad_norm=1.0,
        schedule=LinearTriangle(0.1)
    )

    # Dataset
    train = SnliTrain()
    dev = SnliDev()

    # Data loaders
    train_loader = TorchDataIterator(train, batch_size=args.batch_size)
    dev_loader = TorchDataIterator(dev, batch_size=args.batch_size)

    # Trainer
    trainer = Trainer(
        opt,
        train_loader,
        [EvalDataset(dev_loader, "dev")],
        num_train_epochs=args.epochs
    )

    # Train the model
    trainer.train(model, args.output_dir)

    # Save the trained model
    torch.save(model.state_dict(), f"{args.output_dir}/electra_snli_model.pth")
    print(f"Model saved to {args.output_dir}/electra_snli_model.pth")


if __name__ == "__main__":
    main()
