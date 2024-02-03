import os
import torch
import torch.nn as nn
import time
import argparse
from torch.utils.data import DataLoader

from ranger import Ranger
from Logger import CSVLogger
from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")
    parser.add_argument(
        "--path",
        type=str,
        default="../",
        help="path of csv file with DNA sequences and labels",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="size of each batch during training"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="weight dacay used in optimizer"
    )
    parser.add_argument(
        "--ntoken",
        type=int,
        default=4,
        help="number of tokens to represent DNA nucleotides (should always be 4)",
    )
    parser.add_argument(
        "--nclass",
        type=int,
        default=2,
        help="number of classes from the linear decoder",
    )
    parser.add_argument(
        "--ninp", type=int, default=512, help="ninp for transformer encoder"
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="nhead for transformer encoder"
    )
    parser.add_argument(
        "--nhid", type=int, default=2048, help="nhid for transformer encoder"
    )
    parser.add_argument(
        "--nlayers", type=int, default=6, help="nlayers for transformer encoder"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="saving checkpoints per save_freq epochs",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="transformer dropout"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=3200, help="training schedule warmup steps"
    )
    parser.add_argument(
        "--lr_scale", type=float, default=0.1, help="learning rate scale"
    )
    parser.add_argument(
        "--nmute", type=int, default=18, help="number of mutations during training"
    )
    parser.add_argument(
        "--kmers",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="k-mers to be aggregated",
    )
    # parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
    parser.add_argument(
        "--kmer_aggregation", dest="kmer_aggregation", action="store_true"
    )
    parser.add_argument(
        "--no_kmer_aggregation", dest="kmer_aggregation", action="store_false"
    )
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument(
        "--nfolds", type=int, default=5, help="number of cross validation folds"
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold to train")
    parser.add_argument("--val_freq", type=int, default=1, help="valid frequency")
    parser.add_argument(
        "--stride", type=int, default=1, help="stride used in k-mer convolution"
    )
    parser.add_argument(
        "--viral_loss_weight",
        type=int,
        default=1,
        help="stride used in k-mer convolution",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--error_beta", type=float, default=5, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--error_alpha", type=float, default=0, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--noise_filter",
        type=float,
        default=0.25,
        help="noise filter",
    )
    opts = parser.parse_args()
    return opts


def train_fold():
    # get arguments
    opts = get_args()

    # gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate datasets
    json_path = os.path.join(opts.path, "train.json")

    json = pd.read_json(json_path, lines=True)
    json = json[json.signal_to_noise > opts.noise_filter]
    ids = np.asarray(json.id.to_list())

    error_weights = get_errors(json)
    error_weights = opts.error_alpha + np.exp(-error_weights * opts.error_beta)
    train_indices, val_indices = get_train_val_indices(
        json, opts.fold, seed=42, nfolds=opts.nfolds
    )

    _, labels = get_data(json)
    sequences = np.asarray(json.sequence)
    train_seqs, val_seqs = sequences[train_indices], sequences[val_indices]
    train_labels, val_labels = labels[train_indices], labels[val_indices]
    train_ids, val_ids = ids[train_indices], ids[val_indices]
    train_ew, val_ew = error_weights[train_indices], error_weights[val_indices]

    dataset = RNADataset(
        train_seqs,
        train_labels,
        train_ids,
        train_ew,
        opts.path,
    )
    val_dataset = RNADataset(
        val_seqs,
        val_labels,
        val_ids,
        val_ew,
        opts.path,
        training=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opts.batch_size * 2,
        shuffle=False,
        num_workers=opts.workers,
    )

    # checkpointing
    checkpoints_folder = "checkpoints_fold{}".format((opts.fold))
    csv_file = "logs/log_fold{}.csv".format((opts.fold))
    columns = ["epoch", "train_loss", "val_loss"]
    logger = CSVLogger(columns, csv_file)

    # build model and logger
    model = RNADegformer(
        opts.ntoken,
        opts.nclass,
        opts.ninp,
        opts.nhead,
        opts.nhid,
        opts.nlayers,
        opts.kmer_aggregation,
        kmers=opts.kmers,
        stride=opts.stride,
        dropout=opts.dropout,
    ).to(device)

    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion = weighted_MCRMSE
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("pretrain_weights/epoch0.ckpt"))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of paramters: {}".format(pytorch_total_params))

    # training loop
    cos_epoch = int(opts.epochs * 0.25)
    total_steps = len(dataloader)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (opts.epochs - cos_epoch) * total_steps
    )
    best_loss = 100000

    for epoch in range(opts.epochs):
        model.train(True)
        t = time.time()
        total_loss = 0
        optimizer.zero_grad()
        step = 0

        for data in dataloader:
            step += 1
            lr = get_lr(optimizer)
            src = data["data"].to(device)
            labels = data["labels"].to(device) #.float()
            bpps = data["bpp"].to(device)

            output = model(src, bpps)
            ew = data["ew"].to(device)
            loss = criterion(output[:, :68], labels, ew).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss
            print(
                f"Epoch [{epoch + 1}/{opts.epochs}] "
                + f"Step [{step + 1}/{total_steps}] "
                + f"Loss: {total_loss / (step + 1):.3f} "
                + f"Lr:{lr:.6f} Time: {time.time() - t:.1f}",
                end="\r",
                flush=True,
            )

            if epoch > cos_epoch:
                lr_schedule.step()

        print("")
        train_loss = total_loss / (step + 1)
        torch.cuda.empty_cache()

        # validate
        val_loss = validate(
            model, device, val_dataloader, batch_size=opts.batch_size
        )
        to_log = [
            epoch + 1,
            train_loss,
            val_loss,
        ]
        logger.log(to_log)

        if val_loss < best_loss:
            print(f"New best_loss found at epoch {epoch + 1}: {val_loss}")
            best_loss = val_loss
            save_weights(model, optimizer, epoch, checkpoints_folder)

    get_best_weights_from_fold(opts.fold)


if __name__ == "__main__":
    train_fold()
