import os
import torch
import torch.nn as nn
import time
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *
import Metrics as Metrics
from Logger import CSVLogger
from ranger import Ranger


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=150, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of each batch during training"
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
        "--nmute", type=int, default=18, help="number of mutations during training"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="transformer dropout"
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
        "--lr_scale", type=float, default=0.1, help="learning rate scale"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="stride used in k-mer convolution"
    )
    parser.add_argument(
        "--viral_loss_weight",
        type=int,
        default=1,
        # help="stride used in k-mer convolution",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=3200, help="training schedule warmup steps"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="saving checkpoints per save_freq epochs",
    )
    parser.add_argument("--val_freq", type=int, default=1, help="validation frequent")
    parser.add_argument(
        "--nfolds", type=int, default=3, help="number of cross validation folds"
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold to train")

    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")
    parser.add_argument(
        "--path",
        type=str,
        default="../",
        help="path of csv file with DNA sequences and labels",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="number of workers for dataloader"
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
    train = pd.read_json(json_path, lines=True)
    # train_ids = json.id.to_list()
    json_path = os.path.join(opts.path, "test.json")
    test = pd.read_json(json_path, lines=True)

    # data = test
    data = pd.concat([train, test], ignore_index=True)

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    ids = np.asarray(train_data.id.to_list())
    training_dataset = RNADataset(
        seqs=train_data.sequence.to_list(),
        labels=np.zeros(len(train_data)),
        ids=ids,
        ew=np.arange(len(train_data)),
        bpp_path=opts.path,
        pad=True,
        k=opts.kmers[0],
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
    )

    val_ids = np.asarray(val_data.id.to_list())
    val_dataset = RNADataset(
        seqs=val_data.sequence.to_list(),
        labels=np.zeros(len(val_data)),
        ids=val_ids,
        ew=np.arange(len(val_data)),
        bpp_path=opts.path,
        pad=True,
        k=opts.kmers[0],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
    )

    # checkpointing
    checkpoints_folder = "pretrain_weights"

    os.system("mkdir logs")

    csv_file = "logs/pretrain.csv".format((opts.fold))
    columns = ["epoch", "train_loss", "val_loss"]
    logger = CSVLogger(columns, csv_file)

    # build model and logger
    model = RNADegformer(
        ntoken=opts.ntoken,
        nclass=opts.nclass,
        ninp=opts.ninp,
        nhead=opts.nhead,
        nhid=opts.nhid,
        nlayers=opts.nlayers,
        kmer_aggregation=opts.kmer_aggregation,
        kmers=opts.kmers,
        stride=opts.stride,
        dropout=opts.dropout,
        pretrain=True,
    ).to(device)

    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss()
    model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of paramters: {}".format(pytorch_total_params))

    # training loop
    cos_epoch = int(opts.epochs * 0.25)
    total_steps = len(training_dataloader)
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

        for data in training_dataloader:
            step += 1
            lr = get_lr(optimizer)
            src = data["data"]
            labels = data["labels"]
            bpps = data["bpp"].to(device)
            src_mask = data["src_mask"].to(device)

            if np.random.uniform() > 0.5:
                masked = mutate_rna_input(src)
            else:
                masked = mask_rna_input(src)

            src = src.to(device).long()
            masked = masked.to(device).long()

            output = model(masked, bpps, src_mask)

            mask_selection = src[:, :, 0] != 14
            loss = (
                criterion(
                    output[0][mask_selection].reshape(-1, 4),
                    src[:, :, 0][mask_selection].reshape(-1),
                )
                + criterion(
                    output[1][mask_selection].reshape(-1, 3),
                    src[:, :, 1][mask_selection].reshape(-1) - 4,
                )
                + criterion(
                    output[2][mask_selection].reshape(-1, 7),
                    src[:, :, 2][mask_selection].reshape(-1) - 7,
                )
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss

            if epoch > cos_epoch:
                lr_schedule.step()

            print(
                f"Epoch [{epoch + 1}/{opts.epochs}],"
                + f" Step [{step + 1}/{total_steps}]"
                + f" Loss: {(total_loss / (step + 1)):.3f}"
                + f" Lr:{lr:.6f} Time: {(time.time() - t):.1f}",
                end="\r",
                flush=True,
            )

        print("")

        # if (epoch + 1) % opts.save_freq == 0:
        val_loss = []

        for data in val_dataloader:
            src = data["data"]
            labels = data["labels"]
            bpps = data["bpp"].to(device)
            src_mask = data["src_mask"].to(device)

            if np.random.uniform() > 0.5:
                masked = mutate_rna_input(src)
            else:
                masked = mask_rna_input(src)

            src = src.to(device).long()
            masked = masked.to(device).long()

            with torch.no_grad():
                output = model(masked, bpps, src_mask)

            mask_selection = src[:, :, 0] != 14
            loss = (
                criterion(
                    output[0][mask_selection].reshape(-1, 4),
                    src[:, :, 0][mask_selection].reshape(-1),
                )
                + criterion(
                    output[1][mask_selection].reshape(-1, 3),
                    src[:, :, 1][mask_selection].reshape(-1) - 4,
                )
                + criterion(
                    output[2][mask_selection].reshape(-1, 7),
                    src[:, :, 2][mask_selection].reshape(-1) - 7,
                )
            )
            val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        train_loss = total_loss / (step + 1)
        torch.cuda.empty_cache()
        to_log = [epoch + 1, train_loss, val_loss]
        logger.log(to_log)

        if val_loss < best_loss:
            print(f"new best_loss found at epoch {epoch + 1}: {val_loss}")
            best_loss = val_loss
            save_weights(model, optimizer, -1, checkpoints_folder)


if __name__ == "__main__":
    train_fold()
