import pickle
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from Functions import *

tokens = "ACGU().BEHIMSX"


# eterna,'nupack','rnastructure','vienna_2','contrafold_2',
class RNADataset(Dataset):
    def __init__(
        self,
        seqs,
        labels,
        ids,
        ew,
        bpp_path,
        transform=None,
        training=True,
        pad=True,
        k=5,
    ):
        self.transform = transform
        self.seqs = seqs  # .transpose(1,0,2,3)

        self.data = []
        self.labels = labels.astype("float32")
        self.bpp_path = bpp_path
        self.ids = ids
        self.training = training
        self.bpps = []

        dm = get_distance_mask(130)  # .reshape(1,bpps.shape[-1],bpps.shape[-1])
        # len(seqs[-1])

        self.dms = np.asarray([dm for _ in range(12)])  # shape (12, 3, L, L)
        self.lengths = []

        # load single sequence data from file
        for i, id in tqdm(enumerate(self.ids)):
            # bpps have shape (12, L, L)
            bpps = np.load(
                os.path.join(self.bpp_path, "train_test_bpps", id + "_bpp.npy")
            )
            if pad:
                # add 0 to the tail of seqs have 107 len
                bpps = np.pad(
                    bpps,
                    ([0, 0], [0, 130 - bpps.shape[1]], [0, 130 - bpps.shape[2]]),
                    constant_values=0,
                )

            with open(
                os.path.join(self.bpp_path, "train_test_bpps", id + "_struc.p"),
                "rb",
            ) as f:
                structures = pickle.load(f)

            with open(
                os.path.join(self.bpp_path, "train_test_bpps", id + "_loop.p"),
                "rb",
            ) as f:
                loops = pickle.load(f)

            seq = self.seqs[i]
            self.lengths.append(len(seq))
            input = []

            # j in range 12
            for j in range(bpps.shape[0]):
                input_seq = np.asarray([tokens.index(s) for s in seq])
                input_structure = np.asarray([tokens.index(s) for s in structures[j]])
                input_loop = np.asarray([tokens.index(s) for s in loops[j]])

                # input append an array of shape (L, 3)
                input.append(np.stack([input_seq, input_structure, input_loop], -1))

            input = np.asarray(input).astype("int")  # shape (12, L, 3)
            if pad:
                input = np.pad(
                    input,
                    ([0, 0], [0, 130 - input.shape[1]], [0, 0]),
                    constant_values=14,
                )

            self.data.append(input)
            self.bpps.append(np.clip(bpps, 0, 1).astype("float32"))

        self.data = np.asarray(self.data)  # shape (N, 12, L, 3)
        self.lengths = np.asarray(self.lengths)
        self.ew = ew
        self.k = k
        self.pad = pad
        self.src_masks = [
            self.generate_src_mask(self.lengths[i], self.data.shape[-2], self.k)
            for i in range(len(self.data))
        ]  # shape (N, k, L)

    def generate_src_mask(self, L1, L2, k):
        mask = np.ones((k, L2), dtype="int8")
        for i in range(k):
            mask[i, L1 + i + 1 - k :] = 0
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.training:
            bpp_selection = np.random.randint(self.bpps[idx].shape[0])
            bpps = self.bpps[idx][bpp_selection]
            bpps = np.concatenate(
                [bpps.reshape(1, bpps.shape[0], bpps.shape[1]), self.dms[0]], 0
            )
            bpps = bpps.astype("float32")  # shape (4, L, L)

            sample = {
                "data": self.data[idx][bpp_selection],
                "labels": self.labels[idx],
                "bpp": bpps,
                "ew": self.ew[idx],
                "id": self.ids[idx],
                "src_mask": self.src_masks[idx],
            }
        else:
            bpps = self.bpps[idx]  # shape (12, L, L)
            bpps = np.concatenate(
                [
                    bpps.reshape(bpps.shape[0], 1, bpps.shape[1], bpps.shape[2]),
                    self.dms,  # shape (12, 3, L, L)
                ],
                1,
            )
            bpps = bpps.astype("float32")  # shape (12, 4, L, L)
            sample = {
                "data": self.data[idx],
                "labels": self.labels[idx],
                "bpp": bpps,
                "ew": self.ew[idx],
                "id": self.ids[idx],
            }
        # if self.transform:
        #     sample=self.transform(sample)
        return sample


class RNADataset233(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path
        self.tokens = "ACGU().BEHIMSX"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = np.array([self.tokens.index(s) for s in self.df["sequence"][idx]])
        structure = np.array([self.tokens.index(s) for s in self.df["structure"][idx]])
        loop = np.array([self.tokens.index(s) for s in self.df["structure"][idx]])

        src = np.stack([sequence, structure, loop], -1)
        bpp = np.load(os.path.join(self.path, "new_sequences_bpps", f"{idx}.npy"))
        dm = get_distance_mask(len(sequence))

        bpp = np.concatenate([bpp.reshape(1, *bpp.shape), dm], 0)

        return {"src": src, "bpp": bpp}


if __name__ == "__main__":
    pass
