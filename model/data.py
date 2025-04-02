"""Wrapper for a conditional generation dataset present in 2 tab-separated columns:
source[TAB]target
"""
import logging
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, RobertaTokenizer
from tqdm import tqdm

from data_utils import pad_nt_matrix_roberta, pad_nt_matrix_xlnet


class ClassificationData(pl.LightningDataModule):
    def __init__(self, basedir: str, tokenizer_name: str, batch_size: int, num_workers: int = 16):
        super().__init__()
        self.basedir = basedir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print(tokenizer_name)
        if tokenizer_name == "xlnet-base-cased":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

        self.collator = MyCollator(tokenizer_name)

    def train_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/train_with_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/dev_with_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)

    def test_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/test_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)





class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.read_dataset()


    def read_dataset(self):
        logging.info("Reading data from {}".format(self.data_path))
        data = pd.read_json(self.data_path, orient="records", lines=True)

        self.sentences, self.answer_labels, self.nt_idx_matrix = [], [], []
        num_skipped = 0

        for row in tqdm(data.to_dict(orient="records"), desc="Reading dataset samples"):
            if row.get("nt_idx_matrix") is None or row.get("parse_tree") is None:
                num_skipped += 1
                continue  # Skip broken or unparsable samples

            try:
                sentence = row["sentence"]
                tokens = self.tokenizer.tokenize(sentence)
                token_count = len(tokens)
                nt_matrix = torch.tensor(row["nt_idx_matrix"]).long()

                # ✅ Check if nt_idx_matrix width exceeds token count
                if nt_matrix.size(1) > token_count:
                    logging.warning(f"Skipping due to shape mismatch: nt_idx_matrix width {nt_matrix.size(1)} > token count {token_count}")
                    num_skipped += 1
                    continue

                self.answer_labels.append(int(row["label"]))
                self.sentences.append(sentence)
                self.nt_idx_matrix.append(nt_matrix)

            except Exception as e:
                logging.warning(f"Skipping row due to error: {e}")
                num_skipped += 1

        if num_skipped > 0:
            logging.warning(f"⚠️ Skipped {num_skipped} broken or oversized samples")

        encoded_input = self.tokenizer(self.sentences, padding=True, truncation=True, max_length=512)
        self.input_ids = encoded_input["input_ids"]
        self.token_type_ids = encoded_input.get("token_type_ids", [[0] * len(ids) for ids in self.input_ids])




    def __len__(self) -> int:
            return len(self.sentences)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return (self.input_ids[i], self.token_type_ids[i], self.nt_idx_matrix[i], self.answer_labels[i])




class MyCollator(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if "xlnet" in model_name:
            self.pad_fn = pad_nt_matrix_xlnet
        elif "roberta" in model_name:
            self.pad_fn = pad_nt_matrix_roberta
        else:
            raise NotImplementedError

    def __call__(self, batch):
        max_token_len = 0
        max_phrase_len = 0
        num_elems = len(batch)
        for i in range(num_elems):
            tokens, _, idx_m, _ = batch[i]
            max_token_len = max(max_token_len, len(tokens))
            max_phrase_len = max(max_phrase_len, idx_m.size(0))

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        labels = torch.zeros(num_elems).long()
        nt_idx_matrix = []

        for i in range(num_elems):
            toks, _, idx_matrix, label = batch[i]
            # idx_matrix = torch.tensor(idx_matrix).long()
            idx_matrix = self.pad_fn(nt_idx_matrix=idx_matrix,
                                     max_nt_len=max_phrase_len,
                                     max_length=max_token_len)
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            nt_idx_matrix.append(idx_matrix)
            labels[i] = label

        padded_ndx_tensor = torch.stack(nt_idx_matrix, dim=0)
        return [tokens, tokens_mask, padded_ndx_tensor, labels]



if __name__ == "__main__":
    import sys
    dm = ClassificationData(
        basedir=sys.argv[1], model_name=sys.argv[2], batch_size=32)
    for (tokens, tokens_mask, nt_idx_matrix, labels) in dm.train_dataloader():
        print(torch.tensor(tokens_mask[0].tokens).shape)
