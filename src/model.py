import os
import math
from collections import Counter
import re
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def pick_device():
    if torch.cuda.is_available():
        try:
            torch.tensor([0.0], device="cuda").add_(1.0)
            torch.cuda.synchronize()
            return "cuda"
        except Exception as e:
            logging.warning(f"CUDA present but unusable ({e}); falling back to CPU.")
    return "cpu"

DEVICE = pick_device()
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
PREFIX, SUFFIX = "<", ">"
MIN_LANG_COUNT = 40
SEED = 42
K_FOLDS = 3
BATCH_SIZE, EPOCHS, LR = 64, 3, 3e-4

logging.info(f"Using device: {DEVICE}")
logging.info("Loading dataset elalber2000/etymos-es")
ds_dict = load_dataset("elalber2000/etymos-es")
if hasattr(ds_dict, "values"):
    ds = concatenate_datasets(list(ds_dict.values()))
else:
    ds = ds_dict
logging.info(f"Loaded dataset with {len(ds)} examples")

def tok(x): return f"{PREFIX}{x}{SUFFIX}"

cnt_origin = Counter(ds["lang_origin"])
cnt_dest = Counter(ds["lang_dest"])
logging.info(f"Unique origin langs: {len(cnt_origin)}, unique dest langs: {len(cnt_dest)}")

keep = lambda ex : (
    cnt_origin.get(ex["lang_origin"], 0) >= MIN_LANG_COUNT
    and cnt_dest.get(ex["lang_dest"], 0) >= MIN_LANG_COUNT
    )

logging.info("Filtering low-count languages")
ds = ds.filter(keep, num_proc=4)
logging.info(f"Dataset size after filtering: {len(ds)}")

lang_toks = {tok(x) for x in set(ds["lang_origin"])|set(ds["lang_dest"])}
logging.info(f"Special language tokens count: {len(lang_toks)}")

def build_char_vocab(dataset, extra_tokens):
    words = set()
    for ex in dataset:
        words.add(ex["word_origin"])
        words.add(ex["word_dest"])
    vocab = {PAD, BOS, EOS} | extra_tokens
    for ch in words:
        vocab = vocab|set(ch)
    enc_tok = {ch:i for i,ch in enumerate(sorted(vocab))}
    dec_tok = {i:ch for ch,i in enc_tok.items()}
    return vocab, enc_tok, dec_tok

vocab, enc_tok, dec_tok = build_char_vocab(ds, lang_toks)
print(vocab)
print(lang_toks)
logging.info(f"Vocab size: {len(vocab)}")

ds = ds.map(
    lambda ex: {
        "input_text": f"{tok(ex['lang_origin'])}{tok(ex['lang_dest'])}{ex['word_origin']}",
        "target_text": ex["word_dest"],
    }
)
logging.info("Formatted dataset with input_text and target_text")
ds = ds.select_columns(["input_text", "target_text"])

ds = ds.shuffle(seed=42)
logging.info("Shuffled dataset")

specials = sorted(list(lang_toks | {PAD, BOS, EOS}), key=len, reverse=True)
tok_re = re.compile("|".join(map(re.escape, specials)) + r"|.", re.DOTALL)

def encode(text, add_bos=False, add_eos=False):
    toks = tok_re.findall(text)
    ids = [enc_tok.get(t, enc_tok[PAD]) for t in toks]
    if add_bos: ids.insert(0, enc_tok[BOS])
    if add_eos: ids.append(enc_tok[EOS])
    return ids

class HFWordPairs(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[int(i)]
        src = torch.tensor(encode(ex["input_text"], add_bos=False, add_eos=True), dtype=torch.long)
        tgt_in  = torch.tensor(encode(ex["target_text"], add_bos=True,  add_eos=False), dtype=torch.long)
        tgt_out = torch.tensor(encode(ex["target_text"], add_bos=False, add_eos=True), dtype=torch.long)
        return {"src_ids": src, "tgt_in_ids": tgt_in, "tgt_out_ids": tgt_out}

def pad_batch(batch):
    pad_id = enc_tok[PAD]
    def _pad(seqs):
        L = max(x.size(0) for x in seqs)
        mat = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.ones((len(seqs), L), dtype=torch.bool)
        for i, s in enumerate(seqs):
            l = s.size(0)
            mat[i, :l] = s
            mask[i, :l] = False
        return mat, mask
    src, tin, tout = zip(*[(b["src_ids"], b["tgt_in_ids"], b["tgt_out_ids"]) for b in batch])
    src_mat, src_kpm = _pad(src)
    tin_mat,  tin_kpm = _pad(tin)
    tout_mat, _ = _pad(tout)
    return {
        "src_ids": src_mat, "tgt_in_ids": tin_mat, "tgt_out_ids": tout_mat,
        "src_kpm": src_kpm, "tgt_kpm": tin_kpm
    }

class CharSeq2Seq(nn.Module):
    def __init__(self, vocab, d=128, n=4, h=2):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, h, dim_feedforward=4*d, batch_first=True), n)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, h, dim_feedforward=4*d, batch_first=True), n)
        self.out = nn.Linear(d, len(vocab))

    def forward(self, src_ids, tgt_ids, src_kpm=None, tgt_kpm=None, tgt_mask=None):
        E = self.emb(src_ids); D = self.emb(tgt_ids)
        Hs = self.enc(E, src_key_padding_mask=src_kpm)
        cross = self.dec(D, Hs,
                         tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_kpm,
                         memory_key_padding_mask=src_kpm)
        return self.out(cross)

def causal_mask(T, device=None):
    m = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    return m.to(device) if device else m

def make_loader(hf_subset, shuffle=False):
    ds = HFWordPairs(hf_subset)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=(DEVICE=="cuda"), collate_fn=pad_batch)

def run_epoch(model, loader, opt=None):
    pad_id = enc_tok[PAD]
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    train = opt is not None
    model.train(train)
    total_loss, total_toks = 0.0, 0
    to_dev = lambda x: x.to(DEVICE, non_blocking=(DEVICE=="cuda"))
    for batch in loader:
        src = to_dev(batch["src_ids"])
        tin = to_dev(batch["tgt_in_ids"])
        tout= to_dev(batch["tgt_out_ids"])
        src_kpm = to_dev(batch["src_kpm"])
        tgt_kpm = to_dev(batch["tgt_kpm"])
        T = tin.size(1)
        logits = model(src, tin, src_kpm, tgt_kpm, causal_mask(T, DEVICE))
        loss = ce(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item() * tout.numel()
        total_toks += tout.numel()
    loss_tok = total_loss / max(total_toks, 1)
    ppl = math.exp(loss_tok)
    logging.info(f"{'Train' if train else 'Eval'} loss/tok: {loss_tok:.4f}, ppl: {ppl:.2f}")
    return loss_tok, ppl

n = len(ds)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
folds = np.array_split(idx, K_FOLDS)
logging.info(f"Prepared {K_FOLDS}-fold splits with dataset size {n}")

fold_metrics = []

for k in range(K_FOLDS):
    val_idx = folds[k]
    train_idx = np.concatenate([folds[i] for i in range(K_FOLDS) if i != k])

    ds_train = ds.select(train_idx.tolist())
    ds_val   = ds.select(val_idx.tolist())

    train_loader = make_loader(ds_train, shuffle=True)
    val_loader   = make_loader(ds_val,   shuffle=False)

    model = CharSeq2Seq(vocab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"\n- Fold {k+1}/{K_FOLDS} | train={len(ds_train)} | val={len(ds_val)}")
    logging.info(f"Starting fold {k+1}/{K_FOLDS} with train={len(ds_train)}, val={len(ds_val)}")
    best_val = float("inf")
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, opt)
        va_loss, va_ppl = run_epoch(model, val_loader,   opt=None)
        print(f"  Ep {ep:02d}  train loss/tok {tr_loss:.4f} (ppl {tr_ppl:.2f})  "
              f"val loss/tok {va_loss:.4f} (ppl {va_ppl:.2f})")
        logging.info(f"Fold {k+1} Ep {ep}: train loss/tok {tr_loss:.4f} (ppl {tr_ppl:.2f}), val loss/tok {va_loss:.4f} (ppl {va_ppl:.2f})")
        best_val = min(best_val, va_loss)

    fold_metrics.append(best_val)
    logging.info(f"Fold {k+1} best val loss/tok: {best_val:.4f}")

print("\nCross-val summary")
for i, m in enumerate(fold_metrics, 1):
    print(f"Fold {i}: best val loss/tok = {m:.4f}")
print(f"Avg best val loss/tok = {np.mean(fold_metrics):.4f}  ± {np.std(fold_metrics):.4f}")
logging.info(f"Cross-val summary: avg best val loss/tok = {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")

@torch.no_grad()
def greedy_decode(model, src_text, max_new_tokens=30):
    src_ids = torch.tensor(encode(src_text, add_bos=False, add_eos=True), dtype=torch.long)[None, :].to(DEVICE)
    src_kpm = torch.zeros_like(src_ids, dtype=torch.bool)
    ys = torch.tensor([[enc_tok[BOS]]], device=DEVICE)
    for _ in range(max_new_tokens):
        T = ys.size(1)
        logits = model(src_ids, ys, src_kpm, None, causal_mask(T, DEVICE))
        nxt = logits[:, -1, :].argmax(-1)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        if nxt.item() == enc_tok[EOS]:
            break
    out = ys[0].tolist()[1:]
    if enc_tok[EOS] in out:
        out = out[:out.index(enc_tok[EOS])]
    result = "".join(dec_tok[i] for i in out)
    logging.info(f"Greedy decode result: {result}")
    return result

print(greedy_decode("<latín><castellano>lumen"))
