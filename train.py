import os
import glob
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
import wandb
from consts import GENRE_TOKENS

class MIDIGenreDataset(Dataset):
    def __init__(self, data_dir: str, block_size: int, tokenizer: REMI):
        self.examples = []
        self.block_size = block_size
        self.tokenizer = tokenizer

        for tok in GENRE_TOKENS.values():
            if tok not in self.tokenizer.vocab:
                self.tokenizer.add_to_vocab(tok)

        for genre, tok in GENRE_TOKENS.items():
            genre_dir = os.path.join(data_dir, genre)
            for midipath in glob.glob(os.path.join(genre_dir, '*.mid')):
                tok_seq = self.tokenizer.encode(midipath)[0]
                events = tok_seq.ids
                genre_id = self.tokenizer.vocab[tok]
                seq = [genre_id] + events
                # Chop into blocks
                for i in range(0, len(seq) - block_size):
                    block = seq[i : i + block_size]
                    self.examples.append(block)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        block = self.examples[idx]
        x = torch.tensor(block[:-1], dtype=torch.long)
        y = torch.tensor(block[1:], dtype=torch.long)
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',         type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path or identifier of the pretrained MusicGen checkpoint')
    parser.add_argument('--output_dir',       type=str, required=True)
    parser.add_argument('--epochs',           type=int, default=10)
    parser.add_argument('--batch_size',       type=int, default=4)
    parser.add_argument('--block_size',       type=int, default=512)
    parser.add_argument('--learning_rate',    type=float, default=5e-5)
    parser.add_argument('--weight_decay',     type=float, default=0.01)
    parser.add_argument('--wandb_project',    type=str, default='music_transformer')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(project=args.wandb_project)

    processor = AutoProcessor.from_pretrained(args.pretrained_model)
    model = MusicgenForConditionalGeneration.from_pretrained(args.pretrained_model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    wandb.watch(model, log='all', log_freq=200)
    model.train()

    TOKENIZER_PARAMS = {
        "pitch_range": (21, 108),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"] + list(GENRE_TOKENS.values()),
        "use_chords": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "num_tempos": 32,
        "tempo_range": (40, 250),
    }
    tok_config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(tok_config)

    dataset = MIDIGenreDataset(args.data_dir, args.block_size, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        for step, (x, y) in enumerate(loop, start=1):
            x = x.to(device)
            y = y.to(device)
            loss = model(input_ids=x, labels=y).loss
            loss.backward()


            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            if step % 200 == 0:
                weight_norm = torch.sqrt(
                    sum(p.data.norm()**2 for p in model.parameters())
                )
                grad_norms = [p.grad.norm()**2 for p in model.parameters() if p.grad is not None]
                grad_norm = torch.sqrt(torch.sum(torch.stack(grad_norms))) if grad_norms else torch.tensor(0.0)
                non_zero_grads = sum(p.grad is not None for p in model.parameters())
                wandb.log({
                    'train/loss': loss.item(),
                    'weights/norm': weight_norm.item(),
                    'gradients/norm': grad_norm.item(),
                    'gradients/non_zero_count': non_zero_grads
                }, step=(epoch - 1) * len(dataloader) + step)


        avg_loss = total_loss / len(dataloader)
        wandb.log({'epoch/avg_loss': avg_loss})
        ckpt_dir = os.path.join(args.output_dir, f"ckpt-epoch{epoch}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

    print('yes')

if __name__ == '__main__':
    main()
