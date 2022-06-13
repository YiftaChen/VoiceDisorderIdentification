import wandb
from tqdm import tqdm
import wandb

api = wandb.Api()
entity = "yiftachede"
project = "VoiceDisorder"
runs = api.runs(entity + "/" + project)
losses = []
for run in tqdm(runs):
    history = run.scan_history(keys=["valid_loss"])
    loss = [float(row["valid_loss"]) for row in history]
    run.summary["valid_loss.min"] = min(loss)
    history = run.scan_history(keys=["valid_acc"])
    acc = [float(row["valid_acc"]*100) for row in history]
    max_acc = max(acc)
    run.summary["valid_acc.max"] = f"{format(max_acc, '.2f')}%"
    run.summary.update()
