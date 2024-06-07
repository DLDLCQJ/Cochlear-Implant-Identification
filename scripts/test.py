import torch
import torch.nn as nn

from scripts.run_epoch import run_epoch

def test_epoch(args, model, test_dataloader, criterion, fold):
    results_te = []
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}")
        with torch.no_grad():
            te_loss, te_predicts,te_truths = run_epoch(args, model, test_dataloader, criterion, None, None, fold, epoch, is_training=False,is_testing=True)
        results_te.append({
            "fold": fold + 1,
            "epoch": epoch + 1,
            "predictions": te_predicts,
            "labels": te_truths
        })    
    return results_te
