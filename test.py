import torch
from tqdm import tqdm
from torch import nn, optim
from dataset import DualPaperDataset
from model import PaperModel
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, DataCollatorWithPadding
from utils import compute_metrics

def evaluation(dataset, model, tokenizer, aims_ids, n_classes):
    topks = (1, 3, 5, 10) 
    train_score = {k:0 for k in (1,3,5,10)}
    valid_score = {k:0 for k in (1,3,5,10)}
    test_score = {k:0 for k in (1,3,5,10)}
    model.eval()
    accelerator = Accelerator()
    data_collator = DataCollatorWithPadding(tokenizer)
    test_dataset = DualPaperDataset(dataset, tokenizer, n_classes)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=data_collator, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    model, test_loader = accelerator.prepare(
    model, test_loader)
    aims_ids = {k: v.to(accelerator.device) for k, v in aims_ids.items()}

    test_loop = tqdm(test_loader, leave=True)
    for batch in test_loop:
        with torch.no_grad():
            inputs = {k:v.squeeze() for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            logits = model(inputs, aims_ids)
            logits = logits['logits']
            score, num_cnt = compute_metrics(logits, labels)
            test_loop.set_postfix(Top_01=score[1], Top_03=score[3], Top_05=score[5], Top_10=score[10])
            for k in (1, 3, 5, 10):
                test_score[k] += num_cnt[k]
    test_loss /= len(test_loader)
    for k in topks:
        test_score[k] = test_score[k]/len(test_dataset)
    return test_score