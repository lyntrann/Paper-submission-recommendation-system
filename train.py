from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, DataCollatorWithPadding
from model import PaperModel
import numpy as np
from dataset import DualPaperDataset
import torch
from torch import nn, optim
from accelerate.utils import set_seed
from tqdm import tqdm
from utils import count_parameters, compute_metrics, cosine_distance
from transformers import AdamW
from loss import DualLoss, LabelRegLoss
from accelerate import Accelerator

def training_loop(model_args, data_train, data_valid, n_classes, tokenizer, aims_ids, mixed_precision="fp16", seed=42, batch_size=32, state=None):
    set_seed(seed)
    history = {"cl_loss": [], "uniform_loss": [], "ce_loss": [], "accuracy": []}
    accelerator = Accelerator(mixed_precision=mixed_precision)
    data_collator = DataCollatorWithPadding(tokenizer)
    model = PaperModel(**model_args)
    dataset = DualPaperDataset(data_train, tokenizer, n_classes)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    min_loss = np.inf
    valid_dataset = DualPaperDataset(data_valid, tokenizer, n_classes)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, collate_fn=data_collator, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    max_epochs = 3
    print("Model summary:\n")
    print(">> Total params: ", count_parameters(model))
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, weight_decay=1e-2, correct_bias=True)
    num_training_steps = len(data_loader)*max_epochs
    num_warmup_steps = int(num_training_steps*0.1)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 2000, num_training_steps)
    criterion = DualLoss()
    uniform_criterion = LabelRegLoss()
    aims_ids = {k:v.to(accelerator.device) for k, v in aims_ids.items()}
    topks = (1,3,5,10)
    saved_epochs = -1
    if state != None:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        saved_epochs = state['epoch']
    print(f"Saved epochs: {saved_epochs+1}")
    model, uniform_criterion, optimizer, data_loader, valid_loader, lr_scheduler = accelerator.prepare(
    model, uniform_criterion, optimizer, data_loader, valid_loader, lr_scheduler)
    for epoch in range(saved_epochs+1, max_epochs):
        loop = tqdm(data_loader, leave=True, disable=not accelerator.is_local_main_process)
        train_loss = 0.0
        train_score = {k:0 for k in (1,3,5,10)}
        for idx, batch in enumerate(loop):
            inputs = {k:v.squeeze() for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            optimizer.zero_grad()
            outputs = model(inputs, aims_ids)
            label_feats = outputs['label_feats']
            ce_loss, cl_loss =  criterion(outputs, labels)
            uniform_loss = uniform_criterion(label_feats)
            loss = ce_loss + cl_loss + 0.1*uniform_loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            history['ce_loss'].append(ce_loss.item())
            history['cl_loss'].append(cl_loss.item())
            history['uniform_loss'].append(uniform_loss.item())
            train_loss += loss.item()
            
            with torch.no_grad():
                cosine_dist = cosine_distance(label_feats.detach().clone())
                score, num_cnt = compute_metrics(outputs['logits'], labels)
                history['accuracy'].append(score[1])
                for k in topks:
                    train_score[k] += num_cnt[k]
            if idx % 300 == 0:
                accelerator.print(f"Loss: {loss.item()} || Regularize loss: {uniform_loss.item()} || Uniformity: {cosine_dist} || Top 1 acc: {score[1]} || Top 3 acc: {score[3]} || Top 5 acc: {score[5]} || Top 10 acc: {score[10]}")
            loop.set_description('Epoch: {} - lr: {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=round(loss.item(), 3), top01=score[1], top03=score[3], top05=score[5], top10=score[10])
        train_loss = train_loss / (len(dataset))
        for k in topks:
            accelerator.print(f"Train top {k} acc: {train_score[k]/(len(dataset))}", end=" || ")
        accelerator.print("")
        valid_loss = 0.0
        valid_loop = tqdm(valid_loader, leave=True, disable=not accelerator.is_local_main_process)
        valid_score = {k:0 for k in(1,3,5,10)}
        for batch in valid_loop:
            with torch.no_grad():
                inputs = {k:v.squeeze() for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']
                inputs = {k:v.squeeze() for k, v in inputs.items()}
                outputs = model(inputs, aims_ids)
                ce_loss, cl_loss = criterion(outputs, labels)
                loss = ce_loss + cl_loss
                valid_loss += loss.item()
                score, num_cnt = compute_metrics(outputs['logits'], labels)
                for k in topks:
                    valid_score[k] += num_cnt[k]
            valid_loop.set_description('Epoch: {} - lr: {} '.format(epoch+1, optimizer.param_groups[0]['lr']))
            valid_loop.set_postfix(loss=loss.item(), top01=score[1], top03=score[3], top05=score[5], top10=score[10])
        valid_loss /= len(valid_loader)
        for k in topks:
            accelerator.print(f"Valid top {k} acc: {valid_score[k]/len(valid_dataset)}", end=" || ") 
        accelerator.print("")
        print(f'Validation Loss ({min_loss:.6f}--->{valid_loss:.6f})')
        min_loss = valid_loss
        accelerator.save({
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch
                }, "./Epoch:{:0>2} SupCL-SciNCL.pth".format(epoch+1))