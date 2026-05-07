import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None, sample_importance=None,
              collect_preds=False):
    # sample_importance: tensor of shape (N,) where N is the number of samples in the dataset
    # collect_preds: if True and not is_training, collect (sample_idx, pred) pairs and return them

    """
    scheduler is only used inside this function if model is bert.
    Returns (all_idx, all_preds) as numpy arrays when collect_preds=True and not is_training,
    otherwise returns (None, None).
    """

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader
    ###########################################################################
    imp = None
    if is_training and (sample_importance is not None):
        if not torch.is_tensor(sample_importance):
            imp = torch.tensor(sample_importance, dtype=torch.float32).cuda()
        else:
            imp = sample_importance.to(dtype=torch.float32, device='cuda')
    ###########################################################################

    _collect = collect_preds and not is_training
    _all_idx   = [] if _collect else None
    _all_preds = [] if _collect else None

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            idx = batch[3]  # sample indices

            ###############################
            sample_weight = None
            if imp is not None:
                sample_weight = imp[idx]
            ###############################

            if args.model == 'bert':
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1] # [1] returns logits
            else:
                outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training, sample_weight=sample_weight)

            if is_training:
                if args.model == 'bert':
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if _collect:
                _all_idx.extend(idx.cpu().numpy())
                _all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()

    if _collect:
        return np.array(_all_idx), np.array(_all_preds)
    return None, None


_IDENTITY_COLS_TRAIN = ["male", "female", "LGBTQ", "black", "white", "christian", "muslim", "other_religions"]
_CIVILCOMMENTS_CSV_PATH = None  # supplied at runtime via val_split_df / test_split_df

def _compute_16group_row(epoch, split_df, all_idx, all_preds):
    """
    split_df  : metadata_df filtered to the current split (val or test), reset_index
    all_idx   : numpy array of dataset indices from the DataLoader
    all_preds : numpy array of predicted class indices
    Returns a dict {epoch, worst_16group, acc_16g_<attr>_<y>...}
    """
    import pandas as pd
    n = len(split_df)
    correct_arr = np.zeros(n, dtype=np.float32)
    y_arr       = (split_df['toxicity'].values >= 0.5).astype(int)

    pred_arr = np.empty(n, dtype=np.int32)
    pred_arr[:] = -1
    for i, p in zip(all_idx, all_preds):
        if 0 <= i < n:
            pred_arr[i] = p
    valid_mask = (pred_arr >= 0)
    correct_arr[valid_mask] = (pred_arr[valid_mask] == y_arr[valid_mask]).astype(np.float32)

    row = {'epoch': epoch}
    accs = {}
    for attr in _IDENTITY_COLS_TRAIN:
        if attr not in split_df.columns:
            continue
        has_attr = (split_df[attr].fillna(0).values >= 0.5)
        for y_val in [0, 1]:
            mask = has_attr & (y_arr == y_val) & valid_mask
            key = f"acc_16g_{attr}_{y_val}"
            accs[key] = float(correct_arr[mask].mean()) if mask.sum() > 0 else float('nan')
    row['worst_16group'] = min((v for v in accs.values() if not np.isnan(v)), default=float('nan'))
    row.update(accs)
    return row


def _append_16group_csv(csv_path, row):
    """Append one row to the 16-group CSV, creating the file with header if needed."""
    import csv, os
    fieldnames = ['epoch', 'worst_16group'] + [
        f"acc_16g_{attr}_{y}"
        for attr in _IDENTITY_COLS_TRAIN
        for y in [0, 1]
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, float('nan')) for k in fieldnames})


def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset,
          val_16group_csv=None, test_16group_csv=None,
          val_split_df=None, test_split_df=None):
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)

    # BERT uses its own scheduler and optimizer
    if args.model == 'bert':
        from pytorch_transformers import AdamW, WarmupLinearSchedule
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=t_total)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        else:
            scheduler = None

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            sample_importance=dataset.get('sample_importance', None))

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        val_idx, val_preds = run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False,
            collect_preds=(val_16group_csv is not None))
        if val_16group_csv is not None and val_preds is not None:
            row = _compute_16group_row(epoch, val_split_df, val_idx, val_preds)
            _append_16group_csv(val_16group_csv, row)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            test_idx, test_preds = run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                None, test_csv_logger, args,
                is_training=False,
                collect_preds=(test_16group_csv is not None))
            if test_16group_csv is not None and test_preds is not None:
                row = _compute_16group_row(epoch, test_split_df, test_idx, test_preds)
                _append_16group_csv(test_16group_csv, row)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        # Save periodic checkpoints on 1-based epoch boundaries.
        # This avoids creating an unnecessary 0_model.pth at epoch 0.
        if args.save_step > 0 and (epoch + 1) % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % (epoch + 1)))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')
