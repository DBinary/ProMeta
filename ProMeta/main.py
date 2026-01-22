import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

# Imports from our new modules
from config import parse_args
from utils import set_seed, compute_task_metrics, save_results
from dataset import generate_pathway_mask, MetaDataset
from model import ProphetBioGateModel, FocalLoss

def train_step_v2(model, batch, outer_optimizer, config, device):
    """Executes one meta-training step (Inner Loop + Outer Loop)."""
    q_in, q_lb, s_in, s_lb = batch[0], batch[1], batch[2], batch[3]
    s_in, s_lb = s_in.to(device), s_lb.to(device)
    q_in, q_lb = q_in.to(device), q_lb.to(device)

    inner_criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma) 
    outer_criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    
    outer_optimizer.zero_grad()
    
    meta_loss_sum = 0.0
    focal_loss_sum = 0.0
    batch_preds_collect = []
    batch_labels_collect = []
    
    B_tasks = s_in.shape[0]
    all_params = {n: p for n, p in model.named_parameters() if 'alphas' not in n}
    
    for i in range(B_tasks):
        fast_weights = {k: v for k, v in all_params.items()} 
        alphas = model.alphas
        grad_keys = [k for k in fast_weights.keys() if 'classifier' in k or 'gate' in k]
        
        t_s_in, t_s_lb = s_in[i], s_lb[i]
        t_q_in, t_q_lb = q_in[i], q_lb[i]
        
        # Inner Loop
        for _ in range(config.inner_step):
            preds, gate_vals = model.functional_forward(t_s_in.unsqueeze(0), fast_weights)
            cls_loss = inner_criterion(preds.squeeze(0), t_s_lb.unsqueeze(-1))
            l1_loss = config.l1_lambda * torch.norm(gate_vals, p=1)
            
            grads_dict = torch.autograd.grad(
                cls_loss + l1_loss, 
                [fast_weights[n] for n in grad_keys], 
                create_graph=True, allow_unused=True
            )
            for idx, name in enumerate(grad_keys):
                if grads_dict[idx] is not None:
                    fast_weights[name] = fast_weights[name] - alphas[name.replace('.', '_')] * grads_dict[idx]
        
        # Outer Loop
        q_logits, _ = model.functional_forward(t_q_in.unsqueeze(0), fast_weights)
        q_logits = q_logits.squeeze(0)
        
        batch_preds_collect.append(torch.sigmoid(q_logits).detach().cpu())
        batch_labels_collect.append(t_q_lb.detach().cpu())
        
        task_loss = outer_criterion(q_logits, t_q_lb.unsqueeze(-1))
        meta_loss_sum += task_loss
        focal_loss_sum += task_loss.item()
        
    loss_for_backward = meta_loss_sum / B_tasks
    loss_for_backward.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    outer_optimizer.step()
    
    return loss_for_backward.item(), focal_loss_sum / B_tasks, torch.cat(batch_preds_collect), torch.cat(batch_labels_collect)

def evaluate_v2(model, task_loader, config, device, mode="Valid"):
    model.eval()
    task_results = []
    acc_dict = {k: [] for k in ["auroc", "auprc", "f1", "accuracy"]}
    eval_criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
    
    for batch in tqdm(task_loader, desc=f"Eval ({mode})", leave=False):        
        q_in, q_lb, s_in, s_lb = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        term_name = batch[4][0]
        
        if len(torch.unique(s_lb[0])) < 2: continue 

        fast_weights = {n: p.clone().detach().requires_grad_(True) for n, p in model.named_parameters() if 'alphas' not in n}
        alphas = {k: v.clone().detach() for k, v in model.alphas.items()}
        
        for _ in range(config.inner_step + 5): 
            preds, gate_vals = model.functional_forward(s_in, fast_weights)
            loss = eval_criterion(preds.squeeze(0), s_lb.squeeze(0).unsqueeze(-1)) + config.l1_lambda * torch.norm(gate_vals, p=1)
            target_params = [fast_weights[n] for n in fast_weights.keys() if 'gate' in n or 'classifier' in n]
            grads = torch.autograd.grad(loss, target_params, create_graph=False)
            idx = 0
            for name in fast_weights.keys():
                if 'gate' in name or 'classifier' in name:
                    fast_weights[name] = fast_weights[name] - alphas[name.replace('.', '_')] * grads[idx]
                    idx += 1
        
        with torch.no_grad():
            q_logits, _ = model.functional_forward(q_in, fast_weights)
            q_probs = torch.sigmoid(q_logits).squeeze(0)
            if torch.isnan(q_probs).any() or len(torch.unique(q_lb.squeeze(0))) < 2: continue

        try:
            metrics = compute_task_metrics(q_probs, q_lb.squeeze(0))
            if metrics:
                metrics['disease_term'] = str(term_name) 
                task_results.append(metrics)
                for k in acc_dict: 
                    if k in metrics: acc_dict[k].append(metrics[k])
        except: pass
    
    summary = {k: float(np.mean(v)) if len(v)>0 else 0.0 for k,v in acc_dict.items()}
    return summary, task_results

def main():
    args, config = parse_args()
    set_seed(args.random_seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device} | Config: OutLR={args.outer_lr}, InLR={args.inner_lr}")

    # Load Data
    p_data_df = pd.read_csv(args.proteomics_csv)
    p_data_df['EID'] = p_data_df['EID'].apply(lambda x: str(x).strip().replace('.0',''))
    eid_to_idx = {e: i for i, e in enumerate(p_data_df['EID'].values)}
    protein_names = p_data_df.drop(columns=['EID']).columns.tolist()
    proteins = np.nan_to_num(p_data_df.drop(columns=['EID']).values.astype(np.float32))
    
    def load_pkl(n): return pkl.load(open(os.path.join(args.data_dir, n), 'rb'))
    train_case, train_ctrl = load_pkl('term2pre_cases_train.pkl'), load_pkl('term2pre_controls_train.pkl')
    valid_case, valid_ctrl = load_pkl('term2pre_cases_valid.pkl'), load_pkl('term2pre_controls_valid.pkl')
    test_case, test_ctrl = load_pkl('term2pre_cases_test.pkl'), load_pkl('term2pre_controls_test.pkl')
    
    pathway_mask, unknown_indices = generate_pathway_mask(protein_names, args.cpdb_path)
    pathway_mask = pathway_mask.to(device)
    
    train_loader = DataLoader(MetaDataset(proteins, train_case, train_ctrl, eid_to_idx, args.support_size, mode='train', random_seed=args.random_seed), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(MetaDataset(proteins, valid_case, valid_ctrl, eid_to_idx, args.support_size, mode='valid', random_seed=args.random_seed), batch_size=1, shuffle=False)
    test_loader = DataLoader(MetaDataset(proteins, test_case, test_ctrl, eid_to_idx, args.support_size, mode='test', random_seed=args.random_seed), batch_size=1, shuffle=False)

    model = ProphetBioGateModel(proteins.shape[1], config, pathway_mask, unknown_indices).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    
    best_auroc = 0.0
    history = {'train_loss': [], 'val_auroc': [], 'val_auprc': []}
    save_dir = os.path.join(args.output_dir, "checkpoints", f"support_{args.support_size}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"ProMeta_best_seed{args.random_seed}.pth")

    for epoch in range(config.epochs):
        model.train()
        l_sum, focal_sum, train_probs_all, train_labels_all = 0.0, 0.0, [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            loss, focal_val, b_probs, b_labels = train_step_v2(model, batch, optimizer, config, device)
            l_sum += loss
            focal_sum += focal_val
            train_probs_all.append(b_probs); train_labels_all.append(b_labels)
            
        train_metrics = compute_task_metrics(torch.cat(train_probs_all), torch.cat(train_labels_all)) or {"auroc": 0.0}
        val_summary, _ = evaluate_v2(model, valid_loader, config, device)
        
        history['train_loss'].append(l_sum / len(train_loader))
        history['val_auroc'].append(val_summary['auroc'])
        history['val_auprc'].append(val_summary['auprc'])
        
        print(f"Epoch {epoch+1} | Loss: {l_sum/len(train_loader):.4f} | Train AUC: {train_metrics['auroc']:.4f} | Val AUC: {val_summary['auroc']:.4f}")
    
        if val_summary['auroc'] > best_auroc:
            best_auroc = val_summary['auroc']
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ New Best Model Saved (AUROC: {best_auroc:.4f})")
    
    print("\n--- Testing ---")
    if os.path.exists(best_model_path): model.load_state_dict(torch.load(best_model_path))
    test_summary, test_results = evaluate_v2(model, test_loader, config, device, mode="Test")
    print(f"🏆 Final Test AUROC: {test_summary['auroc']:.4f}")
    save_results(test_summary, test_results, args, "ProMeta", args.output_dir, history)

if __name__ == "__main__":
    main()