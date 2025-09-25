import argparse
import json
import os
import shutil
import torch
import tqdm
import time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearnex import patch_sklearn
from lib.oversmooth_metrics import *
import csv

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

from lib.best import *
from lib.transforms import *
from lib.models import *
from lib.utils import *
from lib.dataset import *

RESULTS_FOLDER = '.results'
if not os.path.exists(RESULTS_FOLDER):
  os.makedirs(RESULTS_FOLDER)

def compute_metrics(y_true, y_pred, y_score=None):
    """Compute all required metrics."""
    metrics = {}
    metrics['acc'] = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    if y_score is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_score)
    return metrics

def save_metrics(metrics, split_name, fold_idx, save_path):
    """Save metrics for a specific split and fold."""
    metrics_file = os.path.join(save_path, f'metrics_{split_name}_fold_{fold_idx}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_features(features, split_name, fold_idx, save_path):
    """Save node features for a specific split and fold."""
    features_file = os.path.join(save_path, f'features_{split_name}_fold_{fold_idx}.pt')
    torch.save(features, features_file)

def training_step(model, optimizer, criterion, data, train_mask):
    model.train()
    optimizer.zero_grad()  
    out, dirichlet_energy, metrics = model(data)  # Perform a single forward pass.
    loss = criterion(out[train_mask], data.y[train_mask]) 
    loss.backward()   
    optimizer.step()  
    return loss

def evaluate(model, criterion, data, train_mask, val_mask, test_mask):
    model.eval()
    metrics = {
        "loss":{},
        "acc":{},
        "precision":{},
        "recall":{},
        "f1":{},
        "roc_auc":{},
        "dirichlet_energy": {"real": {},
                          "imag": {}},
        "dirichlet_energy_ratio": {"real": {},
                          "imag": {}},
        "effective_rank": {
            "before": {},
            "after": {},
            "ratio": {}
        },
        "class_mix_score": {
            "before": {},
            "after": {},
            "ratio": {}
        }
    }
    
    with torch.no_grad():
        start_time = time.time()
        out, dirichlet_energy, oversmooth_metrics = model(data)
        inference_time = time.time() - start_time
        
        metrics["dirichlet_energy"]["real"] = dirichlet_energy[-1].real
        metrics["dirichlet_energy_ratio"]["real"] = (dirichlet_energy[-1]/dirichlet_energy[0]).real
        if type(dirichlet_energy)==torch.cfloat:
            metrics["dirichlet_energy"]["imag"] = dirichlet_energy[-1].imag
            metrics["dirichlet_energy_ratio"]["imag"] = (dirichlet_energy[-1]/dirichlet_energy[0]).imag
        
        # Save oversmooth metrics for each split
        for split, mask in zip(["train", "val", "test"], [train_mask, val_mask, test_mask]):
            metrics["effective_rank"]["before"][split] = oversmooth_metrics["before"]["effective_rank"].item()
            metrics["effective_rank"]["after"][split] = oversmooth_metrics["after"]["effective_rank"].item()
            metrics["effective_rank"]["ratio"][split] = oversmooth_metrics["ratios"]["effective_rank_ratio"].item()
            
            metrics["class_mix_score"]["before"][split] = oversmooth_metrics["before"]["class_mix_score"].item()
            metrics["class_mix_score"]["after"][split] = oversmooth_metrics["after"]["class_mix_score"].item()
            metrics["class_mix_score"]["ratio"][split] = oversmooth_metrics["ratios"]["class_mix_score_ratio"].item()
        
        pred_class = out.argmax(dim=1)
        pred_score = out.softmax(1)[:, -1] if data.y.max().item()==1 else None
        
        for split, mask in zip(["train", "val", "test"], [train_mask, val_mask, test_mask]):
            metrics["loss"][split] = criterion(out[mask], data.y[mask]).item()
            split_metrics = compute_metrics(
                data.y[mask].cpu().numpy(),
                pred_class[mask].cpu().numpy(),
                pred_score[mask].cpu().numpy() if pred_score is not None else None
            )
            for metric_name, value in split_metrics.items():
                metrics[metric_name][split] = value
        
        metrics["inference_time"] = inference_time
        return metrics, out

def main(options):
    # Set global random seed for reproducibility
    seed_all(12345)

    #Delete processed file
    print(f'Deleting preprocessed files')
    if options["dataset"] in ["Cora", "Citeseer", "Pubmed"]:
        shutil.rmtree(f'.data/{options["dataset"]}/geom-gcn/processed', ignore_errors=True)
    elif options["dataset"] in ["Squirrel", "Chameleon"]:
        shutil.rmtree(f'.data/{options["dataset"]}/geom_gcn/processed', ignore_errors=True)
    elif options["dataset"] in ["Minesweeper", "Tolokers", "Roman-empire", "Amazon-ratings", "Questions"]:
        tmp = options["dataset"].lower().replace("-", "_")
        shutil.rmtree(f'.data/{tmp}/processed', ignore_errors=True)
    
    patch_sklearn()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f'device: {colortext(device, "c")}.')
    
    # Create dataset specific directory
    dataset_dir = f'results/{options["dataset"]}_bestopt/folds_results'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(dataset_dir, 'config.json'), 'w') as f:
        json.dump(options, f, indent=4)
    
    # Build transform and dataset
    transform = build_transform(
        normalize_features=options["normalize_features"],
        norm_ord=options["norm_ord"], 
        norm_dim=options["norm_dim"],
        undirected=options["undirected"],
        self_loops=options["self_loops"],
        lcc=options["lcc"],
        sparsity=options["sparsity"],
        sklearn=options["sklearn"],
        verbose=options["verbose"]  
    )
    
    dataset, data, metric_name = build_dataset(
        dataset_name=options["dataset"], 
        transform=transform,
        verbose=options["verbose"]
    )

    criterion = torch.nn.CrossEntropyLoss()
    num_splits = options["num_split"]
    print("num_splits", num_splits)
    
    # Initialize results storage
    all_fold_metrics = []
    all_losses = []
    all_best_val_features = []
    
    # Training
    for nsplit in range(num_splits):
        seed_all(12345)
        model = fLode(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_channels=options["hidden_channels"], 
            num_layers=options["num_layers"], 
            exponent=options["exponent"], 
            spectral_shift=options["spectral_shift"], 
            step_size=options["step_size"], 
            channel_mixing=options["channel_mixing"], 
            input_dropout=options["input_dropout"], 
            decoder_dropout=options["decoder_dropout"], 
            init=options["init"], 
            dtype=torch.float if options["real"] else torch.cfloat, 
            eq=options["equation"],
            encoder_layers=options["encoder_layers"],
            decoder_layers=options["decoder_layers"],
            gcn=options["gcn"],
            no_sharing=options["no_sharing"],
            layer_norm=options["layer_norm"]
        ).to(device)
        
        if options["verbose"]:
            print("Model")
            print(f'| num. params: {colortext(compute_num_params(model), "c")}')
        
        optimizer = getattr(torch.optim, options["optimizer"])(
            model.parameters(),  
            lr=options["learning_rate"],
            weight_decay=options["weight_decay"]    
        )

        data = dataset.data.clone()
        train_mask = data.train_mask[:, nsplit].to(torch.bool)
        val_mask = data.val_mask[:, nsplit].to(torch.bool)
        test_mask = data.test_mask[:, nsplit].to(torch.bool)
        data = data.to(device)
        
        best_val_metrics = None
        best_test_metrics = None
        best_val_model_state = None
        best_val_features = None
        best_test_features = None
        best_val_loss = None
        best_val_acc = 0
        best_test_acc = 0
        losses = []
        
        print(f"\n===== Split {nsplit+1}/10 =====")
        
        with tqdm.trange(1, options["num_epochs"]+1) as progress:
            early_stopping_counter = 0
            for epoch in progress:
                loss = training_step(
                    data=data,
                    model=model, 
                    optimizer=optimizer, 
                    criterion=criterion, 
                    train_mask=train_mask
                )
                losses.append(loss)
                
                with torch.no_grad():
                    evaluation_metrics, out = evaluate(
                        model=model, 
                        criterion=criterion, 
                        data=data, 
                        train_mask=train_mask,
                        val_mask=val_mask,
                        test_mask=test_mask
                    )
                    
                    if evaluation_metrics["acc"]["val"] > best_val_acc:
                        best_val_acc = evaluation_metrics["acc"]["val"]
                        best_val_metrics = evaluation_metrics
                        best_val_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        best_val_features = out[val_mask].cpu()
                        best_val_loss = loss
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    
                    if evaluation_metrics["acc"]["test"] > best_test_acc:
                        best_test_acc = evaluation_metrics["acc"]["test"]
                        best_test_metrics = evaluation_metrics
                        best_test_features = out[test_mask].cpu()
                
                description = (
                    f'Loss: {loss:.4f}, '
                    + metric_name 
                    + ' (train, val, test): ('
                    + "{:.4f}, ".format(evaluation_metrics[metric_name]["train"])
                    + "{:.4f}, ".format(evaluation_metrics[metric_name]["val"])
                    + "{:.4f})".format(evaluation_metrics[metric_name]["test"])
                )
                progress.set_description(description)
                
                if early_stopping_counter >= options["patience"]:
                    break
        
        # Save fold results
        fold_result = {
            'split': nsplit,
            'val': best_val_metrics,
            'test': best_test_metrics,
            'params': options,
            'model_param_count': compute_num_params(model),
            'best_val_loss': best_val_loss
        }
        all_fold_metrics.append(fold_result)
        all_losses.append(losses)
        
        # Save model state
        time_value = options['num_layers']  # 使用num_layers作为time值
        torch.save(best_val_model_state, f'{dataset_dir}/model_split{nsplit}_time{time_value}.pt')
        
        # Save features
        if best_val_features is not None:
            torch.save(best_val_features, f'{dataset_dir}/val_features_split{nsplit}_time{time_value}.pt')
        if best_test_features is not None:
            torch.save(best_test_features, f'{dataset_dir}/test_features_split{nsplit}_time{time_value}.pt')
    
    # Compute and save average metrics
    val_accs = [f['val']['acc']['val'] for f in all_fold_metrics]
    test_accs = [f['test']['acc']['test'] for f in all_fold_metrics]
    val_precisions = [f['val']['precision']['val'] for f in all_fold_metrics]
    test_precisions = [f['test']['precision']['test'] for f in all_fold_metrics]
    val_recalls = [f['val']['recall']['val'] for f in all_fold_metrics]
    test_recalls = [f['test']['recall']['test'] for f in all_fold_metrics]
    val_f1s = [f['val']['f1']['val'] for f in all_fold_metrics]
    test_f1s = [f['test']['f1']['test'] for f in all_fold_metrics]
    
    # Compute average oversmooth metrics
    val_effrank_before = [f['val']['effective_rank']['before']['val'] for f in all_fold_metrics]
    val_effrank_after = [f['val']['effective_rank']['after']['val'] for f in all_fold_metrics]
    val_effrank_ratio = [f['val']['effective_rank']['ratio']['val'] for f in all_fold_metrics]
    test_effrank_before = [f['test']['effective_rank']['before']['test'] for f in all_fold_metrics]
    test_effrank_after = [f['test']['effective_rank']['after']['test'] for f in all_fold_metrics]
    test_effrank_ratio = [f['test']['effective_rank']['ratio']['test'] for f in all_fold_metrics]
    
    val_classmix_before = [f['val']['class_mix_score']['before']['val'] for f in all_fold_metrics]
    val_classmix_after = [f['val']['class_mix_score']['after']['val'] for f in all_fold_metrics]
    val_classmix_ratio = [f['val']['class_mix_score']['ratio']['val'] for f in all_fold_metrics]
    test_classmix_before = [f['test']['class_mix_score']['before']['test'] for f in all_fold_metrics]
    test_classmix_after = [f['test']['class_mix_score']['after']['test'] for f in all_fold_metrics]
    test_classmix_ratio = [f['test']['class_mix_score']['ratio']['test'] for f in all_fold_metrics]
    
    summary = {
        'val_acc_mean': float(np.mean(val_accs)),
        'val_acc_std': float(np.std(val_accs)),
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'val_precision_mean': float(np.mean(val_precisions)),
        'val_precision_std': float(np.std(val_precisions)),
        'test_precision_mean': float(np.mean(test_precisions)),
        'test_precision_std': float(np.std(test_precisions)),
        'val_recall_mean': float(np.mean(val_recalls)),
        'val_recall_std': float(np.std(val_recalls)),
        'test_recall_mean': float(np.mean(test_recalls)),
        'test_recall_std': float(np.std(test_recalls)),
        'val_f1_mean': float(np.mean(val_f1s)),
        'val_f1_std': float(np.std(val_f1s)),
        'test_f1_mean': float(np.mean(test_f1s)),
        'test_f1_std': float(np.std(test_f1s)),
        'val_effrank_before_mean': float(np.mean(val_effrank_before)),
        'val_effrank_before_std': float(np.std(val_effrank_before)),
        'val_effrank_after_mean': float(np.mean(val_effrank_after)),
        'val_effrank_after_std': float(np.std(val_effrank_after)),
        'val_effrank_ratio_mean': float(np.mean(val_effrank_ratio)),
        'val_effrank_ratio_std': float(np.std(val_effrank_ratio)),
        'test_effrank_before_mean': float(np.mean(test_effrank_before)),
        'test_effrank_before_std': float(np.std(test_effrank_before)),
        'test_effrank_after_mean': float(np.mean(test_effrank_after)),
        'test_effrank_after_std': float(np.std(test_effrank_after)),
        'test_effrank_ratio_mean': float(np.mean(test_effrank_ratio)),
        'test_effrank_ratio_std': float(np.std(test_effrank_ratio)),
        'val_classmix_before_mean': float(np.mean(val_classmix_before)),
        'val_classmix_before_std': float(np.std(val_classmix_before)),
        'val_classmix_after_mean': float(np.mean(val_classmix_after)),
        'val_classmix_after_std': float(np.std(val_classmix_after)),
        'val_classmix_ratio_mean': float(np.mean(val_classmix_ratio)),
        'val_classmix_ratio_std': float(np.std(val_classmix_ratio)),
        'test_classmix_before_mean': float(np.mean(test_classmix_before)),
        'test_classmix_before_std': float(np.std(test_classmix_before)),
        'test_classmix_after_mean': float(np.mean(test_classmix_after)),
        'test_classmix_after_std': float(np.std(test_classmix_after)),
        'test_classmix_ratio_mean': float(np.mean(test_classmix_ratio)),
        'test_classmix_ratio_std': float(np.std(test_classmix_ratio)),
        'all_params': options,
    }
    all_fold_metrics = convert_to_json_serializable(all_fold_metrics)
    
    # Save results with time value in filenames
    time_value = options['num_layers']  # 使用num_layers作为time值
    with open(f'{dataset_dir}/fold_metrics_time{time_value}.json', 'w') as f:
        json.dump(all_fold_metrics, f, indent=2)
    with open(f'{dataset_dir}/summary_time{time_value}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(f'{dataset_dir}/losses_time{time_value}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'fold_{i}' for i in range(num_splits)])
        for row in zip(*all_losses):
            writer.writerow(row)
    
    print(f'Overall performances (mean, std)')
    print(f'Validation Accuracy: {summary["val_acc_mean"]:.4f} ± {summary["val_acc_std"]:.4f}')
    print(f'Test Accuracy: {summary["test_acc_mean"]:.4f} ± {summary["test_acc_std"]:.4f}')
    print(f'Validation F1: {summary["val_f1_mean"]:.4f} ± {summary["val_f1_std"]:.4f}')
    print(f'Test F1: {summary["test_f1_mean"]:.4f} ± {summary["test_f1_std"]:.4f}')
    print(f'Validation Effective Rank (before/after): {summary["val_effrank_before_mean"]:.4f} ± {summary["val_effrank_before_std"]:.4f} / {summary["val_effrank_after_mean"]:.4f} ± {summary["val_effrank_after_std"]:.4f}')
    print(f'Test Effective Rank (before/after): {summary["test_effrank_before_mean"]:.4f} ± {summary["test_effrank_before_std"]:.4f} / {summary["test_effrank_after_mean"]:.4f} ± {summary["test_effrank_after_std"]:.4f}')
    print(f'Validation Class Mix Score (before/after): {summary["val_classmix_before_mean"]:.4f} ± {summary["val_classmix_before_std"]:.4f} / {summary["val_classmix_after_mean"]:.4f} ± {summary["val_classmix_after_std"]:.4f}')
    print(f'Test Class Mix Score (before/after): {summary["test_classmix_before_mean"]:.4f} ± {summary["test_classmix_before_std"]:.4f} / {summary["test_classmix_after_mean"]:.4f} ± {summary["test_classmix_after_std"]:.4f}')

def convert_to_json_serializable(obj):
    # for k, v in obj.items():
    #     try:
    #         json.dumps(v)
    #     except TypeError as e:
    #         print(f"Key '{k}' is not serializable: {type(v)} – {e}")

    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, range):
        return list(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}  
    else:
        return obj


def run_single_split(opt):
    """Run a single split of the experiment and save results."""
    split_id = opt['split_id']
    dataset_name = opt['dataset']
    num_layers = opt['num_layers']
    
    # Create dataset specific directory
    dataset_dir = f'results/{dataset_name}_bestopt'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Set up dataset and model
    transform = build_transform(
        normalize_features=opt["normalize_features"],
        norm_ord=opt["norm_ord"], 
        norm_dim=opt["norm_dim"],
        undirected=opt["undirected"],
        self_loops=opt["self_loops"],
        lcc=opt["lcc"],
        sparsity=opt["sparsity"],
        sklearn=opt["sklearn"],
        verbose=opt["verbose"]  
    )
    
    dataset, data, metric_name = build_dataset(
        dataset_name=opt["dataset"], 
        transform=transform,
        verbose=opt["verbose"]
    )
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = fLode(
        in_channels=dataset.num_features,
        out_channels=dataset.num_classes,
        hidden_channels=opt["hidden_channels"], 
        num_layers=opt["num_layers"], 
        exponent=opt["exponent"], 
        spectral_shift=opt["spectral_shift"], 
        step_size=opt["step_size"], 
        channel_mixing=opt["channel_mixing"], 
        input_dropout=opt["input_dropout"], 
        decoder_dropout=opt["decoder_dropout"], 
        init=opt["init"], 
        dtype=torch.float if opt["real"] else torch.cfloat, 
        eq=opt["equation"],
        encoder_layers=opt["encoder_layers"],
        decoder_layers=opt["decoder_layers"],
        gcn=opt["gcn"],
        no_sharing=opt["no_sharing"],
        layer_norm=opt["layer_norm"]
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, opt["optimizer"])(
        model.parameters(),  
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"]    
    )

    data = dataset.data.clone()
    train_mask = data.train_mask[:, split_id].to(torch.bool)
    val_mask = data.val_mask[:, split_id].to(torch.bool)
    test_mask = data.test_mask[:, split_id].to(torch.bool)
    data = data.to(device)
    
    best_val_metrics = None
    best_test_metrics = None
    best_val_model_state = None
    best_val_features = None
    best_test_features = None
    best_val_loss = None
    best_val_acc = 0
    best_test_acc = 0
    losses = []
    
    print(f"\n===== Split {split_id}/9 (single split mode) =====")
    print(f"Model parameters: {compute_num_params(model)}")
    
    for epoch in range(1, opt["num_epochs"] + 1):
        start_time = time.time()
        loss = training_step(
            data=data,
            model=model, 
            optimizer=optimizer, 
            criterion=criterion, 
            train_mask=train_mask
        )
        losses.append(loss)
        
        with torch.no_grad():
            evaluation_metrics, out = evaluate(
                model=model, 
                criterion=criterion, 
                data=data, 
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
            if evaluation_metrics["acc"]["val"] > best_val_acc:
                best_val_acc = evaluation_metrics["acc"]["val"]
                best_val_metrics = evaluation_metrics
                best_val_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_val_features = out[val_mask].cpu()
                best_val_loss = loss
                
            if evaluation_metrics["acc"]["test"] > best_test_acc:
                best_test_acc = evaluation_metrics["acc"]["test"]
                best_test_metrics = evaluation_metrics
                best_test_features = out[test_mask].cpu()
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {evaluation_metrics["acc"]["val"]:.4f}, Test Acc: {evaluation_metrics["acc"]["test"]:.4f}')
    
    # Save results for this split
    fold_result = {
        'split': split_id,
        'val': best_val_metrics,
        'test': best_test_metrics,
        'params': opt,
        'model_param_count': compute_num_params(model),
        'best_val_loss': best_val_loss.item() if isinstance(best_val_loss, torch.Tensor) else best_val_loss
    }
    
    # Convert tensors to native Python types before saving
    fold_result = convert_to_json_serializable(fold_result)
    
    print("\nDebug: Content of fold_result before saving:")
    print(json.dumps(fold_result, indent=2))
    
    # Save model state
    torch.save(best_val_model_state, f'{dataset_dir}/model_split{split_id}_layers{num_layers}.pt')
    
    # Save features
    if best_val_features is not None:
        torch.save(best_val_features, f'{dataset_dir}/val_features_split{split_id}_layers{num_layers}.pt')
    if best_test_features is not None:
        torch.save(best_test_features, f'{dataset_dir}/test_features_split{split_id}_layers{num_layers}.pt')
    
    # Save metrics and losses
    with open(f'{dataset_dir}/single_split_summary_layers{num_layers}.json', 'w') as f:
        json.dump(fold_result, f, indent=2)
    
    with open(f'{dataset_dir}/losses_layers{num_layers}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i, loss in enumerate(losses):
            writer.writerow([i+1, loss.item() if isinstance(loss, torch.Tensor) else loss])
    
    return fold_result

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true', help='Flag to print useful information.')
    parser.add_argument('-b', '--best', dest="best", action='store_true', help='Flag to use the hyperparams from "lib.best".')
    #Dataset
    parser.add_argument('--dataset', dest="dataset", default='chameleon', type=str, help='Which dataset to use (default chameleon).') 
    # Transforms
    parser.add_argument('-n', '--normalize_features', dest="normalize_features", action='store_true', help='Normalizes features.')
    parser.add_argument('--norm_ord', type=norm_ord_type, default=2, help='p-norm w.r.t. which normalize the features (default 2). Check torch.linalg.norm for the possible values. Note that we allow norm_ord="sum" to retrieve the behaviour of NormalizeFeatures() from torch_geometric.transforms.NormalizeFeatures().') 
    parser.add_argument('--norm_dim', type=int, default=0, help='Dimension w.r.t. which normalize the features (default 0).') 
    parser.add_argument('-u', '--undirected', dest="undirected", action='store_true', help='Make the graph undirected.')
    parser.add_argument('--self_loops', type=float, default=0., help='Value for the self loops (default 0.0).')
    parser.add_argument('-l', '--lcc', dest="lcc", action='store_true', help='Consider only the largest connected component.') 
    parser.add_argument('--sparsity', type=float, default=0.0, help='(1-sparsity)*num_nodes singular values will be computed and stored.') 
    parser.add_argument('--sklearn', dest="sklearn", action='store_true', help='Use the scikit-learn-intelex.extmath library to compute the svd. Useful when the graph is too large, i.e., when the torch.linalg.svd() would cause an out-of-memory error.') 
    # Model
    parser.add_argument('--layer_norm', action="store_true", help="Apply layer normalization")
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels (default 64).') 
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers (default 3).') 
    parser.add_argument('--exponent', type=float_or_learnable, default="learnable", help='Value of \alpha (float or "learnable", default "learnable").') 
    parser.add_argument('--spectral_shift', type=float_or_learnable, default=0.0, help='Value of spectral_shift (float (default 0.0).') 
    parser.add_argument('--step_size', type=float_or_learnable, default="learnable", help='Value of step_size (float or "learnable", default "learnable").') 
    parser.add_argument('--channel_mixing', type=str, default="d", help='Which parametrization of channel_mixing to use: "d" for diagonal, "s" for symmetric, "f" for full. (defaul "d")') 
    parser.add_argument('--no_sharing', dest="no_sharing", action='store_true', help='If channel mixing matrix should be different for each layer.') 
    parser.add_argument('--init', type=str, default="normal", help='Which initialization to use for channel_mixing. Check the ones implemented in torch.nn.init. (default "normal")') 
    parser.add_argument('-r', '--real', dest="real", action='store_true', help='The dtype of learnable parameters will be real.') 
    parser.add_argument('--equation', type=str, default="ms", choices=["ms", "s", "mh", "h"], help='Equation to solve: "h" for heat eq., "mh" for minus heat eq., "s" for Schroedinger eq., "ms" "s" for minus Schroedinger eq. (default "ms")') 
    parser.add_argument('--encoder_layers', type=int, default=1, help='Number of encoding layers before the neural ODE (default 1).') 
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoding layers after the neural ODE (default 1).') 
    parser.add_argument('--input_dropout', type=float, default=0.0, help='Dropout of the first encoding layer (default 0.).') 
    parser.add_argument('--decoder_dropout', type=float, default=0.0, help='Dropout of the last decoding layer (default 0.).')
    # Optimizer 
    parser.add_argument('--optimizer', type=str, default="Adam", help='Which optimizer to use (default "Adam").')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate (default 1e-2).')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default 5e-4).')
    # Training
    parser.add_argument('--num_epochs', type=int, default=1000, help='Maximal number of epochs (default 1000).')
    parser.add_argument('--patience', type=int, default=200, help='Patience for early stopping: stops after "patience" consecutive epochs in which the validation accuracy did not increase. (default 200)')
    # Num split
    parser.add_argument('--num_split', type=int, default=10, help='Which splits to consider (default range(10))')
    #Ablation
    parser.add_argument('--gcn', dest="gcn", action='store_true', help='The model is converted to a gcn implementing the (possibly) fractional sna.') 
    # Add split_id argument
    parser.add_argument('--split_id', type=int, default=None, 
                      help='If set, only run the specified split (0-9) and save summary.json for that split.')

    options = vars(parser.parse_args())

    seed_all(12345)

    import sys
    # result_path = f"./results/{options["dataset"]}/folds_results/fold_metrics_time{options['num_layers']}.json"
    # if os.path.exists(result_path):
    #     print(f"Result already exists at {result_path}, skipping this run.")
    #     sys.exit(0)
    
    if options["best"]:
        best_hyperparams = best_hyperparams[options["dataset"]]
        if ("directed" in best_hyperparams.keys()):
            choice = "undirected" if options['undirected'] else "directed"
            best_hyperparams=best_hyperparams[choice]
    #   options={
    #     **options,
    #     **best_hyperparams
        options = {
            **best_hyperparams,
            **options
            }
        print(f"Using best opt for {options['dataset']}")
        
    
    
    print(f'Options')
    if options["verbose"]:
      for k in sorted(options.keys()):
        print(f'| {k}: {options[k]}')
    
    # If real, then the equation must be "h" or "mh"
    if options["real"] and (options["equation"] != options["equation"][0]+'h'):
      print(f'Changing equation from {options["equation"]} to {options["equation"][0]+"h"}')
      options["equation"] = options["equation"][0]+'h'
    
    if options.get('split_id') is not None:
        run_single_split(options)
    else:
        main(options)
