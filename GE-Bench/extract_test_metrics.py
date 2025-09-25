#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import ast

def parse_complex_value(value):
    """
    Parse complex-valued metrics that might be stored as strings or dicts.
    Returns the magnitude (most commonly used) for complex values, or the original value for scalars.
    
    Args:
        value: The metric value, could be a string representation of dict, actual dict, or scalar
        
    Returns:
        The magnitude if complex, original value if scalar
    """
    if isinstance(value, str):
        # Try to parse string representation of complex dict
        try:
            if value.startswith('{') and 'real' in value and 'imag' in value:
                # Use ast.literal_eval for safe evaluation
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict) and 'real' in parsed and 'imag' in parsed:
                    real = float(parsed['real'])
                    imag = float(parsed['imag'])
                    magnitude = np.sqrt(real**2 + imag**2)
                    return magnitude
        except (ValueError, SyntaxError, KeyError):
            pass
        
        # If string parsing fails, try to convert to float
        try:
            return float(value)
        except ValueError:
            return value  # Keep as string if can't convert
            
    elif isinstance(value, dict):
        # Already a dict, check if it's complex-valued
        if 'real' in value and 'imag' in value:
            try:
                real = float(value['real'])
                imag = float(value['imag'])
                magnitude = np.sqrt(real**2 + imag**2)
                return magnitude
            except (ValueError, TypeError):
                return value  # Return original if can't process
        else:
            # Other type of dict, return as is for further processing
            return value
    else:
        # Scalar value, return as is
        return value

def safe_numeric_value(value):
    """
    Safely extract a numeric value from potentially complex metric values.
    Used for mean/std calculations.
    """
    if isinstance(value, (int, float)) and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return value
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    else:
        return None

def extract_metrics_to_csv(base_dir,time_values, val_output_file="val_metrics_summary.csv", test_output_file="test_metrics_summary.csv"):
    """
    Extract validation and test metrics from multiple fold_metrics_time*.json files and create separate CSV tables.
    
    Args:
        base_dir: Directory containing the JSON files
        val_output_file: Output CSV filename for validation metrics
        test_output_file: Output CSV filename for test metrics
    """
    
    # Define the time values we expect
    time_values=time_values
    
    # Find all JSON files matching the pattern
    json_files = []
    for time_val in time_values:
        pattern = os.path.join(base_dir, f"fold_metrics_time{time_val}.json")
        if os.path.exists(pattern):
            json_files.append((time_val, pattern))
    
    if not json_files:
        print(f"No JSON files found in {base_dir}")
        return None, None
    
    print(f"Found {len(json_files)} JSON files:")
    for time_val, filepath in json_files:
        print(f"  Time {time_val}: {filepath}")
    
    # Collect all data separately for val and test
    val_data = []
    test_data = []
    
    for time_val, filepath in sorted(json_files):
        print(f"\nProcessing time={time_val}...")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Store split data for this time value
            time_val_split_data = []
            time_test_split_data = []
            
            # Process each split
            for split_data in data:
                split_id = split_data['split']
                val_metrics = split_data['val']
                test_metrics = split_data['test']
                
                # Create validation row
                val_row = {
                    'time': time_val,
                    'split': split_id
                }
                
                # Create test row
                test_row = {
                    'time': time_val,
                    'split': split_id
                }
                
                # Add validation metrics
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, dict):
                        # Handle nested metrics (like effective_rank, class_mix_score)
                        for sub_key, sub_value in metric_value.items():
                            if isinstance(sub_value, dict):
                                # Handle double nested (like effective_rank->before->val)
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    processed_value = parse_complex_value(sub_sub_value)
                                    val_row[f'{metric_name}_{sub_key}_{sub_sub_key}'] = processed_value
                            else:
                                processed_value = parse_complex_value(sub_value)
                                val_row[f'{metric_name}_{sub_key}'] = processed_value
                    else:
                        processed_value = parse_complex_value(metric_value)
                        val_row[metric_name] = processed_value
                
                # Add test metrics
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, dict):
                        # Handle nested metrics (like effective_rank, class_mix_score)
                        for sub_key, sub_value in metric_value.items():
                            if isinstance(sub_value, dict):
                                # Handle double nested (like effective_rank->before->test)
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    processed_value = parse_complex_value(sub_sub_value)
                                    test_row[f'{metric_name}_{sub_key}_{sub_sub_key}'] = processed_value
                            else:
                                processed_value = parse_complex_value(sub_value)
                                test_row[f'{metric_name}_{sub_key}'] = processed_value
                    else:
                        processed_value = parse_complex_value(metric_value)
                        test_row[metric_name] = processed_value
                
                val_data.append(val_row)
                test_data.append(test_row)
                time_val_split_data.append(val_row)
                time_test_split_data.append(test_row)
            
            # Calculate mean and std for validation metrics
            if time_val_split_data:
                val_columns = [col for col in time_val_split_data[0].keys() if col not in ['time', 'split']]
                
                val_mean_row = {'time': time_val, 'split': 'mean'}
                val_std_row = {'time': time_val, 'split': 'std'}
                
                for col in val_columns:
                    values = [safe_numeric_value(row[col]) for row in time_val_split_data if col in row and safe_numeric_value(row[col]) is not None]
                    if values:
                        val_mean_row[col] = np.mean(values)
                        val_std_row[col] = np.std(values)
                    else:
                        val_mean_row[col] = None
                        val_std_row[col] = None
                
                val_data.append(val_mean_row)
                val_data.append(val_std_row)
            
            # Calculate mean and std for test metrics
            if time_test_split_data:
                test_columns = [col for col in time_test_split_data[0].keys() if col not in ['time', 'split']]
                
                test_mean_row = {'time': time_val, 'split': 'mean'}
                test_std_row = {'time': time_val, 'split': 'std'}
                
                for col in test_columns:
                    values = [safe_numeric_value(row[col]) for row in time_test_split_data if col in row and safe_numeric_value(row[col]) is not None]
                    if values:
                        test_mean_row[col] = np.mean(values)
                        test_std_row[col] = np.std(values, ddof=1)
                    else:
                        test_mean_row[col] = None
                        test_std_row[col] = None
                
                test_data.append(test_mean_row)
                test_data.append(test_std_row)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Add empty rows for missing time values
    if val_data:
        # Get all possible columns from existing data
        all_columns = set()
        for row in val_data:
            all_columns.update(row.keys())
        
        # For each missing time value, add empty rows for each split and mean/std
        for time_val in time_values:
            if time_val not in [row['time'] for row in val_data]:
                # Add empty rows for each split
                for split_id in range(10):
                    empty_row = {'time': time_val, 'split': split_id}
                    for col in all_columns:
                        if col not in ['time', 'split']:
                            empty_row[col] = None
                    val_data.append(empty_row)
                    test_data.append(empty_row)
                
                # Add empty mean and std rows
                empty_mean_row = {'time': time_val, 'split': 'mean'}
                empty_std_row = {'time': time_val, 'split': 'std'}
                for col in all_columns:
                    if col not in ['time', 'split']:
                        empty_mean_row[col] = None
                        empty_std_row[col] = None
                val_data.append(empty_mean_row)
                val_data.append(empty_std_row)
                test_data.append(empty_mean_row)
                test_data.append(empty_std_row)
    
    if not val_data and not test_data:
        print("No data extracted!")
        return None, None
    
    # Process validation data
    val_df = None
    if val_data:
        val_df = pd.DataFrame(val_data)
        
        # Sort by time, then by split (but keep mean and std at the end for each time)
        def sort_key(row):
            time_val = row['time']
            split_val = row['split']
            if split_val == 'mean':
                return (time_val, 10)
            elif split_val == 'std':
                return (time_val, 11)
            else:
                return (time_val, split_val)
        
        val_df['sort_key'] = val_df.apply(sort_key, axis=1)
        val_df = val_df.sort_values('sort_key').drop('sort_key', axis=1)
        
        # Reorder columns: time, split, then all metrics
        val_metric_columns = [col for col in val_df.columns if col not in ['time', 'split']]
        val_column_order = ['time', 'split'] + sorted(val_metric_columns)
        val_df = val_df[val_column_order]
        
        # Save validation CSV
        val_output_path = os.path.join(base_dir, val_output_file)
        val_df['time'] = val_df['time'].astype(int)
        val_df.to_csv(val_output_path, index=False, float_format='%.6f')
        
        print(f"\nValidation data extracted successfully!")
        print(f"Val Shape: {val_df.shape}")
        print(f"Val Columns: {list(val_df.columns)}")
        print(f"Saved to: {val_output_path}")
    
    # Process test data
    test_df = None
    if test_data:
        test_df = pd.DataFrame(test_data)
        
        # Sort by time, then by split (but keep mean and std at the end for each time)
        test_df['sort_key'] = test_df.apply(sort_key, axis=1)
        test_df = test_df.sort_values('sort_key').drop('sort_key', axis=1)
        
        # Reorder columns: time, split, then all metrics
        test_metric_columns = [col for col in test_df.columns if col not in ['time', 'split']]
        test_column_order = ['time', 'split'] + sorted(test_metric_columns)
        test_df = test_df[test_column_order]
        
        # Save test CSV
        test_output_path = os.path.join(base_dir, test_output_file)
        test_df['time'] = test_df['time'].astype(int)
        test_df.to_csv(test_output_path, index=False, float_format='%.6f')
        
        print(f"\nTest data extracted successfully!")
        print(f"Test Shape: {test_df.shape}")
        print(f"Test Columns: {list(test_df.columns)}")
        print(f"Saved to: {test_output_path}")
    
    # Display summary statistics for both validation and test
    if val_df is not None:
        val_numeric_df = val_df[val_df['split'].apply(lambda x: str(x).isdigit())]
        if not val_numeric_df.empty:
            print(f"\nValidation Summary by time:")
            basic_metrics = [col for col in val_numeric_df.columns if col in ['acc', 'f1', 'precision', 'recall']]
            if basic_metrics:
                val_summary = val_numeric_df.groupby('time')[basic_metrics].agg(['mean', 'std']).round(4)
                print(val_summary)
    
    if test_df is not None:
        test_numeric_df = test_df[test_df['split'].apply(lambda x: str(x).isdigit())]
        if not test_numeric_df.empty:
            print(f"\nTest Summary by time:")
            basic_metrics = [col for col in test_numeric_df.columns if col in ['acc', 'f1', 'precision', 'recall']]
            if basic_metrics:
                test_summary = test_numeric_df.groupby('time')[basic_metrics].agg(['mean', 'std']).round(4)
                print(test_summary)
    
    return val_df, test_df

def extract_metrics_to_csv_new_format(base_dir, time_values, val_output_file="val_metrics_summary.csv", test_output_file="test_metrics_summary.csv"):
    """
    Extract validation and test metrics from JSON files with a different format.
    In this format, both val and test metrics are under the 'val' key, with 'val' and 'test' subkeys.
    
    Args:
        base_dir: Directory containing the JSON files
        time_values: List of time values to process
        val_output_file: Output CSV filename for validation metrics
        test_output_file: Output CSV filename for test metrics
    """
    
    def process_complex_value(value):
        """Helper function to process complex values"""
        if isinstance(value, dict):
            if 'real' in value and 'imag' in value:
                try:
                    real = float(value['real'])
                    imag = float(value['imag'])
                    return np.sqrt(real**2 + imag**2)
                except (ValueError, TypeError):
                    return None
            elif value == {}:  # Empty dict
                return None
        return value

    def process_special_metric(metric_value):
        """Process special metrics like class_mix_score and effective_rank"""
        if not isinstance(metric_value, dict):
            return metric_value
            
        result = {}
        for stage in ['before', 'after']:
            if stage in metric_value:
                stage_value = metric_value[stage]
                if isinstance(stage_value, dict):
                    for split_type in ['val', 'test']:
                        if split_type in stage_value:
                            split_value = stage_value[split_type]
                            if isinstance(split_value, dict) and 'real' in split_value and 'imag' in split_value:
                                result[f'{stage}'] = process_complex_value(split_value)
                            else:
                                result[f'{stage}'] = split_value
        
        if 'ratio' in metric_value:
            ratio_value = metric_value['ratio']
            if isinstance(ratio_value, dict):
                for split_type in ['val', 'test']:
                    if split_type in ratio_value:
                        split_value = ratio_value[split_type]
                        if isinstance(split_value, dict) and 'real' in split_value and 'imag' in split_value:
                            result['ratio'] = process_complex_value(split_value)
                        else:
                            result['ratio'] = split_value
        
        return result

    # Find all JSON files matching the pattern
    json_files = []
    for time_val in time_values:
        # pattern = os.path.join(base_dir, f"fold_metrics_nlayers{time_val}.json")
        pattern = os.path.join(base_dir, f"single_split_summary_time{time_val}.json")
        if os.path.exists(pattern):
            json_files.append((time_val, pattern))
    
    if not json_files:
        print(f"No JSON files found in {base_dir}")
        return None, None
    
    print(f"Found {len(json_files)} JSON files:")
    for time_val, filepath in json_files:
        print(f"  Time {time_val}: {filepath}")
    
    # Collect all data separately for val and test
    val_data = []
    test_data = []
    
    for time_val, filepath in sorted(json_files):
        print(f"\nProcessing time={time_val}...")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Store split data for this time value
            time_val_split_data = []
            time_test_split_data = []
            
            # Process each split
            for split_data in data:
                split_id = split_data['split']
                metrics = split_data['val']  # All metrics are under 'val'
                
                # Create validation row
                val_row = {
                    'time': time_val,
                    'split': split_id
                }
                
                # Create test row
                test_row = {
                    'time': time_val,
                    'split': split_id
                }
                
                # Process metrics for both val and test
                for metric_name, metric_value in metrics.items():
                    if metric_name == 'inference_time':
                        # Handle inference_time separately as it's a direct value
                        val_row[metric_name] = metric_value
                        test_row[metric_name] = metric_value
                    elif metric_name in ['class_mix_score', 'effective_rank']:
                        # Process special metrics
                        processed_metric = process_special_metric(metric_value)
                        if isinstance(processed_metric, dict):
                            for stage, value in processed_metric.items():
                                val_row[f'{metric_name}_{stage}'] = value
                                test_row[f'{metric_name}_{stage}'] = value
                    elif isinstance(metric_value, dict):
                        # Handle metrics with train/val/test subkeys
                        if 'val' in metric_value and 'test' in metric_value:
                            # Add validation metric
                            val_metric = metric_value['val']
                            if isinstance(val_metric, dict):
                                for key, value in val_metric.items():
                                    processed_value = process_complex_value(value)
                                    if processed_value is not None:
                                        val_row[f'{metric_name}_{key}'] = processed_value
                            else:
                                val_row[metric_name] = val_metric
                            
                            # Add test metric
                            test_metric = metric_value['test']
                            if isinstance(test_metric, dict):
                                for key, value in test_metric.items():
                                    processed_value = process_complex_value(value)
                                    if processed_value is not None:
                                        test_row[f'{metric_name}_{key}'] = processed_value
                            else:
                                test_row[metric_name] = test_metric
                        else:
                            # Handle direct complex numbers or empty dicts
                            processed_value = process_complex_value(metric_value)
                            if processed_value is not None:
                                val_row[metric_name] = processed_value
                                test_row[metric_name] = processed_value
                    else:
                        # Handle direct values
                        val_row[metric_name] = metric_value
                        test_row[metric_name] = metric_value
                
                val_data.append(val_row)
                test_data.append(test_row)
                time_val_split_data.append(val_row)
                time_test_split_data.append(test_row)
            
            # Calculate mean and std for validation metrics
            if time_val_split_data:
                val_columns = [col for col in time_val_split_data[0].keys() if col not in ['time', 'split']]
                
                val_mean_row = {'time': time_val, 'split': 'mean'}
                val_std_row = {'time': time_val, 'split': 'std'}
                
                for col in val_columns:
                    values = [safe_numeric_value(row[col]) for row in time_val_split_data if col in row and safe_numeric_value(row[col]) is not None]
                    if values:
                        val_mean_row[col] = np.mean(values)
                        val_std_row[col] = np.std(values)
                    else:
                        val_mean_row[col] = None
                        val_std_row[col] = None
                
                val_data.append(val_mean_row)
                val_data.append(val_std_row)
            
            # Calculate mean and std for test metrics
            if time_test_split_data:
                test_columns = [col for col in time_test_split_data[0].keys() if col not in ['time', 'split']]
                
                test_mean_row = {'time': time_val, 'split': 'mean'}
                test_std_row = {'time': time_val, 'split': 'std'}
                
                for col in test_columns:
                    values = [safe_numeric_value(row[col]) for row in time_test_split_data if col in row and safe_numeric_value(row[col]) is not None]
                    if values:
                        test_mean_row[col] = np.mean(values)
                        test_std_row[col] = np.std(values, ddof=1)
                    else:
                        test_mean_row[col] = None
                        test_std_row[col] = None
                
                test_data.append(test_mean_row)
                test_data.append(test_std_row)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Add empty rows for missing time values
    if val_data:
        # Get all possible columns from existing data
        all_columns = set()
        for row in val_data:
            all_columns.update(row.keys())
        
        # For each missing time value, add empty rows for each split and mean/std
        for time_val in time_values:
            if time_val not in [row['time'] for row in val_data]:
                # Add empty rows for each split
                for split_id in range(10):
                    empty_row = {'time': time_val, 'split': split_id}
                    for col in all_columns:
                        if col not in ['time', 'split']:
                            empty_row[col] = None
                    val_data.append(empty_row)
                    test_data.append(empty_row)
                
                # Add empty mean and std rows
                empty_mean_row = {'time': time_val, 'split': 'mean'}
                empty_std_row = {'time': time_val, 'split': 'std'}
                for col in all_columns:
                    if col not in ['time', 'split']:
                        empty_mean_row[col] = None
                        empty_std_row[col] = None
                val_data.append(empty_mean_row)
                val_data.append(empty_std_row)
                test_data.append(empty_mean_row)
                test_data.append(empty_std_row)
    
    if not val_data and not test_data:
        print("No data extracted!")
        return None, None
    
    # Process validation data
    val_df = None
    if val_data:
        val_df = pd.DataFrame(val_data)
        
        # Sort by time, then by split (but keep mean and std at the end for each time)
        def sort_key(row):
            time_val = row['time']
            split_val = row['split']
            if split_val == 'mean':
                return (time_val, 10)
            elif split_val == 'std':
                return (time_val, 11)
            else:
                return (time_val, split_val)
        
        val_df['sort_key'] = val_df.apply(sort_key, axis=1)
        val_df = val_df.sort_values('sort_key').drop('sort_key', axis=1)
        
        # Reorder columns: time, split, then all metrics
        val_metric_columns = [col for col in val_df.columns if col not in ['time', 'split']]
        val_column_order = ['time', 'split'] + sorted(val_metric_columns)
        val_df = val_df[val_column_order]
        
        # Save validation CSV
        val_output_path = os.path.join(base_dir, val_output_file)
        val_df['time'] = val_df['time'].astype(int)
        val_df.to_csv(val_output_path, index=False, float_format='%.6f')
        
        print(f"\nValidation data extracted successfully!")
        print(f"Val Shape: {val_df.shape}")
        print(f"Val Columns: {list(val_df.columns)}")
        print(f"Saved to: {val_output_path}")
    
    # Process test data
    test_df = None
    if test_data:
        test_df = pd.DataFrame(test_data)
        
        # Sort by time, then by split (but keep mean and std at the end for each time)
        test_df['sort_key'] = test_df.apply(sort_key, axis=1)
        test_df = test_df.sort_values('sort_key').drop('sort_key', axis=1)
        
        # Reorder columns: time, split, then all metrics
        test_metric_columns = [col for col in test_df.columns if col not in ['time', 'split']]
        test_column_order = ['time', 'split'] + sorted(test_metric_columns)
        test_df = test_df[test_column_order]
        
        # Save test CSV
        test_output_path = os.path.join(base_dir, test_output_file)
        test_df['time'] = test_df['time'].astype(int)
        test_df.to_csv(test_output_path, index=False, float_format='%.6f')
        
        print(f"\nTest data extracted successfully!")
        print(f"Test Shape: {test_df.shape}")
        print(f"Test Columns: {list(test_df.columns)}")
        print(f"Saved to: {test_output_path}")
    
    return val_df, test_df

def create_pivot_table(df, metric_type, output_file=None, base_dir=None):
    """
    Create a pivot table where each row is a time value and columns are split_metric combinations.
    
    Args:
        df: DataFrame with metrics
        metric_type: 'val' or 'test' to specify the type of metrics
        output_file: Output filename (if None, will be auto-generated)
        base_dir: Base directory for saving
    """
    if df is None or df.empty:
        print(f"No {metric_type} data to create pivot table")
        return None
        
    if base_dir is None:
        base_dir = os.path.dirname(output_file) if output_file and os.path.dirname(output_file) else '.'
    
    if output_file is None:
        output_file = f"{metric_type}_metrics_pivot.csv"
    
    # Filter out mean and std rows for pivot table (only use numeric splits)
    numeric_df = df[df['split'].apply(lambda x: str(x).isdigit())]
    
    # Get metric columns (exclude time and split)
    metric_columns = [col for col in numeric_df.columns if col not in ['time', 'split']]
    
    pivot_data = []
    
    for time_val in sorted(numeric_df['time'].unique()):
        time_data = numeric_df[numeric_df['time'] == time_val]
        
        row = {'time': time_val}
        
        # For each split and each metric, create a column
        for split_id in range(10):  # Assuming 10 splits (0-9)
            split_data = time_data[time_data['split'] == split_id]
            
            if not split_data.empty:
                for col in metric_columns:
                    column_name = f'split{split_id}_{col}'
                    row[column_name] = split_data[col].iloc[0]
            else:
                # Fill with NaN if split data is missing
                for col in metric_columns:
                    column_name = f'split{split_id}_{col}'
                    row[column_name] = None
        
        # Calculate mean and std for each metric across all splits
        for col in metric_columns:
            # Get values for this metric across all splits
            metric_values = []
            for split_id in range(10):
                column_name = f'split{split_id}_{col}'
                if column_name in row and row[column_name] is not None:
                    numeric_val = safe_numeric_value(row[column_name])
                    if numeric_val is not None:
                        metric_values.append(numeric_val)
            
            if metric_values:
                row[f'mean_{col}'] = np.mean(metric_values)
                row[f'std_{col}'] = np.std(metric_values, ddof=1)  # Sample std
            else:
                row[f'mean_{col}'] = None
                row[f'std_{col}'] = None
        
        pivot_data.append(row)
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Sort columns: time first, then split0_metric1, split0_metric2, ..., split1_metric1, ..., then mean_metric1, std_metric1, ...
    time_col = ['time']
    split_cols = [col for col in pivot_df.columns if col.startswith('split')]
    stats_cols = [col for col in pivot_df.columns if col.startswith('mean_') or col.startswith('std_')]
    
    # Sort split columns by split number and metric name
    split_cols.sort(key=lambda x: (int(x.split('_')[0].replace('split', '')), x.split('_', 1)[1]))
    
    # Sort stats columns by metric name and type (mean before std)
    stats_cols.sort(key=lambda x: (x.split('_', 1)[1], x.split('_')[0]))
    
    column_order = time_col + split_cols + stats_cols
    pivot_df = pivot_df[column_order]
    
    # Save pivot table
    pivot_output_path = os.path.join(base_dir, output_file)
    pivot_df.to_csv(pivot_output_path, index=False, float_format='%.6f')
    
    print(f"\n{metric_type.capitalize()} pivot table saved to: {pivot_output_path}")
    print(f"{metric_type.capitalize()} pivot shape: {pivot_df.shape}")
    
    return pivot_df

if __name__ == "__main__":
    # Set the base directory where the JSON files are located
    model = "BRICK"       
    datasets=["Squirrel"]
    time_values = [2, 4, 8, 16, 32, 64, 128]
    for dataset in datasets:
        # base_dir = f"/ram/USERS/jiaqi/Akorn/GE_bench/{model}/src/heterophilic_graphs/results/{dataset}"    
        base_dir = f"/ram/USERS/jiaqi/Akorn/akorn/results/{dataset}"    

        
        print("Extracting validation and test metrics from JSON files...")
        print(f"Base directory: {base_dir}")
        
        # val_df, test_df = extract_metrics_to_csv(base_dir, time_values, f"{model}_{dataset}_val_summary.csv", f"{model}_{dataset}_test_summary.csv")

        # Extract data to separate CSV files using the new format function
        val_df, test_df = extract_metrics_to_csv_new_format(base_dir, time_values, f"{model}_{dataset}_val_summary.csv", f"{model}_{dataset}_test_summary.csv")
        
        if val_df is not None:
            # Create validation pivot table
            print("\nCreating validation pivot table...")
            val_pivot_df = create_pivot_table(val_df, "val", f"{model}_{dataset}_val_pivot.csv", base_dir)
            
            print("\nFirst few rows of the validation summary table:")
            print(val_df.head(15))
        
        if test_df is not None:
            # Create test pivot table
            print("\nCreating test pivot table...")
            test_pivot_df = create_pivot_table(test_df, "test", f"{model}_{dataset}_test_pivot.csv", base_dir)
            
            print("\nFirst few rows of the test summary table:")
            print(test_df.head(15)) 