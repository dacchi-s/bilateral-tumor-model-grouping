import pandas as pd
import numpy as np
import argparse
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpStatus, value, PULP_CBC_CMD

def preprocess_data(data, num_groups):
    """
    Preprocess data to ensure it can be divided equally into groups
    Remove samples with high outlier scores
    """
    N = len(data)
    target_size = (N // num_groups) * num_groups  # Target sample size
    
    if N == target_size:
        return data, []  # No removal needed
    
    # Calculate outlier scores
    def calculate_outlier_scores(df):
        scores = []
        for idx, row in df.iterrows():
            # Calculate Z-scores
            z_left = abs((row['tumor_volume_left'] - df['tumor_volume_left'].mean()) / df['tumor_volume_left'].std())
            z_right = abs((row['tumor_volume_right'] - df['tumor_volume_right'].mean()) / df['tumor_volume_right'].std())
            ratio = row['tumor_volume_left'] / row['tumor_volume_right']
            z_ratio = abs((ratio - (df['tumor_volume_left'] / df['tumor_volume_right']).mean()) / 
                         (df['tumor_volume_left'] / df['tumor_volume_right']).std())
            
            # Total score (average of 3 Z-scores)
            total_score = (z_left + z_right + z_ratio) / 3
            scores.append({'mouse': row['mouse'], 'score': total_score})
        
        return pd.DataFrame(scores)
    
    # Calculate outlier scores
    outlier_scores = calculate_outlier_scores(data)
    outlier_scores = outlier_scores.sort_values('score', ascending=False)
    
    # Calculate number of samples to remove
    num_to_remove = N - target_size
    
    # Remove necessary number of mice with highest scores
    mice_to_remove = outlier_scores.head(num_to_remove)['mouse'].tolist()
    processed_data = data[~data['mouse'].isin(mice_to_remove)].copy()
    
    return processed_data, mice_to_remove

def analyze_data(data):
    """Analyze data and estimate optimal parameters"""
    # Calculate basic statistics
    stats = {
        'left': {
            'mean': data['tumor_volume_left'].mean(),
            'std': data['tumor_volume_left'].std(),
            'cv': data['tumor_volume_left'].std() / data['tumor_volume_left'].mean()
        },
        'right': {
            'mean': data['tumor_volume_right'].mean(),
            'std': data['tumor_volume_right'].std(),
            'cv': data['tumor_volume_right'].std() / data['tumor_volume_right'].mean()
        }
    }
    
    # Calculate weights
    total_cv = stats['left']['cv'] + stats['right']['cv']
    weights = {
        'w_L': stats['left']['cv'] / total_cv,
        'w_R': stats['right']['cv'] / total_cv,
        'w_Ratio': 1.0  # Weight for ratio is fixed at 1.0
    }
    
    # Estimate convergence condition from data variation
    gap_rel = min(0.1, max(0.01, (stats['left']['cv'] + stats['right']['cv']) / 4))
    
    return stats, weights, gap_rel

def standardize_data(data):
    """Standardize data"""
    result = data.copy()
    for col in ['tumor_volume_left', 'tumor_volume_right']:
        mean = data[col].mean()
        std = data[col].std()
        result[col] = (data[col] - mean) / std
    return result

def main(file_path, num_groups, verbose, max_time=300):
    # Add verbose_output list
    verbose_output = []
    
    # Load data
    data = pd.read_csv(file_path)
    
    # 1. Data preprocessing (adjustment for equal group sizes)
    processed_data, removed_mice = preprocess_data(data, num_groups)
    
    if verbose:
        output_text = f"Data preprocessing results:\n"
        output_text += f"Original data count: {len(data)}\n"
        output_text += f"Processed data count: {len(processed_data)}\n"
        if removed_mice:
            output_text += f"Excluded mice: {', '.join(map(str, removed_mice))}\n"
        verbose_output.append(output_text)
        print(output_text)
    
    # 2. Parameter optimization
    stats, weights, gap_rel = analyze_data(processed_data)
    
    if verbose:
        output_text = f"Data analysis results:\n"
        output_text += f"Weights - Left: {weights['w_L']:.3f}, Right: {weights['w_R']:.3f}\n"
        output_text += f"Estimated convergence condition: {gap_rel:.3f}\n"
        verbose_output.append(output_text)
        print(output_text)
    
    # 3. Data standardization
    std_data = standardize_data(processed_data)
    
    # 4. Prepare required data
    mice = processed_data['mouse'].tolist()
    L_i = std_data['tumor_volume_left'].tolist()
    R_i = std_data['tumor_volume_right'].tolist()
    Ratio_i = (processed_data['tumor_volume_left'] / processed_data['tumor_volume_right']).tolist()
    N = len(mice)
    G = num_groups
    
    # Calculate totals and means
    Total_L = sum(L_i)
    Total_R = sum(R_i)
    Total_Ratio = sum(Ratio_i)
    Mean_L = Total_L / G
    Mean_R = Total_R / G
    Mean_Ratio = Total_Ratio / G
    
    # Problem definition
    prob = LpProblem("Tumor_Volume_Grouping", LpMinimize)
    
    # Decision variables
    x = LpVariable.dicts("x", [(i, g) for i in range(N) for g in range(G)], cat=LpBinary)
    dev_L = LpVariable.dicts("dev_L", [g for g in range(G)], lowBound=0)
    dev_R = LpVariable.dicts("dev_R", [g for g in range(G)], lowBound=0)
    dev_Ratio = LpVariable.dicts("dev_Ratio", [g for g in range(G)], lowBound=0)
    
    # Group size variables
    group_size = LpVariable.dicts("group_size", [g for g in range(G)], lowBound=0)
    
    # Constraints
    # 1. Each mouse is assigned to only one group
    for i in range(N):
        prob += lpSum(x[(i, g)] for g in range(G)) == 1
    
    # 2. Group size constraints
    min_size = N // G  # Minimum group size
    max_size = -(-N // G)  # Maximum group size (ceiling division)
    
    for g in range(G):
        prob += lpSum(x[(i, g)] for i in range(N)) >= min_size
        prob += lpSum(x[(i, g)] for i in range(N)) <= max_size
        
        # Record group size
        prob += group_size[g] == lpSum(x[(i, g)] for i in range(N))
        
        # Calculate deviations
        L_g = lpSum(x[(i, g)] * L_i[i] for i in range(N))
        prob += L_g - Mean_L <= dev_L[g]
        prob += Mean_L - L_g <= dev_L[g]
        
        R_g = lpSum(x[(i, g)] * R_i[i] for i in range(N))
        prob += R_g - Mean_R <= dev_R[g]
        prob += Mean_R - R_g <= dev_R[g]
        
        Ratio_g = lpSum(x[(i, g)] * Ratio_i[i] for i in range(N))
        prob += Ratio_g - Mean_Ratio <= dev_Ratio[g]
        prob += Mean_Ratio - Ratio_g <= dev_Ratio[g]
    
    # Objective function
    prob += (weights['w_L'] * lpSum(dev_L[g] for g in range(G)) +
            weights['w_R'] * lpSum(dev_R[g] for g in range(G)) +
            weights['w_Ratio'] * lpSum(dev_Ratio[g] for g in range(G)))
    
    # Solve the problem
    if verbose:
        verbose_output.append("Running optimization...\n")
        print("Running optimization...")
    
    solver = PULP_CBC_CMD(options=[f'sec {max_time}', f'ratio {gap_rel}'])
    prob.solve(solver)
    
    # Extract and save results
    group_assignments = {}
    for i in range(N):
        for g in range(G):
            if x[(i, g)].varValue == 1:
                group_assignments[mice[i]] = g + 1
                break
    
    # Add results to dataframe
    processed_data['Group'] = processed_data['mouse'].map(group_assignments)
    
    # Add information about excluded mice
    if removed_mice:
        removed_data = data[data['mouse'].isin(removed_mice)].copy()
        removed_data['Group'] = 'Excluded'
        final_data = pd.concat([processed_data, removed_data])
    else:
        final_data = processed_data
    
    # Save results
    output_file = file_path.replace('.csv', '_grouped.csv')
    final_data.to_csv(output_file, index=False)
    
    # Summarize results by group (excluding excluded group)
    grouped_data = processed_data.groupby('Group')
    all_stats = []
    
    for group, df in grouped_data:
        mice_in_group = df['mouse'].tolist()
        tumor_volumes_left = df['tumor_volume_left'].tolist()
        tumor_volumes_right = df['tumor_volume_right'].tolist()
        ratios = df['tumor_volume_left'] / df['tumor_volume_right']
        sum_left = df['tumor_volume_left'].sum()
        sum_right = df['tumor_volume_right'].sum()
        sum_ratio = ratios.sum()
        avg_left = df['tumor_volume_left'].mean()
        avg_right = df['tumor_volume_right'].mean()
        avg_ratio = ratios.mean()
        std_left = df['tumor_volume_left'].std()
        std_right = df['tumor_volume_right'].std()
        std_ratio = ratios.std()
        
        stats = {
            'Group': int(group),
            'Sum Left': sum_left,
            'Sum Right': sum_right,
            'Sum Ratio': sum_ratio,
            'Deviation Left': abs(sum_left - Total_L/G),
            'Deviation Right': abs(sum_right - Total_R/G),
            'Deviation Ratio': abs(sum_ratio - Total_Ratio/G)
        }
        all_stats.append(stats)
    
    if verbose:
        # Convert statistics to dataframe
        stats_df = pd.DataFrame(all_stats)
        
        output_text = f"\nResults saved to {output_file}\n"
        output_text += f"Optimization status: {LpStatus[prob.status]}\n"
        output_text += f"Objective function value: {value(prob.objective):.4f}\n\n"
        
        output_text += "Group statistics:\n"
        output_text += stats_df.to_string(index=False) + "\n\n"
        
        # Detailed statistics by group
        for g in range(G):
            group_data = processed_data[processed_data['Group'] == g + 1]
            output_text += f"\nGroup {g + 1} detailed statistics:\n"
            output_text += f"Size: {len(group_data)}\n"
            output_text += f"Mouse IDs: {', '.join(map(str, group_data['mouse'].tolist()))}\n"
            output_text += f"Left tumors - Mean: {group_data['tumor_volume_left'].mean():.2f}, "
            output_text += f"Std dev: {group_data['tumor_volume_left'].std():.2f}\n"
            output_text += f"Right tumors - Mean: {group_data['tumor_volume_right'].mean():.2f}, "
            output_text += f"Std dev: {group_data['tumor_volume_right'].std():.2f}\n"
            ratios = group_data['tumor_volume_left'] / group_data['tumor_volume_right']
            output_text += f"Left/Right ratio - Mean: {ratios.mean():.2f}, "
            output_text += f"Std dev: {ratios.std():.2f}\n"
        
        # Excluded group statistics (if any)
        if removed_mice:
            excluded_data = final_data[final_data['Group'] == 'Excluded']
            output_text += f"\nExcluded group statistics:\n"
            output_text += f"Size: {len(excluded_data)}\n"
            output_text += f"Mouse IDs: {', '.join(map(str, excluded_data['mouse'].tolist()))}\n"
            output_text += f"Left tumors - Mean: {excluded_data['tumor_volume_left'].mean():.2f}, "
            output_text += f"Std dev: {excluded_data['tumor_volume_left'].std():.2f}\n"
            output_text += f"Right tumors - Mean: {excluded_data['tumor_volume_right'].mean():.2f}, "
            output_text += f"Std dev: {excluded_data['tumor_volume_right'].std():.2f}\n"
            ratios = excluded_data['tumor_volume_left'] / excluded_data['tumor_volume_right']
            output_text += f"Left/Right ratio - Mean: {ratios.mean():.2f}, "
            output_text += f"Std dev: {ratios.std():.2f}\n"
        
        verbose_output.append(output_text)
        print(output_text)
        
        # Save detailed log to text file
        verbose_file_path = file_path.replace('.csv', '_verbose_output.txt')
        with open(verbose_file_path, 'w', encoding='utf-8') as f:
            for line in verbose_output:
                f.write(line + '\n')
        print(f"Detailed calculation process saved to {verbose_file_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized mouse grouping')
    parser.add_argument('file_path', type=str, help='Path to input CSV file')
    parser.add_argument('--num_groups', type=int, default=4, help='Number of groups (default: 4)')
    parser.add_argument('--verbose', action='store_true', help='Display detailed output')
    parser.add_argument('--max_time', type=int, default=300, help='Maximum execution time (seconds)')
    args = parser.parse_args()
    
    main(args.file_path, args.num_groups, args.verbose, args.max_time)
