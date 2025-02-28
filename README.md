# Bilateral Tumor Model Grouping

## Overview
This tool helps researchers create balanced groups of mice for bilateral tumor studies. It uses mathematical optimization to make groups with similar tumor volumes on both sides (left and right), ensuring fair experimental conditions.

## Features

Creates balanced groups with similar total tumor volumes for bilateral (left and right) tumors
Automatically removes outliers if needed
Considers both left and right tumor volumes and their ratio
Generates detailed statistics for each group
Customizable number of groups

## Requirements

Python 3.6 or higher
Required packages: pandas, numpy, pulp, argparse

## Installation

```
# Clone this repository
git clone https://github.com/yourusername/bilateral-tumor-model-grouping.git
cd bilateral-tumor-model-grouping

# Install required packages
pip install pandas numpy pulp

```
## Input Data Format
The tool requires a CSV file with at least the following columns:
- mouse: Mouse ID
- tumor_volume_left: Left tumor volume
- tumor_volume_right: Right tumor volume

Example:
```
mouse,tumor_volume_left,tumor_volume_right
1,120.5,95.2
2,145.3,110.7
...
```
## Usage
Basic usage:
```
python bilateral_model_grouping.py your_data.csv
```
Advanced options:
```
python bilateral_model_grouping.py your_data.csv --num_groups 5 --verbose --max_time 600
```
## Parameters

- file_path: Path to your input CSV file
- --num_groups: Number of groups to create (default: 4)
- --verbose: Show detailed output and statistics
- --max_time: Maximum solver runtime in seconds (default: 300)

## How It Works

1. Data Preprocessing: Removes outliers if needed to ensure equal group sizes
2. Parameter Analysis: Calculates optimal weights based on data variation
3. Standardization: Standardizes tumor volumes for fair comparison
4. Optimization: Uses mathematical optimization to find the best grouping that balances both left and right tumors
5. Output: Creates a CSV file with group assignments and statistics

## Output Files

- *_grouped.csv: Original data with group assignments
- *_verbose_output.txt: Detailed statistics and calculation process (when using --verbose)

## Example
```
python tumor_grouping.py sample_data.csv --verbose
```
This will:

1. Read bilateral tumor data from sample_data.csv
2. Create optimal groups with balanced tumor volumes on both sides
3. Save results to sample_data_grouped.csv
4. Generate detailed statistics in sample_data_verbose_output.txt
