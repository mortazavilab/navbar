import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_well_id(well_id):
    if not isinstance(well_id, str) or not re.match(r"^[A-H]([1-9]|1[0-2])$", well_id, re.IGNORECASE): 
        return None, None
    row_char = well_id[0].upper()
    col_str = well_id[1:]
    row_idx = ord(row_char) - ord('A')
    col_idx = int(col_str) - 1
    return row_idx, col_idx

def natural_sort_key(s):
    match = re.search(r'bc(\d+)_well', s)
    return int(match.group(1)) if match else 999999

def create_heatmap_grid(well_counts):
    heatmap_data = pd.DataFrame(np.nan, index=list('ABCDEFGH'), columns=range(1, 13))
    for well_id, count in well_counts.items():
        row_idx, col_idx = parse_well_id(well_id)
        if row_idx is not None and col_idx is not None:
            row_char = chr(ord('A') + row_idx)
            col_num = col_idx + 1
            heatmap_data.loc[row_char, col_num] = count
    return heatmap_data

def create_plate_heatmap_fig(title, label, well_counts):
    heatmap_data = create_heatmap_grid(well_counts)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    mask = heatmap_data.isnull()
    sns.heatmap(
        heatmap_data, annot=True, fmt=".0f", cmap="Reds", mask=mask,
        linewidths=0, linecolor='none', cbar=True, cbar_kws={'label': label},
        square=False, ax=ax
    )
    ax.set_title(title, pad=30)
    ax.set_xlabel("Plate Column")
    ax.set_ylabel("Plate Row")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis='both', which='major', length=0)
    ax.grid(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig