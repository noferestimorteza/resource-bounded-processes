import pandas as pd
import numpy as np

def parse_key_value_pairs(s):
    """Parse key=value pairs including handling arrays"""
    try:
        s = str(s).strip('[]')
        pairs = []
        current = ""
        in_array = False
        for char in s:
            if char == '[':
                in_array = True
                current += char
            elif char == ']':
                in_array = False
                current += char
            elif char == ',' and not in_array:
                pairs.append(current.strip())
                current = ""
            else:
                current += char
        if current:
            pairs.append(current.strip())
        
        result = {}
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                result[key.strip()] = value.strip()
        return result
    except Exception as e:
        print(f"Error parsing: {s}")
        return {}

def expand_nested_columns(df, column_names):
    """Expand nested columns into individual DataFrame columns"""
    for col in column_names:
        if col in df.columns:
            # Parse each row and convert to DataFrame
            expanded = df[col].apply(parse_key_value_pairs).apply(pd.Series)
            # Add prefix to column names
            expanded = expanded.add_prefix(f"{col}.")
            # Drop original column and join expanded ones
            df = df.drop(col, axis=1).join(expanded)
    return df

# Read CSV file
df = pd.read_csv('PIDTID_features_top5.csv')

nested_columns = ['Contents', 'Trace Packet Header', 'Packet Context', 'Stream Context']
for col in nested_columns:
    if col in df.columns:
        expanded = df[col].apply(parse_key_value_pairs).apply(pd.Series)
        expanded = expanded.add_prefix(f"{col}.")
        df = df.drop(col, axis=1).join(expanded)

# Group by PID and TID and create event numbering
grouped = df.groupby(['PID', 'TID'])
processed_data = []

for (pid, tid), group in grouped:
    # Sort by timestamp if needed
    group = group.sort_values('Timestamp')
    
    # Create new row with numbered events
    new_row = {'PID': pid, 'TID': tid}
    
    for i, (_, row) in enumerate(group.iterrows(), 1):
        for col in row.index:
            if col not in ['PID', 'TID']:
                new_row[f'event{i}.{col}'] = row[col]
    
    processed_data.append(new_row)

# Create final DataFrame
result_df = pd.DataFrame(processed_data)

# Reorder columns to group by event number
cols_ordered = ['PID', 'TID']
max_events = max([int(col.split('.')[0][5:]) for col in result_df.columns if col.startswith('event')])
for event_num in range(1, max_events + 1):
    cols_ordered.extend([col for col in result_df.columns if col.startswith(f'event{event_num}.')])

result_df = result_df[cols_ordered]

# Load the category labels CSV
labels_df = pd.read_csv('PIDTID_labeled.csv')  # Assuming format: PID, TID, Category

# Merge categories with your data
final_df = pd.merge(
    result_df,
    labels_df,
    on=['PID', 'TID'],
    how='left'  # Keep all rows from result_df even if no match in labels
)

# Fill missing categories
final_df['Category'] = final_df['Category'].fillna('Unknown')

# Save the final output
final_df.to_csv('final_events_with_categories.csv', index=False)
