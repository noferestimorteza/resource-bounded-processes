import pandas as pd
import re

def extract_features(events_file, pid_tid_pairs):
    features = []
    
    # Read and process the large events file efficiently
    for chunk in pd.read_csv(events_file, sep='\t', chunksize=10000, encoding="utf-8"):
        for _, row in chunk.iterrows():
            # Determine where to get PID/TID based on event type
            if str(row['Event type']).startswith('syscall'):
                # For syscall events, extract from Stream Context
                pid_match = re.search(r'pid=([0-9]+)',row['Stream Context'])
                tid_match = re.search(r'tid=([0-9]+)',row['Stream Context'])
                pid = int(pid_match.group(1)) if pid_match else None
                tid = int(tid_match.group(1)) if tid_match else None
            else:
                # For non-syscall events, use the columns directly
                pid = row['PID'] if pd.notna(row['PID']) else None
                tid = row['TID'] if pd.notna(row['TID']) else None
            # Only proceed if we have valid PID/TID values
            if pid is not None and tid is not None:
                # Check if this PID/TID is in our labeled set
                # Most pandas-idiomatic way
                if not pid_tid_pairs[(pid_tid_pairs['PID'] == pid) & (pid_tid_pairs['TID'] == tid)].empty:
                    event_features = {
                    'PID': pid,
                    'TID': tid,
                    **{col: row[col] for col in row.index if col not in ['PID', 'TID']}  # Include all columns from the row
                    }
                    features.append(event_features)
    
    # Group by PID/TID and take first 5 events
    if features:  # Only proceed if we found any matching events
        df = pd.DataFrame(features)
        grouped = df.groupby(['PID', 'TID'])
        return grouped.head(5).reset_index(drop=True)
    return pd.DataFrame()  # Return empty DataFrame if no matches


# Example usage:
if __name__ == "__main__":
    # Load the PID/TID pairs from your labeled data
    labeled_df = pd.read_csv('PIDTID_labeled.csv')
    pid_tid_pairs = labeled_df[['PID', 'TID']].drop_duplicates()


    # Call the function with your test input
    result = extract_features('holedata.csv', pid_tid_pairs)
    
    # Print the results
    print("Extracted Features:")
    print("=" * 50)
    result.to_csv('PIDTID_features_top5.csv')