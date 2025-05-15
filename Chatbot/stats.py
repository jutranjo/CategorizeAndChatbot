def handle_single_category(df, category, whole_df, z_threshold=2.0):
    """
    Detect and prints spikes in category activity using z-score analysis.

    Args:
        df (pd.DataFrame): Filtered DataFrame for the category and time window.
        category (str): The unique category.
        whole_df (pd.DataFrame): The entire dataset (unfiltered).
        z_threshold (float): Threshold for z-score spike detection.
    """
    print(f"Analyzing category: {category}")

    # 1. Full time series from all data
    full_cat_df = whole_df[whole_df['category'] == category].copy()
    full_cat_df['day'] = full_cat_df['timestamp'].dt.floor('D')
    full_daily_counts = full_cat_df.groupby('day').size()

    # Compute global mean and std
    mean = full_daily_counts.mean()
    std = full_daily_counts.std()

    print(f"Global stats — Mean: {mean:.2f}, Std Dev: {std:.2f}")

    # 2. Daily counts in filtered time frame
    filtered_df = df.copy()
    filtered_df['day'] = filtered_df['timestamp'].dt.floor('D')
    filtered_counts = filtered_df.groupby('day').size()

    # 3. Compute z-scores
    z_scores = (filtered_counts - mean) / std
    spike_days = z_scores[abs(z_scores) >= z_threshold]

    # Output results
    if not spike_days.empty:
        print(f"\nSpike Detection (z ≥ {z_threshold}):")
        for date, z in spike_days.items():
            print(f"- {date.date()}: z = {z:.2f}, count = {filtered_counts[date]}")
    else:
        print("No significant spikes detected.")

def describe_filtered_data(filtered_df, entire_df):
    """
    Prints the number of messages and unique users in the filtered DataFrame.

    Args:
        df (pd.DataFrame): A filtered DataFrame containing at least 'id_user' and 'message' columns.

    Returns:
        tuple: (number of messages, number of unique users)
    """
    num_messages = len(filtered_df)
    num_users = filtered_df['id_user'].nunique()
    
    print(f"Summary:")
    if 'timestamp' in filtered_df.columns:
        start_time = filtered_df['timestamp'].min()
        end_time = filtered_df['timestamp'].max()

        fmt = "%b %d, %Y at %H:%M"
        print(f"- Time range:     {start_time.strftime(fmt)} → {end_time.strftime(fmt)}")


    print(f"- Total messages: {num_messages}")
    print(f"- Unique users:   {num_users}")
    
    unique_categories = filtered_df['category'].dropna().unique()
    if len(unique_categories) == 1:
        category = unique_categories[0]
        handle_single_category(filtered_df, category, entire_df) 

    print("\nFirst few entries:")
    preview = filtered_df[['timestamp', 'id_user', 'source', 'category', 'message']].head(10)
    print(preview.to_string(index=False))
