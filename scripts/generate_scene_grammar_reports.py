"""
Scene Grammar Report Generator
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This script processes scene grammar data and segmentation reports, merging them to generate
comprehensive reports. It reads segmentation files and scene grammar information, then merges,
splits, and processes the data to produce a final CSV report.

Usage:
    Run the script with the desired dataset and trainset flag to generate the scene grammar report.
    For example:
        python this_script.py
        (By default, it calls main('adek20k'))
"""

import pandas as pd

def get_category(seg_file, scegram_info_file):
    """
    Extract and merge scene category information from segmentation and scene grammar info files.
    
    Parameters:
        seg_file (str): Path to the segmentation report CSV file.
        scegram_info_file (str): Path to the scene grammar info Excel file.
        
    Returns:
        pd.DataFrame: Merged DataFrame with scene category information.
    """
    seg_df = pd.read_csv(seg_file)
    scegram_info_df = pd.read_excel(scegram_info_file)

    # Extract scene info from the filename column in seg_df (e.g., "Scene1")
    seg_df['scene_info'] = seg_df['filename'].str.extract(r'(Scene\d+)')

    # Remove '.png' from the sce_file_name and extract scene info in scegram_info_df
    scegram_info_df['scene_info'] = (
        scegram_info_df['sce_file_name']
        .str.replace('.png', '', regex=False)
        .str.extract(r'(Scene\d+)')
    )

    # Merge on the extracted 'scene_info'
    df_merged = pd.merge(seg_df, scegram_info_df[['scene_info', 'sce_category_lower']], on='scene_info', how='left')
    
    # Rename column to more meaningful 'scene_category' and drop redundant columns
    df_merged.rename(columns={'sce_category_lower': 'scene_category'}, inplace=True)
    df_merged = df_merged.drop(columns=['scene_info']).drop_duplicates()
    
    # Sort for consistent ordering
    df_merged = df_merged.sort_values(by=['scene_id', 'consistency'])
    print(df_merged.head(20))
    return df_merged

def split_objects(segmentation_df):
    """
    Split comma-separated object names in the 'object_name' column into individual rows.
    
    Parameters:
        segmentation_df (pd.DataFrame): DataFrame containing a column 'object_name' with multiple names.
        
    Returns:
        pd.DataFrame: Expanded DataFrame with each object name in a separate row.
    """
    expanded_rows = [
        {**row, 'object_name': name} 
        for _, row in segmentation_df.iterrows() 
        for name in row['object_name'].split(', ')
    ]
    segmentation_df_expanded = pd.DataFrame(expanded_rows)
    return segmentation_df_expanded

def add_obj_id_scegram(df):
    """
    Add an object ID to each object based on scene_id and consistency.
    
    The function sorts the DataFrame by 'scene_id' and 'consistency_scegram' and then assigns
    incremental IDs to objects within each scene.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame to update.
        
    Returns:
        pd.DataFrame: DataFrame with a new 'object_id' column.
    """
    df = df.sort_values(by=['scene_id', 'consistency_scegram'])
    df['object_id'] = -1  # Initialize with -1
    
    # Loop through each scene and assign incremental IDs
    for scene_id, group in df.groupby('scene_id'):
        obj_id_counter = 0  # Reset counter for each scene
        for i, row in group.iterrows():
            # Assign new object ID based on consistency condition ('ABS' gets new ID)
            condition = row['consistency_scegram']
            if condition == 'ABS':
                df.at[i, 'object_id'] = obj_id_counter
                obj_id_counter += 1
            else:
                df.at[i, 'object_id'] = obj_id_counter
    # Sort DataFrame back by scene_id and object_id, then reset index
    df = df.sort_values(by=['scene_id', 'object_id']).reset_index(drop=True)
    return df

def merge_scene_grammar_scores(object_pair_df, segmentation_df, df_anch, df_diag, dataset, filter=False, trainset=True):
    """
    Merge scene grammar scores with segmentation data and save the final report to a CSV file.
    
    Parameters:
        object_pair_df (str): Path to CSV file containing scene grammar object pairs.
        segmentation_df (pd.DataFrame): Segmentation DataFrame (possibly from get_category).
        df_anch (str): Path to CSV file containing anchor scores.
        df_diag (str): Path to CSV file containing diagnosticity scores.
        dataset (str): Name of the dataset ('adek20k' or 'scegram').
        filter (bool, optional): Whether to filter segmentation data based on shared objects.
        trainset (bool, optional): Flag indicating whether to process training set data.
    """
    object_pair_df = pd.read_csv(object_pair_df)
    df_anch = pd.read_csv(df_anch)
    df_diag = pd.read_csv(df_diag)

    unique_objects = pd.concat([object_pair_df['obj_a'], object_pair_df['obj_b']]).unique()
    print(f"Unique objects in scene grammar objects: {len(unique_objects)}")

    # For ADEK20K, split objects; otherwise, use the segmentation DataFrame as is.
    if dataset == "adek20k":
        segmentation_df_expanded = split_objects(segmentation_df)
    else:
        segmentation_df_expanded = segmentation_df

    segmentation_df_expanded['scene_category'] = segmentation_df_expanded['scene_category'].str.replace('_', '', regex=False)
    
    # Determine shared objects between segmentation and scene grammar data
    unique_objects_df2 = segmentation_df_expanded['object_name'].unique()
    shared_objects = set(unique_objects).intersection(unique_objects_df2)
    print(f"Shared unique objects: {len(shared_objects)}")

    if filter:
        segmentation_df_expanded_filtered = segmentation_df_expanded[segmentation_df_expanded['object_name'].isin(shared_objects)]
        print(f'Len before filtering: {len(segmentation_df_expanded)}, len after filtering: {len(segmentation_df_expanded_filtered)}')
        segmentation_df_expanded_filtered.to_csv(f'results/segmentation_reports/{dataset.capitalize()}/consolidated_segmentation_report_filtered.csv', index=False)
        segmentation_df_expanded = segmentation_df_expanded_filtered

    # Rename columns and clean whitespace in anchor and diagnosticity data
    df_anch = df_anch.rename(columns={'sceneCat': 'scene_category', 'objName': 'object_name'}).replace({' ': ''}, regex=True)
    df_diag = df_diag.rename(columns={'sceneCat': 'scene_category', 'objName': 'object_name'}).replace({' ': ''}, regex=True)

    # Merge segmentation data with both score DataFrames
    merged = pd.merge(segmentation_df_expanded, df_anch, on=['scene_category', 'object_name'], how='left')
    merged = pd.merge(merged, df_diag, on=['scene_category', 'object_name'], how='left')
    print(merged.head(20))
    merged.drop_duplicates(inplace=True)

    # For ADEK20K, assign scene and object IDs; for SCEGRAM, rename columns and filter scenes
    if dataset == 'adek20k':
        merged['scene_id'] = merged.groupby('filename').ngroup()
        merged['object_id'] = merged.groupby('scene_id').cumcount()
    elif dataset == "scegram":
        merged = merged.rename(columns={'consistency_x': 'consistency_scegram', 'consistency_y': 'consistency_score'})
        merged = add_obj_id_scegram(merged)
        # Drop extra scene_category columns if they exist
        for col in merged.columns:
            if 'scene_category.' in col:
                merged = merged.drop(col, axis=1)
        # Remove specific scene IDs (11 and 56) due to data issues
        merged = merged[merged['scene_id'] != 11]
        merged = merged[merged['scene_id'] != 56]

    # Save the final merged report to CSV, with different filenames if filtering is applied
    if filter:
        merged.to_csv(f'results/scene_grammar_reports/{dataset.capitalize()}/segmentation_report_with_scores_filtered-trainset_{trainset}.csv', index=False)
    else:
        merged.to_csv(f'results/scene_grammar_reports/{dataset.capitalize()}/segmentation_report_with_scores-trainset_{trainset}.csv', index=False)

def main(dataset, trainset=False):
    """
    Main function to process scene grammar and segmentation data.
    
    Parameters:
        dataset (str): Dataset to process ('scegram' or other).
        trainset (bool, optional): Whether to process training set data.
    """
    if dataset == "scegram":
        seg_file = 'results/segmentation_reports/SCEGRAM/consolidated_segmentation_report.csv'
        scegram_info_file = '../SCEGRAM_info.xlsx'
        segmentation_df = get_category(seg_file, scegram_info_file)
    else:
        segmentation_df = pd.read_csv(f'results/segmentation_reports/{dataset.capitalize()}/consolidated_segmentation_report-trainset_{trainset}.csv')
    
    object_pair_df = 'results/scene_grammar_reports/obj_pairs_data.csv'
    df_anch = 'results/scene_grammar_reports/obj_scene_anch_data.csv'
    df_diag = 'results/scene_grammar_reports/object_in_scene_data.csv'
    
    merge_scene_grammar_scores(object_pair_df, segmentation_df, df_anch, df_diag, dataset, filter=False, trainset=trainset)

if __name__ == "__main__":
    main('adek20k')
