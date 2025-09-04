"""
boundary共存率のチェック
"""
import pandas as pd
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def check_boundary_coexistence():
    """boundary共存率をチェック"""
    events_df = pd.read_csv(project_root / 'subject3/data/processed/events_labeled.csv')
    
    # policy_boundaryがある(town,year)を特定
    boundary_keys = set()
    for _, row in events_df.iterrows():
        if row['event_type'] == 'policy_boundary':
            boundary_keys.add((row['town'], row['year']))
    
    print(f"Total policy_boundary events: {len(boundary_keys)}")
    
    # 各boundaryキーで他のカテゴリが存在するかチェック
    coexistence_count = 0
    coexistence_examples = []
    
    for key in boundary_keys:
        town, year = key
        town_events = events_df[
            (events_df['town'] == town) & 
            (events_df['year'] == year)
        ]
        
        # policy_boundary以外のイベントがあるかチェック
        other_events = town_events[town_events['event_type'] != 'policy_boundary']
        if len(other_events) > 0:
            coexistence_count += 1
            coexistence_examples.append({
                'town': town,
                'year': year,
                'boundary_events': len(town_events[town_events['event_type'] == 'policy_boundary']),
                'other_events': len(other_events),
                'other_types': list(other_events['event_type'].unique())
            })
    
    print(f"Boundary coexistence rate: {coexistence_count}/{len(boundary_keys)} = {coexistence_count/len(boundary_keys):.1%}")
    
    if coexistence_examples:
        print("\nExamples of boundary coexistence:")
        for i, example in enumerate(coexistence_examples[:10]):  # 最初の10例
            print(f"  {i+1}. {example['town']} {example['year']}: {example['other_types']}")
    
    return coexistence_count, len(boundary_keys), coexistence_examples

def check_unknown_directions():
    """unknown directionの件数をチェック"""
    events_df = pd.read_csv(project_root / 'subject3/data/processed/events_labeled.csv')
    
    unknown_count = len(events_df[events_df['effect_direction'] == 'unknown'])
    print(f"\nUnknown direction events: {unknown_count}")
    
    if unknown_count > 0:
        print("Examples of unknown direction events:")
        unknown_events = events_df[events_df['effect_direction'] == 'unknown']
        for _, row in unknown_events.head(10).iterrows():
            print(f"  {row['town']} {row['year']}: {row['event_type']} - {row.get('raw_cause_text', 'N/A')}")

if __name__ == "__main__":
    print("Boundary Coexistence Check")
    print("=" * 50)
    
    coexistence_count, total_boundary, examples = check_boundary_coexistence()
    check_unknown_directions()
    
    print(f"\nSummary:")
    print(f"- Boundary coexistence: {coexistence_count}/{total_boundary} ({coexistence_count/total_boundary:.1%})")
    print(f"- Unknown directions: {len([e for e in examples if 'unknown' in str(e)])}")
