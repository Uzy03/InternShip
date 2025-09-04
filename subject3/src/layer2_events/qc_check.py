"""
イベントラベリングのQCチェック
- Consistency: 大幅減かつpolicy_boundaryがincreaseだけ → 警告
- Direction drift: disasterでincreaseのみ & 同年Δが大幅減 → 警告  
- Transit timing: transit_increaseがt1>tになっているか
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def load_data():
    """データを読み込み"""
    events_df = pd.read_csv(project_root / 'subject3/data/processed/events_labeled.csv')
    panel_df = pd.read_csv(project_root / 'subject3/data/processed/panel_raw.csv')
    major_changes_df = pd.read_csv(project_root / 'subject3/data/processed/major_population_changes.csv')
    
    return events_df, panel_df, major_changes_df

def check_consistency(events_df, major_changes_df):
    """Consistencyチェック"""
    print("=== Consistency Check ===")
    
    # 大幅減（|Δ|≥30%）のケースを特定
    major_decreases = major_changes_df[
        major_changes_df['growth_pct'].abs() >= 0.3
    ].copy()
    
    warnings = []
    for _, change_row in major_decreases.iterrows():
        town = change_row['town']
        year = change_row['year']
        change_rate = change_row['growth_pct']
        
        # 該当年度のイベントをチェック
        town_events = events_df[
            (events_df['town'] == town) & 
            (events_df['year'] == year)
        ]
        
        # policy_boundaryがincreaseのみの場合
        boundary_events = town_events[town_events['event_type'] == 'policy_boundary']
        if len(boundary_events) > 0 and all(boundary_events['effect_direction'] == 'increase'):
            warnings.append({
                'town': town,
                'year': year,
                'change_rate': change_rate,
                'issue': 'Major decrease with policy_boundary increase only'
            })
    
    if warnings:
        print(f"⚠️  {len(warnings)} consistency warnings found:")
        for w in warnings:
            print(f"  {w['town']} {w['year']}: {w['change_rate']:.1%} change, {w['issue']}")
    else:
        print("✅ No consistency issues found")
    
    return warnings

def check_direction_drift(events_df, major_changes_df):
    """Direction driftチェック"""
    print("\n=== Direction Drift Check ===")
    
    warnings = []
    for _, change_row in major_changes_df.iterrows():
        town = change_row['town']
        year = change_row['year']
        change_rate = change_row['growth_pct']
        
        # 該当年度のdisasterイベントをチェック
        disaster_events = events_df[
            (events_df['town'] == town) & 
            (events_df['year'] == year) &
            (events_df['event_type'] == 'disaster')
        ]
        
        # disasterがincreaseのみ & 同年Δが大幅減
        if (len(disaster_events) > 0 and 
            all(disaster_events['effect_direction'] == 'increase') and
            change_rate < -0.3):
            warnings.append({
                'town': town,
                'year': year,
                'change_rate': change_rate,
                'issue': 'Disaster increase with major population decrease'
            })
    
    if warnings:
        print(f"⚠️  {len(warnings)} direction drift warnings found:")
        for w in warnings:
            print(f"  {w['town']} {w['year']}: {w['change_rate']:.1%} change, {w['issue']}")
    else:
        print("✅ No direction drift issues found")
    
    return warnings

def check_transit_timing(events_df):
    """Transit timingチェック"""
    print("\n=== Transit Timing Check ===")
    
    transit_events = events_df[events_df['event_type'] == 'transit']
    
    # transit_increaseのラグチェック
    increase_events = transit_events[transit_events['effect_direction'] == 'increase']
    
    timing_issues = []
    for _, event in increase_events.iterrows():
        # 開通系はt1>tが基本
        if event['lag_t1'] == 0 and event['lag_t'] == 1:
            timing_issues.append({
                'town': event['town'],
                'year': event['year'],
                'issue': 'Transit increase with t=1, t1=0 (should be t1=1)'
            })
    
    if timing_issues:
        print(f"⚠️  {len(timing_issues)} transit timing issues found:")
        for issue in timing_issues:
            print(f"  {issue['town']} {issue['year']}: {issue['issue']}")
    else:
        print("✅ No transit timing issues found")
    
    return timing_issues

def check_boundary_override(events_df):
    """Boundary overrideチェック"""
    print("\n=== Boundary Override Check ===")
    
    # (town, year)でboundaryが存在する場合、他のカテゴリが存在しないかチェック
    boundary_keys = set()
    for _, row in events_df.iterrows():
        if row['event_type'] == 'policy_boundary':
            boundary_keys.add((row['town'], row['year']))
    
    override_issues = []
    for _, row in events_df.iterrows():
        key = (row['town'], row['year'])
        if key in boundary_keys and row['event_type'] != 'policy_boundary':
            override_issues.append({
                'town': row['town'],
                'year': row['year'],
                'event_type': row['event_type'],
                'issue': 'Event not overridden by policy_boundary'
            })
    
    if override_issues:
        print(f"⚠️  {len(override_issues)} boundary override issues found:")
        for issue in override_issues:
            print(f"  {issue['town']} {issue['year']}: {issue['event_type']} - {issue['issue']}")
    else:
        print("✅ All boundary overrides working correctly")
    
    return override_issues

def main():
    """メイン処理"""
    print("Event Labeling QC Check")
    print("=" * 50)
    
    # データ読み込み
    events_df, panel_df, major_changes_df = load_data()
    
    print(f"Loaded {len(events_df)} events, {len(major_changes_df)} major changes")
    
    # 各種チェック実行
    consistency_warnings = check_consistency(events_df, major_changes_df)
    direction_warnings = check_direction_drift(events_df, major_changes_df)
    timing_issues = check_transit_timing(events_df)
    override_issues = check_boundary_override(events_df)
    
    # サマリー
    total_issues = (len(consistency_warnings) + len(direction_warnings) + 
                   len(timing_issues) + len(override_issues))
    
    print(f"\n=== Summary ===")
    print(f"Total issues found: {total_issues}")
    
    if total_issues == 0:
        print("🎉 All QC checks passed!")
    else:
        print("⚠️  Some issues need attention")
    
    return {
        'consistency_warnings': consistency_warnings,
        'direction_warnings': direction_warnings,
        'timing_issues': timing_issues,
        'override_issues': override_issues,
        'total_issues': total_issues
    }

if __name__ == "__main__":
    result = main()
