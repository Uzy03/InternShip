"""
Post-processing sanitization for events matrix

This module implements post-processing steps to clean and consolidate
the events matrix before TWFE analysis:
1. Small population zeroing
2. Inc/Dec consolidation (dec→t, inc→t1)
3. Pretrend guard (suppress transit_dec in structurally declining areas)
4. Consecutive year consolidation (keep only first year of transit_dec_t)
5. Policy boundary vs transit collision resolution (NEW)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ===== ハードコード・パラメータ =====
SMALL_POP_THRESH = 300
SMALL_DELTA_THRESH = 50
PRETREND_YEARS = 2             # t-1, t-2 を使う
PRETREND_SUM_THRESH = -120     # (Δt-1 + Δt-2) ≤ -120 なら transit_dec_* を 0
CONSOLIDATE_BASES = ["transit", "public_edu_medical", "disaster"]

P_EVENTS = "subject3/data/processed/events_matrix_signed.csv"
P_FEATS  = "subject3/data/processed/features_panel.csv"
P_LAB_CLEAN = "subject3/data/processed/events_labeled_clean.csv"
P_REPORT = "data/processed/events_sanitize_report.csv"

def main():
    """メイン処理"""
    global P_EVENTS, P_FEATS, P_LAB_CLEAN, P_REPORT
    
    print("=== イベント後処理開始 ===")
    
    # プロジェクトルートからの相対パスに調整
    project_root = Path(__file__).parent.parent.parent
    if not Path(P_EVENTS).exists():
        P_EVENTS = project_root / P_EVENTS
    if not Path(P_FEATS).exists():
        P_FEATS = project_root / P_FEATS
    if not Path(P_LAB_CLEAN).exists():
        P_LAB_CLEAN = project_root / P_LAB_CLEAN
    
    print(f"イベントファイル: {P_EVENTS}")
    print(f"特徴量ファイル: {P_FEATS}")
    print(f"ラベルクリーンファイル: {P_LAB_CLEAN}")
    
    # ===== 読み込み =====
    em = pd.read_csv(P_EVENTS)
    fp = pd.read_csv(P_FEATS).sort_values(["town","year"])
    labs = pd.read_csv(P_LAB_CLEAN)
    
    print(f"イベントデータ: {len(em)} 行, {len(em.columns)} 列")
    print(f"特徴量データ: {len(fp)} 行, {len(fp.columns)} 列")
    print(f"ラベルクリーンデータ: {len(labs)} 行, {len(labs.columns)} 列")
    
    # Δ人数が無ければ作る
    if "delta_people" not in fp.columns:
        if "pop_total" not in fp.columns:
            raise ValueError("features_panel.csv に pop_total が必要です。")
        fp["delta_people"] = fp.groupby("town")["pop_total"].diff()
        print("delta_people列を作成完了")
    
    # ===== (A) 小母数ゼロ化 =====
    print("小母数ゼロ化を実行中...")
    mask_small = (fp["pop_total"] < SMALL_POP_THRESH) & (fp["delta_people"].abs() < SMALL_DELTA_THRESH)
    keys_small = fp.loc[mask_small, ["town","year"]].assign(_small=1)
    print(f"小母数条件を満たす行数: {len(keys_small)}")
    
    em = em.merge(keys_small, on=["town","year"], how="left")
    event_cols = [c for c in em.columns if c.startswith("event_")]
    small_rows = em["_small"].fillna(0).eq(1).sum()
    em.loc[em["_small"].fillna(0).eq(1), event_cols] = 0.0
    em.drop(columns=["_small"], inplace=True, errors="ignore")
    print(f"小母数ゼロ化完了: {small_rows} 行のイベントをゼロ化")
    
    # ===== (B) inc/dec の整流（dec→t, inc→t1 集約） =====
    print("inc/dec整流を実行中...")
    for base in CONSOLIDATE_BASES:
        print(f"  {base} の整流中...")
        inc_t, inc_t1 = f"event_{base}_inc_t",  f"event_{base}_inc_t1"
        dec_t, dec_t1 = f"event_{base}_dec_t",  f"event_{base}_dec_t1"
        
        # 存在しない列は0で初期化
        for col in (inc_t, inc_t1, dec_t, dec_t1):
            if col not in em.columns:
                em[col] = 0.0
                print(f"    新規列作成: {col}")
        
        # 整流前の値を記録
        dec_t_before = em[dec_t].sum()
        dec_t1_before = em[dec_t1].sum()
        inc_t_before = em[inc_t].sum()
        inc_t1_before = em[inc_t1].sum()
        
        # dec は t に寄せる
        em[dec_t]  = em[dec_t] + em[dec_t1]
        em[dec_t1] = 0.0
        # inc は t+1 に寄せる
        em[inc_t1] = em[inc_t1] + em[inc_t]
        em[inc_t]  = 0.0
        
        # 整流後の値を記録
        dec_t_after = em[dec_t].sum()
        dec_t1_after = em[dec_t1].sum()
        inc_t_after = em[inc_t].sum()
        inc_t1_after = em[inc_t1].sum()
        
        print(f"    {base}_dec: t={dec_t_before:.1f}→{dec_t_after:.1f}, t1={dec_t1_before:.1f}→{dec_t1_after:.1f}")
        print(f"    {base}_inc: t={inc_t_before:.1f}→{inc_t_after:.1f}, t1={inc_t1_before:.1f}→{inc_t1_after:.1f}")
    
    print("inc/dec整流完了")
    
    # ===== (C) pretrend ガード：構造的減少に偏った transit_dec_* を 0 に =====
    print("事前トレンドガードを実行中...")
    print(f"事前トレンド年数: {PRETREND_YEARS}, 閾値: {PRETREND_SUM_THRESH}")
    
    fp["d1"] = fp.groupby("town")["delta_people"].shift(1)
    fp["d2"] = fp.groupby("town")["delta_people"].shift(2) if PRETREND_YEARS >= 2 else 0
    fp["pretrend_sum"] = fp[["d1","d2"]].sum(axis=1, min_count=1)
    keys_guard = fp.loc[fp["pretrend_sum"] <= PRETREND_SUM_THRESH, ["town","year"]].assign(_guard=1)
    
    print(f"事前トレンドガード条件を満たす行数: {len(keys_guard)}")
    
    em = em.merge(keys_guard, on=["town","year"], how="left")
    for lag in ["t","t1"]:
        col = f"event_transit_dec_{lag}"
        if col in em.columns:
            before_sum = em[col].sum()
            em.loc[em["_guard"].fillna(0).eq(1), col] = 0.0
            after_sum = em[col].sum()
            print(f"    {col}: {before_sum:.1f} → {after_sum:.1f}")
    em.drop(columns=["_guard"], inplace=True, errors="ignore")
    print("事前トレンドガード完了")
    
    # ===== (D) 連続年の transit_dec_t を「最初の年だけ残す」 =====
    print("連続年集約を実行中...")
    if "event_transit_dec_t" in em.columns:
        em = em.sort_values(["town","year"]).reset_index(drop=True)
        to_zero_idx = []
        for town, sub in em.groupby("town", group_keys=False):
            idx = sub.index[sub["event_transit_dec_t"] > 0].tolist()
            if not idx:
                continue
            yrs = sub.loc[idx, "year"].tolist()
            keep = []
            # 連続 run の先頭のみ keep
            for k, (i, y) in enumerate(zip(idx, yrs)):
                if k == 0 or y != yrs[k-1] + 1:
                    keep.append(i)
            drop = set(idx) - set(keep)
            to_zero_idx.extend(list(drop))
        
        if to_zero_idx:
            before_sum = em["event_transit_dec_t"].sum()
            em.loc[to_zero_idx, "event_transit_dec_t"] = 0.0
            after_sum = em["event_transit_dec_t"].sum()
            print(f"    event_transit_dec_t: {before_sum:.1f} → {after_sum:.1f} ({len(to_zero_idx)} 行をゼロ化)")
        else:
            print("    連続年集約対象なし")
    else:
        print("    event_transit_dec_t列が見つかりません")
    print("連続年集約完了")
    
    # ===== (E) policy_boundary vs transit 交差衝突解消 =====
    print("policy_boundary vs transit 交差衝突解消を実行中...")
    
    # 町年ごとの policy_boundary 存在（cleanはソース優先済み）
    pb_keys = set(zip(labs.loc[labs["event_type"]=="policy_boundary","town"],
                      labs.loc[labs["event_type"]=="policy_boundary","year"]))
    
    print(f"policy_boundary が存在する (town,year) 数: {len(pb_keys)}")
    
    # transit のうち、出典なしの元ラベルがあったか（クリーン前は分からないので、ここでは列抑止のみ）
    # → events_matrix は既に clean ベースなので「出典なし transit」を個別に識別できない。
    # 代わりに、policy_boundary が立っている (town,year) では transit をゼロ化（保守的）
    tr_cols_t  = [c for c in em.columns if c.startswith("event_transit_") and c.endswith("_t")]
    tr_cols_t1 = [c for c in em.columns if c.startswith("event_transit_") and c.endswith("_t1")]
    
    before_nonzero = int((em[tr_cols_t + tr_cols_t1].abs() > 0).sum().sum())
    mask_pb = em.apply(lambda r: (r["town"], r["year"]) in pb_keys, axis=1)
    em.loc[mask_pb, tr_cols_t + tr_cols_t1] = 0.0
    after_nonzero = int((em[tr_cols_t + tr_cols_t1].abs() > 0).sum().sum())
    
    print(f"transit ゼロ化: {before_nonzero} → {after_nonzero} (policy_boundary との交差で {before_nonzero - after_nonzero} 個をゼロ化)")
    
    # レポート作成
    report_data = {
        "transit_nonzero_before": before_nonzero,
        "transit_nonzero_after": after_nonzero,
        "policy_boundary_keys": len(pb_keys),
        "transit_zeroed_by_policy_boundary": before_nonzero - after_nonzero
    }
    
    report_df = pd.DataFrame([report_data])
    report_path = project_root / P_REPORT
    # ディレクトリが存在しない場合は作成
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(str(report_path), index=False)
    print(f"監査レポート保存: {report_path}")
    
    print("policy_boundary vs transit 交差衝突解消完了")
    
    # ===== 保存 =====
    print(f"結果を保存中: {P_EVENTS}")
    # ディレクトリが存在しない場合は作成
    Path(P_EVENTS).parent.mkdir(parents=True, exist_ok=True)
    em.to_csv(str(P_EVENTS), index=False)
    
    print("=== イベント後処理完了 ===")
    print(f"最終結果: 行数={len(em)}, 列数={len(em.columns)}, ファイル={P_EVENTS}")
    
    # ===== 受け入れ基準の確認 =====
    print("=== 受け入れ基準確認 ===")
    
    # 整流確認
    print("1. 整流確認:")
    for base in CONSOLIDATE_BASES:
        for direction in ["inc", "dec"]:
            for lag in ["t", "t1"]:
                col = f"event_{base}_{direction}_{lag}"
                if col in em.columns:
                    col_sum = em[col].sum()
                    if direction == "dec" and lag == "t1":
                        if col_sum == 0:
                            print(f"✓ {col} == 0 (整流成功)")
                        else:
                            print(f"✗ {col} = {col_sum} (整流失敗)")
                    elif direction == "inc" and lag == "t":
                        if col_sum == 0:
                            print(f"✓ {col} == 0 (整流成功)")
                        else:
                            print(f"✗ {col} = {col_sum} (整流失敗)")
    
    # 事前トレンドガード確認
    print("2. 事前トレンドガード確認:")
    # pretrend_sum ≤ -120 の条件を満たす行を再計算
    fp_sorted = fp.sort_values(["town", "year"])
    fp_sorted["d1"] = fp_sorted.groupby("town")["delta_people"].shift(1)
    fp_sorted["d2"] = fp_sorted.groupby("town")["delta_people"].shift(2)
    fp_sorted["pretrend_sum"] = fp_sorted[["d1", "d2"]].sum(axis=1, min_count=1)
    
    # ガード条件を満たす行
    guard_condition = fp_sorted["pretrend_sum"] <= PRETREND_SUM_THRESH
    guard_towns_years = fp_sorted.loc[guard_condition, ["town", "year"]]
    
    print(f"pretrend_sum ≤ {PRETREND_SUM_THRESH} の行数: {len(guard_towns_years)}")
    
    # ガード条件を満たす行でevent_transit_dec_*が0になっているか確認
    for lag in ["t", "t1"]:
        col = f"event_transit_dec_{lag}"
        if col in em.columns:
            # ガード条件を満たす行のイベント値を確認
            guard_events = em.merge(guard_towns_years, on=["town", "year"], how="inner")
            if len(guard_events) > 0:
                guard_sum = guard_events[col].sum()
                if guard_sum == 0:
                    print(f"✓ {col} がガード条件行で0 (成功)")
                else:
                    print(f"✗ {col} がガード条件行で{guard_sum} (失敗)")
            else:
                print(f"ガード条件を満たす行がありません")
            
            # 全体の合計も表示
            total_sum = em[col].sum()
            print(f"  全体の{col}: {total_sum:.1f}")
    
    # 連続年集約確認
    print("3. 連続年集約確認:")
    if "event_transit_dec_t" in em.columns:
        # 各町丁で連続年をチェック
        consecutive_found = 0
        for town, sub in em.groupby("town"):
            sub = sub.sort_values("year")
            transit_dec_years = sub[sub["event_transit_dec_t"] > 0]["year"].tolist()
            if len(transit_dec_years) > 1:
                # 連続年をチェック
                for i in range(len(transit_dec_years) - 1):
                    if transit_dec_years[i+1] == transit_dec_years[i] + 1:
                        consecutive_found += 1
                        break
        
        if consecutive_found == 0:
            print("✓ 連続年のtransit_dec_tが最初の年のみ残っている (成功)")
        else:
            print(f"✗ {consecutive_found} 町丁で連続年のtransit_dec_tが残っている (失敗)")
    else:
        print("event_transit_dec_t列が見つかりません")
    
    # policy_boundary vs transit 衝突解消確認
    print("4. policy_boundary vs transit 衝突解消確認:")
    pb_keys_check = set(zip(labs.loc[labs["event_type"]=="policy_boundary","town"],
                           labs.loc[labs["event_type"]=="policy_boundary","year"]))
    
    # policy_boundary が立っている行で transit が 0 になっているか確認
    pb_rows = em[em.apply(lambda r: (r["town"], r["year"]) in pb_keys_check, axis=1)]
    if len(pb_rows) > 0:
        transit_in_pb_rows = pb_rows[tr_cols_t + tr_cols_t1].abs().sum().sum()
        if transit_in_pb_rows == 0:
            print("✓ policy_boundary が立っている行で transit が 0 (衝突解消成功)")
        else:
            print(f"✗ policy_boundary が立っている行で transit が {transit_in_pb_rows} (衝突解消失敗)")
    else:
        print("policy_boundary が立っている行がありません")
    
    # 全体の transit 合計も表示
    total_transit = em[tr_cols_t + tr_cols_t1].abs().sum().sum()
    print(f"  全体の transit 合計: {total_transit:.1f}")
    
    print("=== 受け入れ基準確認完了 ===")

if __name__ == "__main__":
    main()