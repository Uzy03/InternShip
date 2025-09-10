# -*- coding: utf-8 -*-
"""
空間的影響予測モジュール
A町丁でイベント発生時の周辺町丁への影響を予測
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# パス設定
sys.path.append(os.path.dirname(__file__))

# Layer5モジュールのインポート
try:
    from scenario_to_events import scenario_to_events
    from prepare_baseline import prepare_baseline
    from build_future_features import build_future_features
    from forecast_service import forecast_population
except ImportError as e:
    st.error(f"Layer5モジュールのインポートに失敗しました: {e}")
    st.stop()

def batch_spatial_prediction(event_town, event_type, effect_direction, base_year, towns, centroids_df, debug_mode=False):
    """
    バッチ処理による空間的影響予測
    
    Args:
        event_town: イベント発生町丁
        event_type: イベントタイプ
        effect_direction: 効果方向
        base_year: 基準年
        towns: 全町丁リスト
        centroids_df: 重心データ
    
    Returns:
        dict: 空間的影響の結果
    """
    # シナリオ設定（scenario_to_eventsの形式に合わせる）
    scenario = {
        "town": event_town,
        "base_year": base_year,
        "horizons": [1, 2, 3],
        "events": [{
            "year_offset": 0,  # base_yearからのオフセット
            "event_type": event_type,
            "effect_direction": effect_direction,
            "confidence": 1.0,  # 確信度
            "intensity": 1.0,   # 強度
            "lag_t": 0.8,       # 当年効果の割合
            "lag_t1": 0.2,      # 翌年効果の割合
            "note": f"{event_type} ({effect_direction})"
        }],
        "macros": {},
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
    }
    
    # イベント発生町丁の予測（直接効果）
    event_baseline = prepare_baseline(event_town, base_year)
    event_future_events = scenario_to_events(scenario)
    event_future_features = build_future_features(event_baseline, event_future_events, scenario)
    
    # 将来特徴を保存
    features_path = Path("../../data/processed/l5_future_features.csv")
    event_future_features.to_csv(features_path, index=False)
    
    # イベント発生町丁の予測実行
    event_result = forecast_population(
        town=event_town,
        base_year=base_year,
        horizons=[1, 2, 3],
        base_population=float(event_baseline["pop_total"].iloc[0]),
        debug_output_dir=None,
        manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
        apply_event_to_prediction=True
    )
    
    # バッチ処理：全町丁の将来特徴を一括構築
    st.info("🔄 全町丁の将来特徴を一括構築中...")
    
    # 全町丁のベースラインデータを一括取得
    all_baselines = {}
    for town in towns:
        if town != event_town:  # イベント発生町丁は除外
            all_baselines[town] = prepare_baseline(town, base_year)
    
    # 全町丁の将来特徴を一括構築
    all_future_features_list = []
    for town, baseline in all_baselines.items():
        town_scenario = {
            "town": town,
            "base_year": base_year,
            "horizons": [1, 2, 3],
            "events": [],  # イベントなし
            "macros": {},
            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
        }
        town_future_events = scenario_to_events(town_scenario)
        town_future_features = build_future_features(baseline, town_future_events, town_scenario)
        all_future_features_list.append(town_future_features)
    
    # 全町丁の将来特徴を結合
    if all_future_features_list:
        all_future_features = pd.concat(all_future_features_list, ignore_index=True)
        all_future_features.to_csv(features_path, index=False)
        st.info(f"✅ 全町丁の将来特徴を構築完了: {len(all_future_features)}行")
    
    # バッチ処理：全町丁の予測を一括実行
    st.info("🔄 全町丁の予測を一括実行中...")
    
    spatial_impacts = []
    error_count = 0  # エラーカウンター
    
    # 重心データがない場合は簡易的な処理
    if centroids_df.empty or "lat" not in centroids_df.columns or "lon" not in centroids_df.columns:
        if not debug_mode:
            st.warning("重心データが利用できないため、簡易的な空間的影響予測を実行します")
        else:
            st.info("重心データが利用できないため、簡易的な空間的影響予測を実行します")
        
        # 全町丁に対して簡易的な影響を計算
        for town in towns:
            if town == event_town:
                continue
                
            # 簡易的な距離（町丁名の類似度ベース）
            distance = 1.0  # 固定値
            decay_factor = calculate_decay_factor(event_type, distance)
            
            # 周辺町丁の予測実行
            try:
                town_baseline = all_baselines[town]
                town_result = forecast_population(
                    town=town,
                    base_year=base_year,
                    horizons=[1, 2, 3],
                    base_population=float(town_baseline["pop_total"].iloc[0]),
                    debug_output_dir=None,
                    manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                    apply_event_to_prediction=False
                )
                
                # 空間的影響を計算
                if town_result and "path" in town_result:
                    for entry in town_result["path"]:
                        spatial_impacts.append({
                            "town": town,
                            "year": entry["year"],
                            "h": entry["year"] - base_year,
                            "delta": entry["delta_hat"],
                            "pop": entry["pop_hat"],
                            "exp": entry["contrib"]["exp"],
                            "macro": entry["contrib"]["macro"],
                            "inertia": entry["contrib"]["inertia"],
                            "other": entry["contrib"]["other"],
                            "distance": distance,
                            "decay_factor": decay_factor,
                            "spatial_impact": entry["delta_hat"] * decay_factor,
                            "lat": 0.0,  # デフォルト値
                            "lon": 0.0   # デフォルト値
                        })
            except Exception as e:
                error_count += 1
                # デバッグモードでは個別のエラーメッセージを表示しない
                if not debug_mode:
                    st.warning(f"町丁 '{town}' の予測に失敗: {e}")
                continue
    else:
        # イベント発生町丁からの距離を計算
        event_coords = centroids_df[centroids_df["town"] == event_town]
        if not event_coords.empty:
            event_lat = event_coords["lat"].iloc[0]
            event_lon = event_coords["lon"].iloc[0]
            
            # 各町丁への距離を計算
            for _, town_row in centroids_df.iterrows():
                town = town_row["town"]
                if town == event_town:
                    continue
                
                # 簡易的な距離計算（実際の距離計算に置き換え可能）
                lat_diff = town_row["lat"] - event_lat
                lon_diff = town_row["lon"] - event_lon
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                
                # 距離減衰関数（イベントタイプに応じて調整）
                decay_factor = calculate_decay_factor(event_type, distance)
                
                # 周辺町丁の予測実行
                try:
                    town_baseline = all_baselines[town]
                    town_result = forecast_population(
                        town=town,
                        base_year=base_year,
                        horizons=[1, 2, 3],
                        base_population=float(town_baseline["pop_total"].iloc[0]),
                        debug_output_dir=None,
                        manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                        apply_event_to_prediction=False
                    )
                    
                    # 空間的影響を計算
                    if town_result and "path" in town_result:
                        for entry in town_result["path"]:
                            spatial_impacts.append({
                                "town": town,
                                "year": entry["year"],
                                "h": entry["year"] - base_year,
                                "delta": entry["delta_hat"],
                                "pop": entry["pop_hat"],
                                "exp": entry["contrib"]["exp"],
                                "macro": entry["contrib"]["macro"],
                                "inertia": entry["contrib"]["inertia"],
                                "other": entry["contrib"]["other"],
                                "distance": distance,
                                "decay_factor": decay_factor,
                                "spatial_impact": entry["delta_hat"] * decay_factor,
                                "lat": town_row["lat"],
                                "lon": town_row["lon"]
                            })
                except Exception as e:
                    error_count += 1
                    # デバッグモードでは個別のエラーメッセージを表示しない
                    if not debug_mode:
                        st.warning(f"町丁 '{town}' の予測に失敗: {e}")
                    continue
    
    # エラー統計を表示
    if debug_mode and error_count > 0:
        st.info(f"🔧 デバッグモード: {error_count}町丁で予測に失敗しました（エラーログは非表示）")
    
    return {
        "event_town": event_town,
        "event_result": event_result,
        "spatial_impacts": spatial_impacts
    }

def calculate_spatial_impact(event_town, event_type, effect_direction, base_year, towns, centroids_df, debug_mode=False):
    """
    空間的影響を計算する関数
    
    Args:
        event_town: イベント発生町丁
        event_type: イベントタイプ
        effect_direction: 効果方向
        base_year: 基準年
        towns: 全町丁リスト
        centroids_df: 重心データ
    
    Returns:
        dict: 空間的影響の結果
    """
    # シナリオ設定（scenario_to_eventsの形式に合わせる）
    scenario = {
        "town": event_town,
        "base_year": base_year,
        "horizons": [1, 2, 3],
        "events": [{
            "year_offset": 0,  # base_yearからのオフセット
            "event_type": event_type,
            "effect_direction": effect_direction,
            "confidence": 1.0,  # 確信度
            "intensity": 1.0,   # 強度
            "lag_t": 0.8,       # 当年効果の割合
            "lag_t1": 0.2,      # 翌年効果の割合
            "note": f"{event_type} ({effect_direction})"
        }],
        "macros": {},
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
    }
    
    # イベント発生町丁の予測（直接効果）
    event_baseline = prepare_baseline(event_town, base_year)
    event_future_events = scenario_to_events(scenario)
    event_future_features = build_future_features(event_baseline, event_future_events, scenario)
    
    # 将来特徴を保存
    features_path = Path("../../data/processed/l5_future_features.csv")
    event_future_features.to_csv(features_path, index=False)
    
    # イベント発生町丁の予測実行
    event_result = forecast_population(
        town=event_town,
        base_year=base_year,
        horizons=[1, 2, 3],
        base_population=float(event_baseline["pop_total"].iloc[0]),
        debug_output_dir=None,
        manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
        apply_event_to_prediction=True
    )
    
    # 周辺町丁の影響を計算
    spatial_impacts = []
    error_count = 0  # エラーカウンター
    
    # 重心データがない場合は簡易的な処理
    if centroids_df.empty or "lat" not in centroids_df.columns or "lon" not in centroids_df.columns:
        if not debug_mode:
            st.warning("重心データが利用できないため、簡易的な空間的影響予測を実行します")
        else:
            st.info("重心データが利用できないため、簡易的な空間的影響予測を実行します")
        
        # 全町丁に対して簡易的な影響を計算
        for town in towns:
            if town == event_town:
                continue
                
            # 簡易的な距離（町丁名の類似度ベース）
            distance = 1.0  # 固定値
            decay_factor = calculate_decay_factor(event_type, distance)
            
            # 周辺町丁の予測（イベントなし）
            try:
                town_baseline = prepare_baseline(town, base_year)
                town_scenario = {
                    "town": town,
                    "base_year": base_year,
                    "horizons": [1, 2, 3],
                    "events": [],
                    "macros": {},
                    "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                }
                town_future_events = scenario_to_events(town_scenario)
                town_future_features = build_future_features(town_baseline, town_future_events, town_scenario)
                
                # 将来特徴を保存
                town_future_features.to_csv(features_path, index=False)
                
                # 周辺町丁の予測実行
                town_result = forecast_population(
                    town=town,
                    base_year=base_year,
                    horizons=[1, 2, 3],
                    base_population=float(town_baseline["pop_total"].iloc[0]),
                    debug_output_dir=None,
                    manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                    apply_event_to_prediction=False
                )
                
                # 空間的影響を計算
                if town_result and "path" in town_result:
                    for entry in town_result["path"]:
                        spatial_impacts.append({
                            "town": town,
                            "year": entry["year"],
                            "h": entry["year"] - base_year,
                            "delta": entry["delta_hat"],
                            "pop": entry["pop_hat"],
                            "exp": entry["contrib"]["exp"],
                            "macro": entry["contrib"]["macro"],
                            "inertia": entry["contrib"]["inertia"],
                            "other": entry["contrib"]["other"],
                            "distance": distance,
                            "decay_factor": decay_factor,
                            "spatial_impact": entry["delta_hat"] * decay_factor,
                            "lat": 0.0,  # デフォルト値
                            "lon": 0.0   # デフォルト値
                        })
            except Exception as e:
                error_count += 1
                # デバッグモードでは個別のエラーメッセージを表示しない
                if not debug_mode:
                    st.warning(f"町丁 '{town}' の予測に失敗: {e}")
                continue
    else:
        # イベント発生町丁からの距離を計算
        event_coords = centroids_df[centroids_df["town"] == event_town]
        if not event_coords.empty:
            event_lat = event_coords["lat"].iloc[0]
            event_lon = event_coords["lon"].iloc[0]
            
            # 各町丁への距離を計算
            for _, town_row in centroids_df.iterrows():
                town = town_row["town"]
                if town == event_town:
                    continue
                
                # 簡易的な距離計算（実際の距離計算に置き換え可能）
                lat_diff = town_row["lat"] - event_lat
                lon_diff = town_row["lon"] - event_lon
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                
                # 距離減衰関数（イベントタイプに応じて調整）
                decay_factor = calculate_decay_factor(event_type, distance)
                
                # 周辺町丁の予測（イベントなし）
                town_baseline = prepare_baseline(town, base_year)
                town_scenario = {
                    "town": town,
                    "base_year": base_year,
                    "horizons": [1, 2, 3],
                    "events": [],  # イベントなし
                    "macros": {},
                    "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                }
                town_future_events = scenario_to_events(town_scenario)
                town_future_features = build_future_features(town_baseline, town_future_events, town_scenario)
                
                # 将来特徴を保存
                town_future_features.to_csv(features_path, index=False)
                
                # 周辺町丁の予測実行
                town_result = forecast_population(
                    town=town,
                    base_year=base_year,
                    horizons=[1, 2, 3],
                    base_population=float(town_baseline["pop_total"].iloc[0]),
                    debug_output_dir=None,
                    manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                    apply_event_to_prediction=False
                )
                
                # 空間的影響を計算
                if town_result and "path" in town_result:
                    for entry in town_result["path"]:
                        spatial_impacts.append({
                            "town": town,
                            "year": entry["year"],
                            "h": entry["year"] - base_year,
                            "delta": entry["delta_hat"],
                            "pop": entry["pop_hat"],
                            "exp": entry["contrib"]["exp"],
                            "macro": entry["contrib"]["macro"],
                            "inertia": entry["contrib"]["inertia"],
                            "other": entry["contrib"]["other"],
                            "distance": distance,
                            "decay_factor": decay_factor,
                            "spatial_impact": entry["delta_hat"] * decay_factor,
                            "lat": town_row["lat"],
                            "lon": town_row["lon"]
                        })
    
    # エラー統計を表示
    if debug_mode and error_count > 0:
        st.info(f"🔧 デバッグモード: {error_count}町丁で予測に失敗しました（エラーログは非表示）")
    
    return {
        "event_town": event_town,
        "event_result": event_result,
        "spatial_impacts": spatial_impacts
    }

def calculate_decay_factor(event_type, distance):
    """
    距離減衰関数（イベントタイプに応じて調整）
    
    Args:
        event_type: イベントタイプ
        distance: 距離
    
    Returns:
        float: 減衰係数
    """
    # 基本減衰係数
    base_decay = 1.0 / (1.0 + distance * 0.1)
    
    # イベントタイプ別の調整
    if event_type == "housing":
        # 住宅供給：主に発生地のみ（周辺への影響小）
        return base_decay * 0.3
    elif event_type == "disaster":
        # 災害：広範囲に影響（周辺への影響大）
        return base_decay * 1.5
    elif event_type == "commercial":
        # 商業施設：中程度の周辺影響
        return base_decay * 0.8
    elif event_type == "transit":
        # 交通：沿線への影響
        return base_decay * 1.2
    elif event_type == "employment":
        # 雇用：中程度の周辺影響
        return base_decay * 0.7
    else:
        # その他：標準的な減衰
        return base_decay

def render_spatial_impact_prediction(towns):
    """空間的影響予測のUIとロジックをレンダリング"""
    
    st.header("🌍 空間的影響予測")
    st.markdown("A町丁でイベント発生時の周辺町丁への影響を予測します")
    
    # イベント設定UI
    st.subheader("🎯 イベント設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # イベント発生町丁の選択
        event_town = st.selectbox(
            "イベント発生町丁",
            towns,
            index=0,
            help="イベントが発生する町丁を選択"
        )
    
    with col2:
        # イベントタイプの選択
        EVENT_TYPE_MAPPING = {
            "housing_inc": "住宅供給の増加（竣工）",
            "housing_dec": "住宅の減少・喪失",
            "commercial_inc": "商業施設の増加", 
            "transit_inc": "交通利便の向上",
            "transit_dec": "交通利便の低下",
            "public_edu_medical_inc": "公共・教育・医療の増加",
            "public_edu_medical_dec": "公共・教育・医療の減少",
            "employment_inc": "雇用機会の増加",
            "employment_dec": "雇用機会の減少",
            "disaster_inc": "災害被害・リスクの増加",
            "disaster_dec": "災害リスクの低下（防災整備）"
        }
        
        event_type_display = st.selectbox(
            "イベントタイプ",
            list(EVENT_TYPE_MAPPING.values()),
            help="人口に影響を与えるイベントの種類を選択"
        )
        
        # 表示名から内部キーに変換
        event_type_full = [k for k, v in EVENT_TYPE_MAPPING.items() if v == event_type_display][0]
        
        # 内部キーからevent_typeとeffect_directionに分割
        if event_type_full.endswith("_inc"):
            event_type = event_type_full[:-4]
            effect_direction = "increase"
        elif event_type_full.endswith("_dec"):
            event_type = event_type_full[:-4]
            effect_direction = "decrease"
        else:
            event_type = event_type_full
            effect_direction = "increase"
    
    # 基準年の設定
    base_year = st.slider("基準年", 2020, 2030, 2025)
    
    # 処理モードの設定
    st.subheader("⚙️ 処理設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        debug_mode = st.checkbox("デバッグモード（最初の10町丁のみ）", value=True, help="チェックすると最初の10町丁のみで処理します（高速）")
    
    with col2:
        use_batch = st.checkbox("バッチ処理を使用", value=True, help="チェックすると効率的なバッチ処理を使用します")
    
    if debug_mode:
        st.info("🔧 デバッグモード: 最初の10町丁のみで処理します")
        towns_to_process = towns[:10]
    else:
        st.warning("⚠️ 本格モード: 全町丁で処理します（時間がかかります）")
        towns_to_process = towns
    
    if use_batch:
        st.info("⚡ バッチ処理: 効率的な一括処理を使用します")
    else:
        st.info("🔄 個別処理: 従来の個別処理を使用します")
    
    # 予測実行ボタン
    if st.button("🚀 空間的影響予測を実行", type="primary"):
        # 重心データの読み込み
        centroids_path = Path("../../data/processed/town_centroids.csv")
        centroids_df = pd.DataFrame()
        
        if centroids_path.exists():
            try:
                centroids_df = pd.read_csv(centroids_path, usecols=["town", "lat", "lon"])
                st.info(f"重心データを読み込みました: {len(centroids_df)}町丁")
            except Exception as e:
                if not debug_mode:
                    st.warning(f"重心データの読み込みに失敗: {e}")
                st.info("重心データなしで実行します（距離計算はスキップ）")
        else:
            if not debug_mode:
                st.warning("重心データファイルが見つかりません")
            st.info("重心データなしで実行します（距離計算はスキップ）")
        
        # 空間的影響を計算
        import time
        start_time = time.time()
        
        if use_batch:
            with st.spinner("バッチ処理で空間的影響を計算中..."):
                result = batch_spatial_prediction(
                    event_town, event_type, effect_direction, 
                    base_year, towns_to_process, centroids_df, debug_mode
                )
        else:
            with st.spinner("個別処理で空間的影響を計算中..."):
                result = calculate_spatial_impact(
                    event_town, event_type, effect_direction, 
                    base_year, towns_to_process, centroids_df, debug_mode
                )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result and result["spatial_impacts"]:
            # 結果の表示
            st.success(f"✅ 空間的影響予測が完了しました！（処理時間: {processing_time:.1f}秒）")
            
            # 処理モードの情報表示
            if debug_mode:
                st.info(f"🔧 デバッグモード: {len(towns_to_process)}町丁で処理しました（全{len(towns)}町丁中）")
            else:
                st.info(f"📊 本格モード: {len(towns_to_process)}町丁で処理しました")
            
            if use_batch:
                st.info("⚡ バッチ処理を使用しました")
            else:
                st.info("🔄 個別処理を使用しました")
            
            # イベント発生町丁の直接効果
            st.subheader("🎯 イベント発生町丁の直接効果")
            if result["event_result"] and "path" in result["event_result"]:
                event_data = []
                for entry in result["event_result"]["path"]:
                    event_data.append({
                        "年": entry["year"],
                        "Δ人口": f"{entry['delta_hat']:.1f}人",
                        "人口": f"{entry['pop_hat']:.1f}人",
                        "期待効果": f"{entry['contrib']['exp']:.1f}人",
                        "マクロ": f"{entry['contrib']['macro']:.1f}人",
                        "慣性": f"{entry['contrib']['inertia']:.1f}人",
                        "その他": f"{entry['contrib']['other']:.1f}人"
                    })
                
                event_df = pd.DataFrame(event_data)
                st.dataframe(event_df, use_container_width=True)
            
            # 空間的影響の可視化
            st.subheader("🗺️ 空間的影響の可視化")
            
            # 年を選択
            years = sorted(list(set([impact["year"] for impact in result["spatial_impacts"]])))
            selected_year = st.selectbox("表示年", years)
            
            # 選択年のデータをフィルタ
            year_data = [impact for impact in result["spatial_impacts"] if impact["year"] == selected_year]
            
            if year_data:
                impacts_df = pd.DataFrame(year_data)
                
                # 重心データがある場合はマップ表示
                if not centroids_df.empty and "lat" in centroids_df.columns and "lon" in centroids_df.columns:
                    # マップ表示
                    fig = go.Figure()
                    
                    # イベント発生町丁（赤色で表示）
                    event_coords = centroids_df[centroids_df["town"] == event_town]
                    if not event_coords.empty:
                        fig.add_trace(go.Scattermapbox(
                            lat=[event_coords["lat"].iloc[0]],
                            lon=[event_coords["lon"].iloc[0]],
                            mode='markers',
                            marker=dict(
                                size=30,
                                color='red',
                                opacity=0.9
                            ),
                            text=[f"{event_town}<br>【イベント発生地】"],
                            hovertemplate='%{text}<extra></extra>',
                            name='イベント発生地'
                        ))
                    
                    # 周辺町丁（空間的影響に応じて色とサイズを調整）
                    # 色の設定（空間的影響の符号と大きさに基づく）
                    max_impact = abs(impacts_df['spatial_impact']).max()
                    if max_impact > 0:
                        # 正規化された影響度（0-1）
                        normalized_impacts = abs(impacts_df['spatial_impact']) / max_impact
                        # 色の強度を調整（最小0.3、最大1.0）
                        color_intensities = 0.3 + 0.7 * normalized_impacts
                        # 色の設定（正の影響：オレンジ、負の影響：青）
                        colors = []
                        for i, impact in enumerate(impacts_df['spatial_impact']):
                            if impact > 0:
                                # 正の影響：オレンジ系（強度に応じて調整）
                                colors.append(f'rgba(255, 165, 0, {color_intensities.iloc[i]})')
                            else:
                                # 負の影響：青系（強度に応じて調整）
                                colors.append(f'rgba(0, 0, 255, {color_intensities.iloc[i]})')
                    else:
                        colors = ['rgba(128, 128, 128, 0.5)'] * len(impacts_df)
                    
                    # サイズの設定（影響度に応じて調整）
                    if max_impact > 0:
                        sizes = [max(8, min(25, 8 + 17 * abs(x) / max_impact)) for x in impacts_df['spatial_impact']]
                    else:
                        sizes = [8] * len(impacts_df)
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=impacts_df['lat'],
                        lon=impacts_df['lon'],
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=impacts_df['spatial_impact'],  # 数値で色を指定
                            colorscale='RdBu',  # 赤-青のカラースケール
                            cmin=impacts_df['spatial_impact'].min(),
                            cmax=impacts_df['spatial_impact'].max(),
                            colorbar=dict(
                                title="空間的影響（人）",
                                tickmode="auto",
                                nticks=5
                            ),
                            opacity=0.8
                        ),
                        text=impacts_df['town'] + '<br>空間的影響: ' + impacts_df['spatial_impact'].round(3).astype(str) + '人<br>距離: ' + impacts_df['distance'].round(2).astype(str),
                        hovertemplate='%{text}<extra></extra>',
                        name='周辺町丁'
                    ))
                    
                    # 地図の中心をイベント発生町丁に設定
                    if not event_coords.empty:
                        center_lat = event_coords["lat"].iloc[0]
                        center_lon = event_coords["lon"].iloc[0]
                    else:
                        center_lat = impacts_df['lat'].mean()
                        center_lon = impacts_df['lon'].mean()
                    
                    # 影響度の範囲を計算
                    min_impact = impacts_df['spatial_impact'].min()
                    max_impact = impacts_df['spatial_impact'].max()
                    
                    fig.update_layout(
                        mapbox=dict(
                            style="open-street-map",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=12
                        ),
                        height=600,
                        title=f"{selected_year}年の空間的影響分布（赤: イベント発生地、赤-青: 影響度）<br>影響度範囲: {min_impact:.3f} ～ {max_impact:.3f}人"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # 重心データがない場合はテーブル表示のみ
                    st.info("重心データが利用できないため、マップ表示はスキップします")
            
            # 空間的影響のテーブル表示
            st.subheader("📊 空間的影響の詳細")
            
            if year_data:
                impacts_df = pd.DataFrame(year_data)
                impacts_df = impacts_df.sort_values('spatial_impact', ascending=False)
                
                # 表示する列を選択（重心データの有無に応じて調整）
                if not centroids_df.empty and "lat" in centroids_df.columns and "lon" in centroids_df.columns:
                    display_columns = ["town", "spatial_impact", "delta", "distance", "decay_factor", "lat", "lon"]
                else:
                    display_columns = ["town", "spatial_impact", "delta", "distance", "decay_factor"]
                
                display_df = impacts_df[display_columns].copy()
                
                # 列名を日本語に変更
                column_mapping = {
                    "town": "町丁",
                    "spatial_impact": "空間的影響",
                    "delta": "Δ人口",
                    "distance": "距離",
                    "decay_factor": "減衰係数",
                    "lat": "緯度",
                    "lon": "経度"
                }
                
                display_df = display_df.rename(columns=column_mapping)
                
                # 数値列を丸める
                numeric_columns = ["空間的影響", "Δ人口", "距離", "減衰係数"]
                for col in numeric_columns:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].round(3)
                
                st.dataframe(display_df, use_container_width=True)
                
                # 統計情報
                st.subheader("📈 統計情報")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("影響を受ける町丁数", len(impacts_df))
                
                with col2:
                    avg_impact = impacts_df['spatial_impact'].mean()
                    st.metric("平均空間的影響", f"{avg_impact:.2f}人")
                
                with col3:
                    max_impact = impacts_df['spatial_impact'].max()
                    st.metric("最大空間的影響", f"{max_impact:.2f}人")
                
                with col4:
                    min_impact = impacts_df['spatial_impact'].min()
                    st.metric("最小空間的影響", f"{min_impact:.2f}人")
        else:
            st.error("空間的影響の計算に失敗しました。")
