# -*- coding: utf-8 -*-
"""
シンプル人口予測ダッシュボード
町丁、イベントタイプ、効果方向のみを選択して予測実行
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import os
from typing import Dict, List, Any

# パス設定
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layer5'))

# Layer5モジュールのインポート
try:
    from scenario_to_events import scenario_to_events  # pyright: ignore[reportMissingImports]
    from prepare_baseline import prepare_baseline
    from build_future_features import build_future_features
    from forecast_service import forecast_population, run_scenario
    from scenario_with_learned_intensity import LearnedScenarioGenerator
except ImportError as e:
    st.error(f"Layer5モジュールのインポートに失敗しました: {e}")
    st.stop()

# ページ設定
st.set_page_config(
    page_title="シンプル人口予測ダッシュボード", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🏘️ シンプル人口予測ダッシュボード")
st.markdown("町丁、イベントタイプ、効果方向を選択して人口予測を実行")

# ビューモード選択
view_mode = st.radio(
    "表示モードを選択",
    ["単一町丁予測", "全地域表示"],
    horizontal=True
)

st.markdown("---")

# メタデータの読み込み
@st.cache_data
def load_metadata():
    """利用可能な町丁のリストを取得"""
    try:
        features_path = Path("../../data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("../data/processed/features_panel.csv")
        
        if not features_path.exists():
            st.error(f"features_panel.csv が見つかりません: {features_path}")
            return []
        
        df = pd.read_csv(features_path, usecols=["town"]).drop_duplicates()
        towns = sorted(df["town"].unique().tolist())
        
        return towns
    except Exception as e:
        st.error(f"メタデータの読み込みに失敗しました: {e}")
        return []

towns = load_metadata()

if not towns:
    st.error("メタデータの読み込みに失敗しました。")
    st.stop()

# 全地域リアルタイム予測関数
@st.cache_data
def run_all_towns_realtime_prediction(event_town: str, event_type: str, effect_direction: str, base_year: int = 2025, horizons: list = [1, 2, 3]):
    """指定されたイベントで全町丁のリアルタイム予測を実行"""
    try:
        # 利用可能な町丁リストを取得
        features_path = Path("../../data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("../data/processed/features_panel.csv")
        
        if not features_path.exists():
            st.error(f"features_panel.csv が見つかりません: {features_path}")
            return None
        
        df = pd.read_csv(features_path, usecols=["town"]).drop_duplicates()
        all_towns = sorted(df["town"].unique().tolist())
        
        # デバッグ用：最初の10町丁のみでテスト（コメントアウト）
        # if len(all_towns) > 10:
        #     st.warning(f"デバッグモード: 最初の10町丁のみでテストします（全{len(all_towns)}町丁中）")
        #     all_towns = all_towns[:10]
        
        # 将来特徴が見つからない町丁のカウンター
        missing_features_count = 0
        max_missing_features = 5  # 最大5町丁まで警告を表示
        
        # ベースラインデータを取得
        baseline_path = Path("../../data/processed/l5_baseline.csv")
        if not baseline_path.exists():
            baseline_path = Path("../data/processed/l5_baseline.csv")
        
        if not baseline_path.exists():
            st.error(f"l5_baseline.csv が見つかりません: {baseline_path}")
            return None
        
        baseline_df = pd.read_csv(baseline_path)
        
        # シナリオを作成
        scenario = {
            "town": event_town,  # イベント発生町丁
            "base_year": base_year,
            "horizons": horizons,
            "events": [{
                "year_offset": 1,
                "event_type": event_type,
                "effect_direction": effect_direction,
                "confidence": 1.0,
                "intensity": 1.0,
                "lag_t": 1,
                "lag_t1": 1,
                "note": f"{event_type} ({effect_direction})"
            }],
            "macros": {},
            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
        }
        
        # 全町丁の予測を実行
        st.info("全町丁の予測を実行中...")
        
        all_results = []
        
        # プログレスバーを表示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # イベント発生町丁の将来特徴を先に構築（他の町丁でも参照される可能性があるため）
        if event_town in all_towns:
            st.info(f"イベント発生町丁 '{event_town}' の将来特徴を構築中...")
            try:
                # イベント発生町丁のシナリオで将来特徴を構築
                event_baseline = prepare_baseline(event_town, base_year)
                event_future_events = scenario_to_events(scenario)
                event_future_features = build_future_features(event_baseline, event_future_events, scenario)
                
                # 将来特徴ファイルを保存
                features_path = Path("../../data/processed/l5_future_features.csv")
                event_future_features.to_csv(features_path, index=False)
                st.success(f"イベント発生町丁 '{event_town}' の将来特徴を構築完了")
            except Exception as e:
                st.warning(f"イベント発生町丁の将来特徴構築に失敗: {e}")
        
        for i, town in enumerate(all_towns):
            # プログレスバーの更新頻度を減らす（10町丁ごと）
            if i % 10 == 0 or i == len(all_towns) - 1:
                status_text.text(f"処理中: {town} ({i+1}/{len(all_towns)})")
                progress_bar.progress((i + 1) / len(all_towns))
            
            # 各町丁用のシナリオを作成
            if town == event_town:
                # イベント発生町丁：イベントありのシナリオ
                town_scenario = {
                    "town": town,
                    "base_year": base_year,
                    "horizons": horizons,
                    "events": scenario["events"],
                    "macros": {},
                    "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                }
            else:
                # その他の町丁：通常の人口予測（イベントなし）
                town_scenario = {
                    "town": town,
                    "base_year": base_year,
                    "horizons": horizons,
                    "events": [],  # イベントなし
                    "macros": {},
                    "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                }
            
            # ベース人口を取得
            town_baseline = baseline_df[baseline_df["town"] == town]
            if not town_baseline.empty and "pop_total" in town_baseline.columns:
                town_scenario["base_population"] = float(town_baseline["pop_total"].iloc[0])
            else:
                town_scenario["base_population"] = 0.0
            
            try:
                # 予測実行（ログを削減）
                if town != event_town:
                    # イベントが発生していない町丁の場合は、個別に将来特徴を構築してから予測
                    try:
                        # 各町丁用のシナリオで将来特徴を構築
                        individual_scenario = {
                            "town": town,
                            "base_year": base_year,
                            "horizons": horizons,
                            "events": [],  # イベントなし
                            "macros": {},
                            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                        }
                        
                        # 個別のベースラインデータを準備
                        individual_baseline = prepare_baseline(town, base_year)
                        
                        # 個別の将来イベント行列を生成（イベントなし）
                        individual_future_events = scenario_to_events(individual_scenario)
                        
                        # 個別の将来特徴を構築
                        individual_future_features = build_future_features(
                            individual_baseline, individual_future_events, individual_scenario
                        )
                        
                        # 一時的に将来特徴ファイルを保存（forecast_populationが読み込むため）
                        temp_features_path = Path("../../data/processed/l5_future_features.csv")
                        individual_future_features.to_csv(temp_features_path, index=False)
                        
                        # 通常の人口予測を実行
                        result = forecast_population(
                            town=town,
                            base_year=base_year,
                            horizons=horizons,
                            base_population=town_scenario["base_population"],
                            debug_output_dir=None,
                            manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                            apply_event_to_prediction=False  # イベントなしの通常予測
                        )
                        
                    except Exception as individual_error:
                        # 個別構築に失敗した場合は、基本予測にフォールバック
                        if missing_features_count <= max_missing_features:
                            st.warning(f"町丁 '{town}' の個別特徴構築に失敗、基本予測を実行: {individual_error}")
                        
                        result = forecast_population(
                            town=town,
                            base_year=base_year,
                            horizons=horizons,
                            base_population=town_scenario["base_population"],
                            debug_output_dir=None,
                            manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                            apply_event_to_prediction=False
                        )
                else:
                    # イベント発生町丁は既に構築された将来特徴を使用して予測
                    result = forecast_population(
                        town=town,
                        base_year=base_year,
                        horizons=horizons,
                        base_population=town_scenario["base_population"],
                        debug_output_dir=None,
                        manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                        apply_event_to_prediction=True  # イベントありの予測
                    )
                
                # 結果の検証
                if result is None or "path" not in result:
                    st.warning(f"町丁 '{town}' の予測結果が無効です (result: {result})")
                    # 無効な結果も含める（NaNなどで）
                    for h_val in horizons:
                        all_results.append({
                            "town": town,
                            "baseline_year": base_year,
                            "year": base_year + h_val,
                            "h": h_val,
                            "delta": float('nan'),
                            "pop": float('nan'),
                            "exp": float('nan'),
                            "macro": float('nan'),
                            "inertia": float('nan'),
                            "other": float('nan'),
                            "pi_delta_low": float('nan'),
                            "pi_delta_high": float('nan'),
                            "pi_pop_low": float('nan'),
                            "pi_pop_high": float('nan'),
                            "is_event_town": (town == event_town),
                        })
                    continue
                
                # 結果をフラット化（forecast_populationは"path"キーを返す）
                for entry in result["path"]:
                    row = {
                        "town": result["town"],
                        "baseline_year": result["baseline_year"],
                        "year": entry["year"],
                        "h": entry["year"] - result["baseline_year"],
                        "delta": entry["delta_hat"],
                        "pop": entry["pop_hat"],
                        "exp": entry["contrib"]["exp"],
                        "macro": entry["contrib"]["macro"],
                        "inertia": entry["contrib"]["inertia"],
                        "other": entry["contrib"]["other"],
                        "pi_delta_low": entry["pi95_delta"][0] if isinstance(entry["pi95_delta"], list) else entry["pi95_delta"],
                        "pi_delta_high": entry["pi95_delta"][1] if isinstance(entry["pi95_delta"], list) else entry["pi95_delta"],
                        "pi_pop_low": entry["pi95_pop"][0] if isinstance(entry["pi95_pop"], list) else entry["pi95_pop"],
                        "pi_pop_high": entry["pi95_pop"][1] if isinstance(entry["pi95_pop"], list) else entry["pi95_pop"],
                        "is_event_town": (town == event_town),  # イベント発生町丁かどうか
                    }
                    all_results.append(row)
                    
            except Exception as e:
                error_msg = str(e)
                if "将来特徴が見つかりません" in error_msg:
                    missing_features_count += 1
                    if missing_features_count <= max_missing_features:
                        st.warning(f"町丁 '{town}' の将来特徴が見つかりません。基本予測を実行します。")
                    elif missing_features_count == max_missing_features + 1:
                        st.warning(f"他にも将来特徴が見つからない町丁がありますが、警告表示を制限します。")
                    
                    # 将来特徴が見つからない場合は、イベントなしの基本予測を実行
                    try:
                        basic_scenario = {
                            "town": town,
                            "base_year": base_year,
                            "horizons": horizons,
                            "events": [],  # イベントなし
                            "macros": {},
                            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0},
                            "base_population": town_scenario["base_population"]
                        }
                        result = run_scenario(basic_scenario, out_path=None)
                        
                        if result and "results" in result:
                            # 基本予測の結果を使用
                            for entry in result["results"]:
                                row = {
                                    "town": result["town"],
                                    "baseline_year": result["baseline_year"],
                                    "year": entry["year"],
                                    "h": entry["year"] - result["baseline_year"],
                                    "delta": entry["delta"],
                                    "pop": entry["pop"],
                                    "exp": entry["contrib"]["exp"],
                                    "macro": entry["contrib"]["macro"],
                                    "inertia": entry["contrib"]["inertia"],
                                    "other": entry["contrib"]["other"],
                                    "pi_delta_low": entry["pi"]["delta_low"],
                                    "pi_delta_high": entry["pi"]["delta_high"],
                                    "pi_pop_low": entry["pi"]["pop_low"],
                                    "pi_pop_high": entry["pi"]["pop_high"],
                                    "is_event_town": (town == event_town),
                                }
                                all_results.append(row)
                            continue
                    except Exception as basic_error:
                        if missing_features_count <= max_missing_features:
                            st.warning(f"町丁 '{town}' の基本予測も失敗")
                        # 基本予測も失敗した場合は、NaNで埋める
                        for h_val in horizons:
                            all_results.append({
                                "town": town,
                                "baseline_year": base_year,
                                "year": base_year + h_val,
                                "h": h_val,
                                "delta": float('nan'),
                                "pop": float('nan'),
                                "exp": float('nan'),
                                "macro": float('nan'),
                                "inertia": float('nan'),
                                "other": float('nan'),
                                "pi_delta_low": float('nan'),
                                "pi_delta_high": float('nan'),
                                "pi_pop_low": float('nan'),
                                "pi_pop_high": float('nan'),
                                "is_event_town": (town == event_town),
                            })
                        continue
                
                # その他のエラーの場合
                if missing_features_count <= max_missing_features:
                    st.warning(f"町丁 '{town}' の予測中にエラー")
                # エラーが発生した町丁も結果に含める（NaNなどで）
                for h_val in horizons:
                    all_results.append({
                        "town": town,
                        "baseline_year": base_year,
                        "year": base_year + h_val,
                        "h": h_val,
                        "delta": float('nan'),
                        "pop": float('nan'),
                        "exp": float('nan'),
                        "macro": float('nan'),
                        "inertia": float('nan'),
                        "other": float('nan'),
                        "pi_delta_low": float('nan'),
                        "pi_delta_high": float('nan'),
                        "pi_pop_low": float('nan'),
                        "pi_pop_high": float('nan'),
                        "is_event_town": (town == event_town),
                    })
                continue
        
        # プログレスバーをクリア
        progress_bar.empty()
        status_text.empty()
        
        # 統計情報を表示
        if missing_features_count > 0:
            st.info(f"📊 将来特徴が見つからなかった町丁: {missing_features_count}町丁（基本予測で代替）")
            st.warning("⚠️ 将来特徴が見つからない町丁は、イベント効果なしの基本予測（人口変化=0）で処理されています。")
        
        if not all_results:
            st.error("予測結果がありません")
            return None
        
        # DataFrameに変換
        result_df = pd.DataFrame(all_results)
        
        # 重心データがあれば結合
        centroids_path = Path("../../data/interim/centroids.csv")
        if not centroids_path.exists():
            centroids_path = Path("../data/interim/centroids.csv")
        
        if centroids_path.exists():
            try:
                centroids_df = pd.read_csv(centroids_path, usecols=["town", "lat", "lon"])
                result_df = pd.merge(result_df, centroids_df, on="town", how="left")
            except Exception as e:
                st.warning(f"重心データの結合に失敗: {e}")
        
        return result_df
        
    except Exception as e:
        st.error(f"全地域予測の実行に失敗しました: {e}")
        return None

# 全地域表示の場合
if view_mode == "全地域表示":
    st.header("🌍 全地域表示 - リアルタイム予測")
    
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
    
    # 予測実行ボタン
    if st.button("🚀 全地域予測実行", type="primary", use_container_width=True):
        with st.spinner("全町丁の予測を実行中..."):
            forecast_df = run_all_towns_realtime_prediction(
                event_town=event_town,
                event_type=event_type,
                effect_direction=effect_direction
            )
    
    # 予測結果が存在する場合のみ表示
    if 'forecast_df' in locals() and forecast_df is not None:
        st.success(f"✅ 予測完了！{len(forecast_df)}件のデータを取得しました。")
        
        # イベント発生町丁のハイライト表示
        st.subheader("🎯 イベント発生町丁")
        event_town_data = forecast_df[forecast_df['is_event_town'] == True]
        if not event_town_data.empty:
            st.info(f"**{event_town}** で **{event_type_display}** が発生")
        
        # フィルタリング設定
        st.subheader("🔍 フィルタリング・表示設定")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 年セレクタ
            available_years = sorted(forecast_df['year'].unique())
            selected_year = st.selectbox("年を選択", available_years, index=len(available_years)-1)
        
        with col2:
            # 指標セレクタ
            metric_options = {
                "delta": "Δ人口",
                "pop": "人口",
                "exp": "期待効果",
                "macro": "マクロ",
                "inertia": "慣性",
                "other": "その他"
            }
            selected_metric = st.selectbox("指標を選択", list(metric_options.keys()), 
                                         format_func=lambda x: metric_options[x])
        
        with col3:
            # 町丁検索
            search_term = st.text_input("町丁名で検索", placeholder="町丁名の一部を入力")
        
        # データフィルタリング
        filtered_df = forecast_df[forecast_df['year'] == selected_year].copy()
        
        if search_term:
            filtered_df = filtered_df[filtered_df['town'].str.contains(search_term, case=False, na=False)]
        
        # ソート設定
        sort_options = {
            "delta_desc": "Δ人口（降順）",
            "delta_asc": "Δ人口（昇順）",
            "pop_desc": "人口（降順）",
            "pop_asc": "人口（昇順）",
            "town_asc": "町丁名（昇順）"
        }
        
        sort_option = st.selectbox("ソート順", list(sort_options.keys()), 
                                  format_func=lambda x: sort_options[x])
        
        if sort_option == "delta_desc":
            filtered_df = filtered_df.sort_values('delta', ascending=False)
        elif sort_option == "delta_asc":
            filtered_df = filtered_df.sort_values('delta', ascending=True)
        elif sort_option == "pop_desc":
            filtered_df = filtered_df.sort_values('pop', ascending=False)
        elif sort_option == "pop_asc":
            filtered_df = filtered_df.sort_values('pop', ascending=True)
        elif sort_option == "town_asc":
            filtered_df = filtered_df.sort_values('town', ascending=True)
        
        # 統計情報
        st.subheader("📊 統計情報")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("表示町丁数", len(filtered_df))
        
        with col2:
            avg_delta = filtered_df['delta'].mean()
            st.metric("平均Δ人口", f"{avg_delta:.1f}人")
        
        with col3:
            max_delta = filtered_df['delta'].max()
            st.metric("最大Δ人口", f"{max_delta:.1f}人")
        
        with col4:
            min_delta = filtered_df['delta'].min()
            st.metric("最小Δ人口", f"{min_delta:.1f}人")
        
        # イベント発生町丁の詳細表示
        if not event_town_data.empty:
            st.subheader("🎯 イベント発生町丁の詳細")
            event_year_data = event_town_data[event_town_data['year'] == selected_year]
            if not event_year_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("人口", f"{event_year_data['pop'].iloc[0]:.1f}人")
                with col2:
                    st.metric("Δ人口", f"{event_year_data['delta'].iloc[0]:.1f}人")
                with col3:
                    st.metric("期待効果", f"{event_year_data['exp'].iloc[0]:.1f}人")
                with col4:
                    st.metric("その他", f"{event_year_data['other'].iloc[0]:.1f}人")
        
        # テーブル表示
        st.subheader(f"📊 {selected_year}年の{metric_options[selected_metric]}ランキング")
        
        # 表示する列を選択
        display_columns = ["town", "pop", "delta", "exp", "macro", "inertia", "other", "is_event_town"]
        
        # 重心データがある場合は追加
        if "lat" in filtered_df.columns and "lon" in filtered_df.columns:
            display_columns.extend(["lat", "lon"])
        
        # 列名を日本語に変更
        column_mapping = {
            "town": "町丁",
            "pop": "人口",
            "delta": "Δ人口",
            "exp": "期待効果",
            "macro": "マクロ",
            "inertia": "慣性",
            "other": "その他",
            "is_event_town": "イベント発生地",
            "lat": "緯度",
            "lon": "経度"
        }
        
        display_df = filtered_df[display_columns].copy()
        display_df = display_df.rename(columns=column_mapping)
        
        # 数値列を丸める
        numeric_columns = ["人口", "Δ人口", "期待効果", "マクロ", "慣性", "その他"]
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(1)
        
        # イベント発生地をハイライト
        def highlight_event_town(row):
            if row['イベント発生地']:
                return ['background-color: #ffeb3b'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_event_town, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # マップ表示（重心データがある場合）
        if "lat" in filtered_df.columns and "lon" in filtered_df.columns:
            st.subheader("🗺️ 空間的影響の可視化")
            
            # イベント発生町丁とその他の町丁を分ける
            event_town_data = filtered_df[filtered_df['is_event_town'] == True]
            other_towns_data = filtered_df[filtered_df['is_event_town'] == False]
            
            fig_map = go.Figure()
            
            # イベント発生町丁（大きく、赤色で表示）
            if not event_town_data.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=event_town_data['lat'],
                    lon=event_town_data['lon'],
                    mode='markers',
                    marker=dict(
                        size=30,
                        color='red',
                        opacity=0.9,
                        line=dict(width=3, color='darkred')
                    ),
                    text=event_town_data['town'] + '<br>Δ人口: ' + event_town_data['delta'].astype(str) + '<br>【イベント発生地】',
                    hovertemplate='%{text}<extra></extra>',
                    name='イベント発生地'
                ))
            
            # その他の町丁（Δの値に応じて色とサイズを調整）
            if not other_towns_data.empty:
                # 色の設定（Δの符号に基づく）
                colors = ['orange' if x > 0 else 'blue' for x in other_towns_data['delta']]
                sizes = [max(5, min(20, abs(x) / 10)) for x in other_towns_data['delta']]  # サイズを正規化
                
                fig_map.add_trace(go.Scattermapbox(
                    lat=other_towns_data['lat'],
                    lon=other_towns_data['lon'],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    text=other_towns_data['town'] + '<br>Δ人口: ' + other_towns_data['delta'].astype(str),
                    hovertemplate='%{text}<extra></extra>',
                    name='その他の町丁'
                ))
            
            # 地図の中心をイベント発生町丁に設定
            if not event_town_data.empty:
                center_lat = event_town_data['lat'].iloc[0]
                center_lon = event_town_data['lon'].iloc[0]
            else:
                center_lat = filtered_df['lat'].mean()
                center_lon = filtered_df['lon'].mean()
            
            fig_map.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=12
                ),
                height=600,
                title=f"{selected_year}年の空間的影響分布（赤: イベント発生地、オレンジ: 正の影響、青: 負の影響）"
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        
        # ヒストグラム表示
        st.subheader("📈 分布ヒストグラム")
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=filtered_df[selected_metric],
            nbinsx=30,
            name=metric_options[selected_metric],
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig_hist.update_layout(
            title=f"{selected_year}年の{metric_options[selected_metric]}分布",
            xaxis_title=metric_options[selected_metric],
            yaxis_title="町丁数",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # CSVダウンロード
        csv_data = display_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "CSVダウンロード",
            data=csv_data,
            file_name=f"forecast_all_{event_town}_{event_type}_{selected_year}.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True
        )
    
    else:
        st.info("👆 上記の設定で「全地域予測実行」ボタンをクリックして、リアルタイム予測を開始してください。")

else:
    # 単一町丁予測モード（既存のコード）
    st.header("🏘️ 単一町丁予測")

    # サイドバー: シナリオ設定（単一町丁予測モードのみ）
    if view_mode == "単一町丁予測":
        st.sidebar.header("🎯 シナリオ設定")

        # 基本設定
        st.sidebar.subheader("基本設定")
town = st.sidebar.selectbox("町丁", towns, index=0, help="予測対象の町丁を選択")

# イベント設定
st.sidebar.subheader("イベント設定")

# イベントタイプのマッピング（内部キー → 表示名）
# effects_coefficients_rate.csvに存在する11個のイベントのみ
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

# イベントタイプの選択（表示名で選択、内部キーで処理）
event_type_display = st.sidebar.selectbox(
    "イベントタイプ", 
    list(EVENT_TYPE_MAPPING.values()),
    help="人口に影響を与えるイベントの種類を選択"
)

# 表示名から内部キーに変換
event_type_full = [k for k, v in EVENT_TYPE_MAPPING.items() if v == event_type_display][0]

# 内部キーからevent_typeとeffect_directionに分割
if event_type_full.endswith("_inc"):
    event_type = event_type_full[:-4]  # "_inc"を除去
    effect_direction = "increase"
elif event_type_full.endswith("_dec"):
    event_type = event_type_full[:-4]  # "_dec"を除去
    effect_direction = "decrease"
else:
    # フォールバック（通常は発生しない）
    event_type = event_type_full
    effect_direction = "increase"

# 強度設定（学習された強度をデフォルトで使用）
use_learned_intensity = True  # デフォルトで学習された強度を使用

# 手動加算パラメータ（固定値）
st.sidebar.subheader("手動加算パラメータ")
st.sidebar.info("手動加算は0に固定されています（純粋なイベント効果を確認するため）")
h1 = 0.0
h2 = 0.0
h3 = 0.0

# 表示用（読み取り専用）
st.sidebar.text_input("h1 (2026年) 手動加算", value="0.0", disabled=True, help="2026年の手動加算人数")
st.sidebar.text_input("h2 (2027年) 手動加算", value="0.0", disabled=True, help="2027年の手動加算人数")
st.sidebar.text_input("h3 (2028年) 手動加算", value="0.0", disabled=True, help="2028年の手動加算人数")

# 固定パラメータ
st.sidebar.subheader("固定パラメータ")
st.sidebar.info("""
**固定設定:**
- 基準年: 2025
- 予測期間: [1, 2, 3]年先
- 年オフセット: 1年（翌年）
- 信頼度: 1.0
- 強度: 1.0
- ラグ効果: 当年・翌年両方
""")

# メインエリア: 現在のシナリオ状況
st.header("📋 現在のシナリオ")

# シナリオ概要を表示
col1, col2 = st.columns(2)
with col1:
    st.metric("選択町丁", town)
with col2:
    st.metric("イベントタイプ", event_type_display)

# シナリオ詳細
st.subheader("📝 シナリオ詳細")

# イベントタイプの詳細説明
EVENT_DESCRIPTIONS = {
    "housing": {
        "increase": "新規のマンション・アパート・戸建てが供給される（分譲マンション竣工、団地入居、宅地造成後の入居）",
        "decrease": "住宅の解体・用途転用・空き家化などで実質的な供給が減る（一斉解体、老朽化で未利用化、住宅→駐車場転用）"
    },
    "commercial": {
        "increase": "店舗・モールなど商業集積が拡大（大型商業施設開業、スーパー新設、商店集積）",
        "decrease": "商業施設の撤退・閉鎖で集積が減少（店舗閉鎖、モール撤退）"
    },
    "transit": {
        "increase": "新駅・増便・道路整備などでアクセスが改善（新駅開業、バス増便、IC供用）",
        "decrease": "路線撤退・減便等でアクセスが悪化（バス減便、路線廃止）"
    },
    "policy_boundary": {
        "increase": "政策境界の変更により区域が拡大",
        "decrease": "政策境界の変更により区域が縮小"
    },
    "public_edu_medical": {
        "increase": "学校・病院など公共系施設が増える（小中学校新設、病院開設、大学キャンパス誘致）",
        "decrease": "統廃合・閉鎖で公共系施設が減る（学校統廃合、病院閉鎖）"
    },
    "employment": {
        "increase": "新規雇用創出・大規模採用（工場稼働、物流拠点開設、事業拡張）",
        "decrease": "事業所撤退・解雇で雇用が減る（事業所閉鎖、工場撤退）"
    },
    "disaster": {
        "increase": "災害発生や被害拡大により魅力が低下（洪水・地震被害、土砂災害）",
        "decrease": "復旧・治水・耐震化等で被害リスクが下がる（堤防整備、河川改修、耐震化）"
    }
}

# 効果の強さと方向の表示
EFFECT_STRENGTH = {
    "housing": {"increase": "弱", "decrease": "強"},
    "commercial": {"increase": "強", "decrease": "中"},
    "transit": {"increase": "弱", "decrease": "中"},
    "policy_boundary": {"increase": "中", "decrease": "中"},
    "public_edu_medical": {"increase": "なし", "decrease": "なし"},
    "employment": {"increase": "中", "decrease": "中"},
    "disaster": {"increase": "中", "decrease": "中"}
}

scenario_details = {
    "町丁": town,
    "基準年": 2025,
    "予測期間": "1-3年先",
    "イベントタイプ": event_type_display,
    "年オフセット": "1年（翌年）",
    "信頼度": "1.0",
    "強度": "1.0",
    "手動加算": f"h1={h1}人, h2={h2}人, h3={h3}人（固定値）",
    "強度設定": "学習された強度（自動最適化）"
}

for key, value in scenario_details.items():
    st.write(f"**{key}**: {value}")

# イベントの詳細説明を表示
st.subheader("📋 選択されたイベントの詳細")
event_key = f"{event_type}_{effect_direction}"
event_description = EVENT_DESCRIPTIONS[event_type][effect_direction]
effect_strength = EFFECT_STRENGTH[event_type][effect_direction]

col1, col2 = st.columns([2, 1])
with col1:
    st.write(f"**説明**: {event_description}")
with col2:
    st.write(f"**推定効果**: {effect_strength}")

st.markdown("---")

# 予測実行セクション
st.header("📊 予測実行")

# 実行ボタン
if st.button("🚀 予測実行", type="primary", use_container_width=True):
    try:
        # シナリオ作成（年次別強度を使用）
        try:
            # 年次別強度を使用
            generator = LearnedScenarioGenerator()
            scenario = generator.create_learned_scenario_with_yearly_intensity(town, event_type, effect_direction)
            scenario["manual_delta"] = {"h1": h1, "h2": h2, "h3": h3}
            
            # 年次別強度を表示
            st.info(f"🤖 年次別強度が適用されました:")
            for i, event in enumerate(scenario["events"]):
                year_name = ["1年目", "2年目", "3年目"][i]
                st.info(f"  {year_name}: intensity={event['intensity']:.3f}, lag_t={event['lag_t']:.3f}, lag_t1={event['lag_t1']:.3f}")
            
        except Exception as e:
            st.warning(f"⚠️ 年次別強度の取得に失敗しました: {e}。デフォルト強度を使用します。")
            # フォールバック: デフォルト強度
            scenario = {
                "town": town,
                "base_year": 2025,
                "horizons": [1, 2, 3],
                "events": [{
                    "year_offset": 1,
                    "event_type": event_type,
                    "effect_direction": effect_direction,
                    "confidence": 1.0,
                    "intensity": 1.0,
                    "lag_t": 1,
                    "lag_t1": 1,
                    "note": f"{event_type} ({effect_direction})"
                }],
                "macros": {},
                "manual_delta": {"h1": h1, "h2": h2, "h3": h3}
            }
        
        # 予測実行（CLIと同じフロー）
        with st.spinner("予測を実行中..."):
            # 出力ディレクトリの設定（ダッシュボード専用）
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: 将来イベント行列の生成
            st.info("Step 1: 将来イベント行列を生成中...")
            future_events = scenario_to_events(scenario)
            future_events.to_csv(output_dir / "l5_future_events.csv", index=False)
            
            # Layer5の標準パスにも保存
            layer5_events_path = Path("../../data/processed/l5_future_events.csv")
            future_events.to_csv(layer5_events_path, index=False)
            
            # Step 2: 基準年データの準備
            st.info("Step 2: 基準年データを準備中...")
            baseline = prepare_baseline(town, 2025)
            baseline.to_csv(output_dir / "l5_baseline.csv", index=False)
            
            # Layer5の標準パスにも保存
            layer5_baseline_path = Path("../../data/processed/l5_baseline.csv")
            baseline.to_csv(layer5_baseline_path, index=False)
            
            # Step 3: 将来特徴の構築
            st.info("Step 3: 将来特徴を構築中...")
            future_features = build_future_features(baseline, future_events, scenario)
            future_features.to_csv(output_dir / "l5_future_features.csv", index=False)
            
            # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
            layer5_features_path = Path("../../data/processed/l5_future_features.csv")
            future_features.to_csv(layer5_features_path, index=False)
            
            # Step 4: 人口予測の実行
            st.info("Step 4: 人口予測を実行中...")
            base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
            if pd.isna(base_population):
                st.warning("ベース人口が不明のため、0を使用します")
                base_population = 0.0
            
            # 手動加算パラメータを辞書形式で準備
            manual_add = {1: float(h1), 2: float(h2), 3: float(h3)}
            
            result = forecast_population(town, 2025, [1, 2, 3], base_population, str(output_dir), manual_add)
        
        # 結果表示
        st.success("✅ 予測が完了しました！")
        
        # 基本情報
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("町丁", result["town"])
        with col2:
            st.metric("基準年", result["base_year"])
        with col3:
            st.metric("予測期間", f"{min(result['horizons'])}-{max(result['horizons'])}年先")

        st.markdown("---")

        # データフレーム作成
        path_df = pd.DataFrame(result["path"])
        explain = result.get("explain", {})

        # ==== 復元チェック（delta_hat ≍ delta_noexp + exp_people_total） ====
        if explain:
            eps = 1e-6
            _cons_rows = []
            for y in sorted(explain.keys()):
                e = explain[y]
                residual = float(e["delta_hat"]) - (float(e["delta_noexp"]) + float(e["exp_people_total"]))
                _cons_rows.append({"年": y, "Δ復元誤差": residual})
            df_cons = pd.DataFrame(_cons_rows)
            if df_cons["Δ復元誤差"].abs().max() <= eps:
                st.success("✅ 復元チェックOK：Δ = 非イベント成分 + 期待効果（率+手動）")
            else:
                st.warning("⚠️ 復元チェックNG：一部の年で Δ が合成と一致していません（下表を確認）")
                st.dataframe(df_cons, use_container_width=True)

        # ==== 期待効果の内訳テーブル ====
        st.subheader("期待効果の内訳（率→人数換算 + 手動）")
        _rows = []
        for y in sorted(explain.keys()):
            e = explain[y]
            _rows.append({
                "年": y,
                "期待効果（率）": f"{e['exp_rate_terms']*100:.2f}%",
                "母数": float(e["base_pop_for_rate"]),
                "人数換算（率×母数）": float(e["exp_people_from_rate"]),
                "手動人数": float(e["exp_people_manual"]),
                "合計（率+手動）": float(e["exp_people_total"]),
                "非イベント成分": float(e["delta_noexp"]),
                "復元Δ": float(e["delta_hat"]),
            })
        if _rows:
            df_explain = pd.DataFrame(_rows).sort_values("年")
            st.dataframe(df_explain, use_container_width=True)
            # CSVダウンロード（UTF-8-SIG でExcel互換）
            _csv = df_explain.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "内訳CSVをダウンロード",
                data=_csv,
                file_name="explain_summary.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True
            )

        # ==== サマリー（率由来合計/手動合計/合計） ====
        sum_rate_people   = float(sum(float(explain[y]["exp_people_from_rate"]) for y in explain))
        sum_manual_people = float(sum(float(explain[y]["exp_people_manual"])     for y in explain))
        sum_total_exp     = float(sum_rate_people + sum_manual_people)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("率由来合計", f"{sum_rate_people:.1f}人")
        with c2: st.metric("手動合計", f"{sum_manual_people:.1f}人")
        with c3: st.metric("期待効果 合計", f"{sum_total_exp:.1f}人")

        st.markdown("---")

        # 人口予測の折れ線グラフ
        st.subheader("📈 人口予測パス")

        fig_pop = go.Figure()

        # 人口パス（線）
        fig_pop.add_trace(go.Scatter(
            x=path_df["year"],
            y=path_df["pop_hat"],
            mode='lines+markers',
            name='予測人口',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10, color='#1f77b4')
        ))

        # 信頼区間（帯）
        if "pi95_pop" in path_df.columns:
            lower = [p[0] if isinstance(p, list) else p for p in path_df["pi95_pop"]]
            upper = [p[1] if isinstance(p, list) else p for p in path_df["pi95_pop"]]
            
            fig_pop.add_trace(go.Scatter(
                x=path_df["year"].tolist() + path_df["year"].tolist()[::-1],
                y=upper + lower[::-1],
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95%信頼区間',
                showlegend=True
            ))

        fig_pop.update_layout(
            title=f"人口予測パス: {result['town']} (基準年: {result['base_year']})",
            xaxis_title="年",
            yaxis_title="人口（人）",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_pop, use_container_width=True)

        # 人口変化量のグラフ
        st.subheader("📊 人口変化量（Δ人口）")

        fig_delta = go.Figure()

        # ホバーに "率・母数・人数換算・手動" を追加
        custom = []
        for y in path_df["year"]:
            e = explain.get(y, {"exp_rate_terms": 0.0, "base_pop_for_rate": 0.0,
                                "exp_people_from_rate": 0.0, "exp_people_manual": 0.0})
            custom.append([e["exp_rate_terms"], e["base_pop_for_rate"], e["exp_people_from_rate"], e["exp_people_manual"]])

        # Δ人口のバー
        fig_delta.add_trace(go.Bar(
            x=path_df["year"],
            y=path_df["delta_hat"],
            name='Δ人口',
            marker_color=['#ff7f0e' if x > 0 else '#d62728' for x in path_df["delta_hat"]],
            text=[f"{x:+.1f}" for x in path_df["delta_hat"]],
            textposition='auto',
            customdata=custom,
            hovertemplate=(
                "年 %{x}<br>"
                "Δ人数: %{y:.2f}<br>"
                "期待効果(率): %{customdata[0]:.4f}（= %{customdata[0]:.2%}）<br>"
                "母数: %{customdata[1]:.1f}<br>"
                "人数換算: %{customdata[2]:.2f}<br>"
                "手動人数: %{customdata[3]:.2f}<extra></extra>"
            )
        ))

        # 信頼区間
        if "pi95_delta" in path_df.columns:
            lower_delta = [p[0] if isinstance(p, list) else p for p in path_df["pi95_delta"]]
            upper_delta = [p[1] if isinstance(p, list) else p for p in path_df["pi95_delta"]]
            
            fig_delta.add_trace(go.Scatter(
                x=path_df["year"],
                y=upper_delta,
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-up'),
                name='95%信頼区間上限',
                showlegend=True
            ))
            
            fig_delta.add_trace(go.Scatter(
                x=path_df["year"],
                y=lower_delta,
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-down'),
                name='95%信頼区間下限',
                showlegend=True
            ))

        fig_delta.update_layout(
            title="年別人口変化量",
            xaxis_title="年",
            yaxis_title="人口変化量（人）",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig_delta, use_container_width=True)
        st.caption("グラフにマウスオーバーすると「率・母数・人数換算・手動」の内訳が表示されます。")

        # 寄与分解のグラフ
        st.subheader("🥧 寄与分解")

        # 寄与データを準備
        contrib_data = []
        for _, row in path_df.iterrows():
            contrib = row["contrib"]
            contrib_data.append({
                "year": row["year"],
                "exp": contrib.get("exp", 0),
                "macro": contrib.get("macro", 0),
                "inertia": contrib.get("inertia", 0),
                "other": contrib.get("other", 0),
                "delta_hat": row["delta_hat"]
            })

        contrib_df = pd.DataFrame(contrib_data)

        # 寄与分解の積み上げバー
        fig_contrib = go.Figure()

        colors = {
            "exp": "#FF6B6B",      # 赤（期待効果）
            "macro": "#4ECDC4",    # 青緑（マクロ）
            "inertia": "#45B7D1",  # 青（慣性）
            "other": "#96CEB4"     # 緑（その他）
        }

        for col in ["exp", "macro", "inertia", "other"]:
            fig_contrib.add_trace(go.Bar(
                x=contrib_df["year"],
                y=contrib_df[col],
                name=col,
                marker_color=colors[col],
                opacity=0.8
            ))

        fig_contrib.update_layout(
            title="寄与分解（積み上げバー）",
            xaxis_title="年",
            yaxis_title="寄与（人）",
            barmode='relative',
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_contrib, use_container_width=True)

        # 年別寄与分解の円グラフ
        st.subheader("🥧 年別寄与分解（円グラフ）")

        selected_year = st.selectbox("年を選択", path_df["year"].tolist())

        year_data = path_df[path_df["year"] == selected_year].iloc[0]
        contrib = year_data["contrib"]

        # 円グラフ用データ
        labels = []
        values = []
        colors_pie = []

        for key, value in contrib.items():
            if abs(value) > 0.1:  # 0に近い値は除外
                labels.append(key)
                values.append(abs(value))
                colors_pie.append(colors.get(key, "#CCCCCC"))

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors_pie,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>(%{value:.1f}人)'
        )])

        fig_pie.update_layout(
            title=f"寄与分解: {selected_year}年",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # 詳細データテーブル
        st.subheader("📋 詳細データ")

        # 表示用データフレームの準備
        display_df = path_df.copy()
        display_df["人口"] = display_df["pop_hat"].round(1)
        display_df["Δ人口"] = display_df["delta_hat"].round(1)
        display_df["期待効果"] = display_df["contrib"].apply(lambda x: x.get("exp", 0)).round(1)
        display_df["マクロ"] = display_df["contrib"].apply(lambda x: x.get("macro", 0)).round(1)
        display_df["慣性"] = display_df["contrib"].apply(lambda x: x.get("inertia", 0)).round(1)
        display_df["その他"] = display_df["contrib"].apply(lambda x: x.get("other", 0)).round(1)
        
        # デバッグファイルの読み込み（内訳透明化）
        try:
            debug_detail_path = Path(f"output/l5_debug_detail_{town.replace(' ', '_')}.csv")
            if debug_detail_path.exists():
                debug_detail_df = pd.read_csv(debug_detail_path)
                
                # 内訳情報を追加
                if not debug_detail_df.empty:
                    # 列名を確認して適切にマッピング
                    year_col = "year" if "year" in debug_detail_df.columns else "年"
                    
                    # 必要な列が存在するかチェック
                    required_cols = [year_col, "exp_people_from_rate", "exp_people_manual", "exp_people_total"]
                    available_cols = [col for col in required_cols if col in debug_detail_df.columns]
                    
                    if len(available_cols) >= 2:  # 年列と少なくとも1つのデータ列が必要
                        # 年でマージ
                        merged_df = display_df.merge(
                            debug_detail_df[available_cols], 
                            left_on="year", 
                            right_on=year_col, 
                            how="left"
                        )
                    else:
                        st.warning(f"デバッグ詳細ファイルに必要な列が見つかりません。利用可能な列: {list(debug_detail_df.columns)}")
                        merged_df = display_df
                    
                    # 内訳列を追加
                    merged_df["期待効果(率由来)"] = merged_df["exp_people_from_rate"].round(1)
                    merged_df["期待効果(手動)"] = merged_df["exp_people_manual"].round(1)
                    merged_df["期待効果(合計)"] = merged_df["exp_people_total"].round(1)
                    
                    display_df = merged_df
                    
                    # 内訳の表示
                    st.subheader("🔍 期待効果の内訳")
                    st.info("期待効果を「率由来」と「手動人数」に分けて表示")
                    
                    # 内訳のグラフ
                    fig_breakdown = go.Figure()
                    
                    # 率由来の期待効果
                    fig_breakdown.add_trace(go.Bar(
                        x=merged_df["year"],
                        y=merged_df["exp_people_from_rate"],
                        name='期待効果(率由来)',
                        marker_color='#FF6B6B',
                        opacity=0.8
                    ))
                    
                    # 手動の期待効果
                    fig_breakdown.add_trace(go.Bar(
                        x=merged_df["year"],
                        y=merged_df["exp_people_manual"],
                        name='期待効果(手動)',
                        marker_color='#4ECDC4',
                        opacity=0.8
                    ))
                    
                    fig_breakdown.update_layout(
                        title="期待効果の内訳",
                        xaxis_title="年",
                        yaxis_title="寄与（人）",
                        barmode='stack',
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_breakdown, use_container_width=True)
                    
        except Exception as e:
            st.warning(f"デバッグ詳細ファイルの読み込みに失敗しました: {e}")

        # 信頼区間の表示
        if "pi95_pop" in display_df.columns:
            display_df["人口95%CI"] = display_df["pi95_pop"].apply(
                lambda x: f"[{x[0]:.1f}, {x[1]:.1f}]" if isinstance(x, list) else f"[{x:.1f}, {x:.1f}]"
            )

        if "pi95_delta" in display_df.columns:
            display_df["Δ人口95%CI"] = display_df["pi95_delta"].apply(
                lambda x: f"[{x[0]:.1f}, {x[1]:.1f}]" if isinstance(x, list) else f"[{x:.1f}, {x:.1f}]"
            )

        # 表示する列を選択
        display_columns = ["年", "人口", "Δ人口", "期待効果", "マクロ", "慣性", "その他"]
        
        # 内訳列があれば追加
        if "期待効果(率由来)" in display_df.columns:
            display_columns.extend(["期待効果(率由来)", "期待効果(手動)", "期待効果(合計)"])
        
        if "人口95%CI" in display_df.columns:
            display_columns.append("人口95%CI")
        if "Δ人口95%CI" in display_df.columns:
            display_columns.append("Δ人口95%CI")

        # 列名を日本語に変更
        display_df = display_df.rename(columns={
            "year": "年",
            "人口": "人口",
            "Δ人口": "Δ人口",
            "期待効果": "期待効果",
            "マクロ": "マクロ",
            "慣性": "慣性",
            "その他": "その他"
        })

        st.dataframe(display_df[display_columns], use_container_width=True)

        # サマリー統計
        st.subheader("📊 サマリー統計")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            final_pop = path_df["pop_hat"].iloc[-1]
            initial_pop = path_df["pop_hat"].iloc[0]
            total_change = final_pop - initial_pop
            st.metric(
                "総人口変化",
                f"{total_change:.1f}人",
                f"{initial_pop:.1f} → {final_pop:.1f}"
            )

        with col2:
            avg_delta = path_df["delta_hat"].mean()
            st.metric(
                "平均年次変化",
                f"{avg_delta:.1f}人/年"
            )

        with col3:
            max_exp = path_df["contrib"].apply(lambda x: x.get("exp", 0)).max()
            st.metric(
                "最大期待効果",
                f"{max_exp:.1f}人"
            )

        with col4:
            # explain機能から期待効果合計を取得
            if explain:
                total_exp = sum(explain[y]["exp_people_total"] for y in explain)
            else:
                total_exp = path_df["contrib"].apply(lambda x: x.get("exp", 0)).sum()
            st.metric(
                "期待効果合計",
                f"{total_exp:.1f}人"
            )
        
        # 内訳サマリー（explain機能から取得）
        if explain:
            st.subheader("🔍 期待効果内訳サマリー")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_rate = sum(explain[y]["exp_people_from_rate"] for y in explain)
                st.metric(
                    "率由来合計",
                    f"{total_rate:.1f}人"
                )
            
            with col2:
                total_manual = sum(explain[y]["exp_people_manual"] for y in explain)
                st.metric(
                    "手動合計",
                    f"{total_manual:.1f}人"
                )
            
            with col3:
                total_combined = sum(explain[y]["exp_people_total"] for y in explain)
                st.metric(
                    "合計",
                    f"{total_combined:.1f}人"
                )

        # ==== Debug: explainの生JSON ====
        with st.expander("Debug: raw explain JSON（開発用）", expanded=False):
            if st.checkbox("表示する（開発時のみ）", value=False):
                # 年キーをソートし、サイズを抑えた形で出す
                _mini = {int(y): {
                    "exp_rate_terms": float(explain[y]["exp_rate_terms"]),
                    "base_pop_for_rate": float(explain[y]["base_pop_for_rate"]),
                    "exp_people_from_rate": float(explain[y]["exp_people_from_rate"]),
                    "exp_people_manual": float(explain[y]["exp_people_manual"]),
                    "exp_people_total": float(explain[y]["exp_people_total"]),
                    "delta_noexp": float(explain[y]["delta_noexp"]),
                    "delta_hat": float(explain[y]["delta_hat"]),
                } for y in sorted(explain.keys())}
                st.json(_mini)

    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        st.exception(e)

# 計算式の説明
with st.expander("📐 計算式の説明", expanded=False):
    st.markdown("""
    **期待効果の計算式:**
    - **期待効果（率）** = イベント由来の率寄与の合計
    - **人数換算** = 期待効果（率） × 母数（通常は前年人口）
    - **合計（率+手動）** = 人数換算 + 手動人数
    - **復元Δ** = 非イベント成分 + 合計（率+手動）
    
    **復元チェック:**
    - Δ人口 = 非イベント成分 + 期待効果（率+手動）
    - この等式が成立することを年ごとに検証
    """)

# ヘルプ
with st.expander("❓ ヘルプ"):
    st.markdown("""
    ### 使用方法
    
    1. **町丁選択**: サイドバーで予測対象の町丁を選択
    2. **イベント設定**: イベントタイプを選択（増加・減少の方向は既に含まれています）
    3. **予測実行**: 「予測実行」ボタンをクリック
    
    ### イベントタイプの説明（11種類）
    
    - **住宅供給の増加（竣工）**: 新規のマンション・アパート・戸建てが供給される（分譲マンション竣工、団地入居、宅地造成後の入居）
    - **住宅の減少・喪失**: 住宅の解体・用途転用・空き家化などで実質的な供給が減る（一斉解体、老朽化で未利用化、住宅→駐車場転用）
    - **商業施設の増加**: 店舗・モールなど商業集積が拡大（大型商業施設開業、スーパー新設、商店集積）
    - **交通利便の向上**: 新駅・増便・道路整備などでアクセスが改善（新駅開業、バス増便、IC供用）
    - **交通利便の低下**: 路線撤退・減便等でアクセスが悪化（バス減便、路線廃止）
    - **公共・教育・医療の増加**: 学校・病院など公共系施設が増える（小中学校新設、病院開設、大学キャンパス誘致）
    - **公共・教育・医療の減少**: 統廃合・閉鎖で公共系施設が減る（学校統廃合、病院閉鎖）
    - **雇用機会の増加**: 新規雇用創出・大規模採用（工場稼働、物流拠点開設、事業拡張）
    - **雇用機会の減少**: 事業所撤退・解雇で雇用が減る（事業所閉鎖、工場撤退）
    - **災害被害・リスクの増加**: 災害発生や被害拡大により魅力が低下（洪水・地震被害、土砂災害）
    - **災害リスクの低下（防災整備）**: 復旧・治水・耐震化等で被害リスクが下がる（堤防整備、河川改修、耐震化）
    
    
    ### 固定パラメータ
    
    - 基準年: 2025年
    - 予測期間: 1-3年先
    - 年オフセット: 1年（翌年）
    - 信頼度: 1.0
    - 強度: 機械学習で自動最適化
    - 手動加算: 0に固定（純粋なイベント効果を確認するため）
    """)

# フッター
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2023 地域科学研究所")
