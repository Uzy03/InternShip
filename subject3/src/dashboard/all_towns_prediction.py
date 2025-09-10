# -*- coding: utf-8 -*-
"""
全地域予測機能
イベント発生町丁を指定して全町丁の人口予測を実行
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
    from scenario_to_events import scenario_to_events
    from prepare_baseline import prepare_baseline
    from build_future_features import build_future_features
    from forecast_service import forecast_population, run_scenario
    from scenario_with_learned_intensity import LearnedScenarioGenerator
except ImportError as e:
    st.error(f"Layer5モジュールのインポートに失敗しました: {e}")
    st.stop()

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
        
        # デバッグ用：最初の10町丁のみでテスト
        debug_towns = all_towns[:11]  # イベント発生町丁 + 10町丁
        if len(debug_towns) < len(all_towns):
            st.warning(f"デバッグモード: 最初の{len(debug_towns)}町丁のみでテストします（全{len(all_towns)}町丁中）")
        
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
        
        # 全町丁の将来特徴を一括構築（空間ラグ効果を計算するため）
        st.info("全町丁の将来特徴を一括構築中...")
        all_future_features = []
        
        try:
            # イベント発生町丁の将来特徴を構築
            st.info(f"イベント発生町丁 '{event_town}' の将来特徴を構築中...")
            event_baseline = prepare_baseline(event_town, base_year)
            event_future_events = scenario_to_events(scenario)
            event_future_features = build_future_features(event_baseline, event_future_events, scenario)
            all_future_features.append(event_future_features)
            st.success(f"イベント発生町丁 '{event_town}' の将来特徴を構築完了: {len(event_future_features)}行")
            
            # その他の町丁の将来特徴を構築（デバッグ用に制限）
            test_towns = [town for town in debug_towns if town != event_town]
            
            for town in test_towns:
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
                    
                    individual_baseline = prepare_baseline(town, base_year)
                    individual_future_events = scenario_to_events(individual_scenario)
                    individual_future_features = build_future_features(individual_baseline, individual_future_events, individual_scenario)
                    all_future_features.append(individual_future_features)
                    
                except Exception as e:
                    st.warning(f"町丁 '{town}' の将来特徴構築に失敗: {e}")
                    continue
            
            # 全町丁の将来特徴を結合
            if all_future_features:
                combined_future_features = pd.concat(all_future_features, ignore_index=True)
                features_path = Path("../../data/processed/l5_future_features.csv")
                combined_future_features.to_csv(features_path, index=False)
                st.success(f"全町丁の将来特徴を構築完了: {len(combined_future_features)}行")
                
                # 保存されたファイルの確認
                saved_features = pd.read_csv(features_path)
                st.info(f"保存された将来特徴ファイル: {len(saved_features)}行, 列: {list(saved_features.columns)}")
                event_town_data = saved_features[saved_features["town"] == event_town]
                st.info(f"イベント町丁 '{event_town}' のデータ: {len(event_town_data)}行")
            else:
                st.error("将来特徴の構築に失敗")
                
        except Exception as e:
            st.error(f"将来特徴の一括構築に失敗: {e}")
            import traceback
            st.error(f"詳細エラー: {traceback.format_exc()}")
        
        # 各町丁の予測実行
        for i, town in enumerate(debug_towns):
            # プログレスバーの更新
            if i % 5 == 0 or i == len(debug_towns) - 1:
                status_text.text(f"処理中: {town} ({i+1}/{len(debug_towns)})")
                progress_bar.progress((i + 1) / len(debug_towns))
            
            # ベース人口を取得
            town_baseline = baseline_df[baseline_df["town"] == town]
            if not town_baseline.empty and "pop_total" in town_baseline.columns:
                base_population = float(town_baseline["pop_total"].iloc[0])
            else:
                base_population = 0.0
            
            try:
                # 予測実行（将来特徴は既に構築済み）
                apply_event = (town == event_town)
                
                result = forecast_population(
                    town=town,
                    base_year=base_year,
                    horizons=horizons,
                    base_population=base_population,
                    debug_output_dir=None,
                    manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                    apply_event_to_prediction=apply_event
                )
                
                if i < 3:  # 最初の3町丁のみデバッグ情報を表示
                    st.info(f"町丁 '{town}' の予測完了: {result is not None} (イベント: {apply_event})")
                    if result:
                        st.info(f"結果キー: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
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
                if i < 3:  # 最初の3町丁のみデバッグ情報を表示
                    st.info(f"町丁 '{town}' の結果を処理中: {len(result['path'])}件のエントリ")
                
                for entry in result["path"]:
                    # 率由来合計を計算（exp + macro + inertia + other）
                    rate_total = entry["contrib"]["exp"] + entry["contrib"]["macro"] + entry["contrib"]["inertia"] + entry["contrib"]["other"]
                    
                    row = {
                        "town": result["town"],
                        "baseline_year": result["base_year"],
                        "year": entry["year"],
                        "h": entry["year"] - result["base_year"],
                        "delta": entry["delta_hat"],
                        "pop": entry["pop_hat"],
                        "exp": entry["contrib"]["exp"],
                        "macro": entry["contrib"]["macro"],
                        "inertia": entry["contrib"]["inertia"],
                        "other": entry["contrib"]["other"],
                        "rate_total": rate_total,  # 率由来合計を追加
                        "pi_delta_low": entry["pi95_delta"][0] if isinstance(entry["pi95_delta"], list) else entry["pi95_delta"],
                        "pi_delta_high": entry["pi95_delta"][1] if isinstance(entry["pi95_delta"], list) else entry["pi95_delta"],
                        "pi_pop_low": entry["pi95_pop"][0] if isinstance(entry["pi95_pop"], list) else entry["pi95_pop"],
                        "pi_pop_high": entry["pi95_pop"][1] if isinstance(entry["pi95_pop"], list) else entry["pi95_pop"],
                        "is_event_town": (town == event_town),  # イベント発生町丁かどうか
                    }
                    all_results.append(row)
                
                if i < 3:  # 最初の3町丁のみデバッグ情報を表示
                    st.info(f"町丁 '{town}' の処理完了: {len(all_results)}件の総結果")
                
            except Exception as e:
                if i < 3:  # 最初の3町丁のみデバッグ情報を表示
                    st.error(f"町丁 '{town}' の予測でエラー: {e}")
                    import traceback
                    st.error(f"詳細エラー: {traceback.format_exc()}")
                
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
        
        # プログレスバーをクリア
        progress_bar.empty()
        status_text.empty()
        
        # デバッグ情報を表示
        st.info(f"処理完了: {len(all_results)}件の結果を取得")
        if len(all_results) > 0:
            st.info(f"最初の結果: {all_results[0]}")
        
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

def render_all_towns_prediction(towns):
    """全地域予測のUIとロジックをレンダリング"""
    
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
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("人口", f"{event_year_data['pop'].iloc[0]:.1f}人")
                with col2:
                    st.metric("Δ人口", f"{event_year_data['delta'].iloc[0]:.1f}人")
                with col3:
                    st.metric("期待効果", f"{event_year_data['exp'].iloc[0]:.1f}人")
                with col4:
                    st.metric("その他", f"{event_year_data['other'].iloc[0]:.1f}人")
                with col5:
                    st.metric("率由来合計", f"{event_year_data['rate_total'].iloc[0]:.1f}人")
        
        # テーブル表示
        st.subheader(f"📊 {selected_year}年の{metric_options[selected_metric]}ランキング")
        
        # 表示する列を選択
        display_columns = ["town", "pop", "delta", "exp", "macro", "inertia", "other", "rate_total", "is_event_town"]
        
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
            "rate_total": "率由来合計",
            "is_event_town": "イベント発生地",
            "lat": "緯度",
            "lon": "経度"
        }
        
        display_df = filtered_df[display_columns].copy()
        display_df = display_df.rename(columns=column_mapping)
        
        # 数値列を丸める
        numeric_columns = ["人口", "Δ人口", "期待効果", "マクロ", "慣性", "その他", "率由来合計"]
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
