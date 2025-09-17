# -*- coding: utf-8 -*-
# src/layer4/improved/run_causal_learning.py
"""
因果関係学習の実行スクリプト
- 特徴量構築 → モデル訓練 → 性能比較 を一括実行

使用方法:
    python run_causal_learning.py

実行手順:
1. 因果関係学習用特徴量を構築
2. 因果関係学習モデルを訓練
3. 従来モデルとの性能比較
"""
import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description):
    """スクリプトを実行"""
    print(f"\n{'='*60}")
    print(f"実行中: {description}")
    print(f"スクリプト: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print("✅ 実行成功")
        if result.stdout:
            print("出力:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 実行失敗")
        print(f"エラーコード: {e.returncode}")
        if e.stdout:
            print("標準出力:")
            print(e.stdout)
        if e.stderr:
            print("標準エラー:")
            print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False

def check_dependencies():
    """依存関係をチェック"""
    print("依存関係をチェック中...")
    
    required_files = [
        "../../../data/processed/features_panel.csv",
        "../../../data/processed/events_matrix_signed.csv",
        "../../../data/processed/town_centroids.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 必要なファイルが見つかりません:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ 依存関係チェック完了")
    return True

def main():
    """メイン処理"""
    print("因果関係学習システムを開始します...")
    print("=" * 60)
    
    # 依存関係チェック
    if not check_dependencies():
        print("\n❌ 依存関係チェックに失敗しました。必要なファイルを準備してください。")
        return
    
    # 実行ステップ
    steps = [
        ("build_causal_features.py", "因果関係学習用特徴量の構築"),
        ("train_causal_lgbm.py", "因果関係学習モデルの訓練"),
        ("compare_causal_vs_traditional.py", "従来モデルとの性能比較")
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for script_name, description in steps:
        if run_script(script_name, description):
            success_count += 1
        else:
            print(f"\n❌ ステップ '{description}' でエラーが発生しました。")
            print("処理を中断します。")
            break
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("実行結果サマリー")
    print(f"{'='*60}")
    print(f"成功ステップ: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 全てのステップが正常に完了しました！")
        print("\n生成されたファイル:")
        print("  - ../../../data/processed/features_causal.csv (因果関係学習用特徴量)")
        print("  - ../../../models/l4_causal_model.joblib (因果関係学習モデル)")
        print("  - ../../../data/processed/l4_causal_predictions.csv (予測結果)")
        print("  - ../../../data/processed/l4_causal_metrics.json (メトリクス)")
        print("  - ../../../data/processed/l4_causal_feature_importance.csv (特徴量重要度)")
        print("  - causal_vs_traditional_comparison.json (比較結果)")
        print("  - causal_vs_traditional_comparison.png (比較プロット)")
        print("  - causal_feature_list.json (特徴量リスト)")
        
        print("\n次のステップ:")
        print("1. 比較結果を確認して因果関係学習の効果を評価")
        print("2. 特徴量重要度を分析して重要な因果関係を特定")
        print("3. 必要に応じて特徴量エンジニアリングを改善")
        
    else:
        print("❌ 一部のステップでエラーが発生しました。")
        print("エラーメッセージを確認して問題を解決してください。")

if __name__ == "__main__":
    main()
