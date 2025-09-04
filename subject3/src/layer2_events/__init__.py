"""
Layer2 Events Module

イベントラベル付けシステム
- manual_investigation_targets.csvからイベント情報を抽出・正規化
- events_labeled.csv（ロング形式）とevents_matrix.csv（ワイド形式）を生成
"""

from .normalize_events import normalize_events

__all__ = ['normalize_events']
