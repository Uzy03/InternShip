#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人口変化原因解明フレームワーク（ハイブリッド設計）
実務で安定して動作する原因特定システム
"""

import pandas as pd
import requests
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib

@dataclass
class Event:
    """事象情報のデータクラス"""
    town: str
    year: int
    event_type: str
    confidence: float
    title: str
    url: str
    publish_date: Optional[str]
    source: str
    snippet: str
    keywords_hit: List[str]
    time_gap: int
    geo_match: float
    population_change: float
    change_type: str

class PopulationCauseInvestigator:
    def __init__(self, data_path="../subject2/processed_data/classified_population_changes.csv"):
        self.data_path = data_path
        self.data = None
        self.target_changes = None
        self.output_dir = "cause_investigation_results"
        
        # 調査対象の分類
        self.target_categories = ['激増', '大幅増加', '大幅減少']
        
        # 事象タイプの定義
        self.event_types = {
            '住宅': ['マンション', '分譲', '共同住宅', '戸建', '団地', '竣工', '入居開始', '完了検査'],
            '商業': ['ショッピングモール', '商業施設', 'SC', '開業', 'リニューアル', '店舗', 'オフィスビル'],
            '公共': ['市民病院', '小学校', '大学', 'キャンパス', '庁舎', '図書館', '体育館', '移転開設'],
            '教育': ['学校', '大学', 'キャンパス', '移転', '新設', '統合'],
            '医療': ['病院', 'クリニック', '診療所', '開設', '移転', '増床'],
            '交通': ['駅前', '新駅', '延伸', '高架化', 'IC', 'バイパス', '開通', '整備'],
            '企業': ['工場', '物流センター', '操業', '竣工', '稼働開始', '誘致', '進出'],
            '災害': ['地震', '震災', '豪雨', '土砂', '浸水', '被災', '仮設', '解体'],
            '政策': ['再開発', '区画整理', '都市計画', '市街地整備', '土地区画整理'],
            '統計': ['住所表示変更', '住居表示', '区制施行', '町名変更', '編入', '合併']
        }
        
        # 検索キーワード（主要語+補助語）
        self.search_keywords = {
            '主要語': ['竣工', '完成', '開業', '建設', '再開発', '分譲', 'マンション', '工場', '物流', 'キャンパス', '病院', 'インター', 'BRT', '地震', '災害'],
            '補助語': ['入居', '移転', '新設', '増床', '統合', '誘致', '進出', '整備', '開通', '被災', '仮設', '解体']
        }
        
        # ソース信頼度スコア
        self.source_credibility = {
            '自治体': 3, '官報': 3, '政府': 3,
            '新聞': 2, 'テレビ': 2, 'ラジオ': 2,
            '企業': 1, 'ブログ': 0, 'SNS': 0
        }
        
        # キャッシュ管理
        self.cache = {}
        self.cache_file = "search_cache.json"
        self.load_cache()
        
        # 既知の原因データベース
        self.known_causes = {
            '慶徳堀町_2013': {
                'event_type': '住宅',
                'title': '分譲マンション「エイルマンション慶徳イクシオ」竣工',
                'confidence': 0.95,
                'source': 'エイルの仲介情報',
                'keywords_hit': ['竣工', 'マンション', '分譲'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '魚屋町１丁目_2020': {
                'event_type': '住宅',
                'title': '「（仮称）魚屋町・紺屋町マンション新築工事」完成',
                'confidence': 0.90,
                'source': '市のCASBEE完了届PDF',
                'keywords_hit': ['完成', 'マンション', '新築工事'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '秋津新町_2019': {
                'event_type': '住宅',
                'title': '「秋津新町マンション 新築工事」完成',
                'confidence': 0.90,
                'source': '完了届記録',
                'keywords_hit': ['完成', 'マンション', '新築工事'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '鍛冶屋町_2014': {
                'event_type': '住宅',
                'title': '大型分譲「グランドオーク唐人町通り」竣工',
                'confidence': 0.90,
                'source': 'エイルの仲介、LIFULL HOME\'S',
                'keywords_hit': ['竣工', '分譲', '大型'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '河原町_2006': {
                'event_type': '住宅',
                'title': '「エバーライフ熊本中央」竣工',
                'confidence': 0.90,
                'source': 'エイルの仲介、LIFULL HOME\'S',
                'keywords_hit': ['竣工', 'エバーライフ'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '千葉城町_2013': {
                'event_type': '住宅',
                'title': '「アンピール熊本城」完成',
                'confidence': 0.90,
                'source': 'LIFULL HOME\'S、熊本市公式サイト',
                'keywords_hit': ['完成', 'アンピール'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '本山３丁目_2009': {
                'event_type': '住宅',
                'title': '熊本駅東側の大規模マンション群完成',
                'confidence': 0.90,
                'source': 'LIFULL HOME\'S、エイルの仲介',
                'keywords_hit': ['完成', 'マンション群', '熊本駅東側'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '春日３丁目_2013': {
                'event_type': '政策',
                'title': '熊本駅周辺再開発の進行',
                'confidence': 0.85,
                'source': '熊本市公式サイト',
                'keywords_hit': ['再開発', '熊本駅周辺'],
                'time_gap': 0,
                'geo_match': 1.0
            },
            '近見１丁目_2001': {
                'event_type': '住宅',
                'title': '集合住宅「プラーノウエスト」など竣工',
                'confidence': 0.90,
                'source': 'ホームメイト',
                'keywords_hit': ['竣工', '集合住宅', 'プラーノウエスト'],
                'time_gap': 0,
                'geo_match': 1.0
            }
        }
    
    def load_cache(self):
        """検索キャッシュを読み込み"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"キャッシュ読み込み完了: {len(self.cache)}件")
            except:
                self.cache = {}
    
    def save_cache(self):
        """検索キャッシュを保存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        print(f"キャッシュ保存完了: {len(self.cache_file)}件")
    
    def load_data(self):
        """分類済み人口変化データを読み込み"""
        print("データの読み込み中...")
        try:
            self.data = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"データ読み込み完了: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません: {self.data_path}")
            return False
    
    def filter_target_changes(self):
        """調査対象の人口変化を抽出"""
        print("調査対象の抽出中...")
        
        if self.data is None:
            print("エラー: データが読み込まれていません")
            return False
        
        # 激増・大幅増加・大幅減少のみを抽出
        self.target_changes = self.data[
            self.data['変化タイプ'].isin(self.target_categories)
        ].copy()
        
        # 変化の大きさでソート
        self.target_changes = self.target_changes.sort_values('変化の大きさ', ascending=False)
        
        print(f"調査対象: {len(self.target_changes)}件")
        return True
    
    def normalize_town_name(self, town_name: str) -> str:
        """町丁名の正規化"""
        # 全角・半角の統一
        normalized = town_name.replace('　', ' ').strip()
        
        # 区名のカッコ除去（例：春日３丁目 → 春日３丁目）
        normalized = re.sub(r'（.*?）', '', normalized)
        normalized = re.sub(r'\(.*?\)', '', normalized)
        
        return normalized
    
    def build_search_query(self, town: str, year: int) -> str:
        """検索クエリを構築"""
        normalized_town = self.normalize_town_name(town)
        
        # 年幅を設定（前後1年）
        year_range = f"{year-1} OR {year} OR {year+1}"
        
        # 主要キーワードを組み合わせ
        main_keywords = ' OR '.join(self.search_keywords['主要語'])
        
        # クエリ構築
        query = f'"{normalized_town}" "熊本市" ({year_range}) ({main_keywords})'
        
        return query
    
    def search_bing_api(self, query: str, max_results: int = 10) -> List[Dict]:
        """Bing検索APIを使用（実際のAPIキーが必要）"""
        # 注意: 実際の使用時はBing Search APIのキーが必要
        # ここではモック実装
        
        print(f"  検索クエリ: {query}")
        
        # モック検索結果（実際のAPI使用時は削除）
        mock_results = self.generate_mock_results(query, max_results)
        
        # キャッシュに保存
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.cache[query_hash] = {
            'query': query,
            'results': mock_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return mock_results
    
    def generate_mock_results(self, query: str, max_results: int) -> List[Dict]:
        """モック検索結果を生成（開発・テスト用）"""
        # 実際のAPI使用時は削除
        town_match = re.search(r'"([^"]+)"', query)
        year_match = re.search(r'(\d{4})', query)
        
        if not town_match or not year_match:
            return []
        
        town = town_match.group(1)
        year = int(year_match.group(1))
        
        # 町丁名に応じたモック結果
        mock_data = {
            '慶徳堀町': [
                {
                    'title': f'{town}マンション竣工 - 熊本市',
                    'url': 'https://example.com/keitoku-mansion',
                    'snippet': f'{year}年に{town}で分譲マンションが竣工し、入居が開始されました。',
                    'date': f'{year}-02-15'
                }
            ],
            '魚屋町１丁目': [
                {
                    'title': f'{town}商業施設完成',
                    'url': 'https://example.com/uoya-commercial',
                    'snippet': f'{year}年に{town}で商業施設が完成し、開業しました。',
                    'date': f'{year}-03-20'
                }
            ]
        }
        
        return mock_data.get(town, [])
    
    def analyze_event_type(self, title: str, snippet: str) -> Tuple[str, List[str]]:
        """事象タイプを分析"""
        text = (title + ' ' + snippet).lower()
        event_scores = {}
        keywords_hit = []
        
        for event_type, keywords in self.event_types.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text:
                    if keyword in self.search_keywords['主要語']:
                        score += 3
                        keywords_hit.append(keyword)
                    else:
                        score += 1
                        keywords_hit.append(keyword)
            
            if score > 0:
                event_scores[event_type] = score
        
        # 最高スコアの事象タイプを選択
        if event_scores:
            best_type = max(event_scores, key=event_scores.get)
            return best_type, keywords_hit
        
        return 'その他', []
    
    def calculate_confidence(self, event: Event) -> float:
        """信頼度スコアを計算"""
        score = 0
        
        # キーワード一致点
        score += len(event.keywords_hit) * 3
        
        # 年の近さ
        if event.time_gap == 0:
            score += 3
        elif abs(event.time_gap) == 1:
            score += 2
        elif abs(event.time_gap) == 2:
            score += 1
        
        # 地名一致度
        score += event.geo_match * 3
        
        # ソース信頼度
        source_score = 0
        for source_type, credibility in self.source_credibility.items():
            if source_type in event.source:
                source_score = max(source_score, credibility)
        score += source_score
        
        # 0-1に正規化
        max_possible_score = 20  # 最大スコアの推定
        confidence = min(score / max_possible_score, 1.0)
        
        return confidence
    
    def investigate_town(self, town: str, year: int, change_type: str, change_rate: float) -> List[Event]:
        """特定の町丁を調査"""
        print(f"調査中: {town} ({year}年) - {change_type}")
        
        events = []
        
        # 1. 既知の原因データベースをチェック
        key = f"{town}_{year}"
        if key in self.known_causes:
            known_cause = self.known_causes[key]
            event = Event(
                town=town,
                year=year,
                event_type=known_cause['event_type'],
                confidence=known_cause['confidence'],
                title=known_cause['title'],
                url='',
                publish_date=None,
                source=known_cause['source'],
                snippet=known_cause['title'],
                keywords_hit=known_cause['keywords_hit'],
                time_gap=known_cause['time_gap'],
                geo_match=known_cause['geo_match'],
                population_change=change_rate,
                change_type=change_type
            )
            events.append(event)
            print(f"  ✓ 既知の原因: {known_cause['title']}")
            return events
        
        # 2. 検索による調査
        print(f"  検索による調査を実行中...")
        
        # 検索クエリを構築
        query = self.build_search_query(town, year)
        
        # 検索実行
        search_results = self.search_bing_api(query)
        
        # 検索結果を分析
        for result in search_results:
            # 事象タイプを分析
            event_type, keywords_hit = self.analyze_event_type(
                result['title'], result['snippet']
            )
            
            # 時間差を計算
            try:
                publish_date = datetime.strptime(result['date'], '%Y-%m-%d')
                time_gap = publish_date.year - year
            except:
                time_gap = 0
            
            # 地名一致度を計算
            geo_match = self.calculate_geo_match(town, result['title'], result['snippet'])
            
            # イベントオブジェクトを作成
            event = Event(
                town=town,
                year=year,
                event_type=event_type,
                confidence=0.0,  # 後で計算
                title=result['title'],
                url=result['url'],
                publish_date=result.get('date'),
                source=self.classify_source(result['url']),
                snippet=result['snippet'],
                keywords_hit=keywords_hit,
                time_gap=time_gap,
                geo_match=geo_match,
                population_change=change_rate,
                change_type=change_type
            )
            
            # 信頼度を計算
            event.confidence = self.calculate_confidence(event)
            events.append(event)
        
        return events
    
    def calculate_geo_match(self, town: str, title: str, snippet: str) -> float:
        """地名一致度を計算"""
        text = (title + ' ' + snippet).lower()
        town_lower = town.lower()
        
        # 完全一致
        if town_lower in text:
            return 1.0
        
        # 部分一致（町丁名の一部が含まれる）
        town_parts = town_lower.replace('丁目', '').replace('町', '').replace('目', '')
        if town_parts in text:
            return 0.7
        
        # 熊本市関連
        if '熊本' in text:
            return 0.3
        
        return 0.0
    
    def classify_source(self, url: str) -> str:
        """URLからソースタイプを分類"""
        domain = urlparse(url).netloc.lower()
        
        if any(gov in domain for gov in ['gov', 'city', 'pref']):
            return '自治体'
        elif any(news in domain for news in ['news', 'nhk', 'asahi', 'yomiuri']):
            return '新聞'
        elif any(company in domain for company in ['company', 'corp', 'inc']):
            return '企業'
        elif any(blog in domain for blog in ['blog', 'note', 'hatena']):
            return 'ブログ'
        else:
            return 'その他'
    
    def run_investigation(self, max_targets: int = None):
        """調査を実行"""
        print("=== 人口変化原因解明調査開始 ===")
        
        # 1. データ読み込み
        if not self.load_data():
            return None
        
        # 2. 調査対象の抽出
        if not self.filter_target_changes():
            return None
        
        # 3. 調査対象の制限（全件調査の場合は制限なし）
        if max_targets and len(self.target_changes) > max_targets:
            print(f"調査対象を上位{max_targets}件に制限")
            self.target_changes = self.target_changes.head(max_targets)
        else:
            print(f"全件調査を実行: {len(self.target_changes)}件")
        
        # 4. 各対象の調査
        all_events = []
        
        for idx, (_, row) in enumerate(self.target_changes.iterrows(), 1):
            print(f"\n[{idx}/{len(self.target_changes)}] 調査中...")
            
            events = self.investigate_town(
                row['町丁名'],
                row['年度'],
                row['変化タイプ'],
                row['変化率(%)']
            )
            
            all_events.extend(events)
            
            # 進捗表示
            if events:
                event_types = [e.event_type for e in events]
                print(f"  完了: {row['町丁名']} ({row['年度']}年) - 事象数: {len(events)}件 - タイプ: {', '.join(event_types)}")
            else:
                print(f"  完了: {row['町丁名']} ({row['年度']}年) - 事象数: 0件")
            
            # 100件ごとに進捗サマリー
            if idx % 100 == 0:
                print(f"\n--- 進捗サマリー ({idx}/{len(self.target_changes)}) ---")
                print(f"  発見済み事象: {len(all_events)}件")
        
        # 5. 結果の整理・重複除去
        unique_events = self.deduplicate_events(all_events)
        
        # 6. 結果保存
        self.save_results(unique_events)
        
        # 7. キャッシュ保存
        self.save_cache()
        
        print("\n=== 人口変化原因解明調査完了 ===")
        return unique_events
    
    def deduplicate_events(self, events: List[Event]) -> List[Event]:
        """重複イベントを除去・統合"""
        # URLベースで重複を除去
        url_groups = {}
        
        for event in events:
            if event.url in url_groups:
                # 既存のイベントと統合
                existing = url_groups[event.url]
                existing.keywords_hit = list(set(existing.keywords_hit + event.keywords_hit))
                existing.confidence = max(existing.confidence, event.confidence)
            else:
                url_groups[event.url] = event
        
        return list(url_groups.values())
    
    def save_results(self, events: List[Event]):
        """結果を保存"""
        print("結果の保存中...")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # イベントを辞書に変換
        events_data = []
        for event in events:
            event_dict = {
                '町丁名': event.town,
                '年度': event.year,
                '事象タイプ': event.event_type,
                '信頼度': event.confidence,
                'タイトル': event.title,
                'URL': event.url,
                '公開日': event.publish_date,
                'ソース': event.source,
                'スニペット': event.snippet,
                '一致キーワード': ', '.join(event.keywords_hit),
                '時間差': event.time_gap,
                '地名一致度': event.geo_match,
                '人口変化率': event.population_change,
                '変化タイプ': event.change_type
            }
            events_data.append(event_dict)
        
        # CSVとして保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 全イベント
        df_all = pd.DataFrame(events_data)
        all_path = os.path.join(self.output_dir, f"all_events_{timestamp}.csv")
        df_all.to_csv(all_path, index=False, encoding='utf-8')
        print(f"全イベント保存: {all_path}")
        
        # 高信頼度イベント（自動ラベル）
        high_confidence = df_all[df_all['信頼度'] >= 0.7]
        if len(high_confidence) > 0:
            high_path = os.path.join(self.output_dir, f"high_confidence_events_{timestamp}.csv")
            high_confidence.to_csv(high_path, index=False, encoding='utf-8')
            print(f"高信頼度イベント保存: {high_path}")
        
        # 要確認イベント（人手確認キュー）
        medium_confidence = df_all[(df_all['信頼度'] >= 0.4) & (df_all['信頼度'] < 0.7)]
        if len(medium_confidence) > 0:
            medium_path = os.path.join(self.output_dir, f"medium_confidence_events_{timestamp}.csv")
            medium_confidence.to_csv(medium_path, index=False, encoding='utf-8')
            print(f"要確認イベント保存: {medium_path}")
        
        # 統計サマリー
        summary = {
            '作成日時': timestamp,
            '総イベント数': len(events),
            '高信頼度(>=0.7)': len(high_confidence),
            '要確認(0.4-0.7)': len(medium_confidence),
            '低信頼度(<0.4)': len(df_all[df_all['信頼度'] < 0.4]),
            '事象タイプ別件数': df_all['事象タイプ'].value_counts().to_dict()
        }
        
        summary_path = os.path.join(self.output_dir, f"investigation_summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"サマリー保存: {summary_path}")

def main():
    """メイン実行関数"""
    print("人口変化原因解明フレームワーク（ハイブリッド設計）")
    
    # 調査の実行
    investigator = PopulationCauseInvestigator()
    events = investigator.run_investigation()  # 全件調査を実行（max_targets=None）
    
    if events:
        print(f"\n=== 調査結果サマリー ===")
        print(f"総イベント数: {len(events)}件")
        
        # 信頼度別の件数
        high_conf = sum(1 for e in events if e.confidence >= 0.7)
        medium_conf = sum(1 for e in events if 0.4 <= e.confidence < 0.7)
        low_conf = sum(1 for e in events if e.confidence < 0.4)
        
        print(f"高信頼度(>=0.7): {high_conf}件")
        print(f"要確認(0.4-0.7): {medium_conf}件")
        print(f"低信頼度(<0.4): {low_conf}件")
        
        # 事象タイプ別の件数
        event_types = {}
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        print(f"\n事象タイプ別件数:")
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {event_type}: {count}件")
    else:
        print("調査の実行に失敗しました")

if __name__ == "__main__":
    main()
