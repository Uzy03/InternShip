"""
イベントラベル付けシステム（Layer2）

manual_investigation_targets.csvからイベント情報を抽出・正規化し、
events_labeled.csv（ロング形式）とevents_matrix.csv（ワイド形式）を生成する。

拡張機能：
- events_labeled.csvの重複解消（ソース優先）
- foreignerイベントの除外
- 正規化後のevents_labeled_clean.csv出力
"""

import pandas as pd
import numpy as np
import re
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'subject3'))

# 相対インポートを試行
try:
    from src.common.io import ensure_parent
except ImportError:
    # 直接実行時のフォールバック
    import os
    current_dir = Path(__file__).parent
    subject3_dir = current_dir.parent.parent
    sys.path.insert(0, str(subject3_dir))
    from src.common.io import ensure_parent

# 拡張機能用の定数
P_IN = "subject3/data/processed/events_labeled.csv"
P_CLEAN = "subject3/data/processed/events_labeled_clean.csv"
P_MAT = "subject3/data/processed/events_matrix_signed.csv"

IGNORE_EVENT_TYPES = {"foreigner"}
REQUIRED_COLS = ["town", "year", "event_type", "confidence", "intensity",
                 "lag_t", "lag_t1", "effect_direction", "source_url", "source_name", "publish_date"]


class EventNormalizer:
    """イベントラベル付けのメインクラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初期化"""
        self.config = config or self._get_default_config()
        self.setup_logging()
        
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            # イベントカテゴリと優先順位
            'categories': {
                'disaster': {'priority': 1, 'name': 'disaster'},
                'policy_boundary': {'priority': 2, 'name': 'policy_boundary'},
                'transit': {'priority': 3, 'name': 'transit'},
                'employment': {'priority': 4, 'name': 'employment'},
                'public_edu_medical': {'priority': 5, 'name': 'public_edu_medical'},
                'commercial': {'priority': 6, 'name': 'commercial'},
                'housing': {'priority': 7, 'name': 'housing'},
                'unknown': {'priority': 8, 'name': 'unknown'}
            },
            
            # 正規表現マップ（カテゴリ判定用）- 優先順位付き
            'regex_map': {
                # 最優先：Boundary/住所系（他カテゴリをオーバーライド）
                'policy_boundary': [
                    r'住居表示|町界町名地番変更|区制施行|編入|合併|分割|[一二三四五六七八九十]丁目新設',
                    r'政令市|境界変更|地番変更|町名変更'
                ],
                # 災害系
                'disaster': [
                    r'地震|余震|豪雨|洪水|浸水|土砂|台風|災害|被災|仮設|避難',
                    r'2016|平成28|熊本地震|罹災|倒壊|液状化|被害'
                ],
                # 交通系（アンカー必須）
                'transit': [
                    r'駅|停留所|電停|路線|線路|開通|延伸|供用|全通|IC|JCT|運休|減便|増便|廃止|改札|踏切',
                    r'バスターミナル|鉄道|高架|道路|バイパス|都市計画道路|移設'
                ],
                # 公共・教育・医療系（アンカー必須）
                'public_edu_medical': [
                    r'小学校|中学校|高校|大学|保育園|幼稚園|病院|クリニック|保健所|図書館|市役所|支所|体育館|郵便局|商工会',
                    r'診療所|庁舎|保育所|こども園|医療|教育|公共施設'
                ],
                # 雇用系
                'employment': [
                    r'工場|物流|倉庫|データセンター|操業|新規雇用|企業|事業所',
                    r'人|名|雇用|就業|労働'
                ],
                # 住宅系
                'housing': [
                    r'マンション|団地|分譲|賃貸|区画整理|竣工|入居|建替|新築|集合住宅|アパート|コーポ',
                    r'戸|棟|世帯|住戸|部屋'
                ],
                # 商業系
                'commercial': [
                    r'商業|モール|SC|百貨店|複合商業|サクラマチ|ゆめタウン|イオンモール|ショッピング',
                    r'開業|出店|新規|店舗|商業施設'
                ]
            },
            
            # 方向判定用正規表現（カテゴリ別）
            'direction_regex' : {
                # 境界は最優先。increase は定義しない（復旧は災害側で扱う）
                'policy_boundary': {
                    'decrease': [
                        r'(住居表示|町界町名地番変更|区制施行|編入|合併|分割|(?:[一二三四五六七八九十]|[0-9０-９]+)丁目新設|区画整理.*換地)'
                    ]
                },

                'disaster': {
                    # 被災・当年マイナス
                    'decrease': [
                        r'(地震|余震|豪雨|大雨|洪水|氾濫|浸水|土砂|土石流|崩落|台風|竜巻|被災|罹災|倒壊|全壊|半壊|液状化|避難(指示|勧告)?|断水|停電)'
                    ],
                    # 復旧・復興・復興住宅など（＋方向）
                    'increase': [
                        r'(復旧|復興|再建|復興住宅|災害公営住宅|復旧工事完了|仮設住宅入居開始)'
                    ]
                },

                'transit': {
                    # 工事・休止・減便など（当年マイナス）
                    'decrease': [
                        r'(?=.*(駅|停留所|電停|路線|線|バス|電車|LRT|IC|JCT|踏切))(?=.*(工事|高架化工事|改良|架け替え|仮設|通行止め|運休|休止|減便|ダイヤ改正.*減便|廃止))'
                    ],
                    # 供用・開通・延伸など（翌年プラス）※住居表示文脈は除外
                    'increase': [
                        r'(?!.*住居表示)(?!.*丁目新設)(?=.*(駅|停留所|電停|路線|線|バス|電車|LRT|IC|JCT))(?=.*(開通|供用開始|延伸|駅新設|停留所新設|快速停車|直通|乗り入れ|増便|ダイヤ改正.*増便|複線化完成|高架化完成))'
                    ]
                },

                'public_edu_medical': {
                    # 施設アンカー ＋ 閉鎖等
                    'decrease': [
                        r'(?=.*(小学校|中学校|高校|大学|保育園|幼稚園|病院|診療所|クリニック|保健所|図書館|市役所|支所|郵便局|体育館))(?=.*(閉鎖|廃止|統合|休止|縮小))'
                    ],
                    # 施設アンカー ＋ 開設等（“新設”だけではヒットしない）
                    'increase': [
                        r'(?=.*(小学校|中学校|高校|大学|保育園|幼稚園|病院|診療所|クリニック|保健所|図書館|市役所|支所|郵便局|体育館))(?=.*(開業|開設|新設|開院|開校|移転新築|拡張|増設|再開))'
                    ]
                },

                'employment': {
                    'decrease': [
                        r'(?=.*(工場|事業所|製造所|物流センター|コールセンター|オフィス|拠点|雇用))(?=.*(閉鎖|廃止|撤退|縮小|解雇|雇止め|リストラ|操業停止))'
                    ],
                    'increase': [
                        r'(?=.*(工場|事業所|製造所|物流センター|コールセンター|オフィス|拠点|雇用))(?=.*(開業|操業開始|新規雇用|採用拡大|拡張|増員|新設))'
                    ]
                },

                'housing': {
                    'decrease': [
                        r'(?=.*(住宅|団地|マンション|アパート|戸建|宅地|家屋|空き家))(?=.*(解体|取り壊し|除却|撤去|立退|取壊|老朽化除却))'
                    ],
                    'increase': [
                        r'(?=.*(住宅|団地|マンション|アパート|戸建|宅地|分譲|共同住宅))(?=.*(竣工|新築|完成|建設|着工|入居開始|分譲開始|建替|供用開始))'
                    ]
                },

                'commercial': {
                    'decrease': [
                        r'(?=.*(商業施設|モール|SC|ショッピングセンター|スーパー|ドラッグストア|コンビニ|店舗|支店))(?=.*(閉店|閉鎖|撤退|廃止))'
                    ],
                    'increase': [
                        r'(?=.*(商業施設|モール|SC|ショッピングセンター|スーパー|ドラッグストア|コンビニ|店舗|支店))(?=.*(開業|出店|新規出店|開店|再開|リニューアル(?:オープン)?|増床))'
                    ]
                },
            },
            
            # ラグルール
            'lag_rules': {
                'housing': {'t': 1, 't1': 1},
                'commercial': {'t': 1, 't1': 1},
                'public_edu_medical': {'t': 0, 't1': 1},
                'employment': {'t': 0, 't1': 1},
                'transit': {'t': 0, 't1': 1},
                'disaster': {'t': 1, 't1': 0},
                'policy_boundary': {'t': 1, 't1': 0},
                'unknown': {'t': 0, 't1': 0}
            },
            
            # 強度抽出ルール
            'intensity_rules': {
                'per_category': {
                    'housing': ['戸', '世帯', '棟', '区画', '住戸', '部屋'],
                    'employment': ['人', '名', '雇用'],
                    'commercial': ['m²', '㎡', '床面積', '人', '名'],
                    'public_edu_medical': ['m²', '㎡', '床面積', '人', '名'],
                    'disaster': ['戸', '人', '世帯'],
                    'transit': [],
                    'policy_boundary': []
                },
                'patterns': [
                    r'(\d+(?:\.\d+)?)\s*(?:戸|世帯|棟|区画|住戸|部屋|人|名|m²|㎡)',
                    r'(\d+(?:\.\d+)?)\s*(?:約|およそ|概ね)',
                    r'(\d+(?:\.\d+)?)\s*～\s*(\d+(?:\.\d+)?)',  # 範囲表現
                ]
            },
            
            # 信頼度計算設定
            'confidence': {
                'base_manual': 1.0,
                'note_doubt': 0.7,
                'gov_bonus': 0.1,
                'weak_source_penalty': 0.2,
                'min': 0.1,
                'max': 1.0,
                'gov_sources': ['熊本市', '県庁', '官報', '公示', '都市計画', '市議会', '自治体', '公報'],
                'doubt_keywords': ['要確認', '未確', '？', '不明', '不詳']
            },
            
            # その他設定
            'intensity_max': 10000,
            'use_signed_intensity': False,
            'construction_keywords': ['工事', '立退', '閉鎖', '撤去', '解体']
        }
    
    def setup_logging(self):
        """ログ設定"""
        log_dir = Path(project_root / 'logs').resolve()
        ensure_parent(log_dir / 'dummy.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_dir / 'events_label_issues.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def normalize_text(self, text: str) -> str:
        """テキスト正規化（全角/半角、大文字小文字）"""
        if pd.isna(text) or text == '':
            return ''
        
        # Unicode正規化
        text = unicodedata.normalize('NFKC', str(text))
        # 小文字化
        text = text.lower()
        return text
    
    def normalize_town_name(self, town_name: str) -> str:
        """町名の正規化（NFKC正規化＋区名カッコ除去）"""
        if pd.isna(town_name) or town_name == '':
            return ''
        
        # Unicode正規化（全角数字→半角数字、全角英字→半角英字など）
        normalized = unicodedata.normalize('NFKC', str(town_name))
        
        # 区名の除去（例：「熊本市中央区上通町」→「上通町」）
        # ただし、町名内のカッコは保持（例：「○○町（旧△△町）」）
        import re
        
        # 区名パターン（中央区、東区、西区、南区、北区など）
        ward_patterns = [
            r'中央区', r'東区', r'西区', r'南区', r'北区',
            r'合志市', r'菊陽町', r'大津町', r'菊池市', r'宇土市',
            r'宇城市', r'阿蘇市', r'天草市', r'山鹿市', r'人吉市',
            r'荒尾市', r'水俣市', r'玉名市', r'八代市', r'上天草市',
            r'下益城郡', r'上益城郡', r'玉名郡', r'菊池郡', r'阿蘇郡',
            r'葦北郡', r'球磨郡', r'天草郡'
        ]
        
        # 区名パターンを除去（区名の後に続く町名を抽出）
        for pattern in ward_patterns:
            # 区名パターンの後に町名が続く場合
            match = re.search(f'{pattern}(.+)', normalized)
            if match:
                normalized = match.group(1).strip()
                break
        
        return normalized.strip()
    
    def determine_event_type(self, raw_text: str, existing_type: Optional[str] = None) -> List[str]:
        """イベントタイプを決定（優先順位付き）"""
        if not raw_text or pd.isna(raw_text):
            return ['unknown']
        
        normalized_text = self.normalize_text(raw_text)
        
        # 既存のevent_typeが指定されている場合
        if existing_type and existing_type.strip():
            existing_type = self.normalize_text(existing_type)
            if existing_type in self.config['categories']:
                return [existing_type]
            else:
                self.logger.warning(f"Unknown event_type: {existing_type}, using regex matching")
        
        # 優先順位付きでマッチング（boundary最優先）
        matched_categories = []
        
        # 1) Boundary最優先チェック
        for pattern in self.config['regex_map']['policy_boundary']:
            if re.search(pattern, normalized_text):
                return ['policy_boundary']  # boundaryがマッチしたら他を返さない
        
        # 2) その他のカテゴリをチェック
        for category, patterns in self.config['regex_map'].items():
            if category == 'policy_boundary':  # 既にチェック済み
                continue
                
            # アンカー語チェック（transit, public_edu_medical）
            if category in ['transit', 'public_edu_medical']:
                has_anchor = False
                for pattern in patterns:
                    if re.search(pattern, normalized_text):
                        has_anchor = True
                        break
                if has_anchor:
                    matched_categories.append(category)
            else:
                # その他のカテゴリは通常のマッチング
                for pattern in patterns:
                    if re.search(pattern, normalized_text):
                        matched_categories.append(category)
                        break
        
        if not matched_categories:
            return ['unknown']
        
        # 優先順位でソート
        matched_categories.sort(key=lambda x: self.config['categories'][x]['priority'])
        return matched_categories
    
    def determine_effect_direction(self, raw_text: str, event_type: str) -> str:
        """効果方向を決定（カテゴリ別）"""
        if not raw_text or pd.isna(raw_text):
            return 'unknown'
        
        normalized_text = self.normalize_text(raw_text)
        
        # カテゴリ別の方向判定
        if event_type in self.config['direction_regex']:
            category_directions = self.config['direction_regex'][event_type]
            
            # increase判定
            if 'increase' in category_directions:
                for pattern in category_directions['increase']:
                    if re.search(pattern, normalized_text):
                        return 'increase'
            
            # decrease判定
            if 'decrease' in category_directions:
                for pattern in category_directions['decrease']:
                    if re.search(pattern, normalized_text):
                        return 'decrease'
        
        # デフォルトルール（カテゴリ別ルールがない場合）
        if event_type == 'policy_boundary':
            return 'decrease'  # 住居表示は基本的にdecrease
        elif event_type == 'disaster':
            return 'decrease'  # 災害は基本的にdecrease
        elif event_type in ['transit', 'public_edu_medical', 'employment', 'housing', 'commercial']:
            return 'increase'  # その他は基本的にincrease
        
        return 'unknown'
    
    def calculate_confidence(self, note: str, source_name: str) -> float:
        """信頼度を計算"""
        confidence = self.config['confidence']['base_manual']
        
        # noteに疑わしいキーワードがある場合
        if note and not pd.isna(note):
            note_normalized = self.normalize_text(note)
            for keyword in self.config['confidence']['doubt_keywords']:
                if keyword in note_normalized:
                    confidence = self.config['confidence']['note_doubt']
                    break
        
        # 信頼できるソースの場合
        if source_name and not pd.isna(source_name):
            source_normalized = self.normalize_text(source_name)
            for gov_source in self.config['confidence']['gov_sources']:
                if gov_source in source_normalized:
                    confidence = min(confidence + self.config['confidence']['gov_bonus'], 
                                   self.config['confidence']['max'])
                    break
        
        # クリップ
        confidence = max(self.config['confidence']['min'], 
                        min(confidence, self.config['confidence']['max']))
        
        return confidence
    
    def extract_intensity(self, raw_text: str, event_type: str) -> float:
        """強度を抽出"""
        if not raw_text or pd.isna(raw_text):
            return 1.0
        
        normalized_text = self.normalize_text(raw_text)
        
        # カテゴリ別の優先順位で抽出
        priority_units = self.config['intensity_rules']['per_category'].get(event_type, [])
        
        for unit in priority_units:
            pattern = f'(\\d+(?:\\.\\d+)?)\\s*{re.escape(unit)}'
            match = re.search(pattern, normalized_text)
            if match:
                return min(float(match.group(1)), self.config['intensity_max'])
        
        # 範囲表現の処理
        range_pattern = r'(\d+(?:\.\d+)?)\s*～\s*(\d+(?:\.\d+)?)'
        range_match = re.search(range_pattern, normalized_text)
        if range_match:
            start = float(range_match.group(1))
            end = float(range_match.group(2))
            return min((start + end) / 2, self.config['intensity_max'])
        
        # 一般的な数値パターン
        for pattern in self.config['intensity_rules']['patterns']:
            match = re.search(pattern, normalized_text)
            if match:
                return min(float(match.group(1)), self.config['intensity_max'])
        
        return 1.0
    
    def determine_lag(self, event_type: str, effect_direction: str, raw_text: str) -> Tuple[int, int]:
        """ラグを決定"""
        base_lag = self.config['lag_rules'].get(event_type, {'t': 0, 't1': 0})
        lag_t = base_lag['t']
        lag_t1 = base_lag['t1']
        
        # 工事由来の一時的減少の例外処理
        if (effect_direction == 'decrease' and 
            event_type in ['transit', 'housing'] and
            raw_text and not pd.isna(raw_text)):
            
            normalized_text = self.normalize_text(raw_text)
            for keyword in self.config['construction_keywords']:
                if keyword in normalized_text:
                    lag_t = 1
                    break
        
        return lag_t, lag_t1
    
    def _apply_override_rules(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """多重ラベル抑制ルールを適用"""
        if events_df.empty:
            return events_df
        
        # boundaryがマッチした(town, year)を特定
        boundary_keys = set()
        for _, row in events_df.iterrows():
            if row['event_type'] == 'policy_boundary':
                boundary_keys.add((row['town'], row['year']))
        
        # boundaryが存在する(town, year)から他のカテゴリを削除
        if boundary_keys:
            # 削除対象のインデックスを特定
            to_remove = []
            for idx, row in events_df.iterrows():
                key = (row['town'], row['year'])
                if key in boundary_keys and row['event_type'] != 'policy_boundary':
                    to_remove.append(idx)
            
            # 削除実行
            if to_remove:
                events_df = events_df.drop(index=to_remove).reset_index(drop=True)
                self.logger.info(f"Removed {len(to_remove)} events overridden by policy_boundary")
        
        # boundaryの符号・ラグを強制修正
        events_df = self._fix_boundary_direction_and_lag(events_df)
        
        # unknown directionの処理
        events_df = self._handle_unknown_directions(events_df)
        
        return events_df
    
    def _handle_unknown_directions(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """unknown directionの処理（ドロップまたは強度0）"""
        if events_df.empty:
            return events_df
        
        # unknown directionのイベントを特定
        unknown_mask = events_df['effect_direction'] == 'unknown'
        unknown_count = unknown_mask.sum()
        
        if unknown_count > 0:
            self.logger.info(f"Found {unknown_count} events with unknown direction, dropping them")
            # unknown directionのイベントをドロップ
            events_df = events_df[~unknown_mask].reset_index(drop=True)
        
        return events_df
    
    def _fix_boundary_direction_and_lag(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """boundaryの符号・ラグを強制修正"""
        if events_df.empty:
            return events_df
        
        # boundaryイベントを特定して修正
        for idx, row in events_df.iterrows():
            if row['event_type'] == 'policy_boundary':
                town = row['town']
                
                # 旧町丁側の判定（「丁目」を含まないが「町」を含む）
                is_old_ward = ('丁目' not in town) and ('町' in town)
                
                if is_old_ward:
                    # 旧町丁側は decrease, lag_t=1, lag_t1=0
                    events_df.at[idx, 'effect_direction'] = 'decrease'
                    events_df.at[idx, 'lag_t'] = 1
                    events_df.at[idx, 'lag_t1'] = 0
                    self.logger.debug(f"Fixed boundary for old ward: {town} -> decrease, lag_t=1")
                else:
                    # 新設側は基本的に decrease（住居表示による分割は旧町丁の減少）
                    events_df.at[idx, 'effect_direction'] = 'decrease'
                    events_df.at[idx, 'lag_t'] = 1
                    events_df.at[idx, 'lag_t1'] = 0
                    self.logger.debug(f"Fixed boundary for new ward: {town} -> decrease, lag_t=1")
        
        return events_df
    
    def process_events(self, input_file: str, panel_file: str, output_dir: str) -> Dict[str, Any]:
        """メイン処理"""
        self.logger.info("Starting event normalization process")
        
        # 入力データ読み込み
        try:
            manual_df = pd.read_csv(input_file)
            panel_df = pd.read_csv(panel_file)
        except Exception as e:
            self.logger.error(f"Failed to read input files: {e}")
            raise
        
        # 必須列チェック
        required_cols = ['町丁名', '年度', '原因']
        missing_cols = [col for col in required_cols if col not in manual_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 列名を正規化
        manual_df = manual_df.rename(columns={
            '町丁名': 'town',
            '年度': 'year',
            '原因': 'raw_cause_text'
        })
        
        # 町名の正規化を適用
        manual_df['town_normalized'] = manual_df['town'].apply(self.normalize_town_name)
        panel_df['town_normalized'] = panel_df['town'].apply(self.normalize_town_name)
        
        # パネルデータとの整合性チェック（正規化後の町名を使用）
        panel_keys = set(zip(panel_df['town_normalized'], panel_df['year']))
        
        # イベント処理
        events_list = []
        stats = {
            'total_processed': 0,
            'unknown_categories': 0,
            'panel_mismatches': 0,
            'intensity_failures': 0,
            'category_counts': {}
        }
        
        for idx, row in manual_df.iterrows():
            stats['total_processed'] += 1
            
            town = row['town']
            town_normalized = row['town_normalized']
            year = row['year']
            raw_cause_text = row['raw_cause_text']
            
            # パネルデータとの整合性チェック（正規化後の町名を使用）
            exists_in_panel = (town_normalized, year) in panel_keys
            if not exists_in_panel:
                stats['panel_mismatches'] += 1
                self.logger.warning(f"Panel mismatch: ({town} -> {town_normalized}, {year}) not found in panel data")
            
            # 空の原因テキストはスキップ
            if pd.isna(raw_cause_text) or raw_cause_text.strip() == '':
                continue
            
            # イベントタイプ決定
            event_types = self.determine_event_type(raw_cause_text)
            
            # 複数カテゴリの場合は分割
            for event_type in event_types:
                if event_type == 'unknown':
                    stats['unknown_categories'] += 1
                
                # 効果方向決定（イベントタイプを渡す）
                effect_direction = self.determine_effect_direction(raw_cause_text, event_type)
                
                # 信頼度計算
                confidence = self.calculate_confidence(
                    row.get('note', ''), 
                    row.get('source_name', '')
                )
                
                # 強度抽出
                try:
                    intensity = self.extract_intensity(raw_cause_text, event_type)
                except Exception as e:
                    intensity = 1.0
                    stats['intensity_failures'] += 1
                    self.logger.warning(f"Intensity extraction failed for row {idx}: {e}")
                
                # 符号処理
                if (self.config['use_signed_intensity'] and 
                    effect_direction == 'decrease'):
                    intensity = -intensity
                
                # ラグ決定
                lag_t, lag_t1 = self.determine_lag(event_type, effect_direction, raw_cause_text)
                
                # イベントレコード作成（正規化後の町名を使用）
                event_record = {
                    'town': town_normalized,
                    'year': year,
                    'event_type': event_type,
                    'confidence': confidence,
                    'intensity': intensity,
                    'lag_t': lag_t,
                    'lag_t1': lag_t1,
                    'effect_direction': effect_direction,
                    'source_url': row.get('source_url', ''),
                    'source_name': row.get('source_name', ''),
                    'publish_date': row.get('publish_date', '')
                }
                
                events_list.append(event_record)
                
                # 統計更新
                if event_type not in stats['category_counts']:
                    stats['category_counts'][event_type] = {'t': 0, 't1': 0}
                if lag_t == 1:
                    stats['category_counts'][event_type]['t'] += 1
                if lag_t1 == 1:
                    stats['category_counts'][event_type]['t1'] += 1
        
        # 重複解決と多重ラベル抑制
        events_df = pd.DataFrame(events_list)
        if not events_df.empty:
            # 1) 多重ラベル抑制（boundaryが他カテゴリをオーバーライド）- 最初に実行
            events_df = self._apply_override_rules(events_df)
            
            # 2) (town, year, event_type)で重複解決
            events_df = events_df.sort_values('confidence', ascending=False)
            events_df = events_df.drop_duplicates(subset=['town', 'year', 'event_type'], keep='first')
        
        # 出力生成
        output_path = Path(output_dir).resolve()
        ensure_parent(output_path / 'dummy.csv')
        
        # ロング形式出力（既存ファイルを上書きしない）
        long_output_file = output_path / 'events_labeled_new.csv'
        events_df.to_csv(str(long_output_file), index=False, encoding='utf-8')
        self.logger.info(f"Long format output saved to: {long_output_file}")
        
        # ワイド形式出力（無向版）
        wide_output_file = output_path / 'events_matrix.csv'
        wide_df = self._create_wide_format(events_df, panel_df)
        wide_df.to_csv(str(wide_output_file), index=False, encoding='utf-8')
        self.logger.info(f"Wide format output saved to: {wide_output_file}")
        
        # ワイド形式出力（有向版）
        wide_signed_output_file = output_path / 'events_matrix_signed.csv'
        wide_signed_df = self._create_signed_wide_format(events_df, panel_df)
        
        # 後処理を適用
        self.logger.info("Applying post-processing filters...")
        
        # 1) 小母数フィルタを適用
        features_panel_file = str(project_root / 'subject3/data/processed/features_panel.csv')
        wide_signed_df = self._apply_small_sample_filter(wide_signed_df, features_panel_file)
        
        # 2) inc/dec同時ヒットの整流を適用
        wide_signed_df = self._apply_inc_dec_rectification(wide_signed_df)
        
        # 後処理後のファイルを保存
        wide_signed_df.to_csv(str(wide_signed_output_file), index=False, encoding='utf-8')
        self.logger.info(f"Signed wide format output (post-processed) saved to: {wide_signed_output_file}")
        
        # 統計ログ出力
        self._log_statistics(stats)
        
        return {
            'events_labeled': events_df,
            'events_matrix': wide_df,
            'statistics': stats
        }
    
    def _create_wide_format(self, events_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
        """ワイド形式のマトリックスを作成"""
        if events_df.empty:
            # 空の場合はパネルデータのキーのみ（正規化後の町名を使用）
            wide_df = panel_df[['town_normalized', 'year']].copy()
            wide_df = wide_df.rename(columns={'town_normalized': 'town'})
            for category in self.config['categories']:
                if category != 'unknown':
                    wide_df[f'event_{category}_t'] = 0.0
                    wide_df[f'event_{category}_t1'] = 0.0
            return wide_df
        
        # パネルデータをベースにワイド形式を作成（正規化後の町名を使用）
        wide_df = panel_df[['town_normalized', 'year']].copy()
        wide_df = wide_df.rename(columns={'town_normalized': 'town'})
        
        # 各カテゴリの列を初期化
        for category in self.config['categories']:
            if category != 'unknown':
                wide_df[f'event_{category}_t'] = 0.0
                wide_df[f'event_{category}_t1'] = 0.0
        
        # イベントデータを集計
        for _, event in events_df.iterrows():
            town = event['town']
            year = event['year']
            event_type = event['event_type']
            intensity = event['intensity']
            lag_t = event['lag_t']
            lag_t1 = event['lag_t1']
            
            # パネルデータに該当する行を探す
            mask = (wide_df['town'] == town) & (wide_df['year'] == year)
            if mask.any() and event_type != 'unknown':
                if lag_t == 1:
                    wide_df.loc[mask, f'event_{event_type}_t'] += intensity
                if lag_t1 == 1:
                    wide_df.loc[mask, f'event_{event_type}_t1'] += intensity
        
        return wide_df
    
    def _create_signed_wide_format(self, events_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
        """有向版ワイド形式のマトリックスを作成（*_inc_* / *_dec_* 列）"""
        if events_df.empty:
            # 空の場合はパネルデータのキーのみ（正規化後の町名を使用）
            wide_df = panel_df[['town_normalized', 'year']].copy()
            wide_df = wide_df.rename(columns={'town_normalized': 'town'})
            for category in self.config['categories']:
                if category != 'unknown':
                    wide_df[f'event_{category}_inc_t'] = 0.0
                    wide_df[f'event_{category}_inc_t1'] = 0.0
                    wide_df[f'event_{category}_dec_t'] = 0.0
                    wide_df[f'event_{category}_dec_t1'] = 0.0
            return wide_df
        
        # パネルデータをベースにワイド形式を作成（正規化後の町名を使用）
        wide_df = panel_df[['town_normalized', 'year']].copy()
        wide_df = wide_df.rename(columns={'town_normalized': 'town'})
        
        # 各カテゴリの有向列を初期化
        for category in self.config['categories']:
            if category != 'unknown':
                wide_df[f'event_{category}_inc_t'] = 0.0
                wide_df[f'event_{category}_inc_t1'] = 0.0
                wide_df[f'event_{category}_dec_t'] = 0.0
                wide_df[f'event_{category}_dec_t1'] = 0.0
        
        # イベントデータを集計
        for _, event in events_df.iterrows():
            town = event['town']
            year = event['year']
            event_type = event['event_type']
            intensity = event['intensity']
            lag_t = event['lag_t']
            lag_t1 = event['lag_t1']
            effect_direction = event['effect_direction']
            
            # パネルデータに該当する行を探す
            mask = (wide_df['town'] == town) & (wide_df['year'] == year)
            if mask.any() and event_type != 'unknown':
                # 方向に応じて列を選択
                direction_suffix = 'inc' if effect_direction == 'increase' else 'dec'
                
                if lag_t == 1:
                    wide_df.loc[mask, f'event_{event_type}_{direction_suffix}_t'] += intensity
                if lag_t1 == 1:
                    wide_df.loc[mask, f'event_{event_type}_{direction_suffix}_t1'] += intensity
        
        return wide_df
    
    def _apply_small_sample_filter(self, events_matrix_signed_df: pd.DataFrame, features_panel_file: str) -> pd.DataFrame:
        """小母数フィルタ：abs(delta_people) < 50 かつ pop_total < 300 の年でイベント強度を0化"""
        try:
            # features_panel.csvを読み込み
            fp = pd.read_csv(features_panel_file)
            
            # delta_peopleが無ければ自前で算出
            if "delta_people" not in fp.columns:
                fp = fp.sort_values(["town", "year"])
                fp["delta_people"] = fp.groupby("town")["pop_total"].diff()
            
            # 小母数マスクを作成
            mask_smallN = (fp["pop_total"] < 300) & (fp["delta_people"].abs() < 50)
            small_keys = fp.loc[mask_smallN, ["town", "year"]].copy()
            
            if not small_keys.empty:
                # イベント列を特定
                event_cols = [c for c in events_matrix_signed_df.columns if c.startswith("event_")]
                
                # 小母数の(town, year)でイベント強度を0化
                events_matrix_signed_df = events_matrix_signed_df.merge(
                    small_keys.assign(_small=1), 
                    on=["town", "year"], 
                    how="left"
                )
                events_matrix_signed_df.loc[events_matrix_signed_df["_small"].eq(1), event_cols] = 0.0
                events_matrix_signed_df = events_matrix_signed_df.drop(columns=["_small"])
                
                self.logger.info(f"[SmallN Filter] Zeroed {int(mask_smallN.sum())} rows, {len(event_cols)} event columns")
            else:
                self.logger.info("[SmallN Filter] No small sample cases found")
                
        except Exception as e:
            self.logger.warning(f"[SmallN Filter] Failed to apply filter: {e}")
        
        return events_matrix_signed_df
    
    def _apply_inc_dec_rectification(self, events_matrix_signed_df: pd.DataFrame) -> pd.DataFrame:
        """inc/dec同時ヒットの整流：transit/public_edu_medicalでdec→t、inc→t1に集約"""
        try:
            # 対象カテゴリ
            target_categories = ["transit", "public_edu_medical"]
            
            for category in target_categories:
                inc_t = f"event_{category}_inc_t"
                inc_t1 = f"event_{category}_inc_t1"
                dec_t = f"event_{category}_dec_t"
                dec_t1 = f"event_{category}_dec_t1"
                
                # 列が存在しない場合は0.0で初期化
                for col in [inc_t, inc_t1, dec_t, dec_t1]:
                    if col not in events_matrix_signed_df.columns:
                        events_matrix_signed_df[col] = 0.0
                
                # 1) decはtに集約（t1のdecをtへ移す）
                events_matrix_signed_df[dec_t] = events_matrix_signed_df[dec_t] + events_matrix_signed_df[dec_t1]
                events_matrix_signed_df[dec_t1] = 0.0
                
                # 2) incはt1に集約（tのincをt1へ移す）
                events_matrix_signed_df[inc_t1] = events_matrix_signed_df[inc_t1] + events_matrix_signed_df[inc_t]
                events_matrix_signed_df[inc_t] = 0.0
            
            self.logger.info(f"[Inc/Dec Rectification] Applied to categories: {target_categories}")
            
        except Exception as e:
            self.logger.warning(f"[Inc/Dec Rectification] Failed to apply rectification: {e}")
        
        return events_matrix_signed_df
    
    def _log_statistics(self, stats: Dict[str, Any]):
        """統計情報をログ出力"""
        self.logger.info("=== Event Normalization Statistics ===")
        self.logger.info(f"Total processed: {stats['total_processed']}")
        self.logger.info(f"Unknown categories: {stats['unknown_categories']}")
        self.logger.info(f"Panel mismatches: {stats['panel_mismatches']}")
        self.logger.info(f"Intensity extraction failures: {stats['intensity_failures']}")
        
        self.logger.info("Category counts:")
        for category, counts in stats['category_counts'].items():
            self.logger.info(f"  {category}: t={counts['t']}, t1={counts['t1']}")
    
    def _has_source(self, row):
        """ソース情報の有無を判定"""
        return (str(row.get("source_url", "")).strip() != "") or (str(row.get("source_name", "")).strip() != "")
    
    def _score_row(self, row):
        """行のスコアを計算（ソース有無 > confidence > intensity > publish_date）"""
        has_src = 1 if self._has_source(row) else 0
        conf = float(row.get("confidence", 0) or 0)
        inten = float(row.get("intensity", 0) or 0)
        # publish_date が不明なら後回しになるように大きな日付に
        pd_val = pd.to_datetime(row.get("publish_date", None), errors="coerce")
        ts = pd.Timestamp.max if pd.isna(pd_val) else pd_val
        # ※ sort_values(ascending=False) による降順優先のため、日付だけは逆に扱う
        return (has_src, conf, inten, -ts.value)
    
    def load_and_clean(self, input_file: str = None) -> pd.DataFrame:
        """events_labeled.csvを読み込み、重複解消とforeigner除外を行う"""
        if input_file is None:
            input_file = str(project_root / P_IN)
        
        df = pd.read_csv(input_file)
        
        # 必須列チェック（無い列は作る）
        for c in REQUIRED_COLS:
            if c not in df.columns:
                df[c] = np.nan
        
        # 無視イベント（foreigner）を除外
        before = len(df)
        df = df[~df["event_type"].isin(IGNORE_EVENT_TYPES)].copy()
        dropped_foreigner = before - len(df)
        
        # key 定義：同一キーは1行に正規化
        # 方向（increase/decrease）が違えば別キーとして扱う（矛盾があればスコアで1行だけ残す）
        key = ["town", "year", "event_type", "effect_direction"]
        df["_score"] = df.apply(self._score_row, axis=1)
        df = df.sort_values("_score", ascending=False)
        
        # ソース優先の重複解消
        best = df.groupby(key, as_index=False).first()
        
        # 監査用保存
        clean_output_path = str(project_root / P_CLEAN)
        ensure_parent(clean_output_path)
        best.to_csv(clean_output_path, index=False)
        self.logger.info(f"[normalize] loaded={before}, dropped_foreigner={dropped_foreigner}, kept_clean={len(best)} → {clean_output_path}")
        
        return best
    
    def build_matrix_signed(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """正規化されたイベントデータから有向版マトリックスを作成"""
        # 方向を符号化
        sign = df_clean["effect_direction"].map({"increase": 1, "decrease": -1}).fillna(0).astype(int)
        # スコア（二重カウント抑止のため confidence×intensity を1.0上限でクリップ）
        score = (df_clean["confidence"].fillna(0).astype(float) * df_clean["intensity"].fillna(0).astype(float)).clip(0, 1)
        df_clean["_signed_t"] = sign * score * df_clean["lag_t"].fillna(0).astype(float)
        df_clean["_signed_t1"] = sign * score * df_clean["lag_t1"].fillna(0).astype(float)
        
        # 符号ありの列を作成するため、event_type + effect_direction の組み合わせで処理
        df_clean["_event_direction"] = df_clean["event_type"] + "_" + df_clean["effect_direction"]
        
        # event_type + direction ごとに T と T+1 を別列で集計（最大1.0に抑制）
        def _pivot(colname, suffix):
            tmp = df_clean.pivot_table(
                index=["town", "year"],
                columns="_event_direction",
                values=colname,
                aggfunc="sum",
                fill_value=0.0
            ).clip(-1.0, 1.0)
            # 列名を符号あり形式に変換
            new_columns = []
            for c in tmp.columns:
                if c.endswith("_increase"):
                    base_name = c.replace("_increase", "")
                    new_columns.append(f"event_{base_name}_inc_{suffix}")
                elif c.endswith("_decrease"):
                    base_name = c.replace("_decrease", "")
                    new_columns.append(f"event_{base_name}_dec_{suffix}")
                else:
                    new_columns.append(f"event_{c}_{suffix}")
            tmp.columns = new_columns
            return tmp.reset_index()
        
        m0 = _pivot("_signed_t", "t")
        m1 = _pivot("_signed_t1", "t1")
        
        mat = m0.merge(m1, on=["town", "year"], how="outer").fillna(0.0)
        return mat


def normalize_events(input_file: str = None, panel_file: str = None, output_dir: str = None) -> Dict[str, Any]:
    """
    イベントラベル付けのメイン関数
    
    Args:
        input_file: manual_investigation_targets.csvのパス
        panel_file: panel_raw.csvのパス  
        output_dir: 出力ディレクトリ
    
    Returns:
        処理結果の辞書
    """
    # デフォルトパス設定（絶対パス）
    if input_file is None:
        input_file = str(project_root / 'subject2-3/manual_investigation_targets.csv')
    if panel_file is None:
        panel_file = str(project_root / 'subject3/data/processed/panel_raw.csv')
    if output_dir is None:
        output_dir = str(project_root / 'subject3/data/processed')
    
    # パスを絶対パスに変換
    input_file = str(Path(input_file).resolve())
    panel_file = str(Path(panel_file).resolve())
    output_dir = str(Path(output_dir).resolve())
    
    # イベント正規化実行
    normalizer = EventNormalizer()
    return normalizer.process_events(input_file, panel_file, output_dir)


def main():
    """拡張版メイン関数：重複解消＋foreigner無視"""
    normalizer = EventNormalizer()
    
    # 1) events_labeled.csvを読み込み、重複解消とforeigner除外
    clean = normalizer.load_and_clean()
    
    # 2) 正規化されたデータから有向版マトリックスを作成
    mat = normalizer.build_matrix_signed(clean)
    
    # 3) マトリックスを保存
    mat_output_path = str(project_root / P_MAT)
    ensure_parent(mat_output_path)
    mat.to_csv(mat_output_path, index=False)
    normalizer.logger.info(f"[normalize] events_matrix_signed saved → {mat_output_path}")
    
    # 4) events_labeled.csvは上書きしない（既存ファイルを保持）
    # cleanデータは既にevents_labeled_clean.csvとして保存済み
    normalizer.logger.info(f"[normalize] events_labeled.csv preserved (not overwritten)")
    
    return {
        'events_clean': clean,
        'events_matrix_signed': mat
    }


if __name__ == "__main__":
    # コマンドライン実行
    result = main()
    print("Event normalization (extended) completed successfully!")
    print(f"Clean events: {len(result['events_clean'])} rows")
    print(f"Matrix shape: {result['events_matrix_signed'].shape}")
