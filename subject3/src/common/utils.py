import re
import pandas as pd


def normalize_town(name: str) -> str:
    """町丁名正規化ユーティリティ
    
    Args:
        name: 正規化対象の町丁名
        
    Returns:
        正規化された町丁名
        
    Raises:
        ValueError: 空文字やNaNの場合
    """
    # 空文字やNaNのチェック
    if pd.isna(name) or str(name).strip() == "":
        raise ValueError("町丁名が空文字またはNaNです")
    
    # 文字列に変換
    name = str(name)
    
    # 前後空白の除去
    name = name.strip()
    
    # 全角スペース→半角
    name = name.replace('　', ' ')
    
    # 連続空白の単一化
    name = re.sub(r'\s+', ' ', name)
    
    # （中央区|東区|西区|南区|北区） の丸括弧付き区名を除去（全角/半角両対応）
    name = re.sub(r'[（(](中央区|東区|西区|南区|北区)[)）]', '', name)
    
    # 全角の丸括弧・数字・ハイフン・長音などの一般的半角化
    # 全角丸括弧 → 半角丸括弧
    name = name.replace('（', '(').replace('）', ')')
    # 全角数字 → 半角数字
    name = name.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    # 全角ハイフン → 半角ハイフン
    name = name.replace('－', '-')
    # 全角長音 → 半角長音
    name = name.replace('ー', '-')
    
    # 再度前後空白を除去（変換後の調整）
    name = name.strip()
    
    return name
