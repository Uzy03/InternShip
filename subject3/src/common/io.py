import logging
import random
from pathlib import Path
from typing import Union
import numpy as np


def resolve_path(rel: str) -> Path:
    """プロジェクトルートからの相対パス解決"""
    # プロジェクトルートを探す（src/ディレクトリの親）
    current = Path(__file__).resolve()
    while current.name != 'src' and current.parent != current:
        current = current.parent
    
    if current.name != 'src':
        raise FileNotFoundError("プロジェクトルート（src/の親ディレクトリ）が見つかりません")
    
    project_root = current.parent
    return project_root / rel


def ensure_parent(path: Union[str, Path]) -> None:
    """親ディレクトリ作成"""
    path = Path(path)
    parent = path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f"ディレクトリの作成に失敗しました: {parent}, エラー: {e}")


def get_logger(run_id: str) -> logging.Logger:
    """ロガー設定、logs/{run_id}/run.log に出力"""
    # ログディレクトリ作成
    log_dir = resolve_path(f"logs/{run_id}")
    ensure_parent(log_dir)
    
    # 既存のハンドラーをクリア（重複防止）
    logger = logging.getLogger(run_id)
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    # ファイルハンドラー
    try:
        file_handler = logging.FileHandler(log_dir / "run.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラー追加
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    except Exception as e:
        # ファイルハンドラー作成に失敗した場合はコンソールのみ
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.warning(f"ファイルログの作成に失敗しました。コンソールログのみ使用します: {e}")
    
    return logger


def set_seed(seed: int) -> None:
    """random/numpy にシードを適用"""
    random.seed(seed)
    np.random.seed(seed)
