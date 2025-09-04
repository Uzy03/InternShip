"""
シナリオ検証スキーマ（pydantic）
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Literal, Optional

EventType = Literal["housing", "commercial", "transit", "policy_boundary", "public_edu_medical", "employment", "disaster"]
Direction = Literal["increase", "decrease"]


class ScenarioEvent(BaseModel):
    """シナリオイベント"""
    year_offset: int = Field(ge=0, le=3, description="年オフセット（0-3年先）")
    event_type: EventType = Field(description="イベントタイプ")
    effect_direction: Direction = Field(description="効果方向")
    confidence: float = Field(ge=0, le=1, description="信頼度（0-1）")
    intensity: float = Field(ge=0, le=1, description="強度（0-1）")
    lag_t: int = Field(ge=0, le=1, description="当年効果（0または1）")
    lag_t1: int = Field(ge=0, le=1, description="翌年効果（0または1）")
    note: Optional[str] = Field(default="", description="備考")


class Scenario(BaseModel):
    """シナリオ全体"""
    town: str = Field(description="町丁名")
    base_year: int = Field(description="基準年")
    horizons: List[int] = Field(default=[1, 2, 3], description="予測期間（1-3年先）")
    events: List[ScenarioEvent] = Field(default_factory=list, description="イベントリスト")
    macros: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="マクロ変数")
    manual_delta: Dict[str, float] = Field(default_factory=dict, description="手動加算（人ベース）")

    @field_validator("horizons")
    @classmethod
    def check_horizons(cls, v):
        """horizonsの検証"""
        assert all(h in (1, 2, 3) for h in v), "horizons must be subset of {1,2,3}"
        return sorted(set(v))

    @field_validator("manual_delta")
    @classmethod
    def check_manual_delta(cls, v):
        """manual_deltaの検証"""
        valid_keys = {"h1", "h2", "h3"}
        for key in v.keys():
            assert key in valid_keys, f"manual_delta keys must be in {valid_keys}"
        return v

    def validate_conflicts(self) -> List[str]:
        """衝突チェック（policy_boundary vs transit）"""
        warnings = []
        
        # 年ごとのイベントをグループ化
        events_by_year = {}
        for event in self.events:
            year = self.base_year + event.year_offset
            if year not in events_by_year:
                events_by_year[year] = []
            events_by_year[year].append(event)
        
        # 各年でpolicy_boundaryとtransitの衝突をチェック
        for year, year_events in events_by_year.items():
            has_policy = any(e.event_type == "policy_boundary" for e in year_events)
            has_transit = any(e.event_type == "transit" for e in year_events)
            
            if has_policy and has_transit:
                warnings.append(f"年 {year}: policy_boundary と transit の衝突を検出。transit が無効化されます。")
        
        return warnings
