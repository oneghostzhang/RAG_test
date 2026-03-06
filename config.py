"""
職能基準知識圖譜 Graph RAG 系統 - 配置管理
"""

from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class GraphRAGConfig:
    """系統配置"""

    # ========================================
    # 路徑配置
    # ========================================
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    RAW_PDF_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "raw_pdf")
    PARSED_JSON_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "parsed_json")
    PARSED_JSON_V2_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "parsed_json_v2")
    GRAPH_DB_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "graph_db")
    VECTORDB_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "vectordb")
    OUTPUTS_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "outputs")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "logs")

    # ========================================
    # ICAP 資料來源路徑
    # ========================================
    ICAP_SOURCE_DIR: Path = field(
        default_factory=lambda: Path(r"C:\Users\User\Downloads\職能基準(ICAP全部)\output")
    )

    # ========================================
    # LLM 模型配置
    # ========================================
    MODEL_PATH: str = r"C:\Users\User\.lmstudio\models\ZoneTwelve\TAIDE-LX-7B-Chat-GGUF\TAIDE-LX-7B-Chat.Q4_K_S.gguf"
    N_CTX: int = 4096
    N_THREADS: int = 8
    TEMPERATURE: float = 0.1  # 降低溫度，更確定性的輸出
    MAX_TOKENS: int = 512     # 限制輸出長度，避免 CPU 推論過久（約 1-2 分鐘內完成）
    LLM_STOP_TOKENS: List[str] = field(default_factory=lambda: [
        "\n問題:", "\n問:", "問題列表", "\n【用戶問題】"
    ])

    # ========================================
    # Embedding 配置
    # ========================================
    EMBEDDING_MODEL: str = "BAAI/bge-base-zh-v1.5"
    EMBEDDING_DIM: int = 768
    EMBEDDING_DEVICE: str = "cpu"

    # ========================================
    # 圖譜配置
    # ========================================
    GRAPH_FILE: str = "competency_graph.gpickle"

    # 節點類型
    NODE_TYPES: List[str] = field(default_factory=lambda: [
        "職能基準",      # CompetencyStandard
        "職類別",        # OccupationCategory
        "職業別",        # OccupationType
        "行業別",        # IndustryType
        "主要職責",      # MainDuty
        "工作任務",      # WorkTask
        "工作產出",      # WorkOutput
        "行為指標",      # BehaviorIndicator
        "知識",          # Knowledge
        "技能",          # Skill
        "態度",          # Attitude
    ])

    # 邊類型
    EDGE_TYPES: List[str] = field(default_factory=lambda: [
        "屬於職類",      # belongs_to_category
        "屬於職業",      # belongs_to_occupation
        "適用行業",      # applies_to_industry
        "包含職責",      # contains_duty
        "包含任務",      # contains_task
        "產出",          # produces
        "要求行為",      # requires_behavior
        "需要知識",      # requires_knowledge
        "需要技能",      # requires_skill
        "要求態度",      # requires_attitude
        "相關於",        # related_to
        "前置於",        # prerequisite_of
        "可晉升至",      # promotes_to
    ])

    # ========================================
    # PDF 解析配置
    # ========================================
    # 編碼格式正則表達式
    CODE_PATTERNS: Dict[str, str] = field(default_factory=lambda: {
        "職能基準代碼": r"[A-Z]{2,3}\d{4}-\d{3}v?\d*",       # TFB7912-002v3
        "主要職責": r"T\d+",                                  # T1, T2
        "工作任務": r"T\d+\.\d+",                             # T1.1, T1.2
        "工作產出": r"O\d+\.\d+\.\d+",                        # O1.1.1
        "行為指標": r"P\d+\.\d+\.\d+",                        # P1.1.1
        "知識": r"K\d{2,3}",                                  # K01, K02
        "技能": r"S\d{2,3}",                                  # S01, S02
        "態度": r"A\d{2,3}",                                  # A01, A02
        "職類別代碼": r"[A-Z]{2,3}",                          # TFB
        "職業別代碼": r"\d{4}",                               # 7912
        "行業別代碼": r"[A-Z]\d{4}",                          # C0891
    })

    # ========================================
    # 日誌配置
    # ========================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "graph_rag.log"

    def __post_init__(self):
        """初始化後處理"""
        # 確保目錄存在
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path) and attr_name.endswith("_DIR"):
                attr.mkdir(parents=True, exist_ok=True)


# ========================================
# 節點類型定義
# ========================================

@dataclass
class CompetencyStandard:
    """職能基準節點"""
    code: str                    # TFB7912-002v3
    name: str                    # 西點麵包烘焙助理
    description: str = ""        # 工作描述
    level: int = 0               # 基準級別 (2-4)


@dataclass
class OccupationCategory:
    """職類別節點"""
    code: str                    # TFB
    name: str                    # 休閒與觀光旅遊/餐飲管理


@dataclass
class OccupationType:
    """職業別節點"""
    code: str                    # 7912
    name: str                    # 麵包點心及糖果製造人員


@dataclass
class IndustryType:
    """行業別節點"""
    code: str                    # C0891
    name: str                    # 烘焙炊蒸食品製造業


@dataclass
class MainDuty:
    """主要職責節點"""
    code: str                    # T1
    name: str                    # 烘焙前置準備


@dataclass
class WorkTask:
    """工作任務節點"""
    code: str                    # T1.1
    name: str                    # 整理環境與設備
    level: int = 0               # 職能級別 (1-7)


@dataclass
class WorkOutput:
    """工作產出節點"""
    code: str                    # O1.1.1
    name: str                    # 環境維護紀錄表


@dataclass
class BehaviorIndicator:
    """行為指標節點"""
    code: str                    # P1.1.1
    description: str             # 依食品安全衛生相關規範...


@dataclass
class Knowledge:
    """知識節點"""
    code: str                    # K01
    name: str                    # 食品安全衛生相關規範
    description: str = ""


@dataclass
class Skill:
    """技能節點"""
    code: str                    # S01
    name: str                    # 器具選用及操作能力
    description: str = ""


@dataclass
class Attitude:
    """態度節點"""
    code: str                    # A01
    name: str                    # 主動積極
    description: str = ""        # 不需他人指示或要求能自動自發做事...


# ========================================
# 全域配置實例
# ========================================
config = GraphRAGConfig()


def get_config() -> GraphRAGConfig:
    """取得全域配置"""
    return config


if __name__ == "__main__":
    cfg = get_config()
    print(f"專案根目錄: {cfg.PROJECT_ROOT}")
    print(f"ICAP 資料來源: {cfg.ICAP_SOURCE_DIR}")
    print(f"節點類型: {cfg.NODE_TYPES}")
    print(f"邊類型: {cfg.EDGE_TYPES}")
