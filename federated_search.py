"""
聯邦搜索模組 - RAGRoute 式查詢路由
基於職類別/行業別對職能基準進行分群，實現高效的查詢路由

參考論文: Efficient Federated Search for Retrieval-Augmented Generation

資料來源: CompetencyJSONStore (JSON 資料存取層)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from loguru import logger

from config import get_config

# 導入資料存取層（優先使用 JSON Store）
try:
    from competency_store import CompetencyJSONStore
    JSON_STORE_AVAILABLE = True
except ImportError:
    CompetencyJSONStore = None
    JSON_STORE_AVAILABLE = False

DATA_SOURCE_AVAILABLE = JSON_STORE_AVAILABLE
if not DATA_SOURCE_AVAILABLE:
    logger.warning("沒有可用的資料來源，聯邦搜索功能受限")

config = get_config()


@dataclass
class CategoryDataSource:
    """職類別資料來源"""
    category_id: str           # 職類別 ID (例如: "工程及技術")
    category_name: str         # 職類別名稱
    standards: List[str] = field(default_factory=list)  # 該職類別下的職能基準代碼
    centroid: Optional[np.ndarray] = None  # 該類別的質心向量
    density: float = 0.0       # 密度指標
    size: int = 0              # 包含的節點數量
    keywords: List[str] = field(default_factory=list)  # 關鍵字


@dataclass
class ICAPMetadata:
    """ICAP 職能基準元資料"""
    code: str                  # 職能基準代碼
    name: str                  # 職能基準名稱
    category: str              # 職類別/領域別
    industry: str              # 行業別
    occupation_class: str = "" # 通俗職業分類（如：餐飲、資訊科技）
    occupation_name: str = ""  # 所屬通俗職務名稱（如：中/西餐烹飪廚師）
    source_url: str = ""       # 來源 URL
    json_path: str = ""        # JSON 檔案路徑
    pdf_path: str = ""         # PDF 檔案路徑


@dataclass
class OccupationDataSource:
    """通俗職業分類資料來源（比職類別更精細）"""
    occupation_id: str         # 通俗職業分類 ID (例如: "餐飲")
    occupation_name: str       # 通俗職業分類名稱
    standards: List[str] = field(default_factory=list)  # 該分類下的職能基準代碼
    centroid: Optional[np.ndarray] = None  # 該分類的質心向量
    density: float = 0.0       # 密度指標
    size: int = 0              # 包含的節點數量
    keywords: List[str] = field(default_factory=list)  # 關鍵字
    parent_category: str = ""  # 所屬職類別


class ICAPMetadataIndex:
    """
    職能基準元資料索引

    資料來源（優先順序）：
    1. occupation_index.json — 預計算索引，最快（直接讀單一 JSON）
    2. CompetencyJSONStore   — 掃描 900+ JSON 重新建立，較慢
    """

    def __init__(self, data_source: Optional['CompetencyJSONStore'] = None):
        """
        初始化索引

        Args:
            data_source: 資料來源（CompetencyJSONStore），僅在 occupation_index.json 不存在時使用
        """
        self.data_source = data_source
        self.json_store = None

        # 判斷資料來源類型
        if data_source is not None and JSON_STORE_AVAILABLE and isinstance(data_source, CompetencyJSONStore):
            self.json_store = data_source

        self.metadata: Dict[str, ICAPMetadata] = {}  # code -> metadata
        self.category_index: Dict[str, List[str]] = defaultdict(list)  # category -> [codes]
        self.industry_index: Dict[str, List[str]] = defaultdict(list)  # industry -> [codes]
        self.occupation_class_index: Dict[str, List[str]] = defaultdict(list)  # occupation_code -> [codes]
        self.occupation_name_index: Dict[str, List[str]] = defaultdict(list)  # occupation_name -> [codes]
        self.name_to_code: Dict[str, str] = {}  # name -> code

        self._index_path = config.DATA_DIR / "competency_metadata_index.pkl"
        self._occupation_json_path = config.DATA_DIR / "occupation_index.json"

    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        建立或載入索引

        載入優先順序：
        1. occupation_index.json（最快，預計算）
        2. pickle 快取
        3. 重新掃描所有 JSON 並儲存 pickle

        Args:
            force_rebuild: 是否強制重新掃描（忽略所有快取）

        Returns:
            是否成功
        """
        if not force_rebuild:
            # 優先嘗試預計算 JSON（最快路徑）
            if self._load_from_occupation_json():
                return True
            # 次選 pickle 快取
            if self._load_index():
                return True

        if self.json_store is not None or JSON_STORE_AVAILABLE:
            return self._build_from_json_store()

        logger.error("沒有可用的資料來源")
        return False

    def _load_from_occupation_json(self) -> bool:
        """從預計算的 occupation_index.json 快速載入索引"""
        if not self._occupation_json_path.exists():
            return False

        try:
            import json as _json
            with open(self._occupation_json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)

            standards_data = data.get("standards", {})
            occupations_data = data.get("occupations", {})
            categories_data = data.get("categories", {})

            if not standards_data:
                return False

            # 重建 metadata 與各索引
            for code, info in standards_data.items():
                name = info.get("name", "")
                if not name:
                    continue

                industry = info.get("industry", [])
                industry_str = "; ".join(industry) if isinstance(industry, list) else (industry or "")

                meta = ICAPMetadata(
                    code=code,
                    name=name,
                    category=info.get("category", ""),
                    industry=industry_str,
                    occupation_class=info.get("occupation_code", ""),
                    occupation_name=info.get("occupation_name", ""),
                    source_url="",
                    json_path="",
                    pdf_path="",
                )
                self.metadata[code] = meta
                self.name_to_code[name] = code

            # 職業別索引
            for occ_name, occ_info in occupations_data.items():
                codes = [s["code"] for s in occ_info.get("standards", [])]
                if codes:
                    self.occupation_class_index[occ_name] = codes
                    self.occupation_name_index[occ_name] = codes

            # 職類別索引
            for cat_name, cat_info in categories_data.items():
                codes = cat_info.get("standards", [])
                if codes:
                    self.category_index[cat_name] = codes

            # 行業別索引（從 standards 重建）
            for code, info in standards_data.items():
                industry = info.get("industry", [])
                if isinstance(industry, list):
                    for ind in industry:
                        if ind:
                            self.industry_index[ind].append(code)
                elif industry:
                    self.industry_index[industry].append(code)

            meta_info = data.get("metadata", {})
            logger.success(
                f"從 occupation_index.json 快速載入：{len(self.metadata)} 個職能基準, "
                f"{len(self.occupation_class_index)} 個職業別, "
                f"{len(self.category_index)} 個職類別"
                f"（產生時間：{meta_info.get('generated_at', '未知')}）"
            )
            return True

        except Exception as e:
            logger.warning(f"occupation_index.json 載入失敗，改用掃描模式: {e}")
            return False

    def _load_index(self) -> bool:
        """載入已存儲的索引"""
        if not self._index_path.exists():
            return False

        try:
            with open(self._index_path, 'rb') as f:
                data = pickle.load(f)

            self.metadata = data.get('metadata', {})
            self.category_index = defaultdict(list, data.get('category_index', {}))
            self.industry_index = defaultdict(list, data.get('industry_index', {}))
            self.occupation_class_index = defaultdict(list, data.get('occupation_class_index', {}))
            self.occupation_name_index = defaultdict(list, data.get('occupation_name_index', {}))
            self.name_to_code = data.get('name_to_code', {})

            logger.info(f"載入元資料索引: {len(self.metadata)} 個職能基準, "
                       f"{len(self.industry_index)} 個行業別, "
                       f"{len(self.occupation_class_index)} 個職業別")
            return True

        except Exception as e:
            logger.warning(f"載入索引失敗: {e}")
            return False

    def _save_index(self):
        """存儲索引"""
        try:
            data = {
                'metadata': self.metadata,
                'category_index': dict(self.category_index),
                'industry_index': dict(self.industry_index),
                'occupation_class_index': dict(self.occupation_class_index),
                'occupation_name_index': dict(self.occupation_name_index),
                'name_to_code': self.name_to_code
            }

            with open(self._index_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"索引已存儲: {self._index_path}")

        except Exception as e:
            logger.error(f"存儲索引失敗: {e}")

    def _build_from_json_store(self) -> bool:
        """從 JSON Store 建立索引"""
        if not JSON_STORE_AVAILABLE:
            logger.error("CompetencyJSONStore 模組不可用")
            return False

        # 建立 JSON Store
        if self.json_store is None:
            try:
                self.json_store = CompetencyJSONStore()
            except Exception as e:
                logger.error(f"無法建立 JSON Store: {e}")
                return False

        logger.info("從 JSON Store 建立索引...")

        try:
            for code, standard in self.json_store.standards.items():
                name = standard.name
                category_name = standard.category_name or ""
                industry_name = standard.industry_name or ""
                occupation_code = standard.occupation_code or ""
                occupation_name = standard.occupation_name or ""
                source_file = standard.source_file or ""

                if not code or not name:
                    continue

                # 建立 metadata
                metadata = ICAPMetadata(
                    code=code,
                    name=name,
                    category=category_name,
                    industry=industry_name,
                    occupation_class=occupation_code,
                    occupation_name=occupation_name,
                    source_url="",
                    json_path="",
                    pdf_path=source_file
                )

                self.metadata[code] = metadata
                self.name_to_code[name] = code

                # 建立索引
                if category_name:
                    self.category_index[category_name].append(code)
                if industry_name:
                    self.industry_index[industry_name].append(code)
                if occupation_name:
                    self.occupation_class_index[occupation_name].append(code)
                    self.occupation_name_index[occupation_name].append(code)

            logger.success(f"索引建立完成（JSON Store）: {len(self.metadata)} 個職能基準, "
                          f"{len(self.category_index)} 個職類別, "
                          f"{len(self.industry_index)} 個行業別, "
                          f"{len(self.occupation_class_index)} 個職業別")

            self._save_index()
            return True

        except Exception as e:
            logger.error(f"從 JSON Store 建立索引失敗: {e}")
            return False

    def get_metadata_by_code(self, code: str) -> Optional[ICAPMetadata]:
        """根據代碼取得元資料"""
        return self.metadata.get(code)

    def get_metadata_by_name(self, name: str) -> Optional[ICAPMetadata]:
        """根據名稱取得元資料"""
        code = self.name_to_code.get(name)
        if code:
            return self.metadata.get(code)

        # 模糊匹配
        for stored_name, stored_code in self.name_to_code.items():
            if name in stored_name or stored_name in name:
                return self.metadata.get(stored_code)

        return None

    def get_standards_by_category(self, category: str) -> List[str]:
        """取得職類別下的所有職能基準代碼"""
        return self.category_index.get(category, [])

    def get_standards_by_industry(self, industry: str) -> List[str]:
        """取得行業別下的所有職能基準代碼"""
        return self.industry_index.get(industry, [])

    def get_all_categories(self) -> List[str]:
        """取得所有職類別"""
        return list(self.category_index.keys())

    def get_all_industries(self) -> List[str]:
        """取得所有行業別"""
        return list(self.industry_index.keys())

    def get_category_stats(self) -> Dict[str, int]:
        """取得職類別統計"""
        return {cat: len(codes) for cat, codes in self.category_index.items()}

    def get_standards_by_occupation_class(self, occupation_class: str) -> List[str]:
        """取得通俗職業分類下的所有職能基準代碼"""
        return self.occupation_class_index.get(occupation_class, [])

    def get_standards_by_occupation_name(self, occupation_name: str) -> List[str]:
        """取得通俗職務名稱下的所有職能基準代碼"""
        return self.occupation_name_index.get(occupation_name, [])

    def get_all_occupation_classes(self) -> List[str]:
        """取得所有通俗職業分類"""
        return list(self.occupation_class_index.keys())

    def get_all_occupation_names(self) -> List[str]:
        """取得所有通俗職務名稱"""
        return list(self.occupation_name_index.keys())

    def get_occupation_class_stats(self) -> Dict[str, int]:
        """取得通俗職業分類統計"""
        return {occ: len(codes) for occ, codes in self.occupation_class_index.items()}


class CategoryRouter(nn.Module):
    """職類別路由神經網路 - 判斷查詢與哪些職類別相關"""

    def __init__(self, embedding_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        # 輸入: query_embedding + category_centroid + [distance, category_size, density]
        input_dim = embedding_dim * 2 + 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


class FederatedSearchManager:
    """
    聯邦搜索管理器
    實現 RAGRoute 式的查詢路由，支援多層路由（通俗職業分類 + 職類別）
    """

    def __init__(
        self,
        metadata_index: ICAPMetadataIndex,
        embedding_model,
        embedding_dim: int = 768,
        threshold: float = 0.3
    ):
        """
        初始化聯邦搜索管理器

        Args:
            metadata_index: ICAP 元資料索引
            embedding_model: Embedding 模型 (SentenceTransformer)
            embedding_dim: 向量維度
            threshold: 路由閾值
        """
        self.metadata_index = metadata_index
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.threshold = threshold

        # 職類別資料來源
        self.category_sources: Dict[str, CategoryDataSource] = {}

        # 通俗職業分類資料來源（更精細的路由）
        self.occupation_sources: Dict[str, OccupationDataSource] = {}

        # 路由器
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.router = CategoryRouter(embedding_dim=embedding_dim)
        self.router.to(self.device)
        self.feature_scaler = StandardScaler()
        self.is_trained = False

        # 通俗職業分類路由器
        self.occupation_router = CategoryRouter(embedding_dim=embedding_dim)
        self.occupation_router.to(self.device)
        self.occupation_feature_scaler = StandardScaler()
        self.is_occupation_trained = False

        # 存儲路徑
        self._router_path = config.VECTORDB_DIR / "category_router.pt"
        self._sources_path = config.VECTORDB_DIR / "category_sources.pkl"
        self._occupation_router_path = config.VECTORDB_DIR / "occupation_router.pt"
        self._occupation_sources_path = config.VECTORDB_DIR / "occupation_sources.pkl"

        # 統計
        self.stats = {
            'total_queries': 0,
            'categories_queried': 0,
            'categories_skipped': 0,
            'occupations_queried': 0,
            'occupations_skipped': 0
        }

    def build_category_sources(
        self,
        graph,
        node_embeddings: Dict[str, np.ndarray],
        force_rebuild: bool = False
    ) -> bool:
        """
        建立職類別資料來源

        Args:
            graph: 知識圖譜
            node_embeddings: 節點向量 {node_id -> embedding}
            force_rebuild: 是否強制重建

        Returns:
            是否成功
        """
        if not force_rebuild and self._load_sources():
            return True

        logger.info("建立職類別資料來源...")

        # 取得所有職類別
        categories = self.metadata_index.get_all_categories()

        for category in categories:
            codes = self.metadata_index.get_standards_by_category(category)

            if not codes:
                continue

            # 收集該職類別下所有相關節點的向量
            category_embeddings = []
            category_keywords = set()

            for code in codes:
                # 找到圖中對應的節點
                for node_id, embedding in node_embeddings.items():
                    if code in node_id:
                        category_embeddings.append(embedding)

                # 收集關鍵字
                metadata = self.metadata_index.get_metadata_by_code(code)
                if metadata:
                    category_keywords.add(metadata.name)
                    if metadata.industry:
                        category_keywords.add(metadata.industry)

            if not category_embeddings:
                continue

            # 計算質心和密度
            embeddings_array = np.array(category_embeddings)
            centroid = embeddings_array.mean(axis=0)
            distances = np.linalg.norm(embeddings_array - centroid, axis=1)
            density = 1.0 / (distances.mean() + 1e-6)

            source = CategoryDataSource(
                category_id=category,
                category_name=category,
                standards=codes,
                centroid=centroid,
                density=density,
                size=len(category_embeddings),
                keywords=list(category_keywords)[:20]
            )

            self.category_sources[category] = source
            logger.debug(f"  {category}: {len(codes)} 個職能基準, {len(category_embeddings)} 個節點")

        logger.success(f"建立完成: {len(self.category_sources)} 個職類別資料來源")

        self._save_sources()
        return True

    def build_occupation_sources(
        self,
        graph,
        node_embeddings: Dict[str, np.ndarray],
        force_rebuild: bool = False
    ) -> bool:
        """
        建立通俗職業分類資料來源（比職類別更精細）

        Args:
            graph: 知識圖譜
            node_embeddings: 節點向量 {node_id -> embedding}
            force_rebuild: 是否強制重建

        Returns:
            是否成功
        """
        if not force_rebuild and self._load_occupation_sources():
            return True

        logger.info("建立通俗職業分類資料來源...")

        # 取得所有通俗職業分類
        occupation_classes = self.metadata_index.get_all_occupation_classes()

        if not occupation_classes:
            logger.warning("沒有找到通俗職業分類資料")
            return False

        for occupation in occupation_classes:
            codes = self.metadata_index.get_standards_by_occupation_class(occupation)

            if not codes:
                continue

            # 收集該通俗職業分類下所有相關節點的向量
            occupation_embeddings = []
            occupation_keywords = set()
            parent_categories = set()

            for code in codes:
                # 找到圖中對應的節點
                for node_id, embedding in node_embeddings.items():
                    if code in node_id:
                        occupation_embeddings.append(embedding)

                # 收集關鍵字和父類別
                metadata = self.metadata_index.get_metadata_by_code(code)
                if metadata:
                    occupation_keywords.add(metadata.name)
                    if metadata.industry:
                        occupation_keywords.add(metadata.industry)
                    if metadata.category:
                        parent_categories.add(metadata.category)

            if not occupation_embeddings:
                continue

            # 計算質心和密度
            embeddings_array = np.array(occupation_embeddings)
            centroid = embeddings_array.mean(axis=0)
            distances = np.linalg.norm(embeddings_array - centroid, axis=1)
            density = 1.0 / (distances.mean() + 1e-6)

            source = OccupationDataSource(
                occupation_id=occupation,
                occupation_name=occupation,
                standards=codes,
                centroid=centroid,
                density=density,
                size=len(occupation_embeddings),
                keywords=list(occupation_keywords)[:20],
                parent_category=list(parent_categories)[0] if parent_categories else ""
            )

            self.occupation_sources[occupation] = source
            logger.debug(f"  {occupation}: {len(codes)} 個職能基準, {len(occupation_embeddings)} 個節點")

        logger.success(f"建立完成: {len(self.occupation_sources)} 個通俗職業分類資料來源")

        self._save_occupation_sources()
        return True

    def _load_sources(self) -> bool:
        """載入已存儲的資料來源"""
        if not self._sources_path.exists():
            return False

        try:
            with open(self._sources_path, 'rb') as f:
                self.category_sources = pickle.load(f)

            logger.info(f"載入職類別資料來源: {len(self.category_sources)} 個")
            return True

        except Exception as e:
            logger.warning(f"載入資料來源失敗: {e}")
            return False

    def _save_sources(self):
        """存儲資料來源"""
        try:
            with open(self._sources_path, 'wb') as f:
                pickle.dump(self.category_sources, f)

            logger.info(f"資料來源已存儲: {self._sources_path}")

        except Exception as e:
            logger.error(f"存儲資料來源失敗: {e}")

    def _load_occupation_sources(self) -> bool:
        """載入已存儲的通俗職業分類資料來源"""
        if not self._occupation_sources_path.exists():
            return False

        try:
            with open(self._occupation_sources_path, 'rb') as f:
                self.occupation_sources = pickle.load(f)

            logger.info(f"載入通俗職業分類資料來源: {len(self.occupation_sources)} 個")
            return True

        except Exception as e:
            logger.warning(f"載入通俗職業分類資料來源失敗: {e}")
            return False

    def _save_occupation_sources(self):
        """存儲通俗職業分類資料來源"""
        try:
            with open(self._occupation_sources_path, 'wb') as f:
                pickle.dump(self.occupation_sources, f)

            logger.info(f"通俗職業分類資料來源已存儲: {self._occupation_sources_path}")

        except Exception as e:
            logger.error(f"存儲通俗職業分類資料來源失敗: {e}")

    def _load_router(self) -> bool:
        """載入已訓練的路由器"""
        if not self._router_path.exists():
            return False

        try:
            ckpt = torch.load(self._router_path, map_location=self.device, weights_only=False)
            self.router.load_state_dict(ckpt['router_state_dict'])
            self.feature_scaler = ckpt.get('scaler', StandardScaler())
            self.is_trained = True
            logger.info("路由器已載入")
            return True

        except Exception as e:
            logger.warning(f"載入路由器失敗: {e}")
            return False

    def _save_router(self):
        """存儲路由器"""
        try:
            torch.save({
                'router_state_dict': self.router.state_dict(),
                'scaler': self.feature_scaler,
                'embedding_dim': self.embedding_dim
            }, self._router_path)

            logger.info(f"路由器已存儲: {self._router_path}")

        except Exception as e:
            logger.error(f"存儲路由器失敗: {e}")

    def _load_occupation_router(self) -> bool:
        """載入已訓練的通俗職業分類路由器"""
        if not self._occupation_router_path.exists():
            return False

        try:
            ckpt = torch.load(self._occupation_router_path, map_location=self.device, weights_only=False)
            self.occupation_router.load_state_dict(ckpt['router_state_dict'])
            self.occupation_feature_scaler = ckpt.get('scaler', StandardScaler())
            self.is_occupation_trained = True
            logger.info("通俗職業分類路由器已載入")
            return True

        except Exception as e:
            logger.warning(f"載入通俗職業分類路由器失敗: {e}")
            return False

    def _save_occupation_router(self):
        """存儲通俗職業分類路由器"""
        try:
            torch.save({
                'router_state_dict': self.occupation_router.state_dict(),
                'scaler': self.occupation_feature_scaler,
                'embedding_dim': self.embedding_dim
            }, self._occupation_router_path)

            logger.info(f"通俗職業分類路由器已存儲: {self._occupation_router_path}")

        except Exception as e:
            logger.error(f"存儲通俗職業分類路由器失敗: {e}")

    def get_router_features(self, query_embedding: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        計算查詢與各職類別的路由特徵

        Args:
            query_embedding: 查詢向量

        Returns:
            [(category_id, feature_vector), ...]
        """
        features = []

        for cat_id, source in self.category_sources.items():
            if source.centroid is None:
                continue

            # 計算距離
            distance = np.linalg.norm(query_embedding - source.centroid)

            # 組合特徵: [query_emb, centroid, distance, size, density]
            feature_vector = np.concatenate([
                query_embedding,
                source.centroid,
                [distance],
                [source.size],
                [source.density]
            ])

            features.append((cat_id, feature_vector))

        return features

    def route_query(
        self,
        query: str,
        top_k: int = 3,
        use_router: bool = True
    ) -> List[Tuple[str, float]]:
        """
        路由查詢到相關的職類別

        Args:
            query: 查詢文字
            top_k: 返回的職類別數量
            use_router: 是否使用訓練的路由器（否則使用質心距離）

        Returns:
            [(category_id, relevance_score), ...]
        """
        # 生成查詢向量
        query_embedding = self.embedding_model.encode([query])[0]

        if use_router and self.is_trained:
            return self._route_with_router(query_embedding, top_k)
        else:
            return self._route_with_centroid(query_embedding, top_k)

    def _route_with_router(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """使用訓練的路由器進行路由"""
        features = self.get_router_features(query_embedding)

        if not features:
            return []

        # 準備輸入
        cat_ids = [f[0] for f in features]
        X = np.array([f[1] for f in features], dtype=np.float32)

        # 標準化
        X_scaled = self.feature_scaler.transform(X)

        # 預測
        self.router.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = torch.sigmoid(self.router(X_tensor)).cpu().numpy().flatten()

        # 排序並過濾
        results = []
        for cat_id, score in zip(cat_ids, scores):
            if score >= self.threshold:
                results.append((cat_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _route_with_centroid(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """使用質心距離進行路由（fallback）"""
        results = []

        for cat_id, source in self.category_sources.items():
            if source.centroid is None:
                continue

            # 計算餘弦相似度
            similarity = np.dot(query_embedding, source.centroid) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(source.centroid) + 1e-8
            )

            results.append((cat_id, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_occupation_router_features(self, query_embedding: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        計算查詢與各通俗職業分類的路由特徵

        Args:
            query_embedding: 查詢向量

        Returns:
            [(occupation_id, feature_vector), ...]
        """
        features = []

        for occ_id, source in self.occupation_sources.items():
            if source.centroid is None:
                continue

            # 計算距離
            distance = np.linalg.norm(query_embedding - source.centroid)

            # 組合特徵: [query_emb, centroid, distance, size, density]
            feature_vector = np.concatenate([
                query_embedding,
                source.centroid,
                [distance],
                [source.size],
                [source.density]
            ])

            features.append((occ_id, feature_vector))

        return features

    def route_query_by_occupation(
        self,
        query: str,
        top_k: int = 5,
        use_router: bool = True
    ) -> List[Tuple[str, float]]:
        """
        路由查詢到相關的通俗職業分類

        Args:
            query: 查詢文字
            top_k: 返回的通俗職業分類數量
            use_router: 是否使用訓練的路由器（否則使用質心距離）

        Returns:
            [(occupation_id, relevance_score), ...]
        """
        if not self.occupation_sources:
            return []

        # 生成查詢向量
        query_embedding = self.embedding_model.encode([query])[0]

        if use_router and self.is_occupation_trained:
            return self._route_occupation_with_router(query_embedding, top_k)
        else:
            return self._route_occupation_with_centroid(query_embedding, top_k)

    def _route_occupation_with_router(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """使用訓練的路由器進行通俗職業分類路由"""
        features = self.get_occupation_router_features(query_embedding)

        if not features:
            return []

        # 準備輸入
        occ_ids = [f[0] for f in features]
        X = np.array([f[1] for f in features], dtype=np.float32)

        # 標準化
        X_scaled = self.occupation_feature_scaler.transform(X)

        # 預測
        self.occupation_router.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = torch.sigmoid(self.occupation_router(X_tensor)).cpu().numpy().flatten()

        # 排序並過濾
        results = []
        for occ_id, score in zip(occ_ids, scores):
            if score >= self.threshold:
                results.append((occ_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _route_occupation_with_centroid(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """使用質心距離進行通俗職業分類路由（fallback）"""
        results = []

        for occ_id, source in self.occupation_sources.items():
            if source.centroid is None:
                continue

            # 計算餘弦相似度
            similarity = np.dot(query_embedding, source.centroid) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(source.centroid) + 1e-8
            )

            results.append((occ_id, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def train_router(
        self,
        node_embeddings: Dict[str, np.ndarray],
        num_epochs: int = 30,
        samples_per_category: int = 50
    ) -> bool:
        """
        訓練路由器

        Args:
            node_embeddings: 節點向量
            num_epochs: 訓練週期
            samples_per_category: 每個職類別的樣本數

        Returns:
            是否成功
        """
        if len(self.category_sources) < 2:
            logger.error("至少需要 2 個職類別才能訓練路由器")
            return False

        logger.info(f"開始訓練路由器 (職類別: {len(self.category_sources)})")

        # 收集訓練數據
        X_train, y_train = [], []

        for cat_id, source in self.category_sources.items():
            logger.debug(f"處理職類別: {cat_id}")

            # 從該職類別中採樣節點
            category_nodes = []
            for code in source.standards:
                for node_id, embedding in node_embeddings.items():
                    if code in node_id:
                        category_nodes.append(embedding)

            if not category_nodes:
                continue

            # 隨機採樣
            sample_size = min(samples_per_category, len(category_nodes))
            indices = np.random.choice(len(category_nodes), sample_size, replace=False)
            sampled_embeddings = [category_nodes[i] for i in indices]

            # 為每個樣本生成特徵
            for emb in sampled_embeddings:
                features = self.get_router_features(emb)

                for feature_cat_id, feature_vector in features:
                    X_train.append(feature_vector)
                    # 正樣本: 查詢來自該職類別
                    y_train.append(1.0 if feature_cat_id == cat_id else 0.0)

        if not X_train:
            logger.error("沒有訓練數據")
            return False

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)

        # 正負樣本平衡
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        pos_weight = neg_count / (pos_count + 1e-6)

        logger.info(f"訓練樣本: {len(X_train)} (正: {int(pos_count)}, 負: {int(neg_count)})")

        # 標準化
        self.feature_scaler.fit(X_train)
        X_scaled = self.feature_scaler.transform(X_train)

        # 訓練
        dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.FloatTensor(y_train)
        )
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(self.router.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        self.router.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.router(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

        self.is_trained = True
        self._save_router()

        logger.success("路由器訓練完成")
        return True

    def train_occupation_router(
        self,
        node_embeddings: Dict[str, np.ndarray],
        num_epochs: int = 30,
        samples_per_occupation: int = 50
    ) -> bool:
        """
        訓練通俗職業分類路由器

        Args:
            node_embeddings: 節點向量
            num_epochs: 訓練週期
            samples_per_occupation: 每個通俗職業分類的樣本數

        Returns:
            是否成功
        """
        if len(self.occupation_sources) < 2:
            logger.warning("通俗職業分類太少，跳過路由器訓練")
            return False

        logger.info(f"開始訓練通俗職業分類路由器 (分類數: {len(self.occupation_sources)})")

        # 收集訓練數據
        X_train, y_train = [], []

        for occ_id, source in self.occupation_sources.items():
            logger.debug(f"處理通俗職業分類: {occ_id}")

            # 從該分類中採樣節點
            occupation_nodes = []
            for code in source.standards:
                for node_id, embedding in node_embeddings.items():
                    if code in node_id:
                        occupation_nodes.append(embedding)

            if not occupation_nodes:
                continue

            # 隨機採樣
            sample_size = min(samples_per_occupation, len(occupation_nodes))
            indices = np.random.choice(len(occupation_nodes), sample_size, replace=False)
            sampled_embeddings = [occupation_nodes[i] for i in indices]

            # 為每個樣本生成特徵
            for emb in sampled_embeddings:
                features = self.get_occupation_router_features(emb)

                for feature_occ_id, feature_vector in features:
                    X_train.append(feature_vector)
                    # 正樣本: 查詢來自該通俗職業分類
                    y_train.append(1.0 if feature_occ_id == occ_id else 0.0)

        if not X_train:
            logger.error("沒有訓練數據")
            return False

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)

        # 正負樣本平衡
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        pos_weight = neg_count / (pos_count + 1e-6)

        logger.info(f"訓練樣本: {len(X_train)} (正: {int(pos_count)}, 負: {int(neg_count)})")

        # 標準化
        self.occupation_feature_scaler.fit(X_train)
        X_scaled = self.occupation_feature_scaler.transform(X_train)

        # 訓練
        dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.FloatTensor(y_train)
        )
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(self.occupation_router.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        self.occupation_router.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.occupation_router(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

        self.is_occupation_trained = True
        self._save_occupation_router()

        logger.success("通俗職業分類路由器訓練完成")
        return True

    def federated_search(
        self,
        query: str,
        vector_search_func,
        top_k_categories: int = 3,
        top_k_occupations: int = 5,
        top_k_results: int = 10,
        use_occupation_routing: bool = True
    ) -> Dict[str, Any]:
        """
        執行聯邦搜索（支援多層路由）

        Args:
            query: 查詢文字
            vector_search_func: 向量搜索函數 (query, filter_nodes) -> results
            top_k_categories: 搜索的職類別數量
            top_k_occupations: 搜索的通俗職業分類數量
            top_k_results: 結果數量
            use_occupation_routing: 是否使用通俗職業分類路由（更精細）

        Returns:
            {
                'query': str,
                'routed_categories': [(cat_id, score), ...],
                'routed_occupations': [(occ_id, score), ...],
                'results': [result_dict, ...],
                'stats': {...}
            }
        """
        self.stats['total_queries'] += 1

        relevant_codes = set()
        routed_categories = []
        routed_occupations = []

        # 1. 優先使用通俗職業分類路由（更精細）
        if use_occupation_routing and self.occupation_sources:
            routed_occupations = self.route_query_by_occupation(query, top_k=top_k_occupations)

            if routed_occupations:
                self.stats['occupations_queried'] += len(routed_occupations)
                self.stats['occupations_skipped'] += len(self.occupation_sources) - len(routed_occupations)

                # 收集通俗職業分類下的職能基準代碼
                for occ_id, score in routed_occupations:
                    source = self.occupation_sources.get(occ_id)
                    if source:
                        relevant_codes.update(source.standards)

                logger.debug(f"通俗職業分類路由: {[o[0] for o in routed_occupations]}")

        # 2. 如果通俗職業分類路由結果不足，補充使用職類別路由
        if len(relevant_codes) < 10:
            routed_categories = self.route_query(query, top_k=top_k_categories)

            if not routed_categories:
                # fallback: 搜索所有類別
                logger.warning("路由失敗，使用全域搜索")
                routed_categories = [(cat_id, 0.5) for cat_id in list(self.category_sources.keys())[:top_k_categories]]

            self.stats['categories_queried'] += len(routed_categories)
            self.stats['categories_skipped'] += len(self.category_sources) - len(routed_categories)

            # 收集職類別下的職能基準代碼
            for cat_id, score in routed_categories:
                source = self.category_sources.get(cat_id)
                if source:
                    relevant_codes.update(source.standards)

        # 3. 執行向量搜索（限制在相關節點）
        results = vector_search_func(query, relevant_codes, top_k_results)

        return {
            'query': query,
            'routed_categories': routed_categories,
            'routed_occupations': routed_occupations,
            'results': results,
            'stats': {
                'categories_searched': len(routed_categories),
                'categories_total': len(self.category_sources),
                'occupations_searched': len(routed_occupations),
                'occupations_total': len(self.occupation_sources),
                'relevant_standards': len(relevant_codes)
            }
        }

    def get_category_info(self, category_id: str) -> Optional[Dict]:
        """取得職類別資訊"""
        source = self.category_sources.get(category_id)
        if not source:
            return None

        return {
            'id': source.category_id,
            'name': source.category_name,
            'standards_count': len(source.standards),
            'node_count': source.size,
            'density': source.density,
            'keywords': source.keywords,
            'standards': source.standards[:10]  # 只返回前10個
        }

    def list_categories(self) -> List[Dict]:
        """列出所有職類別"""
        result = []
        for cat_id, source in self.category_sources.items():
            result.append({
                'id': cat_id,
                'name': source.category_name,
                'standards_count': len(source.standards),
                'node_count': source.size
            })

        result.sort(key=lambda x: x['standards_count'], reverse=True)
        return result

    def get_occupation_info(self, occupation_id: str) -> Optional[Dict]:
        """取得通俗職業分類資訊"""
        source = self.occupation_sources.get(occupation_id)
        if not source:
            return None

        return {
            'id': source.occupation_id,
            'name': source.occupation_name,
            'parent_category': source.parent_category,
            'standards_count': len(source.standards),
            'node_count': source.size,
            'density': source.density,
            'keywords': source.keywords,
            'standards': source.standards[:10]  # 只返回前10個
        }

    def list_occupations(self) -> List[Dict]:
        """列出所有通俗職業分類"""
        result = []
        for occ_id, source in self.occupation_sources.items():
            result.append({
                'id': occ_id,
                'name': source.occupation_name,
                'parent_category': source.parent_category,
                'standards_count': len(source.standards),
                'node_count': source.size
            })

        result.sort(key=lambda x: x['standards_count'], reverse=True)
        return result


# ========================================
# 工具函數
# ========================================

def create_federated_search_system(
    knowledge_graph,
    embedding_model,
    force_rebuild: bool = False,
    existing_embeddings: Dict[str, np.ndarray] = None,
    data_source: Optional['CompetencyJSONStore'] = None,
) -> Optional[FederatedSearchManager]:
    """
    建立聯邦搜索系統（支援多層路由）

    Args:
        knowledge_graph: 知識圖譜實例
        embedding_model: SentenceTransformer 模型
        force_rebuild: 是否強制重建
        existing_embeddings: 已計算的節點向量（可選，用於加速）
        data_source: 資料來源（CompetencyJSONStore）

    Returns:
        FederatedSearchManager 或 None
    """
    # 1. 建立元資料索引
    metadata_index = ICAPMetadataIndex(data_source=data_source)
    if not metadata_index.build_index(force_rebuild=force_rebuild):
        logger.error("無法建立元資料索引")
        return None

    # 2. 建立聯邦搜索管理器
    manager = FederatedSearchManager(
        metadata_index=metadata_index,
        embedding_model=embedding_model,
        embedding_dim=config.EMBEDDING_DIM
    )

    # 3. 嘗試載入已存儲的資料來源和路由器
    sources_loaded = manager._load_sources()
    router_loaded = manager._load_router()
    occupation_sources_loaded = manager._load_occupation_sources()
    occupation_router_loaded = manager._load_occupation_router()

    # 如果都已載入且不強制重建，直接返回
    all_loaded = (sources_loaded and router_loaded and
                  occupation_sources_loaded and occupation_router_loaded)
    if all_loaded and not force_rebuild:
        logger.success(f"聯邦搜索系統已從緩存載入: "
                      f"{len(manager.category_sources)} 個職類別, "
                      f"{len(manager.occupation_sources)} 個通俗職業分類")
        return manager

    # 4. 收集節點向量（使用已有的或重新計算）
    if existing_embeddings:
        logger.info(f"使用已有的節點向量: {len(existing_embeddings)} 個")
        node_embeddings = existing_embeddings
    else:
        logger.info("收集節點向量...")
        node_embeddings = {}

        target_types = ["知識", "技能", "態度", "行為指標", "工作任務", "主要職責", "職能基準"]
        texts = []
        node_ids = []

        for node_type in target_types:
            nodes = knowledge_graph.get_nodes_by_type(node_type)
            for node_id in nodes:
                node_data = knowledge_graph.get_node_data(node_id)
                if node_data:
                    text = f"{node_data.get('name', '')} {node_data.get('description', '')}"
                    if text.strip():
                        texts.append(text)
                        node_ids.append(node_id)

        if texts:
            logger.info(f"向量化 {len(texts)} 個節點...")
            embeddings = embedding_model.encode(texts, show_progress_bar=True)

            for node_id, embedding in zip(node_ids, embeddings):
                node_embeddings[node_id] = embedding

    # 5. 建立職類別資料來源（如果沒有從緩存載入）
    if not sources_loaded or force_rebuild:
        if not manager.build_category_sources(knowledge_graph, node_embeddings, force_rebuild=force_rebuild):
            logger.warning("無法建立職類別資料來源")

    # 6. 建立通俗職業分類資料來源（如果沒有從緩存載入）
    if not occupation_sources_loaded or force_rebuild:
        if node_embeddings:
            manager.build_occupation_sources(knowledge_graph, node_embeddings, force_rebuild=force_rebuild)

    # 7. 訓練職類別路由器（如果沒有從緩存載入）
    if not router_loaded or force_rebuild:
        if node_embeddings:
            logger.info("訓練職類別路由器...")
            manager.train_router(node_embeddings, num_epochs=15)

    # 8. 訓練通俗職業分類路由器（如果沒有從緩存載入）
    if not occupation_router_loaded or force_rebuild:
        if node_embeddings and manager.occupation_sources:
            logger.info("訓練通俗職業分類路由器...")
            manager.train_occupation_router(node_embeddings, num_epochs=15)

    logger.success(f"聯邦搜索系統初始化完成: "
                  f"{len(manager.category_sources)} 個職類別, "
                  f"{len(manager.occupation_sources)} 個通俗職業分類")

    return manager
