"""
Graph RAG 查詢系統
整合知識圖譜遍歷和向量檢索，實現五種查詢類型
支援 LLM 生成答案（使用 TAIDE 模型）
支援向量索引持久化存儲
支援 RAGRoute 式聯邦搜索（按職類別分群）
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss

# LLM 支援
try:
    from langchain_community.llms import LlamaCpp
    LLM_AVAILABLE = True
except ImportError:
    LlamaCpp = None
    LLM_AVAILABLE = False
    logger.warning("langchain_community 未安裝，LLM 功能將無法使用")

from config import get_config
from graph_builder import CompetencyKnowledgeGraph

# 聯邦搜索支援
try:
    from federated_search import (
        FederatedSearchManager,
        ICAPMetadataIndex,
        create_federated_search_system
    )
    FEDERATED_SEARCH_AVAILABLE = True
except ImportError:
    FederatedSearchManager = None
    ICAPMetadataIndex = None
    create_federated_search_system = None
    FEDERATED_SEARCH_AVAILABLE = False
    logger.warning("聯邦搜索模組未載入")

config = get_config()


@dataclass
class QueryResult:
    """查詢結果"""
    query: str
    query_type: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    graph_context: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    sources: List[str] = field(default_factory=list)


class GraphRAGQueryEngine:
    """Graph RAG 查詢引擎"""

    def __init__(
        self,
        knowledge_graph: CompetencyKnowledgeGraph,
        embedding_model: Optional[str] = None
    ):
        """
        初始化查詢引擎

        Args:
            knowledge_graph: 知識圖譜實例
            embedding_model: Embedding 模型名稱
        """
        self.kg = knowledge_graph
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL

        # Embedding 模型和向量索引
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vector_index: Optional[faiss.Index] = None
        self.vector_id_map: Dict[int, str] = {}  # 向量索引 -> 節點ID
        self.node_texts: Dict[str, str] = {}     # 節點ID -> 文本

        # LLM（可選）
        self.llm = None
        self.llm_initialized = False

        # 聯邦搜索（可選）
        self.federated_manager: Optional[FederatedSearchManager] = None
        self.federated_initialized = False

        # JSON Store 快取（避免每次查詢都重新載入）
        self._json_store = None
        try:
            from competency_store import CompetencyJSONStore
            self._json_store = CompetencyJSONStore(config.PARSED_JSON_V2_DIR)
            logger.info(f"JSON Store 已載入: {len(self._json_store.standards)} 個職能基準")
        except Exception as e:
            logger.warning(f"JSON Store 載入失敗: {e}")

        logger.info("Graph RAG 查詢引擎初始化完成")

    def initialize_llm(self, model_path: str = None, callback=None):
        """
        初始化 LLM 模型

        Args:
            model_path: 模型路徑（預設使用 config 設定）
            callback: 進度回調函數
        """
        if not LLM_AVAILABLE:
            logger.error("LlamaCpp 未安裝，無法初始化 LLM")
            return False

        if self.llm_initialized:
            logger.info("LLM 模型已初始化")
            return True

        model_path = model_path or config.MODEL_PATH

        if not Path(model_path).exists():
            logger.error(f"LLM 模型檔案不存在: {model_path}")
            return False

        try:
            if callback:
                callback("載入 LLM 模型中...")

            logger.info(f"載入 LLM 模型: {model_path}")

            # 取得 stop tokens，參考 rag_ui_v3_faiss.py 的實現
            stop_tokens = getattr(config, 'LLM_STOP_TOKENS', [
                "\n問題:", "\n問:", "問題列表", "\n【用戶問題】"
            ])

            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=config.N_CTX,
                n_threads=config.N_THREADS,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                verbose=False,
                stop=stop_tokens
            )
            self.llm_initialized = True
            logger.success("LLM 模型初始化完成")
            return True

        except Exception as e:
            logger.error(f"LLM 模型初始化失敗: {e}")
            return False

    def is_llm_ready(self) -> bool:
        """檢查 LLM 是否已就緒"""
        return self.llm_initialized and self.llm is not None

    def _get_vector_index_path(self) -> Path:
        """取得向量索引存儲路徑"""
        return config.VECTORDB_DIR / "graph_rag_vectors"

    def _get_metadata_path(self) -> Path:
        """取得元數據存儲路徑"""
        return config.VECTORDB_DIR / "graph_rag_metadata.pkl"

    def initialize_embeddings(self, force_rebuild: bool = False):
        """
        初始化 Embedding 模型和向量索引

        Args:
            force_rebuild: 是否強制重建向量索引（忽略已存儲的索引）
        """
        if self.embedding_model is not None and self.vector_index is not None:
            logger.info("Embedding 模型已初始化")
            return

        logger.info(f"載入 Embedding 模型: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # 嘗試載入已存儲的向量索引
        if not force_rebuild and self._load_vector_index():
            logger.success("成功載入已存儲的向量索引")
            return

        # 建立新的向量索引
        self._build_vector_index()

        # 存儲向量索引
        self._save_vector_index()

        logger.success("Embedding 模型和向量索引初始化完成")

    def _load_vector_index(self) -> bool:
        """
        載入已存儲的向量索引

        Returns:
            bool: 是否成功載入
        """
        index_path = self._get_vector_index_path()
        metadata_path = self._get_metadata_path()

        faiss_index_file = index_path / "index.faiss"

        if not faiss_index_file.exists() or not metadata_path.exists():
            logger.info("找不到已存儲的向量索引，將建立新索引")
            return False

        try:
            # 載入 FAISS 索引
            logger.info("載入已存儲的 FAISS 索引...")
            self.vector_index = faiss.read_index(str(faiss_index_file))

            # 載入元數據
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.vector_id_map = metadata.get('vector_id_map', {})
            self.node_texts = metadata.get('node_texts', {})
            stored_node_count = metadata.get('node_count', 0)
            stored_model = metadata.get('embedding_model', '')

            # 驗證索引有效性
            current_node_count = sum(
                len(list(self.kg.get_nodes_by_type(t)))
                for t in ["知識", "技能", "態度", "行為指標", "工作任務", "主要職責", "職能基準"]
            )

            if stored_model != self.embedding_model_name:
                logger.warning(f"Embedding 模型不匹配 ({stored_model} vs {self.embedding_model_name})，需重建索引")
                return False

            if abs(stored_node_count - current_node_count) > current_node_count * 0.1:
                logger.warning(f"節點數量變化過大 ({stored_node_count} vs {current_node_count})，需重建索引")
                return False

            logger.info(f"向量索引載入完成: {self.vector_index.ntotal} 個向量")
            return True

        except Exception as e:
            logger.warning(f"載入向量索引失敗: {e}，將重建索引")
            return False

    def _save_vector_index(self):
        """存儲向量索引到檔案"""
        if self.vector_index is None:
            logger.warning("向量索引不存在，無法存儲")
            return

        index_path = self._get_vector_index_path()
        metadata_path = self._get_metadata_path()

        # 確保目錄存在
        index_path.mkdir(parents=True, exist_ok=True)

        try:
            # 存儲 FAISS 索引
            faiss_index_file = index_path / "index.faiss"
            logger.info(f"存儲 FAISS 索引到: {faiss_index_file}")
            faiss.write_index(self.vector_index, str(faiss_index_file))

            # 計算當前節點數量
            current_node_count = sum(
                len(list(self.kg.get_nodes_by_type(t)))
                for t in ["知識", "技能", "態度", "行為指標", "工作任務", "主要職責", "職能基準"]
            )

            # 存儲元數據
            metadata = {
                'vector_id_map': self.vector_id_map,
                'node_texts': self.node_texts,
                'node_count': current_node_count,
                'embedding_model': self.embedding_model_name,
                'vector_count': self.vector_index.ntotal
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.success(f"向量索引已存儲: {self.vector_index.ntotal} 個向量")

        except Exception as e:
            logger.error(f"存儲向量索引失敗: {e}")

    def _build_vector_index(self):
        """建立 FAISS 向量索引"""
        logger.info("建立向量索引...")

        # 收集需要向量化的文本
        texts = []
        node_ids = []

        # 對知識、技能、態度、行為指標等建立向量
        target_types = ["知識", "技能", "態度", "行為指標", "工作任務", "主要職責", "職能基準"]

        for node_type in target_types:
            nodes = self.kg.get_nodes_by_type(node_type)
            for node_id in nodes:
                node_data = self.kg.get_node_data(node_id)
                if node_data:
                    # 組合節點文本
                    text_parts = [
                        node_data.get("name", ""),
                        node_data.get("description", ""),
                    ]
                    text = " ".join([p for p in text_parts if p])

                    if text.strip():
                        texts.append(text)
                        node_ids.append(node_id)
                        self.node_texts[node_id] = text

        if not texts:
            logger.warning("沒有可向量化的文本")
            return

        # 生成向量
        logger.info(f"向量化 {len(texts)} 個節點...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # 建立 FAISS 索引
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # 使用內積相似度

        # 正規化向量（用於 cosine similarity）
        faiss.normalize_L2(embeddings)
        self.vector_index.add(embeddings)

        # 建立 ID 映射
        self.vector_id_map = {}
        for i, node_id in enumerate(node_ids):
            self.vector_id_map[i] = node_id

        logger.success(f"向量索引建立完成: {len(texts)} 個向量")

    def initialize_federated_search(self, force_rebuild: bool = False, callback=None):
        """
        初始化聯邦搜索系統

        Args:
            force_rebuild: 是否強制重建
            callback: 進度回調函數
        """
        if not FEDERATED_SEARCH_AVAILABLE:
            logger.warning("聯邦搜索模組不可用")
            return False

        if self.federated_initialized and not force_rebuild:
            logger.info("聯邦搜索已初始化")
            return True

        # 確保 Embedding 模型已載入
        if self.embedding_model is None:
            if callback:
                callback("載入 Embedding 模型...")
            self.initialize_embeddings()

        try:
            if callback:
                callback("建立 ICAP 元資料索引...")

            # 嘗試複用已計算的向量（從 vector_id_map 和 node_texts 重建）
            existing_embeddings = None
            if self.vector_index is not None and self.vector_id_map:
                if callback:
                    callback("複用已計算的向量...")
                existing_embeddings = {}

                # 從 FAISS 索引中提取向量
                for idx, node_id in self.vector_id_map.items():
                    try:
                        vector = self.vector_index.reconstruct(int(idx))
                        existing_embeddings[node_id] = vector
                    except Exception:
                        pass

                if existing_embeddings:
                    logger.info(f"複用 {len(existing_embeddings)} 個已計算的向量")

            self.federated_manager = create_federated_search_system(
                knowledge_graph=self.kg,
                embedding_model=self.embedding_model,
                force_rebuild=force_rebuild,
                existing_embeddings=existing_embeddings,
                data_source=self._json_store
            )

            if self.federated_manager:
                self.federated_initialized = True
                logger.success("聯邦搜索系統初始化完成")
                return True
            else:
                logger.warning("聯邦搜索系統初始化失敗")
                return False

        except Exception as e:
            logger.error(f"聯邦搜索初始化錯誤: {e}")
            return False

    def is_federated_ready(self) -> bool:
        """檢查聯邦搜索是否已就緒"""
        return self.federated_initialized and self.federated_manager is not None

    def get_category_list(self) -> List[Dict]:
        """取得所有職類別列表"""
        if not self.is_federated_ready():
            return []
        return self.federated_manager.list_categories()

    def get_category_info(self, category_id: str) -> Optional[Dict]:
        """取得職類別詳細資訊"""
        if not self.is_federated_ready():
            return None
        return self.federated_manager.get_category_info(category_id)

    def get_occupation_list(self) -> List[Dict]:
        """取得所有通俗職業分類列表"""
        if not self.is_federated_ready():
            return []
        return self.federated_manager.list_occupations()

    def get_occupation_info(self, occupation_id: str) -> Optional[Dict]:
        """取得通俗職業分類詳細資訊"""
        if not self.is_federated_ready():
            return None
        return self.federated_manager.get_occupation_info(occupation_id)

    # ========================================
    # 查詢類型 1: 跨職業比較
    # ========================================

    def compare_occupations(
        self,
        occupation1: str,
        occupation2: str
    ) -> QueryResult:
        """
        比較兩個職業的共同和差異技能/知識

        Args:
            occupation1: 職業1名稱或代碼
            occupation2: 職業2名稱或代碼

        Returns:
            QueryResult
        """
        result = QueryResult(
            query=f"比較 {occupation1} 和 {occupation2}",
            query_type="跨職業比較"
        )

        # 找到對應的職能基準節點
        std1 = self._find_standard_node(occupation1)
        std2 = self._find_standard_node(occupation2)

        if not std1 or not std2:
            result.answer = f"找不到職能基準: {occupation1 if not std1 else occupation2}"
            return result

        # 取得兩者的知識和技能
        ks1 = self._get_all_knowledge_skills(std1)
        ks2 = self._get_all_knowledge_skills(std2)

        # 計算交集和差集
        common = ks1["知識"].intersection(ks2["知識"]) | ks1["技能"].intersection(ks2["技能"])
        only1 = (ks1["知識"] - ks2["知識"]) | (ks1["技能"] - ks2["技能"])
        only2 = (ks2["知識"] - ks1["知識"]) | (ks2["技能"] - ks1["技能"])

        # 取得名稱
        common_details = self._get_ks_details(common)
        only1_details = self._get_ks_details(only1)
        only2_details = self._get_ks_details(only2)

        result.results = [
            {"type": "共同能力", "items": common_details},
            {"type": f"{occupation1} 獨有", "items": only1_details},
            {"type": f"{occupation2} 獨有", "items": only2_details},
        ]

        result.graph_context = {
            "standard1": self.kg.get_node_data(std1),
            "standard2": self.kg.get_node_data(std2),
            "overlap_ratio": len(common) / max(len(ks1["知識"] | ks1["技能"]), 1)
        }

        # 生成答案
        result.answer = self._generate_comparison_answer(
            occupation1, occupation2, common_details, only1_details, only2_details
        )

        return result

    def _find_standard_node(self, query: str) -> Optional[str]:
        """根據名稱或代碼找到職能基準節點"""
        standards = self.kg.get_nodes_by_type("職能基準")

        for std_id in standards:
            node_data = self.kg.get_node_data(std_id)
            if node_data:
                if query in node_data.get("name", "") or query == node_data.get("node_id", ""):
                    return std_id

        # 如果精確匹配失敗，嘗試模糊匹配
        for std_id in standards:
            node_data = self.kg.get_node_data(std_id)
            if node_data and query in node_data.get("name", ""):
                return std_id

        return None

    def _get_all_knowledge_skills(self, standard_id: str) -> Dict[str, Set[str]]:
        """取得職能基準的所有知識和技能"""
        result = {"知識": set(), "技能": set()}

        # 遍歷職責 -> 任務 -> 知識/技能
        for _, duty_id, duty_edge in self.kg.graph.out_edges(standard_id, data=True):
            if duty_edge.get("edge_type") != "包含職責":
                continue

            for _, task_id, task_edge in self.kg.graph.out_edges(duty_id, data=True):
                if task_edge.get("edge_type") != "包含任務":
                    continue

                for _, ks_id, ks_edge in self.kg.graph.out_edges(task_id, data=True):
                    if ks_edge.get("edge_type") == "需要知識":
                        result["知識"].add(ks_id)
                    elif ks_edge.get("edge_type") == "需要技能":
                        result["技能"].add(ks_id)

        return result

    def _get_ks_details(self, ks_ids: Set[str]) -> List[Dict]:
        """取得知識/技能詳細資訊"""
        details = []
        for ks_id in ks_ids:
            node_data = self.kg.get_node_data(ks_id)
            if node_data:
                details.append({
                    "code": node_data.get("node_id", ""),
                    "name": node_data.get("name", ""),
                    "type": node_data.get("node_type", "")
                })
        return details

    def _generate_comparison_answer(
        self,
        occ1: str,
        occ2: str,
        common: List[Dict],
        only1: List[Dict],
        only2: List[Dict]
    ) -> str:
        """生成比較答案"""
        answer_parts = []

        answer_parts.append(f"## {occ1} 和 {occ2} 的職能比較\n")

        answer_parts.append(f"### 共同能力 ({len(common)} 項)")
        if common:
            for item in common[:10]:
                answer_parts.append(f"- {item['code']}: {item['name']}")
            if len(common) > 10:
                answer_parts.append(f"  (還有 {len(common) - 10} 項...)")
        else:
            answer_parts.append("  無共同能力")

        answer_parts.append(f"\n### {occ1} 獨有能力 ({len(only1)} 項)")
        if only1:
            for item in only1[:5]:
                answer_parts.append(f"- {item['code']}: {item['name']}")
        else:
            answer_parts.append("  無獨有能力")

        answer_parts.append(f"\n### {occ2} 獨有能力 ({len(only2)} 項)")
        if only2:
            for item in only2[:5]:
                answer_parts.append(f"- {item['code']}: {item['name']}")
        else:
            answer_parts.append("  無獨有能力")

        return "\n".join(answer_parts)

    # ========================================
    # 查詢類型 2: 職涯路徑
    # ========================================

    def find_career_path(
        self,
        from_occupation: str,
        to_occupation: str
    ) -> QueryResult:
        """
        找出職涯晉升路徑和需要補充的能力

        Args:
            from_occupation: 起點職業
            to_occupation: 目標職業

        Returns:
            QueryResult
        """
        result = QueryResult(
            query=f"從 {from_occupation} 晉升到 {to_occupation}",
            query_type="職涯路徑"
        )

        # 找到對應節點
        from_std = self._find_standard_node(from_occupation)
        to_std = self._find_standard_node(to_occupation)

        if not from_std or not to_std:
            result.answer = f"找不到職能基準"
            return result

        # 取得兩者的知識和技能
        from_ks = self._get_all_knowledge_skills(from_std)
        to_ks = self._get_all_knowledge_skills(to_std)

        # 計算需要補充的能力（目標有而起點沒有的）
        gap_knowledge = to_ks["知識"] - from_ks["知識"]
        gap_skills = to_ks["技能"] - from_ks["技能"]

        # 取得詳細資訊
        gap_k_details = self._get_ks_details(gap_knowledge)
        gap_s_details = self._get_ks_details(gap_skills)

        # 檢查圖譜中是否有晉升路徑
        path_info = self._find_graph_path(from_std, to_std)

        result.results = [
            {"type": "需補充知識", "items": gap_k_details},
            {"type": "需補充技能", "items": gap_s_details},
            {"type": "路徑資訊", "data": path_info}
        ]

        result.graph_context = {
            "from_standard": self.kg.get_node_data(from_std),
            "to_standard": self.kg.get_node_data(to_std),
            "path_exists": path_info.get("exists", False)
        }

        # 生成答案
        result.answer = self._generate_career_path_answer(
            from_occupation, to_occupation,
            gap_k_details, gap_s_details, path_info
        )

        return result

    def _find_graph_path(self, from_id: str, to_id: str) -> Dict:
        """在圖譜中尋找路徑"""
        try:
            # 嘗試找最短路徑
            path = nx.shortest_path(self.kg.graph, from_id, to_id)
            return {
                "exists": True,
                "path": path,
                "length": len(path) - 1
            }
        except nx.NetworkXNoPath:
            return {
                "exists": False,
                "path": [],
                "length": -1
            }

    def _generate_career_path_answer(
        self,
        from_occ: str,
        to_occ: str,
        gap_k: List[Dict],
        gap_s: List[Dict],
        path_info: Dict
    ) -> str:
        """生成職涯路徑答案"""
        answer_parts = []

        answer_parts.append(f"## 從 {from_occ} 到 {to_occ} 的職涯發展\n")

        total_gap = len(gap_k) + len(gap_s)
        answer_parts.append(f"### 需要補充的能力 (共 {total_gap} 項)\n")

        if gap_k:
            answer_parts.append(f"**知識 ({len(gap_k)} 項):**")
            for item in gap_k:
                answer_parts.append(f"- {item['code']}: {item['name']}")

        if gap_s:
            answer_parts.append(f"\n**技能 ({len(gap_s)} 項):**")
            for item in gap_s:
                answer_parts.append(f"- {item['code']}: {item['name']}")

        if path_info.get("exists"):
            answer_parts.append(f"\n### 晉升路徑")
            answer_parts.append(f"路徑長度: {path_info['length']} 步")
        else:
            answer_parts.append("\n### 建議")
            answer_parts.append("圖譜中無直接晉升路徑，建議分階段發展相關能力。")

        return "\n".join(answer_parts)

    # ========================================
    # 查詢類型 3: 能力反查
    # ========================================

    def find_occupations_by_ability(
        self,
        ability: str
    ) -> QueryResult:
        """
        根據知識/技能找適合的職業

        Args:
            ability: 知識或技能關鍵字

        Returns:
            QueryResult
        """
        result = QueryResult(
            query=f"具備「{ability}」適合哪些職業",
            query_type="能力反查"
        )

        # 先用向量檢索找相關的知識/技能節點
        related_nodes = self._vector_search(ability, top_k=10)

        # 過濾只保留知識和技能
        ks_nodes = [
            n for n in related_nodes
            if n["node_type"] in ["知識", "技能"]
        ]

        if not ks_nodes:
            result.answer = f"找不到與「{ability}」相關的知識或技能"
            return result

        # 反向查詢：找出需要這些知識/技能的職能基準
        matching_standards = defaultdict(int)

        for ks in ks_nodes:
            ks_id = ks["node_id"]

            # 反向遍歷：知識/技能 -> 任務 -> 職責 -> 職能基準
            for task_id, _, edge_data in self.kg.graph.in_edges(ks_id, data=True):
                if edge_data.get("edge_type") not in ["需要知識", "需要技能"]:
                    continue

                for duty_id, _, _ in self.kg.graph.in_edges(task_id, data=True):
                    for std_id, _, _ in self.kg.graph.in_edges(duty_id, data=True):
                        std_data = self.kg.get_node_data(std_id)
                        if std_data and std_data.get("node_type") == "職能基準":
                            matching_standards[std_id] += 1

        # 排序並取前 10
        sorted_standards = sorted(
            matching_standards.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # 取得詳細資訊
        standard_details = []
        for std_id, count in sorted_standards:
            std_data = self.kg.get_node_data(std_id)
            if std_data:
                standard_details.append({
                    "code": std_data.get("node_id", ""),
                    "name": std_data.get("name", ""),
                    "level": std_data.get("level", 0),
                    "match_count": count
                })

        result.results = [
            {"type": "相關知識/技能", "items": ks_nodes[:5]},
            {"type": "適合職業", "items": standard_details}
        ]

        # 生成答案
        result.answer = self._generate_ability_search_answer(
            ability, ks_nodes[:5], standard_details
        )

        return result

    def _vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """向量檢索"""
        if self.vector_index is None:
            self.initialize_embeddings()

        # 生成查詢向量
        query_vector = self.embedding_model.encode([query])
        faiss.normalize_L2(query_vector)

        # 檢索
        scores, indices = self.vector_index.search(query_vector, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue

            node_id = self.vector_id_map.get(idx)
            if node_id:
                node_data = self.kg.get_node_data(node_id)
                if node_data:
                    results.append({
                        **node_data,
                        "score": float(score),
                        "full_id": node_id
                    })

        return results

    def _generate_ability_search_answer(
        self,
        ability: str,
        related_ks: List[Dict],
        standards: List[Dict]
    ) -> str:
        """生成能力反查答案"""
        answer_parts = []

        answer_parts.append(f"## 與「{ability}」相關的職業\n")

        answer_parts.append("### 相關知識/技能")
        for ks in related_ks:
            answer_parts.append(f"- {ks.get('node_id', '')}: {ks.get('name', '')} (相似度: {ks.get('score', 0):.3f})")

        answer_parts.append(f"\n### 適合的職業 (共 {len(standards)} 個)")
        for std in standards:
            answer_parts.append(f"- **{std['name']}** ({std['code']})")
            answer_parts.append(f"  級別: {std['level']}，匹配能力: {std['match_count']} 項")

        return "\n".join(answer_parts)

    # ========================================
    # 查詢類型 4: 聚合統計
    # ========================================

    def get_top_abilities(
        self,
        top_k: int = 10,
        ability_type: str = "all"
    ) -> QueryResult:
        """
        統計最常被需要的知識/技能

        Args:
            top_k: 返回數量
            ability_type: "knowledge", "skill", "all"

        Returns:
            QueryResult
        """
        result = QueryResult(
            query=f"最常需要的{ability_type}能力 Top {top_k}",
            query_type="聚合統計"
        )

        # 計算節點入度（被需要的次數）
        if ability_type == "knowledge":
            target_types = ["知識"]
        elif ability_type == "skill":
            target_types = ["技能"]
        else:
            target_types = ["知識", "技能"]

        degree_counts = []

        for node_type in target_types:
            nodes = self.kg.get_nodes_by_type(node_type)
            for node_id in nodes:
                in_degree = self.kg.graph.in_degree(node_id)
                node_data = self.kg.get_node_data(node_id)
                if node_data:
                    degree_counts.append({
                        "code": node_data.get("node_id", ""),
                        "name": node_data.get("name", ""),
                        "type": node_type,
                        "count": in_degree
                    })

        # 排序
        sorted_abilities = sorted(
            degree_counts,
            key=lambda x: x["count"],
            reverse=True
        )[:top_k]

        result.results = [
            {"type": "Top 能力", "items": sorted_abilities}
        ]

        # 生成答案
        answer_parts = [f"## 最常被需要的能力 Top {top_k}\n"]
        for i, item in enumerate(sorted_abilities, 1):
            answer_parts.append(
                f"{i}. **{item['name']}** ({item['code']}) - {item['type']}"
            )
            answer_parts.append(f"   被 {item['count']} 個工作任務需要")

        result.answer = "\n".join(answer_parts)

        return result

    # ========================================
    # 查詢類型 5: 語義搜尋 + 圖譜擴展
    # ========================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        expand_depth: int = 1,
        use_federated: bool = False
    ) -> QueryResult:
        """
        語義搜尋並擴展相關圖譜內容

        Args:
            query: 自然語言查詢
            top_k: 向量檢索數量
            expand_depth: 圖譜擴展深度
            use_federated: 是否使用聯邦搜索

        Returns:
            QueryResult
        """
        # 如果啟用聯邦搜索，使用專門的方法
        if use_federated and self.is_federated_ready():
            return self.federated_semantic_search(query, top_k, expand_depth)

        result = QueryResult(
            query=query,
            query_type="語義搜尋"
        )

        # 1. 向量檢索
        vector_results = self._vector_search(query, top_k=top_k)

        if not vector_results:
            result.answer = f"找不到與「{query}」相關的內容"
            return result

        # 2. 圖譜擴展
        expanded_nodes = set()
        expanded_edges = []

        for vr in vector_results:
            node_id = vr.get("full_id")
            if not node_id:
                continue

            # BFS 擴展
            self._expand_graph(node_id, expand_depth, expanded_nodes, expanded_edges)

        # 3. 組織結果
        result.results = [
            {"type": "向量檢索結果", "items": vector_results},
            {"type": "擴展節點", "items": list(expanded_nodes)[:20]},
            {"type": "擴展邊", "items": expanded_edges[:30]}
        ]

        # 生成上下文
        context = self._build_context(vector_results, expanded_nodes)
        result.graph_context = context

        # 收集引用來源
        result.sources = self._collect_sources(vector_results, context)

        # 使用 LLM 生成答案（如果可用）
        if self.llm:
            result.answer = self._generate_llm_answer(query, context)
        else:
            result.answer = self._generate_simple_answer(query, vector_results)

        # 附加引用資料到答案
        result.answer = self._append_sources_to_answer(result.answer, result.sources)

        return result

    def federated_semantic_search(
        self,
        query: str,
        top_k: int = 5,
        expand_depth: int = 1,
        top_k_categories: int = 3,
        top_k_occupations: int = 5,
        use_occupation_routing: bool = True
    ) -> QueryResult:
        """
        聯邦語義搜尋 - 先路由到相關職類別/通俗職業分類，再進行向量搜索

        Args:
            query: 自然語言查詢
            top_k: 向量檢索數量
            expand_depth: 圖譜擴展深度
            top_k_categories: 搜索的職類別數量
            top_k_occupations: 搜索的通俗職業分類數量
            use_occupation_routing: 是否使用通俗職業分類路由（更精細）

        Returns:
            QueryResult
        """
        result = QueryResult(
            query=query,
            query_type="聯邦語義搜尋"
        )

        if not self.is_federated_ready():
            result.answer = "聯邦搜索系統未初始化"
            return result

        # 1. 使用聯邦搜索管理器路由查詢
        def filtered_vector_search(q: str, relevant_codes: set, k: int) -> List[Dict]:
            """過濾的向量搜索"""
            all_results = self._vector_search(q, top_k=k * 3)  # 多取一些再過濾

            # 過濾只保留相關職類別的結果
            filtered = []
            for r in all_results:
                node_id = r.get("full_id", "")
                # 檢查節點是否屬於相關職能基準
                is_relevant = any(code in node_id for code in relevant_codes)
                if is_relevant or not relevant_codes:
                    filtered.append(r)

                if len(filtered) >= k:
                    break

            return filtered

        # 執行聯邦搜索（支援多層路由）
        fed_result = self.federated_manager.federated_search(
            query=query,
            vector_search_func=filtered_vector_search,
            top_k_categories=top_k_categories,
            top_k_occupations=top_k_occupations,
            top_k_results=top_k,
            use_occupation_routing=use_occupation_routing
        )

        vector_results = fed_result.get('results', [])
        routed_categories = fed_result.get('routed_categories', [])
        routed_occupations = fed_result.get('routed_occupations', [])

        if not vector_results:
            # 聯邦路由無結果，回退到全域向量搜索（不限職類別過濾）
            logger.info(f"聯邦搜索路由無結果，回退到全域向量搜索: {query}")
            vector_results = self._vector_search(query, top_k=top_k)
            routed_categories = []
            routed_occupations = []
            if not vector_results:
                result.answer = f"找不到與「{query}」相關的內容"
                return result

        # 2. 圖譜擴展
        expanded_nodes = set()
        expanded_edges = []

        for vr in vector_results:
            node_id = vr.get("full_id")
            if not node_id:
                continue
            self._expand_graph(node_id, expand_depth, expanded_nodes, expanded_edges)

        # 3. 組織結果
        category_info = []
        for cat_id, score in routed_categories:
            cat_data = self.federated_manager.get_category_info(cat_id)
            if cat_data:
                category_info.append({
                    "category": cat_data['name'],
                    "score": score,
                    "standards_count": cat_data['standards_count']
                })

        occupation_info = []
        for occ_id, score in routed_occupations:
            occ_data = self.federated_manager.get_occupation_info(occ_id)
            if occ_data:
                occupation_info.append({
                    "occupation": occ_data['name'],
                    "parent_category": occ_data.get('parent_category', ''),
                    "score": score,
                    "standards_count": occ_data['standards_count']
                })

        result_items = []
        if occupation_info:
            result_items.append({"type": "路由通俗職業分類", "items": occupation_info})
        if category_info:
            result_items.append({"type": "路由職類別", "items": category_info})
        result_items.extend([
            {"type": "向量檢索結果", "items": vector_results},
            {"type": "擴展節點", "items": list(expanded_nodes)[:20]},
            {"type": "擴展邊", "items": expanded_edges[:30]}
        ])
        result.results = result_items

        # 添加路由資訊到上下文
        context = self._build_context(vector_results, expanded_nodes)
        context['routed_categories'] = [cat['category'] for cat in category_info]
        context['routed_occupations'] = [occ['occupation'] for occ in occupation_info]
        context['search_stats'] = fed_result.get('stats', {})
        result.graph_context = context

        # 收集引用來源
        result.sources = self._collect_sources(vector_results, context)

        # 使用 LLM 生成答案
        if self.llm:
            result.answer = self._generate_llm_answer(query, context)
        else:
            result.answer = self._generate_federated_answer(query, vector_results, category_info, occupation_info)

        # 附加引用資料到答案
        result.answer = self._append_sources_to_answer(result.answer, result.sources)

        return result

    def _generate_federated_answer(
        self,
        query: str,
        results: List[Dict],
        categories: List[Dict],
        occupations: List[Dict] = None
    ) -> str:
        """生成聯邦搜索答案（不使用 LLM）"""
        answer_parts = [f"## 與「{query}」相關的職能基準資訊\n"]

        # 顯示搜索的通俗職業分類（更精細）
        if occupations:
            answer_parts.append("### 相關通俗職業分類")
            for occ in occupations:
                parent = f" (屬於 {occ['parent_category']})" if occ.get('parent_category') else ""
                answer_parts.append(f"- **{occ['occupation']}**{parent} (相關度: {occ['score']:.2f}, 包含 {occ['standards_count']} 個職能基準)")
            answer_parts.append("")

        # 顯示搜索的職類別
        if categories:
            answer_parts.append("### 相關職類別")
            for cat in categories:
                answer_parts.append(f"- **{cat['category']}** (相關度: {cat['score']:.2f}, 包含 {cat['standards_count']} 個職能基準)")
            answer_parts.append("")

        # 顯示搜索結果
        answer_parts.append("### 搜索結果")
        for i, r in enumerate(results[:5], 1):
            answer_parts.append(f"#### {i}. {r.get('name', '未知')} ({r.get('node_type', '')})")
            if r.get("description"):
                answer_parts.append(f"   {r['description'][:200]}")
            answer_parts.append(f"   相似度: {r.get('score', 0):.3f}")
            answer_parts.append("")

        return "\n".join(answer_parts)

    def _expand_graph(
        self,
        node_id: str,
        depth: int,
        visited: Set[str],
        edges: List[Tuple]
    ):
        """BFS 擴展圖譜"""
        if depth <= 0 or node_id in visited:
            return

        visited.add(node_id)

        # 出邊
        for _, target, data in self.kg.graph.out_edges(node_id, data=True):
            edges.append((node_id, target, data.get("edge_type", "")))
            self._expand_graph(target, depth - 1, visited, edges)

        # 入邊
        for source, _, data in self.kg.graph.in_edges(node_id, data=True):
            edges.append((source, node_id, data.get("edge_type", "")))
            self._expand_graph(source, depth - 1, visited, edges)

    def _build_context(
        self,
        vector_results: List[Dict],
        expanded_nodes: Set[str]
    ) -> Dict:
        """建構查詢上下文"""
        context = {
            "primary_results": [],
            "related_standards": [],
            "related_knowledge": [],
            "related_skills": [],
            "related_industries": [],  # 行業別
            "standard_details": [],    # 職能基準詳細資訊 (包含 ICAP metadata)
            "task_details": {}         # {standard_code: [task_dict,...]} 主要職責/工作任務
        }
        _task_codes_added: set = set()  # 避免重複加入同一職能基準的任務

        for vr in vector_results:
            result_item = {
                "type": vr.get("node_type", ""),
                "name": vr.get("name", ""),
                "description": vr.get("description", ""),
                "score": vr.get("score", 0)
            }

            # 優先從 ICAP 資料庫查詢精確的 metadata
            std_name = vr.get("name", "")
            db_metadata = self._get_icap_metadata_by_name(std_name)

            if db_metadata:
                if db_metadata.get("standard_code"):
                    result_item["standard_code"] = db_metadata["standard_code"]
                if db_metadata.get("category"):
                    result_item["category"] = db_metadata["category"]
                if db_metadata.get("category_code"):
                    result_item["category_code"] = db_metadata["category_code"]
                if db_metadata.get("occupation"):
                    result_item["occupation"] = db_metadata["occupation"]
                if db_metadata.get("occupation_code"):
                    result_item["occupation_code"] = db_metadata["occupation_code"]
                if db_metadata.get("industry"):
                    result_item["industry"] = db_metadata["industry"]
                if db_metadata.get("industry_code"):
                    result_item["industry_code"] = db_metadata["industry_code"]
                if db_metadata.get("level"):
                    result_item["level"] = db_metadata["level"]
            else:
                # 回退到圖譜中的 metadata
                if vr.get("icap_industry"):
                    result_item["industry"] = vr.get("icap_industry")
                if vr.get("icap_category"):
                    result_item["category"] = vr.get("icap_category")
                if vr.get("icap_occupation_class"):
                    result_item["occupation_code"] = vr.get("icap_occupation_class")
                if vr.get("icap_occupation_name"):
                    result_item["occupation"] = vr.get("icap_occupation_name")

            context["primary_results"].append(result_item)

            # 抓取主要職責/工作任務（限最多 3 個職能基準，避免 prompt 過長）
            if self._json_store and len(_task_codes_added) < 3:
                node_type = vr.get("node_type", "")
                std_code = None

                if node_type == "職能基準":
                    # 直接用名稱查代碼
                    std_code = (result_item.get("standard_code") or
                                (db_metadata.get("standard_code") if db_metadata else None))
                    if not std_code and std_name:
                        s = self._json_store.get_standard_by_name(std_name)
                        if s:
                            std_code = s.get("code")
                elif node_type in ("主要職責", "工作任務", "行為指標"):
                    # 從圖譜節點取 standard_code 屬性
                    std_code = vr.get("standard_code")

                if std_code and std_code not in _task_codes_added:
                    s = self._json_store.get_standard_by_code(std_code)
                    if s and s.get("tasks"):
                        context["task_details"][std_code] = {
                            "name": s.get("name", ""),
                            "tasks": s["tasks"][:8]  # 每個職能基準最多 8 個工作任務
                        }
                        _task_codes_added.add(std_code)

        for node_id in expanded_nodes:
            node_data = self.kg.get_node_data(node_id)
            if not node_data:
                continue

            node_type = node_data.get("node_type", "")
            if node_type == "職能基準":
                std_name = node_data.get("name", "")
                context["related_standards"].append(std_name)

                # 優先從 ICAP 資料庫查詢精確的 metadata
                db_metadata = self._get_icap_metadata_by_name(std_name)

                std_detail = {"name": std_name}
                if db_metadata:
                    if db_metadata.get("standard_code"):
                        std_detail["standard_code"] = db_metadata["standard_code"]
                    if db_metadata.get("category"):
                        std_detail["category"] = db_metadata["category"]
                    if db_metadata.get("category_code"):
                        std_detail["category_code"] = db_metadata["category_code"]
                    if db_metadata.get("occupation"):
                        std_detail["occupation"] = db_metadata["occupation"]
                    if db_metadata.get("occupation_code"):
                        std_detail["occupation_code"] = db_metadata["occupation_code"]
                    if db_metadata.get("industry"):
                        std_detail["industry"] = db_metadata["industry"]
                        if db_metadata["industry"] not in context["related_industries"]:
                            context["related_industries"].append(db_metadata["industry"])
                    if db_metadata.get("industry_code"):
                        std_detail["industry_code"] = db_metadata["industry_code"]
                else:
                    # 回退到圖譜中的 metadata
                    if node_data.get("icap_industry"):
                        std_detail["industry"] = node_data.get("icap_industry")
                        if node_data["icap_industry"] not in context["related_industries"]:
                            context["related_industries"].append(node_data["icap_industry"])
                    if node_data.get("icap_category"):
                        std_detail["category"] = node_data.get("icap_category")

                if len(std_detail) > 1:  # 有額外資訊才加入
                    context["standard_details"].append(std_detail)

            elif node_type == "知識":
                context["related_knowledge"].append(node_data.get("name", ""))
            elif node_type == "技能":
                context["related_skills"].append(node_data.get("name", ""))
            elif node_type == "行業別":
                ind_name = node_data.get("name", "")
                if ind_name and ind_name not in context["related_industries"]:
                    context["related_industries"].append(ind_name)

        return context

    def _get_icap_metadata_by_name(self, name: str) -> Optional[Dict]:
        """
        從職能基準資料庫查詢 metadata（包含完整代碼欄位）

        Args:
            name: 職能基準名稱

        Returns:
            包含代碼和名稱的完整字典，或 None
        """
        # 優先嘗試 JSON Store（有完整的代碼欄位）
        # search_standards 返回 asdict(CompetencyStandard)，是扁平化字典
        # 欄位: code, name, category_code, category_name, occupation_code,
        #        occupation_name, industry_code, industry_name, job_description, level ...
        try:
            json_store = self._json_store  # 使用快取實例，避免重複載入
            if json_store is None:
                from competency_store import CompetencyJSONStore
                json_store = CompetencyJSONStore(config.PARSED_JSON_V2_DIR)
            results = json_store.search_standards(name, limit=1)
            if results:
                r = results[0]
                if r.get("name"):  # 有效的結果
                    return {
                        "standard_code": r.get("code", ""),
                        "category": r.get("category_name", ""),
                        "category_code": r.get("category_code", ""),
                        "occupation": r.get("occupation_name", ""),
                        "occupation_code": r.get("occupation_code", ""),
                        "industry": r.get("industry_name", ""),
                        "industry_code": r.get("industry_code", ""),
                        "description": r.get("job_description", ""),
                        "level": r.get("level", ""),
                    }
        except Exception as e:
            logger.debug(f"JSON Store 查詢失敗: {e}")

        return None

    def get_standard_full_structure(self, standard_code: str) -> Optional[Dict]:
        """
        取得職能基準的完整階層結構（從 JSON Store 讀取）

        Args:
            standard_code: 職能基準代碼

        Returns:
            完整的職能基準結構，包含任務、知識、技能等
        """
        if not self._json_store:
            return None
        try:
            return self._json_store.get_standard_by_code(standard_code)
        except Exception as e:
            logger.debug(f"職能基準結構查詢失敗: {e}")
            return None

    def search_indicators(self, keyword: str, limit: int = 20) -> List[Dict]:
        """
        搜尋行為指標（從 JSON Store 掃描知識/技能項目）

        Args:
            keyword: 搜尋關鍵字
            limit: 回傳數量上限

        Returns:
            匹配的行為指標列表，包含 standard_code, description
        """
        if not self._json_store:
            return []
        results = []
        keyword_lower = keyword.lower()
        for standard in self._json_store.standards.values():
            k_items = standard.knowledge
            s_items = standard.skills
            for item_type, items in (("knowledge", k_items), ("skill", s_items)):
                for item in items:
                    desc = item.get("description", "") or item.get("name", "")
                    if keyword_lower in desc.lower():
                        results.append({
                            "standard_code": standard.code,
                            "standard_name": standard.name,
                            "description": desc,
                            "type": item_type
                        })
                        if len(results) >= limit:
                            return results
        return results

    def search_standards_by_keyword(self, keyword: str, limit: int = 20) -> List[Dict]:
        """
        關鍵字搜尋職能基準名稱和描述（從 JSON Store）

        Args:
            keyword: 搜尋關鍵字
            limit: 回傳數量上限

        Returns:
            匹配的職能基準列表
        """
        if not self._json_store:
            return []
        try:
            return self._json_store.search_standards(keyword, limit)
        except Exception as e:
            logger.debug(f"職能基準搜尋失敗: {e}")
            return []

    def get_standards_by_industry(self, industry_code: str) -> List[Dict]:
        """
        根據行業別代碼查詢職能基準（從 JSON Store）

        Args:
            industry_code: 行業別代碼（如 I5611）

        Returns:
            該行業的職能基準列表
        """
        if not self._json_store:
            return []
        try:
            from dataclasses import asdict
            return [
                asdict(s) for s in self._json_store.standards.values()
                if s.industry_code == industry_code
            ]
        except Exception as e:
            logger.debug(f"行業別查詢失敗: {e}")
            return []

    def get_standards_by_occupation(self, occupation_code: str) -> List[Dict]:
        """
        根據職業別代碼查詢職能基準（從 JSON Store）

        Args:
            occupation_code: 職業別代碼（如 5120）

        Returns:
            該職業的職能基準列表
        """
        if not self._json_store:
            return []
        try:
            from dataclasses import asdict
            return [
                asdict(s) for s in self._json_store.standards.values()
                if s.occupation_code == occupation_code
            ]
        except Exception as e:
            logger.debug(f"職業別查詢失敗: {e}")
            return []

    def get_task_competencies(self, standard_code: str, task_code: str) -> Dict[str, List]:
        """
        查詢特定任務需要的知識和技能（從 JSON Store）

        Args:
            standard_code: 職能基準代碼
            task_code: 任務代碼（如 T1.1）

        Returns:
            包含 knowledge 和 skills 列表的字典
        """
        if not self._json_store:
            return {"knowledge": [], "skills": []}
        try:
            data = self._json_store.get_standard_by_code(standard_code)
            if not data:
                return {"knowledge": [], "skills": []}
            # 找到對應任務後回傳該職能基準的知識/技能
            for task in data.get("tasks", []):
                if task.get("task_id") == task_code:
                    return {
                        "knowledge": data.get("knowledge", []),
                        "skills": data.get("skills", [])
                    }
            return {"knowledge": [], "skills": []}
        except Exception as e:
            logger.debug(f"任務職能查詢失敗: {e}")
            return {"knowledge": [], "skills": []}

    def _generate_llm_answer(self, query: str, context: Dict) -> str:
        """使用 LLM 生成答案（參考 rag_ui_v3_faiss.py 實現）"""
        # 組織上下文資訊（結構化格式，每個職能基準獨立呈現所有欄位）
        primary_info = []
        for r in context.get('primary_results', [])[:5]:
            lines = [f"【職能基準資料】"]
            lines.append(f"名稱: {r.get('name', '未知')}")
            lines.append(f"節點類型: {r.get('type', '')}")
            if r.get('standard_code'):
                lines.append(f"職能基準代碼: {r['standard_code']}")
            if r.get('category') or r.get('category_code'):
                lines.append(f"職類別名稱: {r.get('category', '')}　職類別代碼: {r.get('category_code', '（無）')}")
            if r.get('occupation') or r.get('occupation_code'):
                lines.append(f"職業別名稱: {r.get('occupation', '')}　職業別代碼: {r.get('occupation_code', '（無）')}")
            if r.get('industry') or r.get('industry_code'):
                lines.append(f"行業別名稱: {r.get('industry', '')}　行業別代碼: {r.get('industry_code', '（無）')}")
            if r.get('level'):
                lines.append(f"職能級別: {r['level']}")
            if r.get('description'):
                lines.append(f"工作描述: {r['description'][:200]}")
            primary_info.append("\n".join(lines))

        standard_details = context.get('standard_details', [])[:5]
        standards = context.get('related_standards', [])[:5]
        knowledge = context.get('related_knowledge', [])[:8]
        skills = context.get('related_skills', [])[:8]
        industries = context.get('related_industries', [])[:5]

        # 主要文件內容
        context_text = "\n\n".join(primary_info) if primary_info else "無相關資訊"

        # 從圖譜擴展的職能基準補充（也以結構化格式呈現）
        if standard_details:
            context_text += "\n\n【相關職能基準補充資料】"
            for std in standard_details:
                detail_lines = [f"\n名稱: {std['name']}"]
                if std.get('standard_code'):
                    detail_lines.append(f"職能基準代碼: {std['standard_code']}")
                if std.get('category') or std.get('category_code'):
                    detail_lines.append(f"職類別名稱: {std.get('category', '')}　職類別代碼: {std.get('category_code', '（無）')}")
                if std.get('occupation') or std.get('occupation_code'):
                    detail_lines.append(f"職業別名稱: {std.get('occupation', '')}　職業別代碼: {std.get('occupation_code', '（無）')}")
                if std.get('industry') or std.get('industry_code'):
                    detail_lines.append(f"行業別名稱: {std.get('industry', '')}　行業別代碼: {std.get('industry_code', '（無）')}")
                context_text += "\n".join(detail_lines)

        # 主要職責與工作任務（最重要的結構化資料）
        task_details = context.get("task_details", {})
        if task_details:
            context_text += "\n\n【主要職責與工作任務】"
            for std_code, std_data in task_details.items():
                context_text += f"\n\n▍{std_data['name']}（{std_code}）"
                # 按主要職責分組
                duty_groups: dict = {}
                for t in std_data["tasks"]:
                    main_resp = t.get("main_responsibility", "")
                    duty_groups.setdefault(main_resp, []).append(t)
                for main_resp, tasks in duty_groups.items():
                    context_text += f"\n{main_resp}"
                    for t in tasks:
                        tid = t.get("task_id", "")
                        tname = t.get("task_name", "")
                        level = t.get("level", "")
                        context_text += f"\n  {tid} {tname}"
                        if level:
                            context_text += f"（級別{level}）"
                        # 工作產出
                        output = t.get("output", "")
                        if output:
                            context_text += f"\n    → 產出：{output}"
                        # 行為指標（最多 3 條）
                        behaviors = [b for b in t.get("behaviors", []) if b][:3]
                        for b in behaviors:
                            context_text += f"\n    · {b}"

        # 額外關聯資訊摘要
        if knowledge or skills or industries:
            context_text += "\n\n【其他關聯資訊】"
            if industries:
                context_text += f"\n相關行業別：{', '.join(industries)}"
            if knowledge:
                context_text += f"\n相關知識項目：{', '.join(knowledge[:5])}"
            if skills:
                context_text += f"\n相關技能項目：{', '.join(skills[:5])}"

        prompt = f"""你是專業的職能基準知識助手。請根據提供的文件內容，精確回答問題。

【重要術語定義】
- 職能基準代碼：職能基準的完整唯一識別碼，格式如「TFB5120-003v3」或「HBR2431-001v4」，包含職類別代碼、職業別代碼、流水號與版本號
- 職類別（職能類別）：最大分類，代碼為2-3個英文字母縮寫，例如「TFB」代表「休閒與觀光旅遊／餐飲管理」，「HBR」代表「醫療保健／生技研發」
- 職業別：職能的中分類，代碼為4位數字（對應國際標準職業分類），例如「5120」代表「廚師」，「2431」代表「廣告及行銷專業人員」
- 行業別：從業所屬產業類別，代碼以一個英文大寫字母開頭後接數字，例如「I5611」代表「住宿及餐飲業／餐館」，「C0899」代表「製造業／食品及飼品製造業」
- 職能級別：能力深度與複雜度等級，以數字1至7表示（級別愈高代表能力要求愈深、自主性愈強）；1-2級為基礎操作，3-4級為中階技術，5級以上為高階專業或管理
- 主要職責（T編碼）：職能基準中最高層的工作責任區塊，代碼格式為「T1」「T2」…，描述該職位在某面向上的核心使命，一個職能基準通常包含數個主要職責
- 工作任務（T.子編碼）：隸屬於某主要職責下的具體工作項目，代碼格式為「T1.1」「T1.2」…，描述執行主要職責時需完成的個別任務
- 工作產出（O編碼）：執行工作任務後應產生的具體成果或文件，代碼格式為「O1.1.1」「O2.3.2」…，例如「市場分析報告」「年度行銷預算規劃書」
- 行為指標（P編碼）：衡量工作任務是否達標的可觀察行為描述，代碼格式為「P1.1.1」「P2.3.1」…，說明「能夠…」的具體行動標準
- 職能內涵 K（Knowledge 知識）：執行工作任務所需具備的知識項目，代碼為「K01」「K02」…；知識是理解與判斷的基礎，例如「保健食品相關法規」「市場調查分析」
- 職能內涵 S（Skills 技能）：執行工作任務所需具備的操作或應用能力，代碼為「S01」「S02」…；技能是知識的實際運用，例如「SWOT分析」「企劃簡報技巧」「溝通能力」
- 職能內涵 A（Attitude 態度）：從事該職位工作時應展現的職業態度與工作價值觀，代碼為「A01」「A02」…；例如「積極主動」「客戶服務導向」「持續學習」

【回答原則】

▌代碼查詢原則
1. 問「職類別代碼」→ 只回答職類別代碼（2-3個英文字母縮寫，如 TFB），不回答職業別代碼或行業別代碼
2. 問「職業別代碼」→ 只回答職業別代碼（4位數字，如 5120），不回答職類別代碼
3. 問「行業別代碼」→ 只回答行業別代碼（英文大寫字母開頭+數字，如 I5611），不回答其他代碼
4. 代碼必須來自文件，不得根據職能基準名稱或常識推測

▌職責與任務查詢原則
5. 問「主要職責」→ 以 T 編碼條列呈現，格式：「T1 職責名稱」「T2 職責名稱」…
6. 問「工作任務」→ 以 T.子編碼條列呈現，格式：「T1.1 任務名稱（級別N）」，並標明所屬主要職責
7. 問「工作產出」→ 以 O 編碼條列，格式：「O1.1.1 產出名稱」，並說明對應哪個工作任務
8. 問「行為指標」→ 以 P 編碼條列，格式：「P1.1.1 行為描述」，並說明對應哪個工作任務

▌知識／技能／態度查詢原則
9. 問「知識（K）」→ 以代碼+名稱條列，格式：「K01 知識名稱」，如有多個任務共用則合併去重
10. 問「技能（S）」→ 以代碼+名稱條列，格式：「S01 技能名稱」，如有多個任務共用則合併去重
11. 問「態度（A）」→ 以代碼+名稱條列，格式：「A01 態度名稱」

▌比較與統計查詢原則
12. 問「比較兩個職能基準」→ 分三段呈現：①甲有乙無 ②乙有甲無 ③共同擁有
13. 問「最常見/Top N」→ 統計後按出現次數由高到低排序，標示出現在幾個職能基準中

▌資料可信度原則
14. 文件中有明確記錄的資訊 → 直接引用，可標示「依職能基準資料：…」
15. 文件中找不到的資訊 → 明確說明「本次檢索資料中未包含此資訊」，不以推測或常識補充
16. 若部分資料有、部分無 → 先回答有資料的部分，再說明哪些資訊不在文件中

▌格式原則
17. 優先使用條列式，資訊量多時以小標題分段
18. 引用職能基準時標示名稱與代碼，例如「LED光學設計工程師（LED2151-001v3）」
19. 回答長度以完整呈現文件中的相關資訊為準，不過度精簡也不憑空擴充
文件內容：
{context_text}

問題：{query}

回答："""

        try:
            response = self.llm.invoke(prompt)
            # 處理不同類型的 LLM 返回值
            if isinstance(response, str):
                answer = response.strip()
            elif hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = str(response).strip()
            return answer
        except Exception as e:
            logger.error(f"LLM 生成失敗: {e}")
            return self._generate_simple_answer(query, context.get("primary_results", []))

    def _generate_simple_answer(self, query: str, results: List[Dict]) -> str:
        """生成簡單答案（不使用 LLM）"""
        answer_parts = [f"## 與「{query}」相關的職能基準資訊\n"]

        for i, r in enumerate(results[:5], 1):
            answer_parts.append(f"### {i}. {r.get('name', '未知')} ({r.get('node_type', '')})")
            if r.get("description"):
                answer_parts.append(f"   {r['description'][:200]}")
            answer_parts.append(f"   相似度: {r.get('score', 0):.3f}")
            answer_parts.append("")

        return "\n".join(answer_parts)

    def _collect_sources(self, vector_results: List[Dict], context: Dict) -> List[str]:
        """
        收集引用來源資訊

        Args:
            vector_results: 向量檢索結果
            context: 查詢上下文

        Returns:
            引用來源列表
        """
        sources = []
        seen_names = set()

        # 從向量檢索結果收集
        for r in vector_results[:5]:
            name = r.get("name", "")
            node_type = r.get("node_type", "")
            score = r.get("score", 0)

            if name and name not in seen_names:
                seen_names.add(name)

                # 嘗試從資料庫獲取更多資訊
                db_info = self._get_icap_metadata_by_name(name)

                if db_info and db_info.get("code"):
                    source_info = f"[{db_info['code']}] {name}"
                    if db_info.get("industry"):
                        source_info += f" (行業: {db_info['industry']}"
                        if db_info.get("category"):
                            source_info += f", 領域: {db_info['category']}"
                        source_info += ")"
                    source_info += f" [相似度: {score:.3f}]"
                else:
                    source_info = f"{name} ({node_type}) [相似度: {score:.3f}]"

                sources.append(source_info)

        # 從 context 的 standard_details 補充
        for std in context.get("standard_details", [])[:3]:
            name = std.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                source_info = f"{name}"
                if std.get("industry"):
                    source_info += f" (行業: {std['industry']})"
                sources.append(source_info)

        return sources

    def _append_sources_to_answer(self, answer: str, sources: List[str]) -> str:
        """
        將引用來源附加到答案後面

        Args:
            answer: 原始答案
            sources: 引用來源列表

        Returns:
            附加引用的答案
        """
        if not sources:
            return answer

        sources_text = "\n\n---\n【引用資料來源】\n"
        for i, source in enumerate(sources[:5], 1):
            sources_text += f"{i}. {source}\n"

        return answer + sources_text

    # ========================================
    # 統一查詢介面
    # ========================================

    def query(self, question: str, **kwargs) -> QueryResult:
        """
        統一查詢介面，自動判斷查詢類型

        Args:
            question: 使用者問題
            **kwargs: 額外參數

        Returns:
            QueryResult
        """
        question_lower = question.lower()

        # 判斷查詢類型
        if "比較" in question or "和" in question and ("共同" in question or "差異" in question):
            # 跨職業比較
            # 嘗試提取兩個職業名稱
            import re
            pattern = r"(.+?)(?:和|與|跟)(.+?)(?:需要|有|的|共同|差異)"
            match = re.search(pattern, question)
            if match:
                return self.compare_occupations(match.group(1).strip(), match.group(2).strip())

        elif "晉升" in question or "轉換" in question or "發展" in question:
            # 職涯路徑
            import re
            pattern = r"從(.+?)(?:晉升|轉換|發展)到(.+)"
            match = re.search(pattern, question)
            if match:
                return self.find_career_path(match.group(1).strip(), match.group(2).strip())

        elif "適合" in question or "哪些職業" in question or "什麼職業" in question:
            # 能力反查
            import re
            pattern = r"(?:具備|有)?(.+?)(?:適合|可以從事|能做)"
            match = re.search(pattern, question)
            if match:
                return self.find_occupations_by_ability(match.group(1).strip())

        elif "最常" in question or "top" in question_lower or "排名" in question:
            # 聚合統計
            return self.get_top_abilities(top_k=10)

        # 預設：語義搜尋
        return self.semantic_search(question, **kwargs)


# ========================================
# 命令列工具
# ========================================

def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="Graph RAG 查詢系統")
    parser.add_argument(
        "--graph", "-g",
        type=str,
        default=str(config.GRAPH_DB_DIR / config.GRAPH_FILE),
        help="知識圖譜檔案路徑"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="查詢問題"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="互動模式"
    )

    args = parser.parse_args()

    # 設定日誌
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # 載入圖譜
    kg = CompetencyKnowledgeGraph()
    if not kg.load(args.graph):
        print(f"無法載入圖譜: {args.graph}")
        return

    # 初始化查詢引擎
    engine = GraphRAGQueryEngine(kg)
    engine.initialize_embeddings()

    if args.query:
        # 單次查詢
        result = engine.query(args.query)
        print("\n" + "=" * 60)
        print(f"查詢類型: {result.query_type}")
        print("=" * 60)
        print(result.answer)

    elif args.interactive:
        # 互動模式
        print("\n" + "=" * 60)
        print("職能基準 Graph RAG 查詢系統")
        print("輸入問題進行查詢，輸入 'quit' 退出")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("問題: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break

                if not question:
                    continue

                result = engine.query(question)
                print("\n" + "-" * 40)
                print(f"[{result.query_type}]")
                print("-" * 40)
                print(result.answer)
                print()

            except KeyboardInterrupt:
                break

        print("\n再見！")

    else:
        # 顯示統計
        stats = kg.get_statistics()
        print("\n知識圖譜統計：")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
