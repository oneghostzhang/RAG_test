"""
知識圖譜建構模組（改進版）
負責從解析後的 JSON 資料建立階層式 NetworkX 知識圖譜
支援論文中提到的階層式節點表示法
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict

import networkx as nx
from loguru import logger
from tqdm import tqdm

from config import get_config
from pdf_parser_v2 import ParsedCompetencyStandard, CompetencyPDFParser

config = get_config()


class CompetencyKnowledgeGraph:
    """職能基準知識圖譜（階層式結構）"""

    def __init__(self):
        """初始化知識圖譜"""
        # 使用有向多重圖
        self.graph = nx.MultiDiGraph()

        # 節點索引
        self.node_index: Dict[str, Set[str]] = defaultdict(set)

        # 邊索引
        self.edge_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # 全域知識/技能/態度索引（跨職業共用）
        self.global_knowledge: Dict[str, Dict] = {}
        self.global_skills: Dict[str, Dict] = {}
        self.global_attitudes: Dict[str, Dict] = {}

        # 職類別/職業別/行業別索引（用於找出相似職業）
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.occupation_index: Dict[str, Set[str]] = defaultdict(set)
        self.industry_index: Dict[str, Set[str]] = defaultdict(set)

        logger.info("知識圖譜初始化完成（階層式結構）")

    def add_node(
        self,
        node_id: str,
        node_type: str,
        **attributes
    ) -> str:
        """
        新增節點

        Args:
            node_id: 節點唯一識別碼
            node_type: 節點類型
            **attributes: 節點屬性

        Returns:
            完整節點 ID
        """
        full_id = f"{node_type}:{node_id}"

        if not self.graph.has_node(full_id):
            self.graph.add_node(
                full_id,
                node_type=node_type,
                node_id=node_id,
                **attributes
            )
            self.node_index[node_type].add(full_id)

        return full_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        **attributes
    ):
        """
        新增邊

        Args:
            source_id: 源節點完整 ID
            target_id: 目標節點完整 ID
            edge_type: 邊類型
            **attributes: 邊屬性
        """
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(
                source_id,
                target_id,
                edge_type=edge_type,
                **attributes
            )
            self.edge_index[edge_type].append((source_id, target_id))

    def build_from_parsed_data(
        self,
        parsed_data: ParsedCompetencyStandard
    ) -> bool:
        """
        從解析資料建構階層式圖譜

        圖譜結構：
        職能基準 (中心)
          ├── 職類別 (多個)
          ├── 職業別 (多個)
          ├── 行業別 (多個)
          ├── 態度 (多個)
          └── 主要職責 T1, T2...
                └── 工作任務 T1.1, T1.2...
                      ├── 工作產出 O1.1.1
                      ├── 行為指標 P1.1.1
                      ├── 知識 K01, K02... (共用節點)
                      └── 技能 S01, S02... (共用節點)

        Args:
            parsed_data: 解析後的職能基準資料

        Returns:
            是否成功
        """
        if not parsed_data.parse_success:
            return False

        try:
            # 1. 建立職能基準節點（中心節點）
            standard_info = parsed_data.職能基準
            standard_code = standard_info.get("代碼", "")

            if not standard_code:
                logger.warning("缺少職能基準代碼，跳過")
                return False

            standard_id = self.add_node(
                node_id=standard_code,
                node_type="職能基準",
                name=standard_info.get("名稱", ""),
                description=standard_info.get("工作描述", ""),
                level=standard_info.get("基準級別", 0),
                source_file=parsed_data.source_file
            )

            # 2. 建立職類別節點和邊（支援多個）
            for category in standard_info.get("職類別", []):
                if category.get("代碼") or category.get("名稱"):
                    cat_code = category.get("代碼", "")
                    cat_name = category.get("名稱", "")
                    cat_key = cat_code if cat_code else cat_name

                    category_id = self.add_node(
                        node_id=cat_key,
                        node_type="職類別",
                        code=cat_code,
                        name=cat_name
                    )
                    self.add_edge(standard_id, category_id, "屬於職類")
                    self.category_index[cat_key].add(standard_code)

            # 3. 建立職業別節點和邊（支援多個）
            for occupation in standard_info.get("職業別", []):
                if occupation.get("代碼") or occupation.get("名稱"):
                    occ_code = occupation.get("代碼", "")
                    occ_name = occupation.get("名稱", "")
                    occ_key = occ_code if occ_code else occ_name

                    occupation_id = self.add_node(
                        node_id=occ_key,
                        node_type="職業別",
                        code=occ_code,
                        name=occ_name
                    )
                    self.add_edge(standard_id, occupation_id, "屬於職業")
                    self.occupation_index[occ_key].add(standard_code)

            # 4. 建立行業別節點和邊（支援多個）
            for industry in standard_info.get("行業別", []):
                if industry.get("代碼") or industry.get("名稱"):
                    ind_code = industry.get("代碼", "")
                    ind_name = industry.get("名稱", "")
                    ind_key = ind_code if ind_code else ind_name

                    industry_id = self.add_node(
                        node_id=ind_key,
                        node_type="行業別",
                        code=ind_code,
                        name=ind_name
                    )
                    self.add_edge(standard_id, industry_id, "適用行業")
                    self.industry_index[ind_key].add(standard_code)

            # 5. 建立態度節點
            for a_code, a_info in parsed_data.態度清單.items():
                attitude_id = self.add_node(
                    node_id=a_code,
                    node_type="態度",
                    name=a_info.get("名稱", ""),
                    description=a_info.get("描述", "")
                )
                self.add_edge(standard_id, attitude_id, "要求態度")

                if a_code not in self.global_attitudes:
                    self.global_attitudes[a_code] = {
                        "name": a_info.get("名稱", ""),
                        "description": a_info.get("描述", ""),
                        "standards": set()
                    }
                self.global_attitudes[a_code]["standards"].add(standard_code)

            # 6. 建立主要職責和工作任務（階層式結構）
            for duty in parsed_data.主要職責:
                duty_code = duty.get("代碼", "")
                if not duty_code:
                    continue

                # 使用 "職能基準代碼:職責代碼" 作為唯一識別
                duty_unique_id = f"{standard_code}:{duty_code}"

                duty_id = self.add_node(
                    node_id=duty_unique_id,
                    node_type="主要職責",
                    code=duty_code,
                    name=duty.get("名稱", ""),
                    standard_code=standard_code
                )
                self.add_edge(standard_id, duty_id, "包含職責")

                # 工作任務
                for task in duty.get("工作任務", []):
                    task_code = task.get("代碼", "")
                    if not task_code:
                        continue

                    task_unique_id = f"{standard_code}:{task_code}"

                    task_id = self.add_node(
                        node_id=task_unique_id,
                        node_type="工作任務",
                        code=task_code,
                        name=task.get("名稱", ""),
                        level=task.get("職能級別", 0),
                        standard_code=standard_code
                    )
                    self.add_edge(duty_id, task_id, "包含任務")

                    # 工作產出
                    for output in task.get("工作產出", []):
                        out_code = output.get("代碼", "")
                        if out_code:
                            out_unique_id = f"{standard_code}:{out_code}"
                            output_id = self.add_node(
                                node_id=out_unique_id,
                                node_type="工作產出",
                                code=out_code,
                                name=output.get("名稱", ""),
                                standard_code=standard_code
                            )
                            self.add_edge(task_id, output_id, "產出")

                    # 行為指標
                    for behavior in task.get("行為指標", []):
                        beh_code = behavior.get("代碼", "")
                        if beh_code:
                            beh_unique_id = f"{standard_code}:{beh_code}"
                            behavior_id = self.add_node(
                                node_id=beh_unique_id,
                                node_type="行為指標",
                                code=beh_code,
                                description=behavior.get("描述", ""),
                                standard_code=standard_code
                            )
                            self.add_edge(task_id, behavior_id, "要求行為")

                    # 知識（全域共用節點）
                    for k_code in task.get("知識", []):
                        k_name = parsed_data.知識清單.get(k_code, "")

                        knowledge_id = self.add_node(
                            node_id=k_code,
                            node_type="知識",
                            name=k_name
                        )
                        self.add_edge(task_id, knowledge_id, "需要知識")

                        if k_code not in self.global_knowledge:
                            self.global_knowledge[k_code] = {
                                "name": k_name,
                                "standards": set(),
                                "tasks": set()
                            }
                        self.global_knowledge[k_code]["standards"].add(standard_code)
                        self.global_knowledge[k_code]["tasks"].add(task_unique_id)

                    # 技能（全域共用節點）
                    for s_code in task.get("技能", []):
                        s_name = parsed_data.技能清單.get(s_code, "")

                        skill_id = self.add_node(
                            node_id=s_code,
                            node_type="技能",
                            name=s_name
                        )
                        self.add_edge(task_id, skill_id, "需要技能")

                        if s_code not in self.global_skills:
                            self.global_skills[s_code] = {
                                "name": s_name,
                                "standards": set(),
                                "tasks": set()
                            }
                        self.global_skills[s_code]["standards"].add(standard_code)
                        self.global_skills[s_code]["tasks"].add(task_unique_id)

            return True

        except Exception as e:
            logger.error(f"建構圖譜失敗: {e}")
            return False

    def build_from_v2_data(self, data: Dict[str, Any]) -> bool:
        """
        從新版 JSON 格式 (pdf_parser_v2) 建構圖譜

        新版格式結構:
        - metadata: {code, name, version, source_file, ...}
        - basic_info: {category, occupation, industry, level, ...}
        - competency_tasks: [{task_id, task_name, knowledge, skills, ...}]
        - competency_knowledge: [{code, name}]
        - competency_skills: [{code, name}]
        - competency_attitudes: [{code, name}]

        Args:
            data: 新版 JSON 資料

        Returns:
            是否成功
        """
        if not data.get("parse_success", True):
            return False

        try:
            metadata = data.get("metadata", {})
            basic_info = data.get("basic_info", {})

            # 取得職能基準代碼和名稱
            standard_code = metadata.get("code", "")
            standard_name = metadata.get("name", "")

            # 如果沒有代碼，使用名稱作為代碼
            if not standard_code:
                if standard_name:
                    standard_code = standard_name
                else:
                    logger.warning("缺少職能基準代碼和名稱，跳過")
                    return False

            # 1. 建立職能基準節點（中心節點）
            standard_id = self.add_node(
                node_id=standard_code,
                node_type="職能基準",
                name=standard_name,
                description=basic_info.get("job_description", ""),
                level=basic_info.get("level", 0),
                source_file=metadata.get("source_file", "")
            )

            # 2. 建立職類別節點和邊
            category = basic_info.get("category", "")
            category_code = basic_info.get("category_code", "")
            if category:
                cat_key = category_code if category_code else category
                category_id = self.add_node(
                    node_id=cat_key,
                    node_type="職類別",
                    code=category_code,
                    name=category
                )
                self.add_edge(standard_id, category_id, "屬於職類")
                self.category_index[cat_key].add(standard_code)

            # 3. 建立職業別節點和邊
            occupation = basic_info.get("occupation", "")
            occupation_code = basic_info.get("occupation_code", "")
            if occupation:
                occ_key = occupation_code if occupation_code else occupation
                occupation_id = self.add_node(
                    node_id=occ_key,
                    node_type="職業別",
                    code=occupation_code,
                    name=occupation
                )
                self.add_edge(standard_id, occupation_id, "屬於職業")
                self.occupation_index[occ_key].add(standard_code)

            # 4. 建立行業別節點和邊（支援多個行業）
            industry_val = basic_info.get("industry", "")
            industry_code_val = basic_info.get("industry_code", "")
            industries = industry_val if isinstance(industry_val, list) else ([industry_val] if industry_val else [])
            industry_codes = industry_code_val if isinstance(industry_code_val, list) else [industry_code_val] * len(industries)
            for i, ind_name in enumerate(industries):
                ind_code = industry_codes[i] if i < len(industry_codes) else ""
                if ind_name:
                    ind_key = ind_code if ind_code else ind_name
                    industry_id = self.add_node(
                        node_id=ind_key,
                        node_type="行業別",
                        code=ind_code,
                        name=ind_name
                    )
                    self.add_edge(standard_id, industry_id, "適用行業")
                    self.industry_index[ind_key].add(standard_code)

            # 5. 建立知識/技能/態度清單對照表
            knowledge_map = {k["code"]: k.get("name", "") for k in data.get("competency_knowledge", [])}
            skills_map = {s["code"]: s.get("name", "") for s in data.get("competency_skills", [])}
            attitudes_map = {a["code"]: a.get("name", "") for a in data.get("competency_attitudes", [])}

            # 6. 建立態度節點
            for a_code, a_name in attitudes_map.items():
                attitude_id = self.add_node(
                    node_id=a_code,
                    node_type="態度",
                    name=a_name,
                    description=""
                )
                self.add_edge(standard_id, attitude_id, "要求態度")

                if a_code not in self.global_attitudes:
                    self.global_attitudes[a_code] = {
                        "name": a_name,
                        "description": "",
                        "standards": set()
                    }
                self.global_attitudes[a_code]["standards"].add(standard_code)

            # 7. 建立工作任務（按主要職責分組）
            duty_tasks = defaultdict(list)
            for task in data.get("competency_tasks", []):
                main_resp = task.get("main_responsibility", "")
                duty_tasks[main_resp].append(task)

            for duty_name, tasks in duty_tasks.items():
                # 解析職責代碼和名稱 (格式: "T1研擬展覽行銷策略")
                duty_code = ""
                duty_display_name = duty_name
                if duty_name and duty_name[0] == "T":
                    # 嘗試提取代碼
                    import re
                    match = re.match(r"(T\d+)(.*)", duty_name)
                    if match:
                        duty_code = match.group(1)
                        duty_display_name = match.group(2) or duty_name

                if not duty_code:
                    duty_code = duty_name[:10] if duty_name else "T0"

                duty_unique_id = f"{standard_code}:{duty_code}"

                duty_id = self.add_node(
                    node_id=duty_unique_id,
                    node_type="主要職責",
                    code=duty_code,
                    name=duty_display_name,
                    standard_code=standard_code
                )
                self.add_edge(standard_id, duty_id, "包含職責")

                # 建立工作任務
                for task in tasks:
                    task_code = task.get("task_id", "")
                    if not task_code:
                        continue

                    task_unique_id = f"{standard_code}:{task_code}"

                    task_id = self.add_node(
                        node_id=task_unique_id,
                        node_type="工作任務",
                        code=task_code,
                        name=task.get("task_name", ""),
                        level=task.get("level", 0),
                        standard_code=standard_code
                    )
                    self.add_edge(duty_id, task_id, "包含任務")

                    # 工作產出
                    output = task.get("output", "")
                    if output:
                        out_code = f"O{task_code}"
                        out_unique_id = f"{standard_code}:{out_code}"
                        output_id = self.add_node(
                            node_id=out_unique_id,
                            node_type="工作產出",
                            code=out_code,
                            name=output,
                            standard_code=standard_code
                        )
                        self.add_edge(task_id, output_id, "產出")

                    # 行為指標
                    for i, behavior in enumerate(task.get("behaviors", []), 1):
                        beh_code = f"P{task_code}.{i}"
                        beh_unique_id = f"{standard_code}:{beh_code}"
                        behavior_id = self.add_node(
                            node_id=beh_unique_id,
                            node_type="行為指標",
                            code=beh_code,
                            description=behavior,
                            standard_code=standard_code
                        )
                        self.add_edge(task_id, behavior_id, "要求行為")

                    # 知識（全域共用節點）
                    for k_code in task.get("knowledge", []):
                        k_name = knowledge_map.get(k_code, "")

                        knowledge_id = self.add_node(
                            node_id=k_code,
                            node_type="知識",
                            name=k_name
                        )
                        self.add_edge(task_id, knowledge_id, "需要知識")

                        if k_code not in self.global_knowledge:
                            self.global_knowledge[k_code] = {
                                "name": k_name,
                                "standards": set(),
                                "tasks": set()
                            }
                        self.global_knowledge[k_code]["standards"].add(standard_code)
                        self.global_knowledge[k_code]["tasks"].add(task_unique_id)

                    # 技能（全域共用節點）
                    for s_code in task.get("skills", []):
                        s_name = skills_map.get(s_code, "")

                        skill_id = self.add_node(
                            node_id=s_code,
                            node_type="技能",
                            name=s_name
                        )
                        self.add_edge(task_id, skill_id, "需要技能")

                        if s_code not in self.global_skills:
                            self.global_skills[s_code] = {
                                "name": s_name,
                                "standards": set(),
                                "tasks": set()
                            }
                        self.global_skills[s_code]["standards"].add(standard_code)
                        self.global_skills[s_code]["tasks"].add(task_unique_id)

            return True

        except Exception as e:
            logger.error(f"建構圖譜失敗 (v2 格式): {e}")
            return False

    def build_from_json_directory(
        self,
        json_dir: str | Path,
        limit: Optional[int] = None
    ) -> int:
        """
        從 JSON 目錄批次建構圖譜（支援新舊格式）

        Args:
            json_dir: JSON 檔案目錄
            limit: 限制處理數量

        Returns:
            成功建構的數量
        """
        json_dir = Path(json_dir)
        json_files = list(json_dir.rglob("*.json"))

        if limit:
            json_files = json_files[:limit]

        logger.info(f"找到 {len(json_files)} 個 JSON 檔案")

        success_count = 0
        for json_path in tqdm(json_files, desc="建構圖譜"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 檢測 JSON 格式
                if "metadata" in data and "basic_info" in data:
                    # 新版格式 (pdf_parser_v2)
                    if self.build_from_v2_data(data):
                        success_count += 1
                else:
                    # 舊版格式 (ParsedCompetencyStandard)
                    parsed = ParsedCompetencyStandard(**data)
                    parsed.parse_success = True
                    if self.build_from_parsed_data(parsed):
                        success_count += 1

            except Exception as e:
                logger.error(f"處理失敗 {json_path}: {e}")

        logger.success(f"圖譜建構完成: {success_count} 個職能基準")
        return success_count

    def build_from_pdf_directory(
        self,
        pdf_dir: str | Path,
        limit: Optional[int] = None
    ) -> int:
        """
        直接從 PDF 目錄建構圖譜

        Args:
            pdf_dir: PDF 檔案目錄
            limit: 限制處理數量

        Returns:
            成功建構的數量
        """
        parser = CompetencyPDFParser()
        parsed_results = parser.parse_directory(pdf_dir, limit=limit)

        success_count = 0
        for parsed in tqdm(parsed_results, desc="建構圖譜"):
            if self.build_from_parsed_data(parsed):
                success_count += 1

        logger.success(f"圖譜建構完成: {success_count} 個職能基準")
        return success_count

    def infer_career_paths(self):
        """
        推斷職涯晉升路徑

        基於：
        1. 職能基準的 level
        2. 共用的知識/技能
        3. 相同的職類別/職業別
        """
        logger.info("推斷職涯晉升路徑...")

        standards = self.get_nodes_by_type("職能基準")

        for std1 in standards:
            std1_data = self.graph.nodes[std1]
            std1_level = std1_data.get("level", 0)
            std1_code = std1_data.get("node_id", "")

            std1_ks = self._get_knowledge_skills(std1)

            for std2 in standards:
                if std1 == std2:
                    continue

                std2_data = self.graph.nodes[std2]
                std2_level = std2_data.get("level", 0)
                std2_code = std2_data.get("node_id", "")

                if std2_level <= std1_level:
                    continue

                std2_ks = self._get_knowledge_skills(std2)

                overlap = len(std1_ks & std2_ks)
                if overlap > 0 and len(std1_ks) > 0:
                    overlap_ratio = overlap / len(std1_ks)

                    # 檢查是否有共同的職類別/職業別
                    same_category = self._check_same_category(std1_code, std2_code)

                    # 提高有共同類別的權重
                    if overlap_ratio > 0.3 or (same_category and overlap_ratio > 0.2):
                        self.add_edge(
                            std1, std2, "可晉升至",
                            overlap_ratio=overlap_ratio,
                            shared_competencies=overlap,
                            same_category=same_category
                        )

        logger.success("職涯路徑推斷完成")

    def _check_same_category(self, std1_code: str, std2_code: str) -> bool:
        """檢查兩個職能基準是否有相同的類別"""
        for cat_key, standards in self.category_index.items():
            if std1_code in standards and std2_code in standards:
                return True
        for occ_key, standards in self.occupation_index.items():
            if std1_code in standards and std2_code in standards:
                return True
        return False

    def infer_related_knowledge_skills(self):
        """
        推斷知識/技能之間的關聯關係
        """
        logger.info("推斷知識/技能關聯關係...")

        cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        tasks = self.get_nodes_by_type("工作任務")

        for task_id in tasks:
            k_nodes = [
                target for _, target, data in self.graph.out_edges(task_id, data=True)
                if data.get("edge_type") == "需要知識"
            ]
            s_nodes = [
                target for _, target, data in self.graph.out_edges(task_id, data=True)
                if data.get("edge_type") == "需要技能"
            ]

            all_ks = k_nodes + s_nodes
            for i, ks1 in enumerate(all_ks):
                for ks2 in all_ks[i+1:]:
                    key = tuple(sorted([ks1, ks2]))
                    cooccurrence[key] += 1

        for (node1, node2), count in cooccurrence.items():
            if count >= 3:
                self.add_edge(node1, node2, "相關於", cooccurrence_count=count)
                self.add_edge(node2, node1, "相關於", cooccurrence_count=count)

        logger.success("知識/技能關聯推斷完成")

    def _get_knowledge_skills(self, standard_id: str) -> Set[str]:
        """取得職能基準的所有知識和技能"""
        ks_set = set()

        for _, duty_id, duty_edge in self.graph.out_edges(standard_id, data=True):
            if duty_edge.get("edge_type") != "包含職責":
                continue

            for _, task_id, task_edge in self.graph.out_edges(duty_id, data=True):
                if task_edge.get("edge_type") != "包含任務":
                    continue

                for _, ks_id, ks_edge in self.graph.out_edges(task_id, data=True):
                    if ks_edge.get("edge_type") in ["需要知識", "需要技能"]:
                        ks_set.add(ks_id)

        return ks_set

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """取得指定類型的所有節點"""
        return list(self.node_index.get(node_type, set()))

    def get_node_data(self, node_id: str) -> Optional[Dict]:
        """取得節點資料"""
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "out"
    ) -> List[Tuple[str, Dict]]:
        """
        取得節點的鄰居

        Args:
            node_id: 節點 ID
            edge_type: 過濾邊類型
            direction: "out", "in", "both"

        Returns:
            [(鄰居節點ID, 邊資料), ...]
        """
        neighbors = []

        if direction in ["out", "both"]:
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    neighbors.append((target, data))

        if direction in ["in", "both"]:
            for source, _, data in self.graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    neighbors.append((source, data))

        return neighbors

    def get_hierarchy(self, standard_code: str) -> Dict[str, Any]:
        """
        取得職能基準的完整階層結構

        Args:
            standard_code: 職能基準代碼

        Returns:
            階層結構字典
        """
        standard_id = f"職能基準:{standard_code}"

        if not self.graph.has_node(standard_id):
            return {}

        hierarchy = {
            "standard": self.get_node_data(standard_id),
            "categories": [],
            "occupations": [],
            "industries": [],
            "attitudes": [],
            "duties": []
        }

        for _, neighbor, edge_data in self.graph.out_edges(standard_id, data=True):
            edge_type = edge_data.get("edge_type")
            neighbor_data = self.get_node_data(neighbor)

            if edge_type == "屬於職類":
                hierarchy["categories"].append(neighbor_data)
            elif edge_type == "屬於職業":
                hierarchy["occupations"].append(neighbor_data)
            elif edge_type == "適用行業":
                hierarchy["industries"].append(neighbor_data)
            elif edge_type == "要求態度":
                hierarchy["attitudes"].append(neighbor_data)
            elif edge_type == "包含職責":
                duty_hierarchy = {
                    "duty": neighbor_data,
                    "tasks": []
                }

                for _, task_node, task_edge in self.graph.out_edges(neighbor, data=True):
                    if task_edge.get("edge_type") == "包含任務":
                        task_data = self.get_node_data(task_node)
                        task_hierarchy = {
                            "task": task_data,
                            "outputs": [],
                            "behaviors": [],
                            "knowledge": [],
                            "skills": []
                        }

                        for _, item, item_edge in self.graph.out_edges(task_node, data=True):
                            item_data = self.get_node_data(item)
                            item_type = item_edge.get("edge_type")

                            if item_type == "產出":
                                task_hierarchy["outputs"].append(item_data)
                            elif item_type == "要求行為":
                                task_hierarchy["behaviors"].append(item_data)
                            elif item_type == "需要知識":
                                task_hierarchy["knowledge"].append(item_data)
                            elif item_type == "需要技能":
                                task_hierarchy["skills"].append(item_data)

                        duty_hierarchy["tasks"].append(task_hierarchy)

                hierarchy["duties"].append(duty_hierarchy)

        return hierarchy

    def find_similar_standards(
        self,
        standard_code: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        找出相似的職能基準

        基於共用的知識/技能計算相似度

        Args:
            standard_code: 職能基準代碼
            top_k: 返回數量

        Returns:
            [(職能基準代碼, 相似度), ...]
        """
        standard_id = f"職能基準:{standard_code}"

        if not self.graph.has_node(standard_id):
            return []

        source_ks = self._get_knowledge_skills(standard_id)
        if not source_ks:
            return []

        similarities = []

        for other_standard in self.get_nodes_by_type("職能基準"):
            if other_standard == standard_id:
                continue

            other_ks = self._get_knowledge_skills(other_standard)
            if not other_ks:
                continue

            # Jaccard 相似度
            intersection = len(source_ks & other_ks)
            union = len(source_ks | other_ks)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0:
                other_data = self.get_node_data(other_standard)
                similarities.append((other_data.get("node_id", ""), similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """取得圖譜統計資訊"""
        stats = {
            "節點總數": self.graph.number_of_nodes(),
            "邊總數": self.graph.number_of_edges(),
            "節點類型統計": {},
            "邊類型統計": {},
            "全域知識數": len(self.global_knowledge),
            "全域技能數": len(self.global_skills),
            "全域態度數": len(self.global_attitudes),
            "職類別數": len(self.category_index),
            "職業別數": len(self.occupation_index),
            "行業別數": len(self.industry_index),
        }

        for node_type, nodes in self.node_index.items():
            stats["節點類型統計"][node_type] = len(nodes)

        for edge_type, edges in self.edge_index.items():
            stats["邊類型統計"][edge_type] = len(edges)

        return stats

    def save(self, filepath: str | Path):
        """儲存圖譜"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "graph": nx.node_link_data(self.graph),
            "node_index": {k: list(v) for k, v in self.node_index.items()},
            "edge_index": dict(self.edge_index),
            "global_knowledge": {
                k: {**v, "standards": list(v["standards"]), "tasks": list(v.get("tasks", []))}
                for k, v in self.global_knowledge.items()
            },
            "global_skills": {
                k: {**v, "standards": list(v["standards"]), "tasks": list(v.get("tasks", []))}
                for k, v in self.global_skills.items()
            },
            "global_attitudes": {
                k: {**v, "standards": list(v["standards"])}
                for k, v in self.global_attitudes.items()
            },
            "category_index": {k: list(v) for k, v in self.category_index.items()},
            "occupation_index": {k: list(v) for k, v in self.occupation_index.items()},
            "industry_index": {k: list(v) for k, v in self.industry_index.items()},
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.success(f"圖譜已儲存至 {filepath}")

    def load(self, filepath: str | Path) -> bool:
        """載入圖譜"""
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"圖譜檔案不存在: {filepath}")
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.graph = nx.node_link_graph(data["graph"])
            self.node_index = defaultdict(set, {
                k: set(v) for k, v in data["node_index"].items()
            })
            self.edge_index = defaultdict(list, data["edge_index"])
            self.global_knowledge = {
                k: {**v, "standards": set(v["standards"]), "tasks": set(v.get("tasks", []))}
                for k, v in data["global_knowledge"].items()
            }
            self.global_skills = {
                k: {**v, "standards": set(v["standards"]), "tasks": set(v.get("tasks", []))}
                for k, v in data["global_skills"].items()
            }
            self.global_attitudes = {
                k: {**v, "standards": set(v["standards"])}
                for k, v in data["global_attitudes"].items()
            }
            self.category_index = defaultdict(set, {
                k: set(v) for k, v in data.get("category_index", {}).items()
            })
            self.occupation_index = defaultdict(set, {
                k: set(v) for k, v in data.get("occupation_index", {}).items()
            })
            self.industry_index = defaultdict(set, {
                k: set(v) for k, v in data.get("industry_index", {}).items()
            })

            logger.success(f"圖譜已載入: {filepath}")
            return True

        except Exception as e:
            logger.error(f"載入圖譜失敗: {e}")
            return False

    def enrich_with_icap_metadata(self, icap_source_dir: str | Path) -> int:
        """
        從 ICAP 來源 JSON 補充行業別、職類別、職業別等 metadata 到圖譜中

        ICAP 來源 JSON 包含:
        - fields.所屬行業別: 行業名稱 (如 "餐館")
        - fields.所屬領域別: 職類別/領域 (如 "餐飲管理")
        - fields.通俗職業分類: 通俗職業類別 (如 "餐飲")
        - fields.所屬通俗職務名稱: 通俗職務名稱 (如 "中/西餐烹飪廚師")

        Args:
            icap_source_dir: ICAP 來源資料夾路徑 (包含各職類別子資料夾)

        Returns:
            成功補充的職能基準數量
        """
        icap_source_dir = Path(icap_source_dir)
        if not icap_source_dir.exists():
            logger.warning(f"ICAP 來源目錄不存在: {icap_source_dir}")
            return 0

        # 收集所有 ICAP JSON 檔案 (排除解析後的 JSON)
        icap_json_files = []
        for json_path in icap_source_dir.rglob("*.json"):
            # ICAP JSON 的特徵: 檔名包含日期 (如 xxx-20241220.json)
            if json_path.stem.split("-")[-1].isdigit() and len(json_path.stem.split("-")[-1]) == 8:
                icap_json_files.append(json_path)

        logger.info(f"找到 {len(icap_json_files)} 個 ICAP 來源 JSON 檔案")

        # 建立職能代碼到 metadata 的映射
        code_to_metadata: Dict[str, Dict] = {}

        for json_path in icap_json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                fields = data.get("fields", {})
                code = fields.get("職能項目代碼", "")

                if code:
                    code_to_metadata[code] = {
                        "行業別": fields.get("所屬行業別", ""),
                        "職類別": fields.get("所屬領域別", ""),
                        "通俗職業分類": fields.get("通俗職業分類", ""),
                        "通俗職務名稱": fields.get("所屬通俗職務名稱", ""),
                        "職能項目名稱": fields.get("職能項目名稱", ""),
                    }
            except Exception as e:
                logger.warning(f"讀取 ICAP JSON 失敗 {json_path}: {e}")

        logger.info(f"已建立 {len(code_to_metadata)} 個職能代碼的 metadata 映射")

        # 遍歷圖譜中的職能基準節點，補充 metadata
        enriched_count = 0
        standards = self.get_nodes_by_type("職能基準")

        for std_node_id in standards:
            node_data = self.graph.nodes[std_node_id]
            std_code = node_data.get("node_id", "")

            # 嘗試匹配 (可能版本號不同，如 TFB3433-001v2 vs TFB3433-001v3)
            metadata = None
            if std_code in code_to_metadata:
                metadata = code_to_metadata[std_code]
            else:
                # 嘗試模糊匹配 (去掉版本號)
                base_code = std_code.rsplit("v", 1)[0] if "v" in std_code else std_code
                for code, meta in code_to_metadata.items():
                    if code.startswith(base_code):
                        metadata = meta
                        break

            if not metadata:
                continue

            enriched = False

            # 補充行業別
            industry_name = metadata.get("行業別", "")
            if industry_name and industry_name != "無":
                # 檢查是否已存在此行業別節點
                existing_industries = [
                    target for _, target, data in self.graph.out_edges(std_node_id, data=True)
                    if data.get("edge_type") == "適用行業"
                ]

                if not existing_industries:
                    industry_id = self.add_node(
                        node_id=industry_name,
                        node_type="行業別",
                        code="",
                        name=industry_name
                    )
                    self.add_edge(std_node_id, industry_id, "適用行業")
                    self.industry_index[industry_name].add(std_code)
                    enriched = True

            # 補充職類別 (如果圖譜中沒有)
            category_name = metadata.get("職類別", "")
            if category_name and category_name != "無":
                existing_categories = [
                    target for _, target, data in self.graph.out_edges(std_node_id, data=True)
                    if data.get("edge_type") == "屬於職類"
                ]

                if not existing_categories:
                    category_id = self.add_node(
                        node_id=category_name,
                        node_type="職類別",
                        code="",
                        name=category_name
                    )
                    self.add_edge(std_node_id, category_id, "屬於職類")
                    self.category_index[category_name].add(std_code)
                    enriched = True

            # 補充通俗職業分類
            occupation_class = metadata.get("通俗職業分類", "")
            if occupation_class and occupation_class != "無":
                occ_class_id = self.add_node(
                    node_id=f"通俗分類:{occupation_class}",
                    node_type="通俗職業分類",
                    name=occupation_class
                )
                # 檢查是否已有此邊
                existing_edges = [
                    (s, t) for s, t, d in self.graph.out_edges(std_node_id, data=True)
                    if d.get("edge_type") == "通俗分類" and t == occ_class_id
                ]
                if not existing_edges:
                    self.add_edge(std_node_id, occ_class_id, "通俗分類")
                    enriched = True

            # 補充通俗職務名稱
            occupation_name = metadata.get("通俗職務名稱", "")
            if occupation_name and occupation_name != "無":
                occ_name_id = self.add_node(
                    node_id=f"通俗職務:{occupation_name}",
                    node_type="通俗職務名稱",
                    name=occupation_name
                )
                existing_edges = [
                    (s, t) for s, t, d in self.graph.out_edges(std_node_id, data=True)
                    if d.get("edge_type") == "通俗職務" and t == occ_name_id
                ]
                if not existing_edges:
                    self.add_edge(std_node_id, occ_name_id, "通俗職務")
                    enriched = True

            # 更新節點屬性 (方便查詢時直接取得)
            self.graph.nodes[std_node_id]["icap_industry"] = metadata.get("行業別", "")
            self.graph.nodes[std_node_id]["icap_category"] = metadata.get("職類別", "")
            self.graph.nodes[std_node_id]["icap_occupation_class"] = metadata.get("通俗職業分類", "")
            self.graph.nodes[std_node_id]["icap_occupation_name"] = metadata.get("通俗職務名稱", "")

            if enriched:
                enriched_count += 1

        logger.success(f"已補充 {enriched_count} 個職能基準的 ICAP metadata")
        return enriched_count


# ========================================
# 命令列工具
# ========================================

def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="知識圖譜建構工具（階層式版本）")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(config.ICAP_SOURCE_DIR),
        help="輸入 PDF 目錄路徑"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(config.GRAPH_DB_DIR / config.GRAPH_FILE),
        help="輸出圖譜檔案路徑"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="限制處理數量（測試用）"
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="推斷職涯路徑和知識/技能關聯"
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    kg = CompetencyKnowledgeGraph()
    count = kg.build_from_pdf_directory(args.input, limit=args.limit)

    if args.infer:
        kg.infer_career_paths()
        kg.infer_related_knowledge_skills()

    stats = kg.get_statistics()
    print("\n" + "=" * 60)
    print("知識圖譜統計（階層式結構）")
    print("=" * 60)
    print(f"節點總數: {stats['節點總數']}")
    print(f"邊總數: {stats['邊總數']}")
    print("\n節點類型:")
    for node_type, count in stats["節點類型統計"].items():
        print(f"  {node_type}: {count}")
    print("\n邊類型:")
    for edge_type, count in stats["邊類型統計"].items():
        print(f"  {edge_type}: {count}")
    print("=" * 60)

    kg.save(args.output)
    print(f"\n圖譜已儲存至: {args.output}")


if __name__ == "__main__":
    main()
