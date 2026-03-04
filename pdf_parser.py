"""
職能基準 PDF 解析模組（改進版）
負責從 ICAP 職能基準 PDF 中提取結構化的階層式資料
支援論文中提到的階層式節點表示法
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

import fitz  # PyMuPDF
from loguru import logger
from tqdm import tqdm

from config import get_config

config = get_config()


@dataclass
class WorkTaskData:
    """工作任務資料結構"""
    代碼: str = ""           # T1.1, T1.2 等
    名稱: str = ""           # 任務名稱
    職能級別: int = 0        # 1-7 級
    工作產出: List[Dict[str, str]] = field(default_factory=list)    # [{代碼, 名稱}]
    行為指標: List[Dict[str, str]] = field(default_factory=list)    # [{代碼, 描述}]
    知識: List[str] = field(default_factory=list)                   # [K01, K02, ...]
    技能: List[str] = field(default_factory=list)                   # [S01, S02, ...]


@dataclass
class MainDutyData:
    """主要職責資料結構"""
    代碼: str = ""           # T1, T2 等
    名稱: str = ""           # 職責名稱
    工作任務: List[WorkTaskData] = field(default_factory=list)


@dataclass
class ParsedCompetencyStandard:
    """解析後的職能基準資料結構（改進版）"""

    # 基本資訊
    職能基準: Dict[str, Any] = field(default_factory=dict)

    # 職責和任務（階層式結構）
    主要職責: List[Dict[str, Any]] = field(default_factory=list)

    # 知識技能態度清單（全域對照表）
    知識清單: Dict[str, str] = field(default_factory=dict)    # K01 -> "AIoT 系統基本架構"
    技能清單: Dict[str, str] = field(default_factory=dict)    # S01 -> "資料搜集與分析能力"
    態度清單: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # 補充說明
    補充說明: Dict[str, Any] = field(default_factory=dict)

    # 解析元資料
    source_file: str = ""
    parse_success: bool = False
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """轉換為 JSON 字串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class CompetencyPDFParser:
    """職能基準 PDF 解析器（改進版）"""

    def __init__(self):
        """初始化解析器"""
        self.patterns = config.CODE_PATTERNS

        # 編譯正則表達式
        self.compiled_patterns = {
            key: re.compile(pattern)
            for key, pattern in self.patterns.items()
        }

        # 職能級別對應表
        self.level_map = {
            "第一級": 1, "一級": 1, "1級": 1, "1": 1,
            "第二級": 2, "二級": 2, "2級": 2, "2": 2,
            "第三級": 3, "三級": 3, "3級": 3, "3": 3,
            "第四級": 4, "四級": 4, "4級": 4, "4": 4,
            "第五級": 5, "五級": 5, "5級": 5, "5": 5,
            "第六級": 6, "六級": 6, "6級": 6, "6": 6,
            "第七級": 7, "七級": 7, "7級": 7, "7": 7,
        }

        logger.info("職能基準 PDF 解析器初始化完成（改進版）")

    def parse_pdf(self, pdf_path: str | Path) -> ParsedCompetencyStandard:
        """
        解析職能基準 PDF 檔案

        Args:
            pdf_path: PDF 檔案路徑

        Returns:
            ParsedCompetencyStandard 物件
        """
        pdf_path = Path(pdf_path)
        result = ParsedCompetencyStandard(source_file=str(pdf_path))

        if not pdf_path.exists():
            result.parse_errors.append(f"檔案不存在: {pdf_path}")
            return result

        try:
            doc = fitz.open(str(pdf_path))

            # 收集所有頁面的表格資料
            all_tables = []
            full_text = ""

            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                full_text += f"\n--- 第 {page_num + 1} 頁 ---\n{text}"

                try:
                    tables = page.find_tables()
                    for table in tables:
                        table_data = table.extract()
                        if table_data:
                            all_tables.append({
                                "page": page_num + 1,
                                "data": table_data
                            })
                except Exception as e:
                    logger.debug(f"表格提取失敗 (頁 {page_num + 1}): {e}")

            doc.close()

            # 1. 解析基本資訊（從第一頁表格）
            result.職能基準 = self._parse_basic_info_from_table(all_tables, pdf_path)

            # 2. 解析主要職責和工作任務（階層式結構）
            result.主要職責, k_list, s_list = self._parse_hierarchical_duties(all_tables)

            # 2.5 從全文中補充提取知識和技能代碼（作為備份）
            text_k_codes, text_s_codes = self._extract_ks_codes_from_text(full_text)
            k_list = list(set(k_list + text_k_codes))
            s_list = list(set(s_list + text_s_codes))

            # 3. 解析並建立知識清單（從工作內容表中的知識欄位）
            result.知識清單 = self._build_knowledge_dict(k_list, full_text)

            # 4. 解析並建立技能清單
            result.技能清單 = self._build_skill_dict(s_list, full_text)

            # 5. 解析態度清單
            result.態度清單 = self._parse_attitudes(full_text)

            # 6. 解析補充說明
            result.補充說明 = self._parse_supplementary(full_text)

            result.parse_success = True
            logger.success(f"成功解析: {pdf_path.name}")

        except Exception as e:
            result.parse_errors.append(f"解析錯誤: {str(e)}")
            logger.error(f"解析失敗 {pdf_path}: {e}")

        return result

    def _parse_basic_info_from_table(
        self,
        tables: List[Dict],
        pdf_path: Path
    ) -> Dict[str, Any]:
        """
        從第一頁表格解析基本資訊

        Args:
            tables: 所有表格資料
            pdf_path: PDF 路徑

        Returns:
            基本資訊字典
        """
        info = {
            "代碼": "",
            "名稱": "",
            "職類別": [],      # 改為列表，支援多個職類別
            "職業別": [],      # 改為列表，支援多個職業別
            "行業別": [],      # 改為列表，支援多個行業別
            "工作描述": "",
            "基準級別": 0,
        }

        # 從檔名推測職務名稱
        file_name = pdf_path.stem
        if "-職能基準" in file_name:
            info["名稱"] = file_name.replace("-職能基準", "")

        # 尋找第一頁的基本資訊表格
        first_page_tables = [t for t in tables if t["page"] == 1]

        for table_info in first_page_tables:
            table = table_info["data"]

            for row in table:
                if not row:
                    continue

                row_str = [str(cell).strip() if cell else "" for cell in row]
                row_text = " ".join(row_str)

                # 提取職能基準代碼
                if "職能基準代碼" in row_text:
                    for cell in row:
                        if cell:
                            code_match = re.search(r"[A-Z]{2,3}\d{4}-\d{3}v?\d*", str(cell))
                            if code_match:
                                info["代碼"] = code_match.group()
                                break

                # 提取基準級別
                if "基準級別" in row_text:
                    for cell in row:
                        if cell:
                            level_match = re.search(r"(\d+)", str(cell))
                            if level_match and "基準" not in str(cell):
                                info["基準級別"] = int(level_match.group(1))
                                break

                # 提取工作描述
                if "工作描述" in row_text:
                    for cell in row:
                        if cell and "工作描述" not in str(cell) and len(str(cell)) > 20:
                            desc = str(cell).replace("\n", " ").strip()
                            info["工作描述"] = desc[:500]
                            break

                # 提取職類別
                if "職類別" in row_text and "代碼" not in row_text:
                    names = []
                    codes = []
                    for i, cell in enumerate(row):
                        if cell and "職類別" not in str(cell):
                            cell_str = str(cell)
                            if "代碼" in cell_str:
                                continue
                            # 檢查是否為代碼列
                            if re.match(r"^[A-Z]{2,3}(\n[A-Z]{2,3})*$", cell_str.strip()):
                                codes = cell_str.strip().split("\n")
                            elif len(cell_str) > 3:
                                names = cell_str.strip().split("\n")

                    # 配對名稱和代碼
                    for j, name in enumerate(names):
                        code = codes[j] if j < len(codes) else ""
                        info["職類別"].append({"代碼": code.strip(), "名稱": name.strip()})

                # 提取職業別（支援兩種格式：同行或分開）
                if "職業別" in row_text and "職類別" not in row_text:
                    names = []
                    codes = []

                    # 格式1: 職業別和職業別代碼在同一行的不同儲存格
                    # 例如: | 職業別 | 廚師 | ... | 職業別代碼 | 5120 |
                    if "職業別代碼" in row_text:
                        for i, cell in enumerate(row):
                            cell_str = str(cell).strip() if cell else ""
                            # 找到職業別代碼後的數字
                            if cell_str and re.match(r"^\d{4}$", cell_str):
                                codes.append(cell_str)
                            # 找到職業別名稱（不是標籤、不是代碼）
                            elif cell_str and "職業別" not in cell_str and "代碼" not in cell_str:
                                if len(cell_str) > 1 and not cell_str.isdigit():
                                    names.append(cell_str)
                    else:
                        # 格式2: 原有的多行格式
                        for i, cell in enumerate(row):
                            if cell and "職業別" not in str(cell):
                                cell_str = str(cell)
                                if "代碼" in cell_str:
                                    continue
                                if re.match(r"^\d+(\n\d+)*$", cell_str.strip()):
                                    codes = cell_str.strip().split("\n")
                                elif len(cell_str) > 1:
                                    names = cell_str.strip().split("\n")

                    # 配對名稱和代碼
                    if names:
                        for j, name in enumerate(names):
                            code = codes[j] if j < len(codes) else ""
                            if name.strip():
                                info["職業別"].append({"代碼": code.strip(), "名稱": name.strip()})

                # 提取行業別（支援兩種格式：同行或分開）
                if "行業別" in row_text:
                    names = []
                    codes = []

                    # 格式1: 行業別和行業別代碼在同一行的不同儲存格
                    # 例如: | 行業別 | 住宿及餐飲業 / 餐館 | ... | 行業別代碼 | I5611 |
                    if "行業別代碼" in row_text:
                        for i, cell in enumerate(row):
                            cell_str = str(cell).strip() if cell else ""
                            # 找到行業別代碼（字母+數字，如 I5611）
                            if cell_str and re.match(r"^[A-Z]\d{3,5}$", cell_str):
                                codes.append(cell_str)
                            # 找到行業別名稱
                            elif cell_str and "行業別" not in cell_str and "代碼" not in cell_str:
                                if len(cell_str) > 2 and not re.match(r"^[A-Z]\d+$", cell_str):
                                    names.append(cell_str)
                    else:
                        # 格式2: 原有的多行格式
                        for i, cell in enumerate(row):
                            if cell and "行業別" not in str(cell):
                                cell_str = str(cell)
                                if "代碼" in cell_str:
                                    continue
                                if re.match(r"^[A-Z]\d+(\n[A-Z]\d+)*$", cell_str.strip()):
                                    codes = cell_str.strip().split("\n")
                                elif len(cell_str) > 2:
                                    names = cell_str.strip().split("\n")

                    # 配對名稱和代碼
                    if names:
                        for j, name in enumerate(names):
                            code = codes[j] if j < len(codes) else ""
                            if name.strip():
                                info["行業別"].append({"代碼": code.strip(), "名稱": name.strip()})

        return info

    def _clean_cell_text(self, text: str) -> str:
        """
        清理儲存格文字，移除多餘空格和換行

        Args:
            text: 原始文字

        Returns:
            清理後的文字
        """
        if not text:
            return ""
        # 移除多餘的空格（保留單一空格）
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空格
        text = text.strip()
        return text

    def _parse_hierarchical_duties(
        self,
        tables: List[Dict]
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """
        解析階層式的主要職責和工作任務

        Args:
            tables: 所有表格資料

        Returns:
            (主要職責列表, 知識代碼列表, 技能代碼列表)
        """
        duties = []
        all_knowledge = []
        all_skills = []

        current_duty = None
        current_duty_code = None

        # 尋找工作內容表（通常從第2頁開始）
        for table_info in tables:
            table = table_info["data"]

            # 檢查是否為工作內容表（檢查表頭）
            if not table:
                continue

            header = table[0] if table else []
            header_text = " ".join([str(h) if h else "" for h in header])

            # 判斷是否為工作內容表（更寬鬆的匹配）
            is_work_table = (
                ("主要職責" in header_text and "工作任務" in header_text) or
                ("主要職責" in header_text and "工作產出" in header_text) or
                ("工作任務" in header_text and "行為指標" in header_text)
            )

            if not is_work_table:
                continue

            # 確定欄位索引（表頭可能有不同順序）
            col_indices = self._detect_column_indices(header)

            # 解析表格行（跳過表頭）
            for row_idx, row in enumerate(table[1:], 1):
                if not row or all(cell is None or cell == "" for cell in row):
                    continue

                # 根據檢測到的欄位索引解析
                duty_cell = self._get_cell(row, col_indices.get("主要職責", 0))
                task_cell = self._get_cell(row, col_indices.get("工作任務", 1))
                output_cell = self._get_cell(row, col_indices.get("工作產出", 2))
                behavior_cell = self._get_cell(row, col_indices.get("行為指標", 3))
                level_cell = self._get_cell(row, col_indices.get("職能級別", 4))
                knowledge_cell = self._get_cell(row, col_indices.get("知識", 5))
                skill_cell = self._get_cell(row, col_indices.get("技能", 6))

                # 清理文字
                duty_cell = self._clean_cell_text(duty_cell)
                task_cell = self._clean_cell_text(task_cell)
                output_cell = self._clean_cell_text(output_cell)
                behavior_cell = self._clean_cell_text(behavior_cell)
                level_cell = self._clean_cell_text(level_cell)
                knowledge_cell = self._clean_cell_text(knowledge_cell)
                skill_cell = self._clean_cell_text(skill_cell)

                # 檢測主要職責（T1, T2 格式）
                duty_match = re.match(r"(T\d+)\s*(.+)", duty_cell)
                if duty_match:
                    # 儲存前一個職責
                    if current_duty:
                        duties.append(current_duty)

                    current_duty_code = duty_match.group(1)
                    duty_name = duty_match.group(2).strip()

                    current_duty = {
                        "代碼": current_duty_code,
                        "名稱": duty_name,
                        "工作任務": []
                    }

                # 檢測工作任務（T1.1, T1.2 格式）
                task_match = re.match(r"(T\d+\.\d+)\s*(.+)", task_cell)
                if task_match:
                    task_code = task_match.group(1)
                    task_name = task_match.group(2).strip()

                    # 解析職能級別
                    level = 0
                    if level_cell:
                        level_num = re.search(r"(\d+)", level_cell)
                        if level_num:
                            level = int(level_num.group(1))

                    # 解析工作產出
                    outputs = self._parse_outputs(output_cell)

                    # 解析行為指標
                    behaviors = self._parse_behaviors(behavior_cell)

                    # 解析知識和技能代碼（支援多種分隔方式）
                    k_codes = re.findall(r"K\d{2,3}", knowledge_cell)
                    s_codes = re.findall(r"S\d{2,3}", skill_cell)

                    all_knowledge.extend(k_codes)
                    all_skills.extend(s_codes)

                    # 如果還沒有職責但有任務，創建隱含的職責
                    if not current_duty:
                        duty_code = task_code.split(".")[0]
                        current_duty = {
                            "代碼": duty_code,
                            "名稱": "",
                            "工作任務": []
                        }
                        current_duty_code = duty_code

                    task = {
                        "代碼": task_code,
                        "名稱": task_name,
                        "職能級別": level,
                        "工作產出": outputs,
                        "行為指標": behaviors,
                        "知識": list(set(k_codes)),
                        "技能": list(set(s_codes))
                    }

                    current_duty["工作任務"].append(task)

                # 處理跨行的工作產出和行為指標（沒有新任務代碼的行）
                elif current_duty and current_duty["工作任務"] and (output_cell or behavior_cell or knowledge_cell or skill_cell):
                    last_task = current_duty["工作任務"][-1]

                    # 追加工作產出
                    if output_cell:
                        outputs = self._parse_outputs(output_cell)
                        last_task["工作產出"].extend(outputs)

                    # 追加行為指標
                    if behavior_cell:
                        behaviors = self._parse_behaviors(behavior_cell)
                        last_task["行為指標"].extend(behaviors)

                    # 追加知識和技能
                    if knowledge_cell:
                        k_codes = re.findall(r"K\d{2,3}", knowledge_cell)
                        last_task["知識"].extend(k_codes)
                        last_task["知識"] = list(set(last_task["知識"]))
                        all_knowledge.extend(k_codes)

                    if skill_cell:
                        s_codes = re.findall(r"S\d{2,3}", skill_cell)
                        last_task["技能"].extend(s_codes)
                        last_task["技能"] = list(set(last_task["技能"]))
                        all_skills.extend(s_codes)

        # 儲存最後一個職責
        if current_duty:
            duties.append(current_duty)

        return duties, list(set(all_knowledge)), list(set(all_skills))

    def _detect_column_indices(self, header: List) -> Dict[str, int]:
        """
        檢測表頭欄位的索引位置

        Args:
            header: 表頭列

        Returns:
            欄位名稱到索引的映射
        """
        indices = {}
        column_keywords = {
            "主要職責": ["主要職責", "職責"],
            "工作任務": ["工作任務", "任務"],
            "工作產出": ["工作產出", "產出"],
            "行為指標": ["行為指標", "指標"],
            "職能級別": ["職能級別", "級別", "職能\n級別"],
            "知識": ["知識", "K=knowledge", "K=", "knowledge"],
            "技能": ["技能", "S=skills", "S=skill", "S=", "skills"],
        }

        for i, cell in enumerate(header):
            if cell is None:
                continue
            cell_str = str(cell).strip().lower()

            for col_name, keywords in column_keywords.items():
                for kw in keywords:
                    if kw.lower() in cell_str:
                        if col_name not in indices:
                            indices[col_name] = i
                        break

        # 特殊處理：職能內涵欄位通常包含知識和技能
        for i, cell in enumerate(header):
            if cell is None:
                continue
            cell_str = str(cell)

            # 如果是「職能內涵」欄位，需要進一步分析
            if "職能內涵" in cell_str:
                # 通常職能內涵會分成兩欄：知識和技能
                # 先設定知識欄位
                if "知識" not in indices:
                    indices["知識"] = i

                # 檢查下一欄是否為技能
                if i + 1 < len(header) and "技能" not in indices:
                    next_cell = str(header[i + 1]) if header[i + 1] else ""
                    if "S=" in next_cell or "skill" in next_cell.lower() or "技能" in next_cell:
                        indices["技能"] = i + 1

        # 如果沒有找到，使用預設索引
        defaults = {
            "主要職責": 0,
            "工作任務": 1,
            "工作產出": 2,
            "行為指標": 3,
            "職能級別": 4,
            "知識": 5,
            "技能": 6,
        }

        for col_name, default_idx in defaults.items():
            if col_name not in indices:
                indices[col_name] = default_idx

        return indices

    def _get_cell(self, row: List, index: int) -> str:
        """
        安全地取得儲存格內容

        Args:
            row: 行資料
            index: 欄位索引

        Returns:
            儲存格內容字串
        """
        if index < len(row) and row[index] is not None:
            return str(row[index])
        return ""

    def _extract_ks_codes_from_text(self, full_text: str) -> Tuple[List[str], List[str]]:
        """
        從全文中提取知識和技能代碼（作為表格解析的補充）

        Args:
            full_text: PDF 全文

        Returns:
            (知識代碼列表, 技能代碼列表)
        """
        # 提取所有 K 代碼
        k_codes = re.findall(r"K\d{2,3}", full_text)
        # 提取所有 S 代碼
        s_codes = re.findall(r"S\d{2,3}", full_text)

        return list(set(k_codes)), list(set(s_codes))

    def _parse_outputs(self, cell: str) -> List[Dict[str, str]]:
        """解析工作產出欄位"""
        outputs = []
        if not cell:
            return outputs

        # 清理換行符
        cell_clean = cell.replace("\n", " ")

        # 匹配 O1.1.1 格式
        matches = re.finditer(r"(O\d+\.\d+\.\d+)\s*([^O]*?)(?=O\d+\.\d+\.\d+|$)", cell_clean)
        for match in matches:
            code = match.group(1)
            name = match.group(2).strip()
            if code:
                outputs.append({"代碼": code, "名稱": name})

        return outputs

    def _parse_behaviors(self, cell: str) -> List[Dict[str, str]]:
        """解析行為指標欄位"""
        behaviors = []
        if not cell:
            return behaviors

        # 清理換行符
        cell_clean = cell.replace("\n", " ")

        # 匹配 P1.1.1 格式
        matches = re.finditer(r"(P\d+\.\d+\.\d+)\s*([^P]*?)(?=P\d+\.\d+\.\d+|$)", cell_clean)
        for match in matches:
            code = match.group(1)
            desc = match.group(2).strip()
            if code:
                behaviors.append({"代碼": code, "描述": desc})

        return behaviors

    def _build_knowledge_dict(
        self,
        k_codes: List[str],
        full_text: str
    ) -> Dict[str, str]:
        """
        建立知識代碼對照字典

        Args:
            k_codes: 從表格中提取的知識代碼列表
            full_text: PDF 全文（用於提取知識名稱）

        Returns:
            知識代碼到名稱的映射
        """
        knowledge = {}

        # 從全文中尋找知識名稱（多種模式）
        patterns = [
            # K01職業安全衛生相關規範 格式
            r"(K\d{2,3})([^\nKS\d][^\nKS]*?)(?=\n|K\d{2,3}|S\d{2,3}|$)",
            # K01 職業安全衛生相關規範 格式（有空格）
            r"(K\d{2,3})\s+([^\nKS]+?)(?=\n|K\d{2,3}|S\d{2,3}|$)",
            # K01、職業安全衛生 格式
            r"(K\d{2,3})[、,.\s]+([^\n]+?)(?=K\d{2,3}|S\d{2,3}|$)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                code = match.group(1)
                name = match.group(2).strip()

                # 清理名稱
                name = re.sub(r"[。,，、\s]+$", "", name)
                name = name.split('\n')[0].strip()

                # 移除代碼前綴（如果重複出現）
                name = re.sub(r"^K\d{2,3}\s*", "", name)

                # 移除多餘的空格
                name = re.sub(r'\s+', '', name)

                if code and name and len(name) < 100 and len(name) > 1 and code not in knowledge:
                    knowledge[code] = name

        # 從表格中直接提取（處理 "K01職業安全衛生相關規範" 這種格式）
        table_patterns = [
            r"(K\d{2,3})([^\nKS\d][^\nKS]{2,50})",
        ]

        for pattern in table_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                code = match.group(1)
                name = match.group(2).strip()
                name = re.sub(r'\s+', '', name)

                if code and name and len(name) > 1 and code not in knowledge:
                    knowledge[code] = name

        # 確保所有在表格中出現的代碼都有對應（即使沒找到名稱）
        for code in k_codes:
            if code not in knowledge:
                knowledge[code] = ""

        return knowledge

    def _build_skill_dict(
        self,
        s_codes: List[str],
        full_text: str
    ) -> Dict[str, str]:
        """
        建立技能代碼對照字典

        Args:
            s_codes: 從表格中提取的技能代碼列表
            full_text: PDF 全文

        Returns:
            技能代碼到名稱的映射
        """
        skills = {}

        # 多種模式匹配
        patterns = [
            # S01資料蒐集與分析能力 格式
            r"(S\d{2,3})([^\nKS\d][^\nKS]*?)(?=\n|K\d{2,3}|S\d{2,3}|$)",
            # S01 資料蒐集與分析能力 格式（有空格）
            r"(S\d{2,3})\s+([^\nKS]+?)(?=\n|K\d{2,3}|S\d{2,3}|$)",
            # S01、資料蒐集 格式
            r"(S\d{2,3})[、,.\s]+([^\n]+?)(?=K\d{2,3}|S\d{2,3}|$)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                code = match.group(1)
                name = match.group(2).strip()

                name = re.sub(r"[。,，、\s]+$", "", name)
                name = name.split('\n')[0].strip()
                name = re.sub(r"^S\d{2,3}\s*", "", name)

                # 移除多餘的空格
                name = re.sub(r'\s+', '', name)

                if code and name and len(name) < 100 and len(name) > 1 and code not in skills:
                    skills[code] = name

        # 從表格中直接提取
        table_patterns = [
            r"(S\d{2,3})([^\nKS\d][^\nKS]{2,50})",
        ]

        for pattern in table_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                code = match.group(1)
                name = match.group(2).strip()
                name = re.sub(r'\s+', '', name)

                if code and name and len(name) > 1 and code not in skills:
                    skills[code] = name

        for code in s_codes:
            if code not in skills:
                skills[code] = ""

        return skills

    def _parse_attitudes(self, text: str) -> Dict[str, Dict[str, str]]:
        """
        解析態度清單

        Args:
            text: PDF 全文

        Returns:
            態度代碼到名稱和描述的映射
        """
        attitudes = {}

        patterns = [
            r"(A\d{2,3})[、,.\s]+([^:：\n]+)[：:]\s*([^\n]+)",
            r"(A\d{2,3})\s+([^\n]+)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                code = match.group(1)

                if len(match.groups()) >= 3:
                    name = match.group(2).strip()
                    description = match.group(3).strip()
                else:
                    full_text = match.group(2).strip()
                    if "：" in full_text or ":" in full_text:
                        parts = re.split(r"[：:]", full_text, 1)
                        name = parts[0].strip()
                        description = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        name = full_text
                        description = ""

                if code not in attitudes and name:
                    attitudes[code] = {
                        "名稱": name[:50],
                        "描述": description[:200]
                    }

        return attitudes

    def _parse_supplementary(self, text: str) -> Dict[str, Any]:
        """
        解析補充說明

        Args:
            text: PDF 全文

        Returns:
            補充說明字典
        """
        supplementary = {
            "學歷經驗條件": "",
            "名詞解釋": {}
        }

        # 提取學歷經驗條件
        edu_patterns = [
            r"入門水準[：:\s]*([^\n]+(?:\n[^\n]+)*?)(?=名詞解釋|$)",
            r"學歷[^\n]*經驗[^\n]*條件[：:\s]*([^\n]+(?:\n[^\n]+)*?)(?=名詞解釋|$)",
        ]

        for pattern in edu_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                content = re.sub(r'\n+', '\n', content)
                supplementary["學歷經驗條件"] = content[:500]
                break

        # 提取名詞解釋
        glossary_section = re.search(r"名詞解釋[：:\s]*(.+?)(?=附錄|$)", text, re.DOTALL)
        if glossary_section:
            glossary_text = glossary_section.group(1)
            term_pattern = r"([^：:\n]+)[：:]([^\n]+)"
            terms = re.finditer(term_pattern, glossary_text)
            for term in terms:
                term_name = term.group(1).strip()
                term_def = term.group(2).strip()
                if term_name and len(term_name) < 50:
                    supplementary["名詞解釋"][term_name] = term_def

        return supplementary

    def parse_directory(
        self,
        directory: str | Path,
        output_dir: Optional[str | Path] = None,
        limit: Optional[int] = None
    ) -> List[ParsedCompetencyStandard]:
        """
        批次解析目錄中的所有 PDF

        Args:
            directory: PDF 目錄路徑
            output_dir: JSON 輸出目錄（可選）
            limit: 限制處理數量（測試用）

        Returns:
            解析結果列表
        """
        directory = Path(directory)
        results = []

        pdf_files = list(directory.rglob("*.pdf"))

        if limit:
            pdf_files = pdf_files[:limit]

        logger.info(f"找到 {len(pdf_files)} 個 PDF 檔案")

        for pdf_path in tqdm(pdf_files, desc="解析 PDF"):
            result = self.parse_pdf(pdf_path)
            results.append(result)

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                json_name = pdf_path.stem + ".json"
                json_path = output_dir / json_name

                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(result.to_json())

        success_count = sum(1 for r in results if r.parse_success)
        logger.info(f"解析完成: {success_count}/{len(results)} 成功")

        return results


# ========================================
# 命令列工具
# ========================================

def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="職能基準 PDF 解析器（改進版）")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(config.ICAP_SOURCE_DIR),
        help="輸入 PDF 目錄路徑"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(config.PARSED_JSON_DIR),
        help="輸出 JSON 目錄路徑"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="限制處理數量（測試用）"
    )
    parser.add_argument(
        "--single", "-s",
        type=str,
        default=None,
        help="解析單一 PDF 檔案"
    )

    args = parser.parse_args()

    # 設定日誌
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    pdf_parser = CompetencyPDFParser()

    if args.single:
        result = pdf_parser.parse_pdf(args.single)
        print("\n" + "=" * 60)
        print("解析結果：")
        print("=" * 60)
        print(result.to_json())
    else:
        results = pdf_parser.parse_directory(
            args.input,
            args.output,
            args.limit
        )

        print("\n" + "=" * 60)
        print(f"解析完成！")
        print(f"成功: {sum(1 for r in results if r.parse_success)}")
        print(f"失敗: {sum(1 for r in results if not r.parse_success)}")
        print(f"輸出目錄: {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
