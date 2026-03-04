"""
職能基準 PDF 解析器 v2
使用 pdfplumber 表格提取功能解析職能基準 PDF，輸出結構化 JSON

特點：
- 使用表格提取功能正確處理多欄結構
- 合併跨行內容
- 支援 T/O/P/K/S/A 代碼格式
- 輸出格式相容 competency_db 和 RAG 系統
- 內建 chunks_for_rag 區塊，方便向量化
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from datetime import datetime

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("請安裝 pdfplumber: pip install pdfplumber")


@dataclass
class ParsedCompetencyStandard:
    """解析後的職能基準資料結構（新版格式）"""
    # 元資料
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 基本資訊
    basic_info: Dict[str, Any] = field(default_factory=dict)
    # 扁平化的工作任務列表
    competency_tasks: List[Dict] = field(default_factory=list)
    # 知識清單 (list of objects)
    competency_knowledge: List[Dict] = field(default_factory=list)
    # 技能清單 (list of objects)
    competency_skills: List[Dict] = field(default_factory=list)
    # 態度清單 (list of objects)
    competency_attitudes: List[Dict] = field(default_factory=list)
    # RAG 向量化用的 chunks
    chunks_for_rag: List[Dict] = field(default_factory=list)
    # 解析狀態
    parse_success: bool = False
    parse_errors: List[str] = field(default_factory=list)


class CompetencyPDFParser:
    """職能基準 PDF 解析器"""

    def __init__(self):
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber 未安裝")

    def parse(self, pdf_path: str) -> ParsedCompetencyStandard:
        """解析 PDF 檔案"""
        result = ParsedCompetencyStandard()
        pdf_path = Path(pdf_path)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                all_tables = []

                for page in pdf.pages:
                    text = page.extract_text() or ""
                    all_text += text + "\n"
                    tables = page.extract_tables()
                    all_tables.extend(tables)

                # 1. 解析元資料（版本資訊）
                result.metadata = self._parse_metadata(all_text, pdf_path)

                # 2. 解析基本資訊
                result.basic_info = self._parse_basic_info(all_tables, all_text, pdf_path)

                # 3. 從表格提取主要職責與任務
                resp_data = self._parse_responsibilities_from_tables(all_tables)

                # 4. 扁平化任務結構
                result.competency_tasks = self._flatten_tasks(
                    resp_data['responsibilities'],
                    resp_data['knowledge'],
                    resp_data['skills']
                )

                # 5. 知識清單轉為 list of objects
                result.competency_knowledge = [
                    {"code": code, "name": name, "category": "知識"}
                    for code, name in resp_data['knowledge'].items()
                ]

                # 6. 技能清單轉為 list of objects
                result.competency_skills = [
                    {"code": code, "name": name, "category": "技能"}
                    for code, name in resp_data['skills'].items()
                ]

                # 7. 態度清單
                attitudes = self._extract_attitudes(all_text)
                result.competency_attitudes = [
                    {"code": code, "name": self._extract_attitude_name(desc),
                     "description": self._extract_attitude_desc(desc), "category": "態度"}
                    for code, desc in attitudes.items()
                ]

                # 8. 生成 RAG chunks
                result.chunks_for_rag = self._generate_rag_chunks(result)

                result.parse_success = True

        except Exception as e:
            result.parse_errors.append(f"解析錯誤: {str(e)}")
            result.parse_success = False
            import traceback
            result.parse_errors.append(traceback.format_exc())

        return result

    def _parse_metadata(self, text: str, pdf_path: Path) -> Dict[str, Any]:
        """解析元資料（版本資訊）"""
        metadata = {
            "code": "",
            "name": "",
            "update_date": "",
            "version": "",
            "status": "最新版本",
            "source_file": str(pdf_path)
        }

        # 提取版本號（支援所有職能基準代碼格式，不限 TF 前綴，如 HBR2431-001v4）
        version_match = re.search(r'(V\d+)\s+([A-Z]{2,}\d+-\d+v\d+)', text)
        if version_match:
            metadata["version"] = version_match.group(1)
            metadata["code"] = version_match.group(2)

        # 只有代碼
        if not metadata["code"]:
            code_match = re.search(r'([A-Z]{2,}\d+-\d+v\d+)', text)
            if code_match:
                metadata["code"] = code_match.group(1)
                # 從代碼提取版本
                v_match = re.search(r'v(\d+)$', metadata["code"])
                if v_match:
                    metadata["version"] = f"V{v_match.group(1)}"

        # 提取日期
        date_match = re.search(r'(\d{4}[/年]\d{1,2}[/月]\d{1,2})', text)
        if date_match:
            metadata["update_date"] = date_match.group(1).replace('年', '/').replace('月', '/')

        return metadata

    def _parse_basic_info(self, tables: List, text: str, pdf_path: Path) -> Dict[str, Any]:
        """解析基本資訊"""
        info = {
            "category": "",
            "category_code": "",
            "occupation": "",
            "occupation_code": "",
            "industry": [],
            "industry_code": [],
            "job_description": "",
            "level": 0,
            "requirements": ""
        }

        # 遍歷所有表格尋找基本資訊
        for table in tables:
            for row in table:
                if not row:
                    continue
                row_text = ' '.join([str(c) for c in row if c])

                # 職類別
                if '職類別' in row_text and '職類別代碼' in row_text:
                    name = ""
                    code = ""
                    for i, cell in enumerate(row):
                        if cell == '職類別' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip() and row[j] != '職類別代碼':
                                    name = self._clean_cell(row[j])
                                    break
                        if cell == '職類別代碼' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip():
                                    code = self._clean_cell(row[j])
                                    break
                    if name:
                        info["category"] = name
                        info["category_code"] = code

                # 職業別
                if '職業別' in row_text and '職業別代碼' in row_text:
                    name = ""
                    code = ""
                    for i, cell in enumerate(row):
                        if cell == '職業別' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip() and row[j] != '職業別代碼':
                                    name = self._clean_cell(row[j])
                                    break
                        if cell == '職業別代碼' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip():
                                    code = self._clean_cell(row[j])
                                    break
                    if name:
                        info["occupation"] = name
                        info["occupation_code"] = code

                # 行業別（支援多個：一格多行 或 多行表格列）
                if '行業別' in row_text and '行業別代碼' in row_text:
                    names = []
                    codes = []
                    for i, cell in enumerate(row):
                        if cell == '行業別' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip() and row[j] != '行業別代碼':
                                    names = self._split_cell_values(row[j])
                                    break
                        if cell == '行業別代碼' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip():
                                    # 先以換行符分割，再對每段用正則進一步拆分串聯代碼
                                    raw_parts = self._split_cell_values(row[j])
                                    codes = []
                                    for part in raw_parts:
                                        sub = re.findall(r'[A-Z]\d+', part)
                                        codes.extend(sub if len(sub) > 1 else [part])
                                    break
                    if names:
                        info["industry"].extend(names)
                        info["industry_code"].extend(codes)

                # 工作描述
                if '工作描述' in row_text:
                    for i, cell in enumerate(row):
                        if cell == '工作描述' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip():
                                    info["job_description"] = self._clean_cell(row[j])
                                    break
                            break

                # 基準級別
                if '基準級別' in row_text:
                    for i, cell in enumerate(row):
                        if cell == '基準級別' and i + 1 < len(row):
                            for j in range(i + 1, len(row)):
                                if row[j] and str(row[j]).strip():
                                    try:
                                        info["level"] = int(str(row[j]).strip())
                                    except ValueError:
                                        pass
                                    break
                            break

        # 學歷經驗條件
        edu_patterns = [
            r'學歷[/／]經驗[/／]或能力條件[：:]\s*(.+?)(?=其他補充|⚫|$)',
            r'建議擔任此職類[/／]職業之學歷[/／]經驗[/／]或能力條件[：:]\s*(.+?)(?=其他補充|⚫|$)',
        ]
        for pattern in edu_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                content = re.sub(r'\s+', ' ', content)
                info["requirements"] = content
                break

        return info

    def _clean_cell(self, cell: Any) -> str:
        """清理儲存格內容"""
        if cell is None:
            return ""
        text = str(cell).strip()
        text = text.replace('\n', '')
        # 移除 PDF 擷取時在中文字符之間插入的多餘空格
        text = re.sub(r'(?<=[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef])', '', text)
        return text

    def _split_cell_values(self, cell: Any) -> List[str]:
        """將可能包含多行的儲存格拆分為多個獨立值（支援換行符分隔的多值）"""
        if cell is None:
            return []
        parts = str(cell).split('\n')
        result = []
        for part in parts:
            cleaned = part.strip()
            # 移除 CJK 字符之間的多餘空格
            cleaned = re.sub(
                r'(?<=[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef])\s+'
                r'(?=[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef])',
                '', cleaned
            )
            if cleaned:
                result.append(cleaned)
        return result

    def _parse_responsibilities_from_tables(self, tables: List) -> Dict:
        """從表格提取主要職責"""
        responsibilities = []
        all_knowledge = OrderedDict()
        all_skills = OrderedDict()

        current_resp = None
        current_task = None

        for table in tables:
            if not table or len(table) < 2:
                continue

            # 檢查前三列是否含任務表格關鍵字（部分PDF的主要職責在第二列）
            header_text = ' '.join([
                ' '.join([str(c) for c in row if c])
                for row in table[:3] if row
            ])
            if '主要職責' not in header_text and '工作任務' not in header_text:
                continue

            # 找到真正含有「主要職責」的列，從其下一列開始讀取資料
            data_start = 2
            for ri, row in enumerate(table[:4]):
                if row and any('主要職責' in str(c) or '工作任務' in str(c) for c in row if c):
                    data_start = ri + 1
                    break

            for row in table[data_start:]:
                if not row or len(row) < 4:
                    continue

                col_resp = self._clean_cell(row[0]) if len(row) > 0 else ""
                col_task = self._clean_cell(row[1]) if len(row) > 1 else ""
                col_output = self._clean_cell(row[2]) if len(row) > 2 else ""
                col_indicator = self._clean_cell(row[3]) if len(row) > 3 else ""
                col_level = self._clean_cell(row[4]) if len(row) > 4 else ""

                col_knowledge = ""
                col_skills = ""
                for i, cell in enumerate(row):
                    cell_text = self._clean_cell(cell)
                    if re.search(r'K\d{2}', cell_text):
                        col_knowledge = cell_text
                    if re.search(r'S\d{2}', cell_text):
                        col_skills = cell_text

                # 解析主要職責
                if col_resp and re.match(r'T\d+', col_resp):
                    t_match = re.match(r'(T\d+)(.+)', col_resp)
                    if t_match:
                        if current_resp and current_resp.get("工作任務"):
                            responsibilities.append(current_resp)

                        current_resp = {
                            "代碼": t_match.group(1),
                            "名稱": t_match.group(2).strip(),
                            "工作任務": []
                        }
                        current_task = None

                # 解析工作任務
                if col_task and re.match(r'T\d+\.\d+', col_task):
                    task_match = re.match(r'(T\d+\.\d+)(.+)', col_task)
                    if task_match:
                        if current_task and current_resp:
                            current_resp["工作任務"].append(current_task)

                        level = 0
                        if col_level and col_level.isdigit():
                            level = int(col_level)

                        current_task = {
                            "代碼": task_match.group(1),
                            "名稱": task_match.group(2).strip(),
                            "職能級別": level,
                            "工作產出": [],
                            "行為指標": [],
                            "知識": [],
                            "技能": []
                        }

                # 解析工作產出
                if col_output and current_task:
                    for o_match in re.finditer(r'(O\d+\.\d+\.\d+)([^\nO]*)', col_output):
                        current_task["工作產出"].append({
                            "代碼": o_match.group(1),
                            "名稱": o_match.group(2).strip()
                        })

                # 解析行為指標
                if col_indicator and current_task:
                    for p_match in re.finditer(r'(P\d+\.\d+\.\d+)([^P]*?)(?=P\d+\.\d+\.\d+|$)', col_indicator):
                        desc = p_match.group(2).strip()
                        desc = re.sub(r'\s+', ' ', desc)
                        if desc:
                            current_task["行為指標"].append({
                                "代碼": p_match.group(1),
                                "描述": desc
                            })

                # 解析知識
                if col_knowledge:
                    for k_match in re.finditer(r'(K\d{2})([^\nK]*)', col_knowledge):
                        code = k_match.group(1)
                        name = k_match.group(2).strip()
                        if name and code not in all_knowledge:
                            all_knowledge[code] = name
                        if current_task and code not in current_task["知識"]:
                            current_task["知識"].append(code)

                # 解析技能
                if col_skills:
                    for s_match in re.finditer(r'(S\d{2})([^\nS]*)', col_skills):
                        code = s_match.group(1)
                        name = s_match.group(2).strip()
                        if name.endswith('能') or name.endswith('選用'):
                            name += '力'
                        if name and code not in all_skills:
                            all_skills[code] = name
                        if current_task and code not in current_task["技能"]:
                            current_task["技能"].append(code)

                if col_level and col_level.isdigit() and current_task:
                    if current_task["職能級別"] == 0:
                        current_task["職能級別"] = int(col_level)

        if current_task and current_resp:
            current_resp["工作任務"].append(current_task)
        if current_resp and current_resp.get("工作任務"):
            responsibilities.append(current_resp)

        return {
            'responsibilities': responsibilities,
            'knowledge': dict(all_knowledge),
            'skills': dict(all_skills)
        }

    def _flatten_tasks(self, responsibilities: List[Dict],
                       knowledge_dict: Dict[str, str],
                       skills_dict: Dict[str, str]) -> List[Dict]:
        """扁平化任務結構"""
        tasks = []

        for resp in responsibilities:
            resp_code = resp.get("代碼", "")
            resp_name = resp.get("名稱", "")

            for task in resp.get("工作任務", []):
                # 組合行為描述
                behaviors = [ind.get("描述", "") for ind in task.get("行為指標", [])]

                # 工作產出名稱
                outputs = [out.get("名稱", "") for out in task.get("工作產出", [])]
                output_str = "、".join(outputs) if outputs else None

                flat_task = {
                    "main_responsibility": f"{resp_code}{resp_name}",
                    "task_id": task.get("代碼", ""),
                    "task_name": task.get("名稱", ""),
                    "output": output_str,
                    "behaviors": behaviors,
                    "knowledge": task.get("知識", []),
                    "skills": task.get("技能", []),
                    "level": task.get("職能級別", 0)
                }
                tasks.append(flat_task)

        return tasks

    def _extract_attitudes(self, text: str) -> Dict[str, str]:
        """提取態度清單"""
        attitudes = OrderedDict()
        a_pattern = r'(A\d{2})([^：:\n]+)[：:]([^\nA]+)'
        matches = re.findall(a_pattern, text)

        for code, title, desc in matches:
            title = title.strip()
            desc = desc.strip()
            attitudes[code] = f"{title}：{desc}"

        return dict(attitudes)

    def _extract_attitude_name(self, full_desc: str) -> str:
        """從態度完整描述中提取名稱"""
        if '：' in full_desc:
            return full_desc.split('：')[0].strip()
        return full_desc[:10] if len(full_desc) > 10 else full_desc

    def _extract_attitude_desc(self, full_desc: str) -> str:
        """從態度完整描述中提取描述"""
        if '：' in full_desc:
            return full_desc.split('：', 1)[1].strip()
        return full_desc

    def _generate_rag_chunks(self, result: ParsedCompetencyStandard) -> List[Dict]:
        """生成 RAG 向量化用的 chunks"""
        chunks = []
        code = result.metadata.get("code", "")
        name = result.metadata.get("name", "") or self._extract_name_from_tasks(result)

        # 1. 基本資訊 chunk
        basic = result.basic_info
        _ind = basic.get('industry', [])
        _ind_code = basic.get('industry_code', [])
        ind_display = '; '.join(_ind) if isinstance(_ind, list) else (_ind or '')
        ind_code_display = '; '.join(_ind_code) if isinstance(_ind_code, list) else (_ind_code or '')
        basic_content = f"""職能基準：{name}
代碼：{code}
版本：{result.metadata.get('version', '')}
職類別：{basic.get('category', '')}（代碼：{basic.get('category_code', '')}）
職業別：{basic.get('occupation', '')}（代碼：{basic.get('occupation_code', '')}）
行業別：{ind_display}（代碼：{ind_code_display}）
基準級別：{basic.get('level', '')}
工作描述：{basic.get('job_description', '')}
建議條件：{basic.get('requirements', '')}
更新日期：{result.metadata.get('update_date', '')}"""

        chunks.append({
            "id": "basic_info",
            "type": "基本資訊",
            "content": basic_content,
            "metadata": {
                "code": code,
                "name": name,
                "level": basic.get('level', 0),
                "chunk_type": "overview"
            }
        })

        # 2. 建立 K/S 名稱對照表
        k_names = {k["code"]: k["name"] for k in result.competency_knowledge}
        s_names = {s["code"]: s["name"] for s in result.competency_skills}

        # 3. 每個任務一個 chunk
        for task in result.competency_tasks:
            task_id = task.get("task_id", "")
            task_name = task.get("task_name", "")

            # 展開知識和技能名稱
            k_list = [k_names.get(k, k) for k in task.get("knowledge", [])]
            s_list = [s_names.get(s, s) for s in task.get("skills", [])]

            behaviors_text = "\n".join([f"- {b}" for b in task.get("behaviors", [])]) or "無"

            task_content = f"""主要職責：{task.get('main_responsibility', '')}
任務編號：{task_id}
任務名稱：{task_name}
職能級別：{task.get('level', '')}
工作產出：{task.get('output') or '無特定產出'}

行為指標：
{behaviors_text}

所需知識（{len(k_list)}項）：
{chr(10).join(['- ' + k for k in k_list]) if k_list else '無'}

所需技能（{len(s_list)}項）：
{chr(10).join(['- ' + s for s in s_list]) if s_list else '無'}"""

            chunks.append({
                "id": f"task_{task_id}",
                "type": "工作任務",
                "content": task_content,
                "metadata": {
                    "task_id": task_id,
                    "task_name": task_name,
                    "main_responsibility": task.get("main_responsibility", ""),
                    "code": code,
                    "level": task.get("level", 0),
                    "chunk_type": "task"
                }
            })

        # 4. 按職責分組的知識技能 chunks
        resp_groups = {}
        for task in result.competency_tasks:
            resp = task.get("main_responsibility", "")
            if resp not in resp_groups:
                resp_groups[resp] = {"knowledge": set(), "skills": set()}
            resp_groups[resp]["knowledge"].update(task.get("knowledge", []))
            resp_groups[resp]["skills"].update(task.get("skills", []))

        for i, (resp, items) in enumerate(resp_groups.items(), 1):
            k_list = sorted(items["knowledge"])
            s_list = sorted(items["skills"])

            ks_content = f"""職責領域：{resp}

需具備知識（{len(k_list)}項）：
{chr(10).join([f'- {k}: {k_names.get(k, "")}' for k in k_list]) if k_list else '無'}

需具備技能（{len(s_list)}項）：
{chr(10).join([f'- {s}: {s_names.get(s, "")}' for s in s_list]) if s_list else '無'}"""

            chunks.append({
                "id": f"knowledge_skills_{i}",
                "type": "知識技能",
                "content": ks_content,
                "metadata": {
                    "main_responsibility": resp,
                    "code": code,
                    "chunk_type": "knowledge_skills"
                }
            })

        # 5. 態度 chunk
        if result.competency_attitudes:
            attitudes_text = "\n\n".join([
                f"{a['code']} - {a['name']}：{a['description']}"
                for a in result.competency_attitudes
            ])

            chunks.append({
                "id": "attitudes",
                "type": "職能態度",
                "content": f"職能態度要求（{len(result.competency_attitudes)}項）：\n\n{attitudes_text}",
                "metadata": {
                    "code": code,
                    "count": len(result.competency_attitudes),
                    "chunk_type": "attitudes"
                }
            })

        # 6. 完整摘要 chunk
        summary_content = f"""【{name}職能基準摘要】

一、基本資訊
- 職能代碼：{code}
- 職類別：{basic.get('category', '')}
- 職業別：{basic.get('occupation', '')}
- 基準級別：{basic.get('level', '')}

二、工作描述
{basic.get('job_description', '')}

三、主要職責與任務（{len(result.competency_tasks)}項）
{chr(10).join([f'- {t["task_id"]} {t["task_name"]}' for t in result.competency_tasks])}

四、職能要求
- 知識項目：{len(result.competency_knowledge)}項
- 技能項目：{len(result.competency_skills)}項
- 態度項目：{len(result.competency_attitudes)}項

五、建議條件
{basic.get('requirements', '無')}"""

        chunks.append({
            "id": "summary",
            "type": "完整摘要",
            "content": summary_content,
            "metadata": {
                "code": code,
                "name": name,
                "chunk_type": "summary"
            }
        })

        return chunks

    def _extract_name_from_tasks(self, result: ParsedCompetencyStandard) -> str:
        """從任務中提取職業名稱"""
        if result.competency_tasks:
            resp = result.competency_tasks[0].get("main_responsibility", "")
            # 從 "T1協助研發..." 提取
            match = re.match(r'T\d+(.+)', resp)
            if match:
                return match.group(1)[:10]
        return ""

    def to_json(self, result: ParsedCompetencyStandard, indent: int = 2) -> str:
        """轉換為 JSON 字串"""
        return json.dumps(asdict(result), ensure_ascii=False, indent=indent)

    def save_json(self, result: ParsedCompetencyStandard, output_path: str):
        """儲存為 JSON 檔案"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json(result))

    # ============ 相容舊版格式的方法 ============

    def to_legacy_format(self, result: ParsedCompetencyStandard) -> Dict:
        """轉換為舊版格式（相容 competency_db）"""
        # 重建巢狀的主要職責結構
        responsibilities = []
        resp_map = {}

        for task in result.competency_tasks:
            resp_name = task.get("main_responsibility", "")
            if resp_name not in resp_map:
                # 從 "T1協助..." 解析代碼和名稱
                match = re.match(r'(T\d+)(.+)', resp_name)
                if match:
                    resp_map[resp_name] = {
                        "代碼": match.group(1),
                        "名稱": match.group(2).strip(),
                        "工作任務": []
                    }
                else:
                    resp_map[resp_name] = {
                        "代碼": "",
                        "名稱": resp_name,
                        "工作任務": []
                    }

            # 重建任務結構
            task_obj = {
                "代碼": task.get("task_id", ""),
                "名稱": task.get("task_name", ""),
                "職能級別": task.get("level", 0),
                "工作產出": [],
                "行為指標": [],
                "知識": task.get("knowledge", []),
                "技能": task.get("skills", [])
            }

            # 工作產出
            if task.get("output"):
                for i, out in enumerate(task["output"].split("、"), 1):
                    task_obj["工作產出"].append({
                        "代碼": f"O{task.get('task_id', '')[1:]}.{i}",
                        "名稱": out.strip()
                    })

            # 行為指標
            for i, behavior in enumerate(task.get("behaviors", []), 1):
                task_obj["行為指標"].append({
                    "代碼": f"P{task.get('task_id', '')[1:]}.{i}",
                    "描述": behavior
                })

            resp_map[resp_name]["工作任務"].append(task_obj)

        responsibilities = list(resp_map.values())

        # 建立知識和技能字典
        knowledge_dict = {k["code"]: k["name"] for k in result.competency_knowledge}
        skills_dict = {s["code"]: s["name"] for s in result.competency_skills}
        attitudes_dict = {a["code"]: f"{a['name']}：{a['description']}" for a in result.competency_attitudes}

        return {
            "職能基準": {
                "代碼": result.metadata.get("code", ""),
                "名稱": result.metadata.get("name", ""),
                "職類別": [{"代碼": result.basic_info.get("category_code", ""),
                          "名稱": result.basic_info.get("category", "")}] if result.basic_info.get("category") else [],
                "職業別": [{"代碼": result.basic_info.get("occupation_code", ""),
                          "名稱": result.basic_info.get("occupation", "")}] if result.basic_info.get("occupation") else [],
                "行業別": [{"代碼": c, "名稱": n} for c, n in zip(
                    result.basic_info.get("industry_code", []),
                    result.basic_info.get("industry", [])
                )] if result.basic_info.get("industry") else [],
                "工作描述": result.basic_info.get("job_description", ""),
                "基準級別": result.basic_info.get("level", 0)
            },
            "主要職責": responsibilities,
            "知識清單": knowledge_dict,
            "技能清單": skills_dict,
            "態度清單": attitudes_dict,
            "補充說明": {
                "學歷經驗條件": result.basic_info.get("requirements", ""),
                "名詞解釋": {}
            },
            "source_file": result.metadata.get("source_file", ""),
            "parse_success": result.parse_success,
            "parse_errors": result.parse_errors
        }


def parse_pdf_to_json(pdf_path: str, output_path: str = None, legacy_format: bool = False) -> Dict:
    """
    便利函數：解析 PDF 並輸出 JSON

    Args:
        pdf_path: PDF 檔案路徑
        output_path: 輸出 JSON 路徑（可選）
        legacy_format: 是否使用舊版格式（相容 competency_db）

    Returns:
        解析結果字典
    """
    parser = CompetencyPDFParser()
    result = parser.parse(pdf_path)

    # 從 PDF 檔名提取職業名稱
    pdf_name = Path(pdf_path).stem
    if '-職能基準' in pdf_name:
        result.metadata["name"] = pdf_name.replace('-職能基準', '')

    if legacy_format:
        output_data = parser.to_legacy_format(result)
    else:
        output_data = asdict(result)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"已儲存至: {output_path}")

    return output_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdf_parser_v2.py <pdf_path> [output_json_path] [--legacy]")
        print("  --legacy: 使用舊版格式（相容 competency_db）")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = None
    legacy_format = False

    for arg in sys.argv[2:]:
        if arg == '--legacy':
            legacy_format = True
        else:
            output_path = arg

    result = parse_pdf_to_json(pdf_path, output_path, legacy_format)

    if not output_path:
        print(json.dumps(result, ensure_ascii=False, indent=2))
