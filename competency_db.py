"""
職能基準 SQLite 資料庫模組 (修正版 v4)
提供結構化存儲和高效查詢職能基準資料

支援格式：
- 新版 JSON 格式（pdf_parser_v2 產生，含 metadata, basic_info, competency_tasks 等）
- 舊版 JSON 格式（向後相容，含 職能基準, 主要職責 等）
- 直接從 PDF 匯入（使用 pdf_parser_v2）
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

from loguru import logger
from tqdm import tqdm

from config import get_config

# 嘗試匯入 PDF parser
try:
    from pdf_parser_v2 import CompetencyPDFParser, ParsedCompetencyStandard
    PDF_PARSER_AVAILABLE = True
except ImportError:
    PDF_PARSER_AVAILABLE = False
    CompetencyPDFParser = None
    ParsedCompetencyStandard = None

config = get_config()


@dataclass
class Standard:
    """職能基準資料結構"""
    id: int = 0
    standard_code: str = ""
    standard_name: str = ""
    category_code: str = ""
    category_name: str = ""
    occupation_code: str = ""
    occupation_name: str = ""
    industry_code: str = ""
    industry_name: str = ""
    job_description: str = ""
    level: int = 0
    qualification_requirements: str = ""
    source_file: str = ""
    source_url: str = ""


class CompetencyDatabase:
    """職能基準 SQLite 資料庫"""

    def __init__(self, db_path: Path = None):
        """
        初始化資料庫

        Args:
            db_path: 資料庫檔案路徑，預設為 config 中的設定
        """
        if db_path is None:
            self.db_path = config.DATA_DIR / "competency.db"
        elif isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """取得資料庫連線（context manager）"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """初始化資料庫 schema（只在表格不存在時執行）"""
        # 檢查 standard 表是否已存在
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='standard'
            """)
            if cursor.fetchone():
                # 表格已存在，無需重新初始化
                return

        # 表格不存在，執行完整 schema
        schema_path = Path(__file__).parent / "data" / "competency_schema.sql"

        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            with self._get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()
            logger.info(f"資料庫初始化完成: {self.db_path}")
        else:
            logger.warning(f"Schema 檔案不存在: {schema_path}，使用內建 schema")
            self._create_tables_inline()

    def _create_tables_inline(self):
        """內建的表格建立（備用）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # standard 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS standard (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    standard_code TEXT NOT NULL UNIQUE,
                    standard_name TEXT NOT NULL,
                    category_code TEXT,
                    category_name TEXT,
                    occupation_code TEXT,
                    occupation_name TEXT,
                    industry_code TEXT,
                    industry_name TEXT,
                    category_list TEXT,
                    occupation_list TEXT,
                    industry_list TEXT,
                    job_description TEXT,
                    level INTEGER,
                    dev_update_date TEXT,
                    qualification_requirements TEXT,
                    glossary TEXT,
                    other_notes TEXT,
                    source_file TEXT,
                    source_url TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)

            # responsibility 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS responsibility (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    standard_id INTEGER NOT NULL,
                    t_code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    source_page INTEGER,
                    source_ref TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (standard_id) REFERENCES standard(id) ON DELETE CASCADE,
                    UNIQUE (standard_id, t_code)
                )
            """)

            # task 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    responsibility_id INTEGER NOT NULL,
                    task_code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    competency_level INTEGER,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    source_page INTEGER,
                    source_ref TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (responsibility_id) REFERENCES responsibility(id) ON DELETE CASCADE,
                    UNIQUE (responsibility_id, task_code)
                )
            """)

            # output 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS output (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    o_code TEXT,
                    name TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    source_page INTEGER,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
                    UNIQUE (task_id, o_code)
                )
            """)

            # indicator 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    p_code TEXT NOT NULL,
                    description TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    source_page INTEGER,
                    source_ref TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
                    UNIQUE (task_id, p_code)
                )
            """)

            # competency_item 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS competency_item (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    standard_id INTEGER NOT NULL,
                    type TEXT NOT NULL CHECK (type IN ('K', 'S', 'A')),
                    code TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    source_page INTEGER,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (standard_id) REFERENCES standard(id) ON DELETE CASCADE,
                    UNIQUE (standard_id, type, code)
                )
            """)

            # task_competency_map 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_competency_map (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    competency_item_id INTEGER NOT NULL,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
                    FOREIGN KEY (competency_item_id) REFERENCES competency_item(id) ON DELETE CASCADE,
                    UNIQUE (task_id, competency_item_id)
                )
            """)

            # 建立索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_standard_category_code ON standard(category_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_standard_occupation_code ON standard(occupation_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_standard_industry_code ON standard(industry_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_standard_name ON standard(standard_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_responsibility_standard ON responsibility(standard_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_responsibility ON task(responsibility_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_task ON indicator(task_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_competency_item_standard_type ON competency_item(standard_id, type)")

            # FTS5 表（使用 trigram，若失敗則用 unicode61）
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS indicator_fts USING fts5(
                        description,
                        content='indicator',
                        content_rowid='id',
                        tokenize='trigram'
                    )
                """)
            except sqlite3.OperationalError:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS indicator_fts USING fts5(
                        description,
                        content='indicator',
                        content_rowid='id',
                        tokenize='unicode61'
                    )
                """)

            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS standard_fts USING fts5(
                        standard_name,
                        job_description,
                        qualification_requirements,
                        content='standard',
                        content_rowid='id',
                        tokenize='trigram'
                    )
                """)
            except sqlite3.OperationalError:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS standard_fts USING fts5(
                        standard_name,
                        job_description,
                        qualification_requirements,
                        content='standard',
                        content_rowid='id',
                        tokenize='unicode61'
                    )
                """)

            conn.commit()
            logger.info(f"資料庫初始化完成（內建 schema）: {self.db_path}")

    def _detect_json_format(self, data: Dict) -> str:
        """
        偵測 JSON 格式版本

        Args:
            data: JSON 資料

        Returns:
            'new' 或 'legacy'
        """
        # 新版格式有 metadata 和 basic_info
        if 'metadata' in data and 'basic_info' in data:
            return 'new'
        # 舊版格式有 職能基準
        if '職能基準' in data:
            return 'legacy'
        return 'unknown'

    def _convert_new_to_legacy(self, data: Dict) -> Dict:
        """
        將新版 JSON 格式轉換為舊版格式以便匯入

        Args:
            data: 新版 JSON 資料

        Returns:
            舊版格式 JSON
        """
        metadata = data.get('metadata', {})
        basic_info = data.get('basic_info', {})

        # 重建主要職責結構
        responsibilities = []
        resp_map = {}

        for task in data.get('competency_tasks', []):
            resp_name = task.get('main_responsibility', '')
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
                "代碼": task.get('task_id', ''),
                "名稱": task.get('task_name', ''),
                "職能級別": task.get('level', 0),
                "工作產出": [],
                "行為指標": [],
                "知識": task.get('knowledge', []),
                "技能": task.get('skills', [])
            }

            # 工作產出
            output_str = task.get('output')
            if output_str:
                for i, out in enumerate(output_str.split('、'), 1):
                    task_id = task.get('task_id', '')[1:] if task.get('task_id', '').startswith('T') else ''
                    task_obj["工作產出"].append({
                        "代碼": f"O{task_id}.{i}",
                        "名稱": out.strip()
                    })

            # 行為指標
            for i, behavior in enumerate(task.get('behaviors', []), 1):
                task_id = task.get('task_id', '')[1:] if task.get('task_id', '').startswith('T') else ''
                task_obj["行為指標"].append({
                    "代碼": f"P{task_id}.{i}",
                    "描述": behavior
                })

            resp_map[resp_name]["工作任務"].append(task_obj)

        responsibilities = list(resp_map.values())

        # 建立知識、技能、態度字典
        knowledge_dict = {k['code']: k['name'] for k in data.get('competency_knowledge', [])}
        skills_dict = {s['code']: s['name'] for s in data.get('competency_skills', [])}
        attitudes_dict = {}
        for a in data.get('competency_attitudes', []):
            name = a.get('name', '')
            desc = a.get('description', '')
            attitudes_dict[a['code']] = f"{name}：{desc}" if desc else name

        return {
            "職能基準": {
                "代碼": metadata.get('code', ''),
                "名稱": metadata.get('name', ''),
                "職類別": [{"代碼": basic_info.get('category_code', ''),
                          "名稱": basic_info.get('category', '')}] if basic_info.get('category') else [],
                "職業別": [{"代碼": basic_info.get('occupation_code', ''),
                          "名稱": basic_info.get('occupation', '')}] if basic_info.get('occupation') else [],
                "行業別": [{"代碼": basic_info.get('industry_code', ''),
                          "名稱": basic_info.get('industry', '')}] if basic_info.get('industry') else [],
                "工作描述": basic_info.get('job_description', ''),
                "基準級別": basic_info.get('level', 0)
            },
            "主要職責": responsibilities,
            "知識清單": knowledge_dict,
            "技能清單": skills_dict,
            "態度清單": attitudes_dict,
            "補充說明": {
                "學歷經驗條件": basic_info.get('requirements', ''),
                "名詞解釋": {}
            },
            "source_file": metadata.get('source_file', ''),
            "parse_success": data.get('parse_success', True),
            "parse_errors": data.get('parse_errors', [])
        }

    def import_from_parsed_json(self, json_dir: Path = None) -> int:
        """
        從解析後的 JSON 目錄匯入資料（自動偵測格式）

        Args:
            json_dir: JSON 檔案目錄路徑

        Returns:
            匯入的職能基準數量
        """
        if json_dir is None:
            json_dir = config.PARSED_JSON_DIR
        elif isinstance(json_dir, str):
            json_dir = Path(json_dir)

        if not json_dir.exists():
            logger.error(f"JSON 目錄不存在: {json_dir}")
            return 0

        json_files = list(json_dir.glob("*.json"))
        logger.info(f"找到 {len(json_files)} 個 JSON 檔案")

        imported_count = 0
        new_format_count = 0
        legacy_format_count = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()

            for json_path in tqdm(json_files, desc="匯入職能基準"):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if not data.get("parse_success", False):
                        continue

                    # 偵測並轉換格式
                    json_format = self._detect_json_format(data)
                    if json_format == 'new':
                        data = self._convert_new_to_legacy(data)
                        new_format_count += 1
                    elif json_format == 'legacy':
                        legacy_format_count += 1
                    else:
                        logger.warning(f"無法識別的 JSON 格式: {json_path}")
                        continue

                    standard_info = data.get("職能基準", {})
                    standard_code = standard_info.get("代碼", "")
                    if not standard_code:
                        continue

                    # 提取基本資訊
                    standard_name = standard_info.get("名稱", "")
                    job_description = standard_info.get("工作描述", "")
                    level = standard_info.get("基準級別", 0)

                    # 提取分類資訊（支援多值）
                    category_list = standard_info.get("職類別", [])
                    occupation_list = standard_info.get("職業別", [])
                    industry_list = standard_info.get("行業別", [])

                    # 取第一個作為主要值
                    category_code = category_list[0].get("代碼", "") if category_list else ""
                    category_name = category_list[0].get("名稱", "") if category_list else ""
                    occupation_code = occupation_list[0].get("代碼", "") if occupation_list else ""
                    occupation_name = occupation_list[0].get("名稱", "") if occupation_list else ""
                    industry_code = industry_list[0].get("代碼", "") if industry_list else ""
                    industry_name = industry_list[0].get("名稱", "") if industry_list else ""

                    # 提取補充說明
                    supplement = data.get("補充說明", {})
                    qualification = supplement.get("學歷經驗條件", "")
                    glossary = json.dumps(supplement.get("名詞解釋", {}), ensure_ascii=False) if supplement.get("名詞解釋") else ""

                    source_file = data.get("source_file", "")

                    # 插入或更新 standard
                    cursor.execute("""
                        INSERT INTO standard (
                            standard_code, standard_name,
                            category_code, category_name,
                            occupation_code, occupation_name,
                            industry_code, industry_name,
                            category_list, occupation_list, industry_list,
                            job_description, level,
                            qualification_requirements, glossary,
                            source_file
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(standard_code) DO UPDATE SET
                            standard_name = excluded.standard_name,
                            category_code = excluded.category_code,
                            category_name = excluded.category_name,
                            occupation_code = excluded.occupation_code,
                            occupation_name = excluded.occupation_name,
                            industry_code = excluded.industry_code,
                            industry_name = excluded.industry_name,
                            category_list = excluded.category_list,
                            occupation_list = excluded.occupation_list,
                            industry_list = excluded.industry_list,
                            job_description = excluded.job_description,
                            level = excluded.level,
                            qualification_requirements = excluded.qualification_requirements,
                            glossary = excluded.glossary,
                            source_file = excluded.source_file,
                            updated_at = datetime('now', 'localtime')
                    """, (
                        standard_code, standard_name,
                        category_code, category_name,
                        occupation_code, occupation_name,
                        industry_code, industry_name,
                        json.dumps(category_list, ensure_ascii=False) if category_list else None,
                        json.dumps(occupation_list, ensure_ascii=False) if occupation_list else None,
                        json.dumps(industry_list, ensure_ascii=False) if industry_list else None,
                        job_description, level,
                        qualification, glossary,
                        source_file
                    ))

                    # 取得 standard_id
                    cursor.execute("SELECT id FROM standard WHERE standard_code = ?", (standard_code,))
                    standard_id = cursor.fetchone()[0]

                    # 清除舊的關聯資料（級聯刪除會處理子表）
                    cursor.execute("DELETE FROM responsibility WHERE standard_id = ?", (standard_id,))
                    cursor.execute("DELETE FROM competency_item WHERE standard_id = ?", (standard_id,))

                    # 匯入主要職責
                    for r_idx, duty in enumerate(data.get("主要職責", [])):
                        t_code = duty.get("代碼", "")
                        if not t_code:
                            continue

                        cursor.execute("""
                            INSERT INTO responsibility (standard_id, t_code, name, sort_order)
                            VALUES (?, ?, ?, ?)
                        """, (standard_id, t_code, duty.get("名稱", ""), r_idx))

                        responsibility_id = cursor.lastrowid

                        # 匯入工作任務
                        for t_idx, task in enumerate(duty.get("工作任務", [])):
                            task_code = task.get("代碼", "")
                            if not task_code:
                                continue

                            cursor.execute("""
                                INSERT INTO task (responsibility_id, task_code, name, competency_level, sort_order)
                                VALUES (?, ?, ?, ?, ?)
                            """, (responsibility_id, task_code, task.get("名稱", ""),
                                  task.get("職能級別"), t_idx))

                            task_id = cursor.lastrowid

                            # 匯入工作產出
                            for o_idx, output in enumerate(task.get("工作產出", [])):
                                cursor.execute("""
                                    INSERT INTO output (task_id, o_code, name, sort_order)
                                    VALUES (?, ?, ?, ?)
                                """, (task_id, output.get("代碼"), output.get("名稱", ""), o_idx))

                            # 匯入行為指標
                            for i_idx, indicator in enumerate(task.get("行為指標", [])):
                                p_code = indicator.get("代碼", "")
                                if not p_code:
                                    continue

                                cursor.execute("""
                                    INSERT INTO indicator (task_id, p_code, description, sort_order)
                                    VALUES (?, ?, ?, ?)
                                """, (task_id, p_code, indicator.get("描述", ""), i_idx))

                    # 匯入知識清單
                    knowledge_list = data.get("知識清單", {})
                    knowledge_map = {}  # code -> competency_item_id
                    for k_idx, (code, value) in enumerate(knowledge_list.items()):
                        # 處理不同格式：字串或字典
                        if isinstance(value, dict):
                            title = value.get("名稱", "") or value.get("title", "")
                        else:
                            title = str(value) if value else ""

                        if title:
                            cursor.execute("""
                                INSERT INTO competency_item (standard_id, type, code, title, sort_order)
                                VALUES (?, 'K', ?, ?, ?)
                            """, (standard_id, code, title, k_idx))
                            knowledge_map[code] = cursor.lastrowid

                    # 匯入技能清單
                    skill_list = data.get("技能清單", {})
                    skill_map = {}  # code -> competency_item_id
                    for s_idx, (code, value) in enumerate(skill_list.items()):
                        if isinstance(value, dict):
                            title = value.get("名稱", "") or value.get("title", "")
                        else:
                            title = str(value) if value else ""

                        if title:
                            cursor.execute("""
                                INSERT INTO competency_item (standard_id, type, code, title, sort_order)
                                VALUES (?, 'S', ?, ?, ?)
                            """, (standard_id, code, title, s_idx))
                            skill_map[code] = cursor.lastrowid

                    # 匯入態度清單
                    attitude_list = data.get("態度清單", {})
                    for a_idx, (code, value) in enumerate(attitude_list.items()):
                        if isinstance(value, dict):
                            title = value.get("名稱", "") or value.get("title", "")
                        else:
                            title = str(value) if value else ""

                        if title:
                            cursor.execute("""
                                INSERT INTO competency_item (standard_id, type, code, title, sort_order)
                                VALUES (?, 'A', ?, ?, ?)
                            """, (standard_id, code, title, a_idx))

                    # 建立任務與 K/S 的對應關係
                    competency_map = {**knowledge_map, **skill_map}
                    self._import_task_competency_maps(cursor, data, competency_map)

                    imported_count += 1

                except Exception as e:
                    logger.debug(f"處理失敗 {json_path}: {e}")

            conn.commit()

            # 重建 FTS 索引
            self._rebuild_fts_index(conn)

        logger.success(f"匯入完成: {imported_count} 個職能基準 (新版格式: {new_format_count}, 舊版格式: {legacy_format_count})")
        return imported_count

    def _import_task_competency_maps(self, cursor, data: Dict, competency_map: Dict[str, int]):
        """
        建立任務與 K/S 的對應關係

        Args:
            cursor: 資料庫游標
            data: 完整的職能基準 JSON 資料
            competency_map: K/S 代碼對應到 competency_item_id 的字典
        """
        for duty in data.get("主要職責", []):
            for task in duty.get("工作任務", []):
                task_code = task.get("代碼", "")
                if not task_code:
                    continue

                # 取得 task_id
                cursor.execute("""
                    SELECT t.id FROM task t
                    JOIN responsibility r ON t.responsibility_id = r.id
                    JOIN standard s ON r.standard_id = s.id
                    WHERE t.task_code = ? AND s.standard_code = ?
                """, (task_code, data.get("職能基準", {}).get("代碼", "")))

                task_row = cursor.fetchone()
                if not task_row:
                    continue
                task_id = task_row[0]

                # 處理知識對應
                for k_code in task.get("知識", []):
                    if isinstance(k_code, str) and k_code in competency_map:
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO task_competency_map (task_id, competency_item_id)
                                VALUES (?, ?)
                            """, (task_id, competency_map[k_code]))
                        except Exception:
                            pass

                # 處理技能對應
                for s_code in task.get("技能", []):
                    if isinstance(s_code, str) and s_code in competency_map:
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO task_competency_map (task_id, competency_item_id)
                                VALUES (?, ?)
                            """, (task_id, competency_map[s_code]))
                        except Exception:
                            pass

    def _rebuild_fts_index(self, conn):
        """重建 FTS 索引"""
        cursor = conn.cursor()
        try:
            # 重建 indicator_fts
            cursor.execute("DELETE FROM indicator_fts")
            cursor.execute("INSERT INTO indicator_fts(rowid, description) SELECT id, description FROM indicator")
            cursor.execute("INSERT INTO indicator_fts(indicator_fts) VALUES('optimize')")

            # 重建 standard_fts
            cursor.execute("DELETE FROM standard_fts")
            cursor.execute("""
                INSERT INTO standard_fts(rowid, standard_name, job_description, qualification_requirements)
                SELECT id, standard_name, job_description, qualification_requirements FROM standard
            """)
            cursor.execute("INSERT INTO standard_fts(standard_fts) VALUES('optimize')")

            conn.commit()
            logger.info("FTS 索引重建完成")
        except Exception as e:
            logger.warning(f"重建 FTS 索引時發生錯誤: {e}")

    def import_from_pdf(self, pdf_path: Union[str, Path]) -> bool:
        """
        直接從 PDF 檔案匯入資料庫

        Args:
            pdf_path: PDF 檔案路徑

        Returns:
            是否匯入成功
        """
        if not PDF_PARSER_AVAILABLE:
            logger.error("pdf_parser_v2 模組未載入，無法從 PDF 匯入")
            return False

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF 檔案不存在: {pdf_path}")
            return False

        try:
            # 解析 PDF
            parser = CompetencyPDFParser()
            result = parser.parse(str(pdf_path))

            # 從檔名提取職業名稱
            if '-職能基準' in pdf_path.stem:
                result.metadata["name"] = pdf_path.stem.replace('-職能基準', '')

            if not result.parse_success:
                logger.error(f"PDF 解析失敗: {result.parse_errors}")
                return False

            # 轉換為字典並匯入
            data = asdict(result)
            return self._import_single_standard(data)

        except Exception as e:
            logger.error(f"從 PDF 匯入失敗: {e}")
            return False

    def import_from_pdf_directory(self, pdf_dir: Union[str, Path] = None,
                                   save_json: bool = False,
                                   json_output_dir: Union[str, Path] = None) -> int:
        """
        從 PDF 目錄批次匯入資料庫

        Args:
            pdf_dir: PDF 檔案目錄，預設為 config.RAW_PDF_DIR
            save_json: 是否同時儲存 JSON 檔案
            json_output_dir: JSON 輸出目錄

        Returns:
            匯入的職能基準數量
        """
        if not PDF_PARSER_AVAILABLE:
            logger.error("pdf_parser_v2 模組未載入，無法從 PDF 匯入")
            return 0

        if pdf_dir is None:
            pdf_dir = config.RAW_PDF_DIR
        elif isinstance(pdf_dir, str):
            pdf_dir = Path(pdf_dir)

        if not pdf_dir.exists():
            logger.error(f"PDF 目錄不存在: {pdf_dir}")
            return 0

        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 個 PDF 檔案")

        if save_json and json_output_dir:
            json_output_dir = Path(json_output_dir)
            json_output_dir.mkdir(parents=True, exist_ok=True)

        parser = CompetencyPDFParser()
        imported_count = 0

        for pdf_path in tqdm(pdf_files, desc="解析並匯入 PDF"):
            try:
                result = parser.parse(str(pdf_path))

                # 從檔名提取職業名稱
                if '-職能基準' in pdf_path.stem:
                    result.metadata["name"] = pdf_path.stem.replace('-職能基準', '')

                if not result.parse_success:
                    logger.debug(f"解析失敗: {pdf_path.name}")
                    continue

                # 轉換為字典
                data = asdict(result)

                # 儲存 JSON（如果需要）
                if save_json and json_output_dir:
                    json_path = json_output_dir / f"{pdf_path.stem}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                # 匯入資料庫
                if self._import_single_standard(data):
                    imported_count += 1

            except Exception as e:
                logger.debug(f"處理失敗 {pdf_path.name}: {e}")

        # 重建 FTS 索引
        with self._get_connection() as conn:
            self._rebuild_fts_index(conn)

        logger.success(f"從 PDF 匯入完成: {imported_count}/{len(pdf_files)} 個職能基準")
        return imported_count

    def _import_single_standard(self, data: Dict) -> bool:
        """
        匯入單一職能基準

        Args:
            data: 職能基準資料（新版或舊版格式）

        Returns:
            是否匯入成功
        """
        try:
            # 偵測並轉換格式
            json_format = self._detect_json_format(data)
            if json_format == 'new':
                data = self._convert_new_to_legacy(data)
            elif json_format == 'unknown':
                logger.warning("無法識別的 JSON 格式")
                return False

            if not data.get("parse_success", False):
                return False

            standard_info = data.get("職能基準", {})
            standard_code = standard_info.get("代碼", "")
            standard_name = standard_info.get("名稱", "")

            # 如果沒有代碼，使用名稱作為代碼
            if not standard_code:
                if standard_name:
                    standard_code = standard_name
                else:
                    return False

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 提取基本資訊
                job_description = standard_info.get("工作描述", "")
                level = standard_info.get("基準級別", 0)

                # 提取分類資訊
                category_list = standard_info.get("職類別", [])
                occupation_list = standard_info.get("職業別", [])
                industry_list = standard_info.get("行業別", [])

                category_code = category_list[0].get("代碼", "") if category_list else ""
                category_name = category_list[0].get("名稱", "") if category_list else ""
                occupation_code = occupation_list[0].get("代碼", "") if occupation_list else ""
                occupation_name = occupation_list[0].get("名稱", "") if occupation_list else ""
                industry_code = industry_list[0].get("代碼", "") if industry_list else ""
                industry_name = industry_list[0].get("名稱", "") if industry_list else ""

                # 提取補充說明
                supplement = data.get("補充說明", {})
                qualification = supplement.get("學歷經驗條件", "")
                glossary = json.dumps(supplement.get("名詞解釋", {}), ensure_ascii=False) if supplement.get("名詞解釋") else ""

                source_file = data.get("source_file", "")

                # 插入或更新 standard
                cursor.execute("""
                    INSERT INTO standard (
                        standard_code, standard_name,
                        category_code, category_name,
                        occupation_code, occupation_name,
                        industry_code, industry_name,
                        category_list, occupation_list, industry_list,
                        job_description, level,
                        qualification_requirements, glossary,
                        source_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(standard_code) DO UPDATE SET
                        standard_name = excluded.standard_name,
                        category_code = excluded.category_code,
                        category_name = excluded.category_name,
                        occupation_code = excluded.occupation_code,
                        occupation_name = excluded.occupation_name,
                        industry_code = excluded.industry_code,
                        industry_name = excluded.industry_name,
                        category_list = excluded.category_list,
                        occupation_list = excluded.occupation_list,
                        industry_list = excluded.industry_list,
                        job_description = excluded.job_description,
                        level = excluded.level,
                        qualification_requirements = excluded.qualification_requirements,
                        glossary = excluded.glossary,
                        source_file = excluded.source_file,
                        updated_at = datetime('now', 'localtime')
                """, (
                    standard_code, standard_name,
                    category_code, category_name,
                    occupation_code, occupation_name,
                    industry_code, industry_name,
                    json.dumps(category_list, ensure_ascii=False) if category_list else None,
                    json.dumps(occupation_list, ensure_ascii=False) if occupation_list else None,
                    json.dumps(industry_list, ensure_ascii=False) if industry_list else None,
                    job_description, level,
                    qualification, glossary,
                    source_file
                ))

                # 取得 standard_id
                cursor.execute("SELECT id FROM standard WHERE standard_code = ?", (standard_code,))
                standard_id = cursor.fetchone()[0]

                # 清除舊的關聯資料
                cursor.execute("DELETE FROM responsibility WHERE standard_id = ?", (standard_id,))
                cursor.execute("DELETE FROM competency_item WHERE standard_id = ?", (standard_id,))

                # 匯入主要職責
                for r_idx, duty in enumerate(data.get("主要職責", [])):
                    t_code = duty.get("代碼", "")
                    if not t_code:
                        continue

                    cursor.execute("""
                        INSERT INTO responsibility (standard_id, t_code, name, sort_order)
                        VALUES (?, ?, ?, ?)
                    """, (standard_id, t_code, duty.get("名稱", ""), r_idx))

                    responsibility_id = cursor.lastrowid

                    # 匯入工作任務
                    for t_idx, task in enumerate(duty.get("工作任務", [])):
                        task_code = task.get("代碼", "")
                        if not task_code:
                            continue

                        cursor.execute("""
                            INSERT INTO task (responsibility_id, task_code, name, competency_level, sort_order)
                            VALUES (?, ?, ?, ?, ?)
                        """, (responsibility_id, task_code, task.get("名稱", ""),
                              task.get("職能級別"), t_idx))

                        task_id = cursor.lastrowid

                        # 匯入工作產出
                        for o_idx, output in enumerate(task.get("工作產出", [])):
                            cursor.execute("""
                                INSERT INTO output (task_id, o_code, name, sort_order)
                                VALUES (?, ?, ?, ?)
                            """, (task_id, output.get("代碼"), output.get("名稱", ""), o_idx))

                        # 匯入行為指標
                        for i_idx, indicator in enumerate(task.get("行為指標", [])):
                            p_code = indicator.get("代碼", "")
                            if not p_code:
                                continue

                            cursor.execute("""
                                INSERT INTO indicator (task_id, p_code, description, sort_order)
                                VALUES (?, ?, ?, ?)
                            """, (task_id, p_code, indicator.get("描述", ""), i_idx))

                # 匯入知識清單
                knowledge_list = data.get("知識清單", {})
                knowledge_map = {}
                for k_idx, (code, value) in enumerate(knowledge_list.items()):
                    if isinstance(value, dict):
                        title = value.get("名稱", "") or value.get("title", "")
                    else:
                        title = str(value) if value else ""

                    if title:
                        cursor.execute("""
                            INSERT INTO competency_item (standard_id, type, code, title, sort_order)
                            VALUES (?, 'K', ?, ?, ?)
                        """, (standard_id, code, title, k_idx))
                        knowledge_map[code] = cursor.lastrowid

                # 匯入技能清單
                skill_list = data.get("技能清單", {})
                skill_map = {}
                for s_idx, (code, value) in enumerate(skill_list.items()):
                    if isinstance(value, dict):
                        title = value.get("名稱", "") or value.get("title", "")
                    else:
                        title = str(value) if value else ""

                    if title:
                        cursor.execute("""
                            INSERT INTO competency_item (standard_id, type, code, title, sort_order)
                            VALUES (?, 'S', ?, ?, ?)
                        """, (standard_id, code, title, s_idx))
                        skill_map[code] = cursor.lastrowid

                # 匯入態度清單
                attitude_list = data.get("態度清單", {})
                for a_idx, (code, value) in enumerate(attitude_list.items()):
                    if isinstance(value, dict):
                        title = value.get("名稱", "") or value.get("title", "")
                    else:
                        title = str(value) if value else ""

                    if title:
                        cursor.execute("""
                            INSERT INTO competency_item (standard_id, type, code, title, sort_order)
                            VALUES (?, 'A', ?, ?, ?)
                        """, (standard_id, code, title, a_idx))

                # 建立任務與 K/S 的對應關係
                competency_map = {**knowledge_map, **skill_map}
                self._import_task_competency_maps(cursor, data, competency_map)

                conn.commit()
                return True

        except Exception as e:
            logger.debug(f"匯入單一職能基準失敗: {e}")
            return False

    def get_statistics(self) -> Dict[str, int]:
        """取得資料庫統計資訊"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}
            tables = ['standard', 'responsibility', 'task', 'output', 'indicator', 'competency_item']

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"total_{table}s"] = cursor.fetchone()[0]

            # 額外統計
            cursor.execute("SELECT COUNT(DISTINCT industry_code) FROM standard WHERE industry_code != ''")
            stats["unique_industries"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT occupation_code) FROM standard WHERE occupation_code != ''")
            stats["unique_occupations"] = cursor.fetchone()[0]

            return stats

    def get_standard_by_code(self, standard_code: str) -> Optional[Dict]:
        """根據代碼取得職能基準"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM standard WHERE standard_code = ?", (standard_code,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_standard_by_name(self, name: str) -> Optional[Dict]:
        """根據名稱取得職能基準"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM standard WHERE standard_name = ?", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def search_standards(self, keyword: str, limit: int = 20) -> List[Dict]:
        """全文檢索職能基準"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, bm25(standard_fts) AS score
                FROM standard_fts fts
                JOIN standard s ON s.id = fts.rowid
                WHERE standard_fts MATCH ?
                ORDER BY bm25(standard_fts)
                LIMIT ?
            """, (keyword, limit))
            return [dict(row) for row in cursor.fetchall()]

    def search_indicators(self, keyword: str, limit: int = 20) -> List[Dict]:
        """全文檢索行為指標"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    s.standard_code,
                    s.standard_name,
                    t.task_code,
                    i.p_code,
                    highlight(indicator_fts, 0, '【', '】') AS highlighted_text,
                    i.description,
                    bm25(indicator_fts) AS score
                FROM indicator_fts fts
                JOIN indicator i ON i.id = fts.rowid
                JOIN task t ON t.id = i.task_id
                JOIN responsibility r ON r.id = t.responsibility_id
                JOIN standard s ON s.id = r.standard_id
                WHERE indicator_fts MATCH ?
                ORDER BY bm25(indicator_fts)
                LIMIT ?
            """, (keyword, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_standard_full_structure(self, standard_code: str) -> Optional[Dict]:
        """取得職能基準的完整階層結構"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 取得基本資訊
            cursor.execute("SELECT * FROM standard WHERE standard_code = ?", (standard_code,))
            standard_row = cursor.fetchone()
            if not standard_row:
                return None

            standard = dict(standard_row)
            standard_id = standard['id']

            # 取得職責
            cursor.execute("""
                SELECT * FROM responsibility
                WHERE standard_id = ?
                ORDER BY sort_order
            """, (standard_id,))
            responsibilities = []

            for r_row in cursor.fetchall():
                resp = dict(r_row)
                resp_id = resp['id']

                # 取得任務
                cursor.execute("""
                    SELECT * FROM task
                    WHERE responsibility_id = ?
                    ORDER BY sort_order
                """, (resp_id,))
                tasks = []

                for t_row in cursor.fetchall():
                    task = dict(t_row)
                    task_id = task['id']

                    # 取得產出
                    cursor.execute("""
                        SELECT * FROM output
                        WHERE task_id = ?
                        ORDER BY sort_order
                    """, (task_id,))
                    task['outputs'] = [dict(o) for o in cursor.fetchall()]

                    # 取得指標
                    cursor.execute("""
                        SELECT * FROM indicator
                        WHERE task_id = ?
                        ORDER BY sort_order
                    """, (task_id,))
                    task['indicators'] = [dict(i) for i in cursor.fetchall()]

                    tasks.append(task)

                resp['tasks'] = tasks
                responsibilities.append(resp)

            standard['responsibilities'] = responsibilities

            # 取得職能內涵
            cursor.execute("""
                SELECT * FROM competency_item
                WHERE standard_id = ?
                ORDER BY type, sort_order
            """, (standard_id,))

            competency_items = {'K': [], 'S': [], 'A': []}
            for ci_row in cursor.fetchall():
                ci = dict(ci_row)
                competency_items[ci['type']].append(ci)

            standard['knowledge'] = competency_items['K']
            standard['skills'] = competency_items['S']
            standard['attitudes'] = competency_items['A']

            return standard

    def get_standards_by_category(self, category_code: str) -> List[Dict]:
        """根據職類別代碼取得所有職能基準"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT standard_code, standard_name, industry_name, level
                FROM standard
                WHERE category_code = ?
                ORDER BY standard_code
            """, (category_code,))
            return [dict(row) for row in cursor.fetchall()]

    def get_standards_by_industry(self, industry_code: str) -> List[Dict]:
        """根據行業別代碼取得所有職能基準"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT standard_code, standard_name, occupation_name, level
                FROM standard
                WHERE industry_code = ?
                ORDER BY standard_code
            """, (industry_code,))
            return [dict(row) for row in cursor.fetchall()]

    def get_standards_by_occupation(self, occupation_code: str) -> List[Dict]:
        """根據職業別代碼取得所有職能基準"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT standard_code, standard_name, industry_name, level
                FROM standard
                WHERE occupation_code = ?
                ORDER BY standard_code
            """, (occupation_code,))
            return [dict(row) for row in cursor.fetchall()]


# 便捷函數
def get_database(db_path: Path = None) -> CompetencyDatabase:
    """取得資料庫實例"""
    return CompetencyDatabase(db_path)


if __name__ == "__main__":
    # 測試
    db = CompetencyDatabase()

    # 從解析後的 JSON 匯入
    count = db.import_from_parsed_json()
    print(f"匯入了 {count} 個職能基準")

    # 顯示統計
    stats = db.get_statistics()
    print(f"統計資訊: {stats}")

    # 測試查詢
    result = db.get_standard_by_name("中式烹飪廚師")
    if result:
        print(f"找到: {result['standard_code']} - {result['standard_name']}")
        print(f"  職業別代碼: {result['occupation_code']}")
        print(f"  行業別代碼: {result['industry_code']}")
