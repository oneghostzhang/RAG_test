"""
職能基準 JSON 資料存取層
直接從 JSON 檔案讀取資料，取代 SQLite 資料庫

支援功能：
- 從 PDF 解析並儲存為 JSON
- 從 JSON 目錄載入所有職能基準
- 提供與 CompetencyDatabase 相容的查詢介面
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from loguru import logger
from tqdm import tqdm

_CJK_SPACE_RE = re.compile(
    r'(?<=[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef])'
)

# 台灣行業標準分類 — 大類代碼字母 → 大類名稱（用於分割串聯行業名稱）
_INDUSTRY_MAJOR_CATEGORIES: Dict[str, str] = {
    'A': '農、林、漁、牧業',
    'B': '礦業及土石採取業',
    'C': '製造業',
    'D': '電力及燃氣供應業',
    'E': '用水供應及污染整治業',
    'F': '營建工程業',
    'G': '批發及零售業',
    'H': '運輸及倉儲業',
    'I': '住宿及餐飲業',
    'J': '出版、影音製作、傳播及資通訊服務業',
    'K': '金融及保險業',
    'L': '不動產業',
    'M': '專業、科學及技術服務業',
    'N': '支援服務業',
    'O': '公共行政及國防；強制性社會安全',
    'P': '教育業',
    'Q': '醫療保健及社會工作服務業',
    'R': '藝術、娛樂及休閒服務業',
    'S': '其他服務業',
    'T': '國際組織及外國機構',
}

_INDUSTRY_CODE_RE = re.compile(r'[A-Z]\d+')


def _split_industry_codes(value: str) -> List[str]:
    """將串聯行業代碼字串拆分為列表，例：'C0899G4729' → ['C0899', 'G4729']"""
    if not value:
        return []
    codes = _INDUSTRY_CODE_RE.findall(value)
    return codes if len(codes) > 1 else [value]


def _split_industry_names(name: str, codes: List[str]) -> List[str]:
    """依行業代碼數量將串聯行業名稱拆分為個別名稱。
    策略：
    1. 以下一個大類名稱＋斜線為分割點（最精確）
    2. 若步驟 1 失敗，嘗試所有已知大類名稱＋斜線（處理代碼字母與大類不符的情況）
    3. 退路：以斜線分割，若段數等於剩餘代碼數則直接使用
    4. 最終退路：每個代碼都使用完整名稱（重複）
    """
    if not name or not codes:
        return [name] if name else []
    if len(codes) == 1:
        return [name]

    # 依長度降冪排列，避免短名稱誤匹配
    all_majors_by_len = sorted(_INDUSTRY_MAJOR_CATEGORIES.values(), key=len, reverse=True)
    major_names = [_INDUSTRY_MAJOR_CATEGORIES.get(c[0].upper(), '') for c in codes]

    result = []
    remaining = name

    for i in range(len(codes)):
        if i == len(codes) - 1:
            result.append(remaining.strip())
            break

        found = False

        # 優先以「下一個代碼對應的大類」查找分割點
        candidates = [major_names[i + 1]] if major_names[i + 1] else []
        # 若找不到，備用所有大類（排除已用大類，避免重複）
        candidates_ext = [m for m in all_majors_by_len if m not in candidates]

        for candidate_list in [candidates, candidates_ext]:
            for major in candidate_list:
                # 支援「大類名稱＋（可選空格）＋斜線」的各種格式
                pat = re.compile(re.escape(major) + r'\s*[/／]')
                m = pat.search(remaining, 1)
                if m and m.start() > 0:
                    result.append(remaining[:m.start()].strip())
                    remaining = remaining[m.start():]
                    found = True
                    break
            if found:
                break

        if not found:
            # 退路：以斜線分割，若段數等於剩餘代碼數則直接使用
            slash_parts = re.split(r'[/／]', remaining)
            if len(slash_parts) == len(codes) - i:
                result.append(slash_parts[0].strip())
                remaining = '/'.join(slash_parts[1:])
            else:
                # 最終退路：剩餘代碼全用同一名稱
                for _ in range(len(codes) - i):
                    result.append(remaining.strip())
                remaining = ''
                break

    if len(result) == len(codes):
        return result
    return [name] * len(codes)


def fix_industry_in_json_files(json_dir) -> int:
    """掃描 json_dir 內所有 JSON 檔，將串聯的行業代碼/名稱修正為列表格式。
    同時修正名稱全部相同（回退策略產生的重複值）的情況。
    回傳修正的檔案數量。"""
    json_dir = Path(json_dir)
    fixed = 0
    for json_file in sorted(json_dir.glob('*.json')):
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)

            basic_info = data.get('basic_info')
            if not isinstance(basic_info, dict):
                continue

            raw_code = basic_info.get('industry_code', '')
            raw_name = basic_info.get('industry', '')
            needs_fix = False
            codes = None

            if isinstance(raw_code, list) and len(raw_code) > 1:
                # 代碼已是列表，但名稱可能是錯誤的重複值（回退策略）
                if isinstance(raw_name, list) and len(raw_name) > 1 and len(set(raw_name)) == 1:
                    # 所有名稱相同 → 重新嘗試用原始名稱拆分
                    codes = raw_code
                    needs_fix = True
            elif isinstance(raw_code, str) and raw_code:
                codes = _split_industry_codes(raw_code)
                if len(codes) > 1:
                    needs_fix = True

            if not needs_fix or not codes:
                continue

            # 取得原始未分割的名稱字串
            if isinstance(raw_name, list):
                orig_name = raw_name[0] if raw_name else ''
            else:
                orig_name = raw_name or ''

            names = _split_industry_names(orig_name, codes)

            basic_info['industry_code'] = codes
            basic_info['industry'] = names
            data['basic_info'] = basic_info

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"已修正行業資料: {json_file.name}  → codes={codes}")
            fixed += 1

        except Exception as e:
            logger.warning(f"修正失敗 {json_file.name}: {e}")

    return fixed


def _resolve_industry_code(raw_code, raw_name='') -> str:
    """將 industry_code 欄位（可能為列表或串聯字串）解析為 '; ' 分隔字串，供 CompetencyStandard 使用。"""
    if isinstance(raw_code, list):
        return '; '.join(raw_code)
    if isinstance(raw_code, str) and raw_code:
        codes = _split_industry_codes(raw_code)
        if len(codes) > 1:
            return '; '.join(codes)
    return raw_code or ''


def _resolve_industry_name(raw_name, raw_code='') -> str:
    """將 industry 欄位（可能為列表或串聯字串）解析為 '; ' 分隔字串，供 CompetencyStandard 使用。"""
    if isinstance(raw_name, list):
        return '; '.join(raw_name)
    if isinstance(raw_name, str) and raw_name:
        # 若代碼有多個，嘗試對應拆分名稱
        codes_list = raw_code if isinstance(raw_code, list) else _split_industry_codes(raw_code)
        if len(codes_list) > 1:
            names = _split_industry_names(raw_name, codes_list)
            return '; '.join(names)
    return raw_name or ''


def _normalize_cjk(obj):
    """遞迴移除中文字符之間 PDF 擷取產生的多餘空格"""
    if isinstance(obj, str):
        return _CJK_SPACE_RE.sub('', obj)
    if isinstance(obj, list):
        return [_normalize_cjk(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _normalize_cjk(v) for k, v in obj.items()}
    return obj

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
class CompetencyStandard:
    """職能基準資料結構"""
    code: str = ""
    name: str = ""
    category_code: str = ""
    category_name: str = ""
    occupation_code: str = ""
    occupation_name: str = ""
    industry_code: str = ""
    industry_name: str = ""
    job_description: str = ""
    level: int = 0
    requirements: str = ""
    source_file: str = ""
    # 完整資料
    tasks: List[Dict] = field(default_factory=list)
    knowledge: List[Dict] = field(default_factory=list)
    skills: List[Dict] = field(default_factory=list)
    attitudes: List[Dict] = field(default_factory=list)
    chunks_for_rag: List[Dict] = field(default_factory=list)
    # 原始資料
    raw_data: Dict = field(default_factory=dict)


class CompetencyJSONStore:
    """
    職能基準 JSON 資料存取層

    直接從 JSON 檔案讀取資料，提供與 CompetencyDatabase 相容的介面
    """

    def __init__(self, json_dir: Path = None):
        """
        初始化資料存取層

        Args:
            json_dir: JSON 檔案目錄，預設為 config.DATA_DIR / "parsed_json_v2"
        """
        if json_dir is None:
            self.json_dir = config.DATA_DIR / "parsed_json_v2"
        elif isinstance(json_dir, str):
            self.json_dir = Path(json_dir)
        else:
            self.json_dir = json_dir

        self.json_dir.mkdir(parents=True, exist_ok=True)

        # 資料索引
        self.standards: Dict[str, CompetencyStandard] = {}  # code -> standard
        self.name_to_code: Dict[str, str] = {}  # name -> code
        self.category_index: Dict[str, List[str]] = defaultdict(list)  # category -> [codes]
        self.industry_index: Dict[str, List[str]] = defaultdict(list)  # industry -> [codes]
        self.occupation_index: Dict[str, List[str]] = defaultdict(list)  # occupation -> [codes]

        # 索引快取路徑
        self._index_cache_path = self.json_dir / "_index_cache.pkl"

        # 載入資料
        self._load_all()

    def _load_all(self):
        """載入所有 JSON 檔案"""
        # 嘗試載入快取
        if self._load_index_cache():
            logger.info(f"從快取載入索引: {len(self.standards)} 個職能基準")
            return

        # 重新掃描並建立索引
        self._rebuild_index()

    def _load_index_cache(self) -> bool:
        """載入索引快取"""
        if not self._index_cache_path.exists():
            return False

        try:
            # 檢查快取是否過期（比較 JSON 檔案修改時間）
            cache_mtime = self._index_cache_path.stat().st_mtime
            json_files = list(self.json_dir.glob("*.json"))

            for json_file in json_files:
                if json_file.stat().st_mtime > cache_mtime:
                    logger.info("發現新的 JSON 檔案，重建索引")
                    return False

            with open(self._index_cache_path, 'rb') as f:
                cache = pickle.load(f)

            self.standards = cache.get('standards', {})
            self.name_to_code = cache.get('name_to_code', {})
            self.category_index = defaultdict(list, cache.get('category_index', {}))
            self.industry_index = defaultdict(list, cache.get('industry_index', {}))
            self.occupation_index = defaultdict(list, cache.get('occupation_index', {}))

            return True

        except Exception as e:
            logger.warning(f"載入索引快取失敗: {e}")
            return False

    def _save_index_cache(self):
        """儲存索引快取"""
        try:
            cache = {
                'standards': self.standards,
                'name_to_code': self.name_to_code,
                'category_index': dict(self.category_index),
                'industry_index': dict(self.industry_index),
                'occupation_index': dict(self.occupation_index),
            }

            with open(self._index_cache_path, 'wb') as f:
                pickle.dump(cache, f)

        except Exception as e:
            logger.warning(f"儲存索引快取失敗: {e}")

    def _rebuild_index(self):
        """重建索引"""
        self.standards.clear()
        self.name_to_code.clear()
        self.category_index.clear()
        self.industry_index.clear()
        self.occupation_index.clear()

        json_files = list(self.json_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"JSON 目錄為空: {self.json_dir}")
            return

        logger.info(f"載入 {len(json_files)} 個 JSON 檔案...")

        for json_path in tqdm(json_files, desc="載入職能基準"):
            if json_path.name.startswith("_"):
                continue  # 跳過快取檔案

            try:
                standard = self._load_single_json(json_path)
                if standard:
                    self._add_to_index(standard)
            except Exception as e:
                logger.debug(f"載入失敗 {json_path.name}: {e}")

        logger.success(f"載入完成: {len(self.standards)} 個職能基準")
        self._save_index_cache()

    def _load_single_json(self, json_path: Path) -> Optional[CompetencyStandard]:
        """載入單一 JSON 檔案"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data.get('parse_success', False):
            return None

        # 偵測格式
        if 'metadata' in data and 'basic_info' in data:
            return self._parse_new_format(data, json_path)
        elif '職能基準' in data:
            return self._parse_legacy_format(data, json_path)

        return None

    def _parse_new_format(self, data: Dict, json_path: Path) -> CompetencyStandard:
        """解析新版 JSON 格式"""
        metadata = data.get('metadata', {})
        basic_info = data.get('basic_info', {})

        code = metadata.get('code', '') or metadata.get('name', '') or json_path.stem.replace('-職能基準', '')
        name = metadata.get('name', '') or json_path.stem.replace('-職能基準', '')

        return CompetencyStandard(
            code=code,
            name=name,
            category_code=basic_info.get('category_code', ''),
            category_name=basic_info.get('category', ''),
            occupation_code=basic_info.get('occupation_code', ''),
            occupation_name=basic_info.get('occupation', ''),
            industry_code=_resolve_industry_code(basic_info.get('industry_code', ''), basic_info.get('industry', '')),
            industry_name=_resolve_industry_name(basic_info.get('industry', ''), basic_info.get('industry_code', '')),
            job_description=basic_info.get('job_description', ''),
            level=basic_info.get('level', 0),
            requirements=basic_info.get('requirements', ''),
            source_file=metadata.get('source_file', str(json_path)),
            tasks=_normalize_cjk(data.get('competency_tasks', [])),
            knowledge=_normalize_cjk(data.get('competency_knowledge', [])),
            skills=_normalize_cjk(data.get('competency_skills', [])),
            attitudes=_normalize_cjk(data.get('competency_attitudes', [])),
            chunks_for_rag=_normalize_cjk(data.get('chunks_for_rag', [])),
            raw_data=data
        )

    def _parse_legacy_format(self, data: Dict, json_path: Path) -> CompetencyStandard:
        """解析舊版 JSON 格式"""
        standard_info = data.get('職能基準', {})
        supplement = data.get('補充說明', {})

        # 提取分類資訊
        category_list = standard_info.get('職類別', [])
        occupation_list = standard_info.get('職業別', [])
        industry_list = standard_info.get('行業別', [])

        code = standard_info.get('代碼', '') or standard_info.get('名稱', '') or json_path.stem.replace('-職能基準', '')
        name = standard_info.get('名稱', '') or json_path.stem.replace('-職能基準', '')

        # 扁平化任務
        tasks = []
        for resp in data.get('主要職責', []):
            resp_code = resp.get('代碼', '')
            resp_name = resp.get('名稱', '')
            for task in resp.get('工作任務', []):
                tasks.append({
                    'main_responsibility': f"{resp_code}{resp_name}",
                    'task_id': task.get('代碼', ''),
                    'task_name': task.get('名稱', ''),
                    'level': task.get('職能級別', 0),
                    'output': '、'.join([o.get('名稱', '') for o in task.get('工作產出', [])]),
                    'behaviors': [i.get('描述', '') for i in task.get('行為指標', [])],
                    'knowledge': task.get('知識', []),
                    'skills': task.get('技能', [])
                })

        # 轉換知識清單
        knowledge_dict = data.get('知識清單', {})
        knowledge = [{'code': k, 'name': v, 'category': '知識'}
                     for k, v in knowledge_dict.items() if isinstance(v, str)]

        # 轉換技能清單
        skills_dict = data.get('技能清單', {})
        skills = [{'code': k, 'name': v, 'category': '技能'}
                  for k, v in skills_dict.items() if isinstance(v, str)]

        # 轉換態度清單
        attitudes_dict = data.get('態度清單', {})
        attitudes = []
        for k, v in attitudes_dict.items():
            if isinstance(v, str):
                if '：' in v:
                    name, desc = v.split('：', 1)
                    attitudes.append({'code': k, 'name': name, 'description': desc, 'category': '態度'})
                else:
                    attitudes.append({'code': k, 'name': v, 'description': '', 'category': '態度'})

        return CompetencyStandard(
            code=code,
            name=name,
            category_code=category_list[0].get('代碼', '') if category_list else '',
            category_name=category_list[0].get('名稱', '') if category_list else '',
            occupation_code=occupation_list[0].get('代碼', '') if occupation_list else '',
            occupation_name=occupation_list[0].get('名稱', '') if occupation_list else '',
            industry_code=industry_list[0].get('代碼', '') if industry_list else '',
            industry_name=industry_list[0].get('名稱', '') if industry_list else '',
            job_description=standard_info.get('工作描述', ''),
            level=standard_info.get('基準級別', 0),
            requirements=supplement.get('學歷經驗條件', ''),
            source_file=data.get('source_file', str(json_path)),
            tasks=tasks,
            knowledge=knowledge,
            skills=skills,
            attitudes=attitudes,
            chunks_for_rag=[],
            raw_data=data
        )

    def _add_to_index(self, standard: CompetencyStandard):
        """加入索引"""
        self.standards[standard.code] = standard
        self.name_to_code[standard.name] = standard.code

        if standard.category_name:
            self.category_index[standard.category_name].append(standard.code)
        if standard.industry_name:
            for ind in standard.industry_name.split('; '):
                ind = ind.strip()
                if ind:
                    self.industry_index[ind].append(standard.code)
        if standard.occupation_name:
            self.occupation_index[standard.occupation_name].append(standard.code)

    # ========== 與 CompetencyDatabase 相容的查詢介面 ==========

    def get_standard_by_code(self, code: str) -> Optional[Dict]:
        """根據代碼取得職能基準"""
        standard = self.standards.get(code)
        if standard:
            return asdict(standard)
        return None

    def get_standard_by_name(self, name: str) -> Optional[Dict]:
        """根據名稱取得職能基準"""
        code = self.name_to_code.get(name)
        if code:
            return self.get_standard_by_code(code)
        return None

    def get_all_standards(self) -> List[Dict]:
        """取得所有職能基準"""
        return [asdict(s) for s in self.standards.values()]

    def get_standards_by_category(self, category_name: str) -> List[Dict]:
        """根據職類別取得職能基準"""
        codes = self.category_index.get(category_name, [])
        return [asdict(self.standards[c]) for c in codes if c in self.standards]

    def get_standards_by_industry(self, industry_name: str) -> List[Dict]:
        """根據行業別取得職能基準"""
        codes = self.industry_index.get(industry_name, [])
        return [asdict(self.standards[c]) for c in codes if c in self.standards]

    def get_standards_by_occupation(self, occupation_name: str) -> List[Dict]:
        """根據職業別取得職能基準"""
        codes = self.occupation_index.get(occupation_name, [])
        return [asdict(self.standards[c]) for c in codes if c in self.standards]

    def search_standards(self, keyword: str, limit: int = 20) -> List[Dict]:
        """搜尋職能基準（簡單關鍵字比對）"""
        results = []
        keyword_lower = keyword.lower()

        for standard in self.standards.values():
            score = 0
            # 名稱匹配
            if keyword in standard.name:
                score += 10
            # 工作描述匹配
            if keyword in standard.job_description:
                score += 5
            # 職類別匹配
            if keyword in standard.category_name:
                score += 3
            # 職業別匹配
            if keyword in standard.occupation_name:
                score += 3

            if score > 0:
                result = asdict(standard)
                result['score'] = score
                results.append(result)

        # 排序並限制數量
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def get_statistics(self) -> Dict[str, int]:
        """取得統計資訊"""
        total_tasks = sum(len(s.tasks) for s in self.standards.values())
        total_knowledge = sum(len(s.knowledge) for s in self.standards.values())
        total_skills = sum(len(s.skills) for s in self.standards.values())
        total_attitudes = sum(len(s.attitudes) for s in self.standards.values())

        return {
            'total_standards': len(self.standards),
            'total_tasks': total_tasks,
            'total_knowledge': total_knowledge,
            'total_skills': total_skills,
            'total_attitudes': total_attitudes,
            'unique_categories': len(self.category_index),
            'unique_industries': len(self.industry_index),
            'unique_occupations': len(self.occupation_index)
        }

    def get_all_rag_chunks(self) -> List[Dict]:
        """取得所有 RAG chunks"""
        all_chunks = []
        for standard in self.standards.values():
            for chunk in standard.chunks_for_rag:
                chunk_copy = chunk.copy()
                chunk_copy['standard_code'] = standard.code
                chunk_copy['standard_name'] = standard.name
                all_chunks.append(chunk_copy)
        return all_chunks

    # ========== 資料匯入方法 ==========

    def import_from_pdf(self, pdf_path: Union[str, Path]) -> bool:
        """從 PDF 匯入"""
        if not PDF_PARSER_AVAILABLE:
            logger.error("pdf_parser_v2 模組未載入")
            return False

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF 不存在: {pdf_path}")
            return False

        try:
            parser = CompetencyPDFParser()
            result = parser.parse(str(pdf_path))

            if '-職能基準' in pdf_path.stem:
                result.metadata['name'] = pdf_path.stem.replace('-職能基準', '')

            if not result.parse_success:
                return False

            # 儲存 JSON
            json_path = self.json_dir / f"{pdf_path.stem}.json"
            data = asdict(result)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 加入索引
            standard = self._parse_new_format(data, json_path)
            if standard:
                self._add_to_index(standard)
                self._save_index_cache()

            return True

        except Exception as e:
            logger.error(f"匯入失敗: {e}")
            return False

    def import_from_pdf_directory(self, pdf_dir: Union[str, Path] = None) -> int:
        """從 PDF 目錄批次匯入"""
        if not PDF_PARSER_AVAILABLE:
            logger.error("pdf_parser_v2 模組未載入")
            return 0

        if pdf_dir is None:
            pdf_dir = config.DATA_DIR / "raw_pdf"
        else:
            pdf_dir = Path(pdf_dir)

        if not pdf_dir.exists():
            logger.error(f"PDF 目錄不存在: {pdf_dir}")
            return 0

        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 個 PDF 檔案")

        parser = CompetencyPDFParser()
        imported_count = 0

        for pdf_path in tqdm(pdf_files, desc="解析 PDF"):
            try:
                result = parser.parse(str(pdf_path))

                if '-職能基準' in pdf_path.stem:
                    result.metadata['name'] = pdf_path.stem.replace('-職能基準', '')

                if not result.parse_success:
                    continue

                # 儲存 JSON
                json_path = self.json_dir / f"{pdf_path.stem}.json"
                data = asdict(result)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                # 加入索引
                standard = self._parse_new_format(data, json_path)
                if standard:
                    self._add_to_index(standard)
                    imported_count += 1

            except Exception as e:
                logger.debug(f"處理失敗 {pdf_path.name}: {e}")

        self._save_index_cache()
        logger.success(f"匯入完成: {imported_count}/{len(pdf_files)} 個職能基準")
        return imported_count

    def refresh(self):
        """重新載入所有資料"""
        self._rebuild_index()


# 便捷函數
def get_store(json_dir: Path = None) -> CompetencyJSONStore:
    """取得資料存取層實例"""
    return CompetencyJSONStore(json_dir)
