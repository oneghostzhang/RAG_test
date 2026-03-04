# -*- coding: utf-8 -*-
"""
職能基準知識圖譜 Graph RAG UI
PyQt6 桌面介面，整合五種查詢模式
"""

import sys
import io
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# UTF-8 輸出支持 (Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QFileDialog,
    QListWidget, QSplitter, QGroupBox, QProgressBar, QMessageBox,
    QListWidgetItem, QScrollArea, QFrame, QDialog, QComboBox,
    QTabWidget, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QFont, QTextCursor, QCursor, QDesktopServices

from config import get_config
from graph_builder import CompetencyKnowledgeGraph
from graph_rag import GraphRAGQueryEngine, QueryResult
from pdf_parser_v2 import CompetencyPDFParser, parse_pdf_to_json
from competency_store import CompetencyJSONStore, fix_industry_in_json_files

config = get_config()


# =============================
# 背景執行緒
# =============================

class PDFParseWorker(QThread):
    """PDF 解析背景執行緒（支援新舊版解析器）"""
    progress = pyqtSignal(str, int, int)  # message, current, total
    finished = pyqtSignal(bool, str, int)  # success, message, count

    def __init__(self, pdf_dir: Path, output_dir: Path, limit: int = 0, use_v2: bool = True):
        super().__init__()
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.limit = limit
        self.use_v2 = use_v2

    def run(self):
        try:
            pdf_files = list(self.pdf_dir.glob("*.pdf"))

            if self.limit > 0:
                pdf_files = pdf_files[:self.limit]

            total = len(pdf_files)
            success_count = 0

            if self.use_v2:
                # 使用新版解析器 (pdf_parser_v2)
                for i, pdf_file in enumerate(pdf_files, 1):
                    self.progress.emit(f"解析(v2): {pdf_file.name}", i, total)
                    try:
                        output_file = self.output_dir / f"{pdf_file.stem}.json"
                        data = parse_pdf_to_json(str(pdf_file), str(output_file))
                        if data and data.get("parse_success"):
                            success_count += 1
                    except Exception as e:
                        pass  # 繼續處理下一個
            else:
                # 使用舊版解析器 (pdf_parser)
                parser = CompetencyPDFParser()
                for i, pdf_file in enumerate(pdf_files, 1):
                    self.progress.emit(f"解析: {pdf_file.name}", i, total)
                    try:
                        result = parser.parse_pdf(str(pdf_file))
                        if result and result.parse_success:
                            output_file = self.output_dir / f"{pdf_file.stem}.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(result.to_json())
                            success_count += 1
                    except Exception as e:
                        pass  # 繼續處理下一個

            # 解析完成後，修正既有 JSON 檔中串聯的行業代碼
            try:
                fixed = fix_industry_in_json_files(self.output_dir)
                if fixed:
                    self.progress.emit(f"已修正 {fixed} 個行業資料格式", total, total)
            except Exception:
                pass

            self.finished.emit(True, f"成功解析 {success_count}/{total} 個 PDF", success_count)

        except Exception as e:
            self.finished.emit(False, f"解析失敗: {str(e)}", 0)


class GraphBuildWorker(QThread):
    """圖譜建構背景執行緒"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, object)  # success, message, graph

    def __init__(self, json_dir: Path, graph_path: Path):
        super().__init__()
        self.json_dir = json_dir
        self.graph_path = graph_path

    def run(self):
        try:
            config = get_config()

            # 建構前先修正 JSON 中串聯的行業代碼
            self.progress.emit("檢查並修正行業資料格式...")
            try:
                fixed = fix_industry_in_json_files(self.json_dir)
                if fixed:
                    self.progress.emit(f"已修正 {fixed} 個行業資料格式")
            except Exception:
                pass

            self.progress.emit("初始化知識圖譜...")
            kg = CompetencyKnowledgeGraph()

            self.progress.emit("從 JSON 建構圖譜...")
            kg.build_from_json_directory(str(self.json_dir))

            self.progress.emit("推斷職涯路徑...")
            kg.infer_career_paths()

            # 從 ICAP 來源 JSON 補充 metadata (行業別等)
            self.progress.emit("補充 ICAP metadata (行業別等)...")
            kg.enrich_with_icap_metadata(config.ICAP_SOURCE_DIR)

            self.progress.emit("儲存圖譜...")
            kg.save(str(self.graph_path))

            stats = kg.get_statistics()
            msg = f"圖譜建構完成！節點: {stats.get('total_nodes', 0)}, 邊: {stats.get('total_edges', 0)}"
            self.finished.emit(True, msg, kg)

        except Exception as e:
            self.finished.emit(False, f"建構失敗: {str(e)}", None)


class EmbeddingInitWorker(QThread):
    """Embedding 初始化背景執行緒"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, engine: GraphRAGQueryEngine, force_rebuild: bool = False):
        super().__init__()
        self.engine = engine
        self.force_rebuild = force_rebuild

    def run(self):
        try:
            if self.force_rebuild:
                self.progress.emit("重新建立向量索引...")
            else:
                self.progress.emit("載入 Embedding 模型（嘗試載入已存儲的索引）...")

            self.engine.initialize_embeddings(force_rebuild=self.force_rebuild)

            if self.force_rebuild:
                self.finished.emit(True, "向量索引重建完成並已存儲")
            else:
                # 檢查是載入還是新建
                vector_count = self.engine.vector_index.ntotal if self.engine.vector_index else 0
                self.finished.emit(True, f"Embedding 初始化完成（{vector_count} 個向量）")
        except Exception as e:
            self.finished.emit(False, f"初始化失敗: {str(e)}")


class LLMInitWorker(QThread):
    """LLM 初始化背景執行緒"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, engine: GraphRAGQueryEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.progress.emit("載入 LLM 模型（這可能需要一些時間）...")
            success = self.engine.initialize_llm(callback=lambda msg: self.progress.emit(msg))
            if success:
                self.finished.emit(True, "LLM 模型初始化完成")
            else:
                self.finished.emit(False, "LLM 模型初始化失敗，請檢查模型路徑")
        except Exception as e:
            self.finished.emit(False, f"LLM 初始化失敗: {str(e)}")


class FederatedSearchInitWorker(QThread):
    """聯邦搜索初始化背景執行緒"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, engine: GraphRAGQueryEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.progress.emit("建立 ICAP 元資料索引...")
            success = self.engine.initialize_federated_search(
                callback=lambda msg: self.progress.emit(msg)
            )
            if success:
                # 取得職類別數量
                categories = self.engine.get_category_list()
                count = len(categories) if categories else 0
                self.finished.emit(True, f"聯邦搜索初始化完成，共 {count} 個職類別")
            else:
                self.finished.emit(False, "聯邦搜索初始化失敗")
        except Exception as e:
            self.finished.emit(False, f"聯邦搜索初始化失敗: {str(e)}")


class CommunityVisualizeWorker(QThread):
    """社群視覺化背景執行緒（GraphRAG 風格）"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)  # success, message, file_path

    def __init__(self, kg: CompetencyKnowledgeGraph, output_path: Path,
                 center_node: str = None, max_nodes: int = 300):
        super().__init__()
        self.kg = kg
        self.output_path = output_path
        self.center_node = center_node
        self.max_nodes = max_nodes

    def run(self):
        try:
            self.progress.emit("偵測社群中...")
            from graph_community import generate_community_visualization

            self.progress.emit("生成社群視覺化...")
            generate_community_visualization(
                self.kg.graph,
                str(self.output_path),
                center_node=self.center_node,
                resolution=1.0,
                max_nodes=self.max_nodes,
                title="職能基準知識圖譜 - 社群視覺化"
            )
            self.finished.emit(True, "社群視覺化完成", str(self.output_path))

        except ImportError as e:
            self.finished.emit(False, f"請先安裝必要套件: pip install python-louvain\n{str(e)}", "")
        except Exception as e:
            self.finished.emit(False, f"社群視覺化失敗: {str(e)}", "")


class GraphVisualizeWorker(QThread):
    """圖譜視覺化背景執行緒（改進版 - 支援過濾和限制）"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)  # success, message, file_path

    def __init__(self, kg: CompetencyKnowledgeGraph, output_path: Path, center_node: str = None,
                 depth: int = 2, node_limit: int = 100, selected_types: list = None):
        super().__init__()
        self.kg = kg
        self.output_path = output_path
        self.center_node = center_node
        self.depth = depth
        self.node_limit = node_limit
        self.selected_types = selected_types or ["職能基準", "主要職責", "工作任務", "知識", "技能"]

    def run(self):
        try:
            self.progress.emit("建立清晰分層視覺化...")
            self._generate_improved_visualization()
            self.finished.emit(True, "視覺化完成", str(self.output_path))

        except ImportError as e:
            self.finished.emit(False, f"請先安裝必要套件: {str(e)}", "")
        except Exception as e:
            self.finished.emit(False, f"視覺化失敗: {str(e)}", "")

    def _generate_improved_visualization(self):
        """生成改進的視覺化 HTML"""

        # 選擇要顯示的節點
        if self.center_node:
            nodes_to_show = self._get_hierarchical_nodes_filtered(
                self.center_node, self.depth, self.selected_types, self.node_limit
            )
        else:
            # 顯示前幾個職能基準的完整結構
            standards = list(self.kg.get_nodes_by_type("職能基準"))[:3]
            nodes_to_show = set()
            for std in standards:
                nodes_to_show.update(self._get_hierarchical_nodes_filtered(
                    std, 2, self.selected_types, self.node_limit // 3
                ))

        self.progress.emit(f"處理 {len(nodes_to_show)} 個節點（限制: {self.node_limit}）...")

        # 收集節點和邊資料
        nodes_data = []
        edges_data = []

        # 節點層級映射（用於分層顯示）
        level_map = {
            "職能基準": 0,
            "職類別": 1, "職業別": 1, "行業別": 1,
            "主要職責": 2, "態度": 2,
            "工作任務": 3,
            "工作產出": 4, "行為指標": 4,
            "知識": 5, "技能": 5
        }

        # 節點配色（高對比度）
        color_map = {
            "職能基準": {"bg": "#c0392b", "border": "#922b21", "font": "#ffffff"},
            "職類別": {"bg": "#d35400", "border": "#a04000", "font": "#ffffff"},
            "職業別": {"bg": "#f39c12", "border": "#d68910", "font": "#000000"},
            "行業別": {"bg": "#16a085", "border": "#117a65", "font": "#ffffff"},
            "主要職責": {"bg": "#2980b9", "border": "#1a5276", "font": "#ffffff"},
            "工作任務": {"bg": "#8e44ad", "border": "#6c3483", "font": "#ffffff"},
            "知識": {"bg": "#27ae60", "border": "#1e8449", "font": "#ffffff"},
            "技能": {"bg": "#f1c40f", "border": "#d4ac0d", "font": "#000000"},
            "態度": {"bg": "#7f8c8d", "border": "#616a6b", "font": "#ffffff"},
            "行為指標": {"bg": "#3498db", "border": "#2471a3", "font": "#ffffff"},
            "工作產出": {"bg": "#e67e22", "border": "#ca6f1e", "font": "#ffffff"}
        }

        # 節點形狀和大小（全部使用 box 確保文字在節點內）
        shape_map = {
            "職能基準": {"shape": "box", "size": 40},
            "職類別": {"shape": "box", "size": 30},
            "職業別": {"shape": "box", "size": 30},
            "行業別": {"shape": "box", "size": 30},
            "主要職責": {"shape": "box", "size": 35},
            "工作任務": {"shape": "box", "size": 30},
            "知識": {"shape": "box", "size": 25},
            "技能": {"shape": "box", "size": 25},
            "態度": {"shape": "box", "size": 25},
            "行為指標": {"shape": "box", "size": 22},
            "工作產出": {"shape": "box", "size": 22}
        }

        for node_id in nodes_to_show:
            node_data = self.kg.get_node_data(node_id)
            if node_data:
                node_type = node_data.get("node_type", "未知")
                name = node_data.get("name", "") or node_data.get("code", node_id)
                display_name = name[:20] + "..." if len(name) > 20 else name

                colors = color_map.get(node_type, {"bg": "#bdc3c7", "border": "#95a5a6", "font": "#000"})
                shape_info = shape_map.get(node_type, {"shape": "dot", "size": 20})
                level = level_map.get(node_type, 5)

                # 建立 tooltip（使用純文字格式）
                tooltip_lines = [f"【{node_type}】"]
                if name:
                    tooltip_lines.append(f"名稱: {name}")
                if node_data.get("code"):
                    tooltip_lines.append(f"代碼: {node_data.get('code')}")
                if node_data.get("level"):
                    tooltip_lines.append(f"職能級別: {node_data.get('level')}")
                if node_data.get("description"):
                    desc = node_data.get("description")[:100]
                    tooltip_lines.append(f"描述: {desc}...")
                tooltip = "\n".join(tooltip_lines)

                # 對於 box 形狀，需要設定適當的寬度和高度
                node_config = {
                    "id": node_id,
                    "label": display_name,
                    "title": tooltip,
                    "color": {"background": colors["bg"], "border": colors["border"]},
                    "font": {
                        "color": colors["font"],
                        "size": 12,
                        "face": "Microsoft JhengHei, Arial, sans-serif",
                        "bold": True if node_type in ["職能基準", "主要職責"] else False
                    },
                    "shape": shape_info["shape"],
                    "level": level,
                    "borderWidth": 4 if node_type == "職能基準" else 2,
                    "group": node_type,
                    # box 形狀的寬高設定
                    "widthConstraint": {"minimum": 80, "maximum": 200},
                    "heightConstraint": {"minimum": 30},
                    "margin": {"top": 8, "bottom": 8, "left": 10, "right": 10},
                    # 完整節點資料（供模態框使用）
                    "fullData": {
                        "nodeType": node_type,
                        "name": name,
                        "code": node_data.get("code", ""),
                        "level": node_data.get("level", ""),
                        "description": node_data.get("description", ""),
                        "bgColor": colors["bg"]
                    }
                }

                # 職能基準節點較大
                if node_type == "職能基準":
                    node_config["widthConstraint"] = {"minimum": 120, "maximum": 250}
                    node_config["font"]["size"] = 14

                nodes_data.append(node_config)

        # 收集邊
        edge_colors = {
            "包含職責": "#3498db", "包含任務": "#9b59b6",
            "需要知識": "#27ae60", "需要技能": "#f1c40f",
            "屬於職類": "#e67e22", "屬於職業": "#f39c12", "適用行業": "#1abc9c",
            "產出": "#eb984e", "要求行為": "#5dade2", "要求態度": "#95a5a6",
            "可晉升至": "#e74c3c"
        }

        for source, target, edge_data in self.kg.graph.edges(data=True):
            if source in nodes_to_show and target in nodes_to_show:
                edge_type = edge_data.get("edge_type", "")
                edges_data.append({
                    "from": source,
                    "to": target,
                    "title": edge_type,
                    "color": edge_colors.get(edge_type, "#bdc3c7"),
                    "width": 2.5 if edge_type in ["包含職責", "包含任務"] else 1.5,
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}}
                })

        # 生成 HTML
        html_content = self._generate_html(nodes_data, edges_data)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _get_hierarchical_nodes_filtered(self, center_node: str, depth: int,
                                          selected_types: list, node_limit: int) -> set:
        """取得以中心節點為起點的階層節點（帶過濾和限制）

        只沿著出邊方向展開，避免通過共享節點拉入其他職能基準
        """
        nodes = set([center_node])
        current_level = {center_node}

        # 調試：檢查圖譜類型和中心節點的邊數量
        import networkx as nx
        is_directed = isinstance(self.kg.graph, nx.DiGraph)
        total_edges = self.kg.graph.number_of_edges()
        out_edges = list(self.kg.graph.out_edges(center_node)) if is_directed else []
        in_edges = list(self.kg.graph.in_edges(center_node)) if is_directed else []
        neighbors = list(self.kg.graph.neighbors(center_node)) if self.kg.graph.has_node(center_node) else []

        self.progress.emit(f"圖譜類型: {'有向' if is_directed else '無向'}, 總邊數: {total_edges}")
        self.progress.emit(f"中心節點: {center_node[:50]}")
        self.progress.emit(f"出邊: {len(out_edges)}, 入邊: {len(in_edges)}, 鄰居: {len(neighbors)}")

        # 優先順序：主要職責 > 工作任務 > 知識/技能 > 其他（不包含職能基準，因為只往下展開）
        priority_order = ["主要職責", "工作任務", "知識", "技能", "工作產出", "行為指標", "態度", "職類別", "職業別", "行業別"]

        for d in range(depth):
            next_level = set()
            for node in current_level:
                # 只沿著出邊方向展開（職能基準 → 主要職責 → 工作任務 → 知識/技能）
                for _, target, _ in self.kg.graph.out_edges(node, data=True):
                    target_data = self.kg.get_node_data(target)
                    if target_data:
                        target_type = target_data.get("node_type", "")
                        # 不要把其他職能基準拉進來
                        if target_type in selected_types and target_type != "職能基準":
                            next_level.add(target)

            # 按優先順序排序並限制數量
            if len(nodes) + len(next_level) > node_limit:
                # 對 next_level 排序，優先保留重要類型
                sorted_next = sorted(next_level, key=lambda n: (
                    priority_order.index(self.kg.get_node_data(n).get("node_type", ""))
                    if self.kg.get_node_data(n) and self.kg.get_node_data(n).get("node_type", "") in priority_order
                    else 999
                ))
                remaining = node_limit - len(nodes)
                next_level = set(sorted_next[:remaining])

            nodes.update(next_level)
            current_level = next_level

            if len(nodes) >= node_limit:
                break

        return nodes

    def _get_hierarchical_nodes(self, center_node: str, depth: int) -> set:
        """取得以中心節點為起點的階層節點（舊版本，保留相容性）"""
        return self._get_hierarchical_nodes_filtered(center_node, depth,
            ["職能基準", "主要職責", "工作任務", "知識", "技能", "工作產出"], 200)

    def _generate_html(self, nodes_data: list, edges_data: list) -> str:
        """生成完整的 HTML 視覺化頁面"""
        import json

        nodes_json = json.dumps(nodes_data, ensure_ascii=False)
        edges_json = json.dumps(edges_data, ensure_ascii=False)

        html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>職能基準知識圖譜視覺化</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Microsoft JhengHei', 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
        }}
        .header {{
            background: rgba(255,255,255,0.1);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
        }}
        .header h1 {{
            color: #fff;
            font-size: 1.5em;
            font-weight: 600;
        }}
        .controls {{
            display: flex;
            gap: 10px;
        }}
        .controls button {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }}
        .btn-primary {{ background: #3498db; color: #fff; }}
        .btn-primary:hover {{ background: #2980b9; }}
        .btn-secondary {{ background: #95a5a6; color: #fff; }}
        .btn-secondary:hover {{ background: #7f8c8d; }}

        .main-container {{
            display: flex;
            height: calc(100vh - 70px);
        }}

        .legend {{
            width: 220px;
            background: rgba(255,255,255,0.95);
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 10px rgba(0,0,0,0.2);
        }}
        .legend h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.1em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
            border: 2px solid rgba(0,0,0,0.2);
        }}
        .legend-section {{
            margin-top: 20px;
        }}
        .legend-section h4 {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-bottom: 8px;
        }}

        #network {{
            flex: 1;
            background: #f8f9fa;
        }}

        .stats {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(255,255,255,0.95);
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            font-size: 0.9em;
        }}
        .stats span {{
            margin-right: 20px;
            color: #2c3e50;
        }}
        .stats strong {{ color: #3498db; }}

        /* 搜索框樣式 */
        .search-box {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .search-box input {{
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            width: 250px;
            font-size: 0.9em;
            background: rgba(255,255,255,0.9);
        }}
        .search-box input:focus {{
            outline: 2px solid #3498db;
        }}
        .btn-search {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            background: #27ae60;
            color: #fff;
            transition: all 0.3s;
        }}
        .btn-search:hover {{ background: #219a52; }}
        .btn-clear {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            background: #e74c3c;
            color: #fff;
            transition: all 0.3s;
        }}
        .btn-clear:hover {{ background: #c0392b; }}
        .search-result {{
            color: #ecf0f1;
            font-size: 0.85em;
            margin-left: 10px;
        }}

        /* 高亮節點樣式（通過 JS 動態修改） */
        .highlighted {{
            border: 3px solid #f1c40f !important;
        }}

        /* vis.js tooltip 樣式覆蓋 */
        div.vis-tooltip {{
            background-color: rgba(50, 50, 50, 0.95) !important;
            color: #fff !important;
            border: 1px solid #555 !important;
            border-radius: 6px !important;
            padding: 10px 14px !important;
            font-size: 13px !important;
            line-height: 1.6 !important;
            white-space: pre-line !important;
            max-width: 400px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
            font-family: 'Microsoft JhengHei', 'Segoe UI', sans-serif !important;
        }}

        /* 節點詳細資訊模態框 */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .modal-overlay.active {{
            display: flex;
        }}
        .modal-content {{
            background: #fff;
            border-radius: 12px;
            max-width: 600px;
            max-height: 80vh;
            width: 90%;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            animation: modalSlideIn 0.3s ease;
        }}
        @keyframes modalSlideIn {{
            from {{ transform: translateY(-20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        .modal-header {{
            padding: 20px 25px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .modal-header h2 {{
            font-size: 1.3em;
            color: #2c3e50;
            margin: 0;
        }}
        .modal-type-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            color: #fff;
            margin-left: 10px;
        }}
        .modal-close {{
            background: none;
            border: none;
            font-size: 1.8em;
            cursor: pointer;
            color: #95a5a6;
            transition: color 0.2s;
            line-height: 1;
        }}
        .modal-close:hover {{
            color: #e74c3c;
        }}
        .modal-body {{
            padding: 25px;
            overflow-y: auto;
            max-height: calc(80vh - 80px);
        }}
        .modal-field {{
            margin-bottom: 18px;
        }}
        .modal-field-label {{
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        .modal-field-value {{
            font-size: 1em;
            color: #2c3e50;
            line-height: 1.6;
            background: #f8f9fa;
            padding: 12px 15px;
            border-radius: 6px;
            border-left: 3px solid #3498db;
        }}
        .modal-field-value.description {{
            white-space: pre-wrap;
            word-break: break-word;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>職能基準知識圖譜視覺化</h1>
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="搜索節點 (輸入名稱或編號)..." onkeyup="handleSearch(event)">
            <button class="btn-search" onclick="searchNodes()">搜索</button>
            <button class="btn-clear" onclick="clearSearch()">清除</button>
            <span id="searchResult" class="search-result"></span>
        </div>
        <div class="controls">
            <button class="btn-primary" onclick="network.fit()">適應視窗</button>
            <button class="btn-secondary" onclick="togglePhysics()">切換物理引擎</button>
            <button class="btn-secondary" onclick="exportPNG()">匯出圖片</button>
        </div>
    </div>

    <div class="main-container">
        <div class="legend">
            <h3>節點類型圖例</h3>
            <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div>職能基準</div>
            <div class="legend-item"><div class="legend-color" style="background:#e67e22"></div>職類別</div>
            <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div>職業別</div>
            <div class="legend-item"><div class="legend-color" style="background:#1abc9c"></div>行業別</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498db"></div>主要職責</div>
            <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div>工作任務</div>
            <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div>知識</div>
            <div class="legend-item"><div class="legend-color" style="background:#f1c40f"></div>技能</div>
            <div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div>態度</div>
            <div class="legend-item"><div class="legend-color" style="background:#5dade2"></div>行為指標</div>
            <div class="legend-item"><div class="legend-color" style="background:#eb984e"></div>工作產出</div>

            <div class="legend-section">
                <h4>操作說明</h4>
                <p style="font-size:0.85em; color:#666; line-height:1.6">
                    • 滾輪縮放<br>
                    • 拖曳移動畫布<br>
                    • 懸停查看簡要資訊<br>
                    • <strong>雙擊節點查看完整內容</strong>
                </p>
            </div>
        </div>

        <div id="network"></div>
    </div>

    <div class="stats">
        <span>節點數: <strong id="nodeCount">0</strong></span>
        <span>連結數: <strong id="edgeCount">0</strong></span>
    </div>

    <!-- 節點詳細資訊模態框 -->
    <div id="nodeModal" class="modal-overlay" onclick="closeModalOnOverlay(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-header">
                <div>
                    <h2 id="modalTitle">節點名稱</h2>
                </div>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body" id="modalBody">
                <!-- 動態內容 -->
            </div>
        </div>
    </div>

    <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        document.getElementById('nodeCount').textContent = nodes.length;
        document.getElementById('edgeCount').textContent = edges.length;

        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};

        var options = {{
            physics: {{
                enabled: true,
                hierarchicalRepulsion: {{
                    centralGravity: 0.0,
                    springLength: 250,
                    springConstant: 0.01,
                    nodeDistance: 220,
                    damping: 0.09,
                    avoidOverlap: 0.8
                }},
                solver: 'hierarchicalRepulsion',
                stabilization: {{
                    enabled: true,
                    iterations: 300,
                    updateInterval: 25
                }}
            }},
            layout: {{
                hierarchical: {{
                    enabled: true,
                    levelSeparation: 320,
                    nodeSpacing: 200,
                    treeSpacing: 280,
                    direction: 'LR',
                    sortMethod: 'directed',
                    shakeTowards: 'leaves'
                }}
            }},
            nodes: {{
                shape: 'box',
                font: {{
                    size: 12,
                    face: 'Microsoft JhengHei, Arial',
                    color: '#ffffff',
                    multi: true
                }},
                shadow: {{
                    enabled: true,
                    color: 'rgba(0,0,0,0.2)',
                    size: 6,
                    x: 2,
                    y: 2
                }},
                margin: {{
                    top: 10,
                    bottom: 10,
                    left: 12,
                    right: 12
                }},
                widthConstraint: {{
                    minimum: 80,
                    maximum: 200
                }}
            }},
            edges: {{
                smooth: {{
                    enabled: true,
                    type: 'cubicBezier',
                    roundness: 0.5
                }},
                font: {{
                    size: 10,
                    align: 'middle'
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                hideEdgesOnDrag: true,
                navigationButtons: true,
                keyboard: {{ enabled: true }}
            }}
        }};

        var network = new vis.Network(container, data, options);
        var physicsEnabled = true;

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}

        function exportPNG() {{
            var canvas = container.getElementsByTagName('canvas')[0];
            var link = document.createElement('a');
            link.download = 'knowledge_graph.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                console.log('Selected:', node);
            }}
        }});

        // 雙擊事件 - 顯示節點詳細資訊
        network.on('doubleClick', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                showNodeDetail(node);
            }}
        }});

        // 顯示節點詳細資訊模態框
        function showNodeDetail(node) {{
            var modal = document.getElementById('nodeModal');
            var titleEl = document.getElementById('modalTitle');
            var bodyEl = document.getElementById('modalBody');

            var data = node.fullData || {{}};
            var nodeType = data.nodeType || '未知';
            var bgColor = data.bgColor || '#3498db';

            // 設定標題
            titleEl.innerHTML = (data.name || node.label || '未命名') +
                '<span class="modal-type-badge" style="background:' + bgColor + '">' + nodeType + '</span>';

            // 建立內容
            var bodyHtml = '';

            if (data.code) {{
                bodyHtml += '<div class="modal-field">' +
                    '<div class="modal-field-label">代碼</div>' +
                    '<div class="modal-field-value">' + data.code + '</div>' +
                    '</div>';
            }}

            if (data.name && data.name !== node.label) {{
                bodyHtml += '<div class="modal-field">' +
                    '<div class="modal-field-label">完整名稱</div>' +
                    '<div class="modal-field-value">' + data.name + '</div>' +
                    '</div>';
            }}

            if (data.level) {{
                bodyHtml += '<div class="modal-field">' +
                    '<div class="modal-field-label">職能級別</div>' +
                    '<div class="modal-field-value">' + data.level + '</div>' +
                    '</div>';
            }}

            if (data.description) {{
                bodyHtml += '<div class="modal-field">' +
                    '<div class="modal-field-label">描述 / 說明</div>' +
                    '<div class="modal-field-value description">' + data.description + '</div>' +
                    '</div>';
            }}

            if (!bodyHtml) {{
                bodyHtml = '<div class="modal-field"><div class="modal-field-value">沒有更多詳細資訊</div></div>';
            }}

            bodyEl.innerHTML = bodyHtml;
            modal.classList.add('active');
        }}

        // 關閉模態框
        function closeModal() {{
            document.getElementById('nodeModal').classList.remove('active');
        }}

        // 點擊遮罩關閉
        function closeModalOnOverlay(event) {{
            if (event.target.id === 'nodeModal') {{
                closeModal();
            }}
        }}

        // ESC 鍵關閉模態框
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});

        network.once('stabilizationIterationsDone', function() {{
            network.setOptions({{ physics: {{ enabled: false }} }});
            physicsEnabled = false;
        }});

        // ===== 搜索和高亮功能 =====
        var originalColors = {{}};  // 儲存原始顏色
        var highlightedNodes = [];  // 當前高亮的節點
        var searchIndex = 0;        // 搜索結果索引

        function handleSearch(event) {{
            if (event.key === 'Enter') {{
                searchNodes();
            }}
        }}

        function searchNodes() {{
            var query = document.getElementById('searchInput').value.trim().toLowerCase();
            if (!query) {{
                document.getElementById('searchResult').textContent = '請輸入搜索關鍵字';
                return;
            }}

            // 恢復之前高亮的節點
            clearHighlight();

            // 搜索匹配的節點
            var allNodes = nodes.get();
            highlightedNodes = allNodes.filter(function(node) {{
                var label = (node.label || '').toLowerCase();
                var id = (node.id || '').toLowerCase();
                var title = (node.title || '').toLowerCase();
                return label.includes(query) || id.includes(query) || title.includes(query);
            }});

            if (highlightedNodes.length === 0) {{
                document.getElementById('searchResult').textContent = '找不到匹配的節點';
                return;
            }}

            // 高亮所有匹配的節點
            var updates = [];
            highlightedNodes.forEach(function(node) {{
                originalColors[node.id] = {{
                    color: node.color,
                    borderWidth: node.borderWidth || 1,
                    size: node.size || 25
                }};
                updates.push({{
                    id: node.id,
                    borderWidth: 4,
                    color: {{
                        background: node.color ? (node.color.background || node.color) : '#f1c40f',
                        border: '#f1c40f',
                        highlight: {{ background: '#f39c12', border: '#e67e22' }}
                    }},
                    size: (node.size || 25) * 1.3
                }});
            }});
            nodes.update(updates);

            // 聚焦到第一個匹配節點
            searchIndex = 0;
            focusOnNode(highlightedNodes[0].id);

            document.getElementById('searchResult').textContent =
                '找到 ' + highlightedNodes.length + ' 個匹配 (按搜索跳到下一個)';
        }}

        function focusOnNode(nodeId) {{
            network.focus(nodeId, {{
                scale: 1.2,
                animation: {{
                    duration: 500,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
            network.selectNodes([nodeId]);
        }}

        function clearSearch() {{
            document.getElementById('searchInput').value = '';
            document.getElementById('searchResult').textContent = '';
            clearHighlight();
            network.unselectAll();
        }}

        function clearHighlight() {{
            // 恢復原始樣式
            var updates = [];
            Object.keys(originalColors).forEach(function(nodeId) {{
                var orig = originalColors[nodeId];
                updates.push({{
                    id: nodeId,
                    borderWidth: orig.borderWidth,
                    color: orig.color,
                    size: orig.size
                }});
            }});
            if (updates.length > 0) {{
                nodes.update(updates);
            }}
            originalColors = {{}};
            highlightedNodes = [];
            searchIndex = 0;
        }}

        // 連續按搜索按鈕跳到下一個匹配節點
        document.querySelector('.btn-search').addEventListener('click', function(e) {{
            if (highlightedNodes.length > 1) {{
                var currentQuery = document.getElementById('searchInput').value.trim().toLowerCase();
                var allNodes = nodes.get();
                var matchingNow = allNodes.filter(function(node) {{
                    var label = (node.label || '').toLowerCase();
                    var id = (node.id || '').toLowerCase();
                    return label.includes(currentQuery) || id.includes(currentQuery);
                }});

                if (matchingNow.length === highlightedNodes.length) {{
                    // 同一個搜索，跳到下一個
                    searchIndex = (searchIndex + 1) % highlightedNodes.length;
                    focusOnNode(highlightedNodes[searchIndex].id);
                    document.getElementById('searchResult').textContent =
                        '第 ' + (searchIndex + 1) + '/' + highlightedNodes.length + ' 個匹配';
                }}
            }}
        }});
    </script>
</body>
</html>'''
        return html


class QueryWorker(QThread):
    """查詢背景執行緒"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, float)  # QueryResult, elapsed_time

    def __init__(self, engine: GraphRAGQueryEngine, query_type: str, **kwargs):
        super().__init__()
        self.engine = engine
        self.query_type = query_type
        self.kwargs = kwargs

    def run(self):
        start_time = time.time()
        try:
            self.progress.emit(f"執行 {self.query_type} 查詢...")

            if self.query_type == "跨職業比較":
                result = self.engine.compare_occupations(
                    self.kwargs.get("occupation1", ""),
                    self.kwargs.get("occupation2", "")
                )
            elif self.query_type == "職涯路徑":
                result = self.engine.find_career_path(
                    self.kwargs.get("from_occupation", ""),
                    self.kwargs.get("to_occupation", "")
                )
            elif self.query_type == "能力反查":
                result = self.engine.find_occupations_by_ability(
                    self.kwargs.get("ability", "")
                )
            elif self.query_type == "聚合統計":
                result = self.engine.get_top_abilities(
                    top_k=self.kwargs.get("top_k", 10),
                    ability_type=self.kwargs.get("ability_type", "all")
                )
            elif self.query_type == "聯邦語義搜尋":
                result = self.engine.federated_semantic_search(
                    self.kwargs.get("query", ""),
                    top_k=self.kwargs.get("top_k", 5),
                    expand_depth=self.kwargs.get("expand_depth", 1),
                    top_k_categories=self.kwargs.get("top_k_categories", 3),
                    top_k_occupations=self.kwargs.get("top_k_occupations", 5),
                    use_occupation_routing=self.kwargs.get("use_occupation_routing", True)
                )
            else:  # 語義搜尋
                result = self.engine.semantic_search(
                    self.kwargs.get("query", ""),
                    top_k=self.kwargs.get("top_k", 5),
                    expand_depth=self.kwargs.get("expand_depth", 1)
                )

            elapsed = time.time() - start_time
            self.finished.emit(result, elapsed)

        except Exception as e:
            elapsed = time.time() - start_time
            error_result = QueryResult(
                query=str(self.kwargs),
                query_type=self.query_type,
                answer=f"查詢失敗: {str(e)}"
            )
            self.finished.emit(error_result, elapsed)


# =============================
# 彈出視窗
# =============================

class ResultDialog(QDialog):
    """查詢結果詳細視窗"""
    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 900, 700)

        layout = QVBoxLayout(self)

        # 標題
        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft JhengHei", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2C3E50; padding: 10px;")
        layout.addWidget(title_label)

        # 內容
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMarkdown(content)
        text_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                padding: 15px;
                background-color: white;
                font-size: 11pt;
                line-height: 1.6;
            }
        """)
        layout.addWidget(text_edit)

        # 關閉按鈕
        close_btn = QPushButton("關閉")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 30px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)


class GraphStatsDialog(QDialog):
    """圖譜統計視窗（階層式結構）"""
    def __init__(self, stats: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("知識圖譜統計")
        self.setGeometry(300, 200, 600, 550)

        layout = QVBoxLayout(self)

        # 標題
        title = QLabel("知識圖譜統計資訊（階層式結構）")
        title.setFont(QFont("Microsoft JhengHei", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #2C3E50; padding: 10px;")
        layout.addWidget(title)

        # 統計內容
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)

        content = "## 基本統計\n\n"
        content += f"- **總節點數**: {stats.get('節點總數', 0):,}\n"
        content += f"- **總邊數**: {stats.get('邊總數', 0):,}\n\n"

        content += "## 全域索引統計\n\n"
        content += f"- **全域知識數**: {stats.get('全域知識數', 0):,}\n"
        content += f"- **全域技能數**: {stats.get('全域技能數', 0):,}\n"
        content += f"- **全域態度數**: {stats.get('全域態度數', 0):,}\n"
        content += f"- **職類別數**: {stats.get('職類別數', 0):,}\n"
        content += f"- **職業別數**: {stats.get('職業別數', 0):,}\n"
        content += f"- **行業別數**: {stats.get('行業別數', 0):,}\n\n"

        content += "## 節點類型分布\n\n"
        node_type_stats = stats.get('節點類型統計', {})
        for node_type, count in sorted(node_type_stats.items(), key=lambda x: x[1], reverse=True):
            content += f"- {node_type}: {count:,}\n"

        content += "\n## 邊類型分布\n\n"
        edge_type_stats = stats.get('邊類型統計', {})
        for edge_type, count in sorted(edge_type_stats.items(), key=lambda x: x[1], reverse=True):
            content += f"- {edge_type}: {count:,}\n"

        stats_text.setMarkdown(content)
        stats_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
                font-size: 10pt;
            }
        """)
        layout.addWidget(stats_text)

        # 關閉按鈕
        close_btn = QPushButton("關閉")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 25px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)


# =============================
# 主視窗
# =============================

class GraphRAGMainWindow(QMainWindow):
    """Graph RAG 主視窗"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("職能基準知識圖譜 Graph RAG 系統")
        self.setGeometry(50, 50, 1500, 900)

        # 核心物件
        self.kg: Optional[CompetencyKnowledgeGraph] = None
        self.engine: Optional[GraphRAGQueryEngine] = None
        self.search_history: List[Dict] = []
        self.result_dialogs: List[QDialog] = []

        # 設置樣式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
            }
        """)

        # 建立 UI
        self.init_ui()

        # 嘗試載入現有圖譜
        QTimer.singleShot(500, self.try_load_existing_graph)

    def init_ui(self):
        """初始化 UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 標題
        title = QLabel("職能基準知識圖譜 Graph RAG 系統")
        title.setFont(QFont("Microsoft JhengHei", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #2C3E50; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # 主分割區域
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左側面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # 右側面板
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([450, 1050])
        main_layout.addWidget(splitter)

        # 狀態列
        self.statusBar().showMessage("就緒")

    def create_left_panel(self) -> QWidget:
        """建立左側面板（系統管理）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 8)

        # 通用 GroupBox 樣式
        def get_group_style(color):
            return f"""
                QGroupBox {{
                    font-size: 10pt;
                    font-weight: bold;
                    border: 2px solid {color};
                    border-radius: 6px;
                    margin-top: 12px;
                    padding: 5px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 8px;
                    padding: 0 3px;
                    background-color: white;
                }}
            """

        # ===== 系統狀態 =====
        status_group = QGroupBox("系統狀態")
        status_group.setStyleSheet(get_group_style("#4CAF50") + "QGroupBox { background-color: #F1F8E9; }")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(4)
        status_layout.setContentsMargins(8, 8, 8, 8)

        self.graph_status_label = QLabel("圖譜狀態: 未載入")
        self.graph_status_label.setStyleSheet("font-size: 9pt;")
        status_layout.addWidget(self.graph_status_label)

        self.embedding_status_label = QLabel("Embedding: 未初始化")
        self.embedding_status_label.setStyleSheet("font-size: 9pt;")
        status_layout.addWidget(self.embedding_status_label)

        self.llm_status_label = QLabel("LLM: 未初始化")
        self.llm_status_label.setStyleSheet("font-size: 9pt;")
        status_layout.addWidget(self.llm_status_label)

        self.pdf_count_label = QLabel("PDF 檔案: 0 個")
        self.pdf_count_label.setStyleSheet("font-size: 9pt;")
        status_layout.addWidget(self.pdf_count_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # ===== 資料管理 =====
        data_group = QGroupBox("資料管理")
        data_group.setStyleSheet(get_group_style("#2196F3"))
        data_layout = QVBoxLayout()
        data_layout.setSpacing(8)
        data_layout.setContentsMargins(8, 8, 8, 8)

        # PDF 解析按鈕
        self.parse_pdf_btn = QPushButton("解析 PDF")
        self.parse_pdf_btn.setFixedHeight(32)
        self.parse_pdf_btn.clicked.connect(self.parse_pdfs)
        self.parse_pdf_btn.setStyleSheet(self._get_button_style("#FF9800", "#F57C00"))
        data_layout.addWidget(self.parse_pdf_btn)

        # 解析數量限制
        limit_layout = QHBoxLayout()
        limit_label = QLabel("數量:")
        limit_label.setStyleSheet("font-size: 9pt;")
        limit_layout.addWidget(limit_label)
        self.parse_limit_spin = QSpinBox()
        self.parse_limit_spin.setRange(0, 10000)
        self.parse_limit_spin.setValue(0)
        self.parse_limit_spin.setSpecialValueText("全部")
        self.parse_limit_spin.setFixedHeight(24)
        limit_layout.addWidget(self.parse_limit_spin)
        data_layout.addLayout(limit_layout)

        # 建構和載入放同一行
        graph_btns = QHBoxLayout()
        graph_btns.setSpacing(5)

        self.build_graph_btn = QPushButton("重建圖譜")
        self.build_graph_btn.setFixedHeight(32)
        self.build_graph_btn.clicked.connect(self.build_graph)
        self.build_graph_btn.setStyleSheet(self._get_button_style("#4CAF50", "#45A049"))
        self.build_graph_btn.setToolTip("從 JSON 重新建構知識圖譜（覆蓋現有圖譜）")
        graph_btns.addWidget(self.build_graph_btn)

        self.load_graph_btn = QPushButton("載入圖譜")
        self.load_graph_btn.setFixedHeight(32)
        self.load_graph_btn.clicked.connect(self.load_graph)
        self.load_graph_btn.setStyleSheet(self._get_button_style("#2196F3", "#1976D2"))
        self.load_graph_btn.setToolTip("從檔案載入已儲存的知識圖譜")
        graph_btns.addWidget(self.load_graph_btn)

        self.update_db_btn = QPushButton("資料統計")
        self.update_db_btn.setFixedHeight(32)
        self.update_db_btn.clicked.connect(self.update_database)
        self.update_db_btn.setStyleSheet(self._get_button_style("#FF9800", "#F57C00"))
        self.update_db_btn.setToolTip("顯示 JSON 資料庫統計資訊")
        graph_btns.addWidget(self.update_db_btn)

        data_layout.addLayout(graph_btns)

        # 初始化 Embedding 和 LLM 放同一行
        init_btns = QHBoxLayout()
        init_btns.setSpacing(5)

        self.init_embedding_btn = QPushButton("更新 Embedding")
        self.init_embedding_btn.setFixedHeight(32)
        self.init_embedding_btn.clicked.connect(self.init_embeddings)
        self.init_embedding_btn.setStyleSheet(self._get_button_style("#9C27B0", "#7B1FA2"))
        self.init_embedding_btn.setToolTip("重新建立向量索引（圖譜有更新時使用）")
        init_btns.addWidget(self.init_embedding_btn)

        self.init_llm_btn = QPushButton("載入 LLM")
        self.init_llm_btn.setFixedHeight(32)
        self.init_llm_btn.clicked.connect(self.init_llm)
        self.init_llm_btn.setStyleSheet(self._get_button_style("#FF5722", "#E64A19"))
        self.init_llm_btn.setToolTip("載入 LLM 語言模型（TAIDE）以啟用 AI 回答")
        init_btns.addWidget(self.init_llm_btn)

        data_layout.addLayout(init_btns)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # ===== 視覺化 =====
        viz_group = QGroupBox("視覺化")
        viz_group.setStyleSheet(get_group_style("#E91E63"))
        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(8)
        viz_layout.setContentsMargins(8, 8, 8, 8)

        # 統計和視覺化按鈕放同一行
        viz_btns = QHBoxLayout()
        viz_btns.setSpacing(5)

        self.stats_btn = QPushButton("統計資訊")
        self.stats_btn.setFixedHeight(32)
        self.stats_btn.clicked.connect(self.show_graph_stats)
        self.stats_btn.setStyleSheet(self._get_button_style("#607D8B", "#546E7A"))
        self.stats_btn.setEnabled(False)
        viz_btns.addWidget(self.stats_btn)

        self.visualize_btn = QPushButton("視覺化")
        self.visualize_btn.setFixedHeight(32)
        self.visualize_btn.clicked.connect(self.visualize_graph)
        self.visualize_btn.setStyleSheet(self._get_button_style("#E91E63", "#C2185B"))
        self.visualize_btn.setEnabled(False)
        viz_btns.addWidget(self.visualize_btn)

        # 社群視覺化按鈕
        self.community_viz_btn = QPushButton("社群視覺化")
        self.community_viz_btn.setFixedHeight(32)
        self.community_viz_btn.clicked.connect(self.visualize_community)
        self.community_viz_btn.setStyleSheet(self._get_button_style("#9C27B0", "#7B1FA2"))
        self.community_viz_btn.setEnabled(False)
        viz_btns.addWidget(self.community_viz_btn)

        viz_layout.addLayout(viz_btns)

        # 視覺化控制 - 深度和節點上限在同一行
        control_row = QHBoxLayout()
        control_row.setSpacing(10)

        depth_label = QLabel("深度:")
        depth_label.setStyleSheet("font-size: 9pt;")
        control_row.addWidget(depth_label)
        self.viz_depth_spin = QSpinBox()
        self.viz_depth_spin.setRange(1, 5)
        self.viz_depth_spin.setValue(2)
        self.viz_depth_spin.setFixedHeight(24)
        self.viz_depth_spin.setFixedWidth(50)
        control_row.addWidget(self.viz_depth_spin)

        viz_limit_label = QLabel("上限:")
        viz_limit_label.setStyleSheet("font-size: 9pt;")
        control_row.addWidget(viz_limit_label)
        self.viz_node_limit_spin = QSpinBox()
        self.viz_node_limit_spin.setRange(10, 500)
        self.viz_node_limit_spin.setValue(100)
        self.viz_node_limit_spin.setSingleStep(10)
        self.viz_node_limit_spin.setFixedHeight(24)
        self.viz_node_limit_spin.setFixedWidth(60)
        control_row.addWidget(self.viz_node_limit_spin)
        control_row.addStretch()

        viz_layout.addLayout(control_row)

        # 節點類型過濾 - 更緊湊的排列
        type_label = QLabel("節點類型:")
        type_label.setStyleSheet("font-size: 9pt;")
        viz_layout.addWidget(type_label)

        self.viz_type_checks = {}
        type_grid = QHBoxLayout()
        type_grid.setSpacing(3)

        col1 = QVBoxLayout()
        col1.setSpacing(0)
        for t in ["職能基準", "主要職責", "工作任務"]:
            cb = QCheckBox(t)
            cb.setChecked(True)
            cb.setStyleSheet("font-size: 8pt;")
            self.viz_type_checks[t] = cb
            col1.addWidget(cb)
        type_grid.addLayout(col1)

        col2 = QVBoxLayout()
        col2.setSpacing(0)
        for t in ["知識", "技能", "工作產出"]:
            cb = QCheckBox(t)
            cb.setChecked(t in ["知識", "技能"])
            cb.setStyleSheet("font-size: 8pt;")
            self.viz_type_checks[t] = cb
            col2.addWidget(cb)
        type_grid.addLayout(col2)

        viz_layout.addLayout(type_grid)

        # 全選/取消按鈕
        select_btns = QHBoxLayout()
        select_btns.setSpacing(5)
        select_all_btn = QPushButton("全選")
        select_all_btn.setFixedHeight(24)
        select_all_btn.clicked.connect(lambda: self._set_all_viz_types(True))
        select_none_btn = QPushButton("取消")
        select_none_btn.setFixedHeight(24)
        select_none_btn.clicked.connect(lambda: self._set_all_viz_types(False))
        select_btns.addWidget(select_all_btn)
        select_btns.addWidget(select_none_btn)
        viz_layout.addLayout(select_btns)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # ===== 職能基準列表 =====
        list_group = QGroupBox("職能基準列表")
        list_group.setStyleSheet(get_group_style("#795548"))
        list_layout = QVBoxLayout()
        list_layout.setContentsMargins(8, 8, 8, 8)

        # 搜索過濾框
        filter_layout = QHBoxLayout()
        self.standard_filter_input = QLineEdit()
        self.standard_filter_input.setPlaceholderText("輸入關鍵字過濾...")
        self.standard_filter_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 9pt;
            }
            QLineEdit:focus {
                border-color: #795548;
            }
        """)
        self.standard_filter_input.textChanged.connect(self.filter_standard_list)
        filter_layout.addWidget(self.standard_filter_input)

        clear_filter_btn = QPushButton("✕")
        clear_filter_btn.setFixedWidth(28)
        clear_filter_btn.setToolTip("清除過濾")
        clear_filter_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        clear_filter_btn.clicked.connect(lambda: self.standard_filter_input.clear())
        filter_layout.addWidget(clear_filter_btn)
        list_layout.addLayout(filter_layout)

        # 計數標籤
        self.standard_count_label = QLabel("共 0 項")
        self.standard_count_label.setStyleSheet("font-size: 8pt; color: #888;")
        list_layout.addWidget(self.standard_count_label)

        self.standard_list = QListWidget()
        self.standard_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-size: 9pt;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
        """)
        self.standard_list.itemDoubleClicked.connect(self.on_standard_double_clicked)
        list_layout.addWidget(self.standard_list)

        # 儲存所有項目用於過濾 (name, std_id)
        self.all_standard_items: List[tuple] = []

        list_group.setLayout(list_layout)
        layout.addWidget(list_group, 1)  # stretch factor 讓列表可以擴展

        # ===== 進度條 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        layout.addWidget(self.progress_bar)

        return panel

    def create_right_panel(self) -> QWidget:
        """建立右側面板（查詢介面）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ===== 查詢模式選擇 =====
        mode_group = QGroupBox("查詢模式")
        mode_layout = QVBoxLayout()

        # 使用 Tab 來區分不同查詢模式
        self.query_tabs = QTabWidget()
        self.query_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #F5F5F5;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #4A90E2;
                color: white;
            }
        """)

        # Tab 1: 跨職業比較
        compare_tab = self.create_compare_tab()
        self.query_tabs.addTab(compare_tab, "跨職業比較")

        # Tab 2: 職涯路徑
        career_tab = self.create_career_tab()
        self.query_tabs.addTab(career_tab, "職涯路徑")

        # Tab 3: 能力反查
        ability_tab = self.create_ability_tab()
        self.query_tabs.addTab(ability_tab, "能力反查")

        # Tab 4: 聚合統計
        stats_tab = self.create_stats_tab()
        self.query_tabs.addTab(stats_tab, "聚合統計")

        # Tab 5: 語義搜尋
        semantic_tab = self.create_semantic_tab()
        self.query_tabs.addTab(semantic_tab, "語義搜尋")

        mode_layout.addWidget(self.query_tabs)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # ===== 查詢結果 =====
        result_group = QGroupBox("查詢結果")
        result_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4A90E2;
                background-color: #E3F2FD;
            }
        """)
        result_layout = QVBoxLayout()

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                padding: 15px;
                background-color: white;
                font-size: 11pt;
                line-height: 1.6;
            }
        """)
        self.result_display.setMinimumHeight(300)
        self.result_display.mouseDoubleClickEvent = lambda e: self.show_result_popup()
        result_layout.addWidget(self.result_display)

        # 結果操作列（複製 / 清除 + 查詢時間）
        result_action_row = QHBoxLayout()

        copy_btn = QPushButton("複製結果")
        copy_btn.setFixedHeight(26)
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #546E7A; }
        """)
        copy_btn.clicked.connect(self._copy_result)
        result_action_row.addWidget(copy_btn)

        clear_btn = QPushButton("清除")
        clear_btn.setFixedHeight(26)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #BDBDBD;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #9E9E9E; }
        """)
        clear_btn.clicked.connect(self.result_display.clear)
        result_action_row.addWidget(clear_btn)

        result_action_row.addStretch()

        self.query_time_label = QLabel("")
        self.query_time_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 9pt;
                padding: 4px;
            }
        """)
        self.query_time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        result_action_row.addWidget(self.query_time_label)

        result_layout.addLayout(result_action_row)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        return panel

    def create_compare_tab(self) -> QWidget:
        """建立跨職業比較 Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("職業 1:"))
        self.compare_occ1_input = QLineEdit()
        self.compare_occ1_input.setPlaceholderText("例如: 烘焙助理")
        layout.addWidget(self.compare_occ1_input)

        layout.addWidget(QLabel("職業 2:"))
        self.compare_occ2_input = QLineEdit()
        self.compare_occ2_input.setPlaceholderText("例如: 餐飲服務人員")
        layout.addWidget(self.compare_occ2_input)

        compare_btn = QPushButton("比較職業")
        compare_btn.clicked.connect(self.do_compare_query)
        compare_btn.setStyleSheet(self._get_button_style("#4A90E2", "#357ABD"))
        layout.addWidget(compare_btn)

        layout.addStretch()
        return tab

    def create_career_tab(self) -> QWidget:
        """建立職涯路徑 Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("起點職業:"))
        self.career_from_input = QLineEdit()
        self.career_from_input.setPlaceholderText("例如: 餐飲服務人員")
        layout.addWidget(self.career_from_input)

        layout.addWidget(QLabel("目標職業:"))
        self.career_to_input = QLineEdit()
        self.career_to_input.setPlaceholderText("例如: 餐廳主管")
        layout.addWidget(self.career_to_input)

        career_btn = QPushButton("規劃職涯路徑")
        career_btn.clicked.connect(self.do_career_query)
        career_btn.setStyleSheet(self._get_button_style("#4A90E2", "#357ABD"))
        layout.addWidget(career_btn)

        layout.addStretch()
        return tab

    def create_ability_tab(self) -> QWidget:
        """建立能力反查 Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("知識/技能關鍵字:"))
        self.ability_input = QLineEdit()
        self.ability_input.setPlaceholderText("例如: 食品安全")
        layout.addWidget(self.ability_input)

        ability_btn = QPushButton("搜尋適合職業")
        ability_btn.clicked.connect(self.do_ability_query)
        ability_btn.setStyleSheet(self._get_button_style("#4A90E2", "#357ABD"))
        layout.addWidget(ability_btn)

        layout.addStretch()
        return tab

    def create_stats_tab(self) -> QWidget:
        """建立聚合統計 Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("能力類型:"))
        self.stats_type_combo = QComboBox()
        self.stats_type_combo.addItems(["全部", "知識", "技能"])
        layout.addWidget(self.stats_type_combo)

        layout.addWidget(QLabel("顯示數量:"))
        self.stats_topk_spin = QSpinBox()
        self.stats_topk_spin.setRange(5, 50)
        self.stats_topk_spin.setValue(10)
        layout.addWidget(self.stats_topk_spin)

        stats_btn = QPushButton("查詢統計排名")
        stats_btn.clicked.connect(self.do_stats_query)
        stats_btn.setStyleSheet(self._get_button_style("#4A90E2", "#357ABD"))
        layout.addWidget(stats_btn)

        layout.addStretch()
        return tab

    def create_semantic_tab(self) -> QWidget:
        """建立語義搜尋 Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("自然語言查詢:"))
        self.semantic_input = QLineEdit()
        self.semantic_input.setPlaceholderText("例如: 如何成為烘焙師傅？")
        self.semantic_input.returnPressed.connect(self.do_semantic_query)
        layout.addWidget(self.semantic_input)

        # 進階設定
        adv_layout = QHBoxLayout()
        adv_layout.addWidget(QLabel("檢索數量:"))
        self.semantic_topk_spin = QSpinBox()
        self.semantic_topk_spin.setRange(1, 20)
        self.semantic_topk_spin.setValue(5)
        adv_layout.addWidget(self.semantic_topk_spin)

        adv_layout.addWidget(QLabel("擴展深度:"))
        self.semantic_depth_spin = QSpinBox()
        self.semantic_depth_spin.setRange(0, 3)
        self.semantic_depth_spin.setValue(1)
        adv_layout.addWidget(self.semantic_depth_spin)
        layout.addLayout(adv_layout)

        # 聯邦搜索選項
        fed_layout = QHBoxLayout()
        self.federated_checkbox = QCheckBox("啟用聯邦搜索")
        self.federated_checkbox.setToolTip(
            "啟用後，系統會先判斷查詢與哪些通俗職業分類/職類別相關，\n"
            "再在相關分類中進行搜索，提高搜索效率和相關性。"
        )
        fed_layout.addWidget(self.federated_checkbox)

        self.occupation_routing_checkbox = QCheckBox("使用通俗職業分類路由")
        self.occupation_routing_checkbox.setChecked(True)
        self.occupation_routing_checkbox.setEnabled(False)  # 預設灰化，需先啟用聯邦搜索
        self.occupation_routing_checkbox.setToolTip(
            "通俗職業分類（如：餐飲、資訊科技）比職類別更精細，\n"
            "可以提供更精確的搜索結果。\n"
            "（需先啟用聯邦搜索）"
        )
        fed_layout.addWidget(self.occupation_routing_checkbox)

        fed_layout.addStretch()
        layout.addLayout(fed_layout)

        # 聯邦搜索參數
        fed_param_layout = QHBoxLayout()
        fed_param_layout.addWidget(QLabel("職類別數:"))
        self.federated_topk_spin = QSpinBox()
        self.federated_topk_spin.setRange(1, 10)
        self.federated_topk_spin.setValue(3)
        self.federated_topk_spin.setToolTip("搜索最相關的 N 個職類別")
        fed_param_layout.addWidget(self.federated_topk_spin)

        fed_param_layout.addWidget(QLabel("通俗職業數:"))
        self.occupation_topk_spin = QSpinBox()
        self.occupation_topk_spin.setRange(1, 15)
        self.occupation_topk_spin.setValue(5)
        self.occupation_topk_spin.setToolTip("搜索最相關的 N 個通俗職業分類")
        fed_param_layout.addWidget(self.occupation_topk_spin)

        fed_param_layout.addStretch()
        layout.addLayout(fed_param_layout)

        # 聯邦搜索啟用狀態連動：灰化/啟用相關控件（在所有控件建立後才連接）
        def _on_federated_toggled(checked: bool):
            self.occupation_routing_checkbox.setEnabled(checked)
            self.federated_topk_spin.setEnabled(checked)
            self.occupation_topk_spin.setEnabled(checked)

        self.federated_checkbox.stateChanged.connect(
            lambda state: _on_federated_toggled(bool(state))
        )
        _on_federated_toggled(self.federated_checkbox.isChecked())  # 初始同步

        # 聯邦搜索狀態標籤
        self.federated_status_label = QLabel("聯邦搜索: 未初始化")
        self.federated_status_label.setStyleSheet("color: #888888; font-size: 9pt;")
        layout.addWidget(self.federated_status_label)

        # 按鈕列
        btn_layout = QHBoxLayout()

        semantic_btn = QPushButton("語義搜尋")
        semantic_btn.clicked.connect(self.do_semantic_query)
        semantic_btn.setStyleSheet(self._get_button_style("#4A90E2", "#357ABD"))
        btn_layout.addWidget(semantic_btn)

        init_federated_btn = QPushButton("初始化聯邦搜索")
        init_federated_btn.clicked.connect(self.init_federated_search)
        init_federated_btn.setStyleSheet(self._get_button_style("#27AE60", "#219A52"))
        btn_layout.addWidget(init_federated_btn)

        layout.addLayout(btn_layout)

        # 職類別資訊按鈕
        category_btn = QPushButton("查看職類別列表")
        category_btn.clicked.connect(self.show_category_list)
        category_btn.setStyleSheet(self._get_button_style("#9B59B6", "#8E44AD"))
        layout.addWidget(category_btn)

        layout.addStretch()
        return tab

    def _get_button_style(self, bg_color: str, hover_color: str) -> str:
        """取得按鈕樣式"""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 10pt;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
            }}
        """

    def _set_all_viz_types(self, checked: bool):
        """設定所有視覺化類型勾選框"""
        for cb in self.viz_type_checks.values():
            cb.setChecked(checked)

    def _get_selected_viz_types(self) -> list:
        """取得已勾選的視覺化類型"""
        return [t for t, cb in self.viz_type_checks.items() if cb.isChecked()]

    # =============================
    # 事件處理
    # =============================

    def try_load_existing_graph(self):
        """嘗試載入現有圖譜，並自動初始化 Embedding（若索引已存在）"""
        graph_path = config.GRAPH_DB_DIR / config.GRAPH_FILE
        if graph_path.exists():
            self.load_graph(str(graph_path))
            # 圖譜載入成功後，若向量索引已存在則自動靜默載入
            if self.engine:
                index_path = config.VECTORDB_DIR / "graph_rag_vectors" / "index.faiss"
                if index_path.exists():
                    self._auto_init_embeddings()

        # 更新 PDF 計數
        pdf_count = len(list(config.RAW_PDF_DIR.glob("*.pdf")))
        self.pdf_count_label.setText(f"PDF 檔案: {pdf_count} 個")

    def parse_pdfs(self):
        """解析 PDF 檔案（使用新版 v2 解析器）"""
        if not config.RAW_PDF_DIR.exists() or not list(config.RAW_PDF_DIR.glob("*.pdf")):
            QMessageBox.warning(self, "警告", f"PDF 目錄為空: {config.RAW_PDF_DIR}")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.parse_pdf_btn.setEnabled(False)

        limit = self.parse_limit_spin.value()
        # 使用新版解析器，輸出到 parsed_json_v2 目錄
        self.worker = PDFParseWorker(
            config.RAW_PDF_DIR,
            config.PARSED_JSON_V2_DIR,
            limit,
            use_v2=True
        )
        self.worker.progress.connect(self.on_parse_progress)
        self.worker.finished.connect(self.on_parse_finished)
        self.worker.start()

    def on_parse_progress(self, msg: str, current: int, total: int):
        """PDF 解析進度"""
        self.statusBar().showMessage(msg)
        self.progress_bar.setValue(int(current / total * 100))

    def on_parse_finished(self, success: bool, msg: str, count: int):
        """PDF 解析完成"""
        self.progress_bar.setVisible(False)
        self.parse_pdf_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "失敗", msg)

        self.statusBar().showMessage("就緒")

    def build_graph(self):
        """建構知識圖譜（優先使用 v2 目錄）"""
        # 優先檢查 v2 目錄，若無則使用舊目錄
        json_dir = config.PARSED_JSON_V2_DIR
        json_count = len(list(json_dir.glob("*.json")))

        if json_count == 0:
            # 嘗試使用舊目錄
            json_dir = config.PARSED_JSON_DIR
            json_count = len(list(json_dir.glob("*.json")))

        if json_count == 0:
            QMessageBox.warning(self, "警告", "請先解析 PDF 檔案")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.build_graph_btn.setEnabled(False)

        graph_path = config.GRAPH_DB_DIR / config.GRAPH_FILE
        self.worker = GraphBuildWorker(json_dir, graph_path)
        self.worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.finished.connect(self.on_graph_build_finished)
        self.worker.start()

    def on_graph_build_finished(self, success: bool, msg: str, kg: Optional[CompetencyKnowledgeGraph]):
        """圖譜建構完成"""
        self.progress_bar.setVisible(False)
        self.build_graph_btn.setEnabled(True)

        if success and kg:
            self.kg = kg
            self.engine = GraphRAGQueryEngine(kg)
            self.update_graph_status()
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "失敗", msg)

        self.statusBar().showMessage("就緒")

    def load_graph(self, path: Optional[str] = None):
        """載入圖譜"""
        if path is None:
            path = str(config.GRAPH_DB_DIR / config.GRAPH_FILE)

        if not Path(path).exists():
            QMessageBox.warning(self, "警告", f"圖譜檔案不存在: {path}")
            return

        try:
            self.statusBar().showMessage("載入圖譜...")
            self.kg = CompetencyKnowledgeGraph()
            self.kg.load(path)

            # 檢查是否需要補充 ICAP metadata
            # 如果圖譜中行業別數量為 0，則自動補充
            stats = self.kg.get_statistics()
            if stats.get("行業別數", 0) == 0:
                self.statusBar().showMessage("補充 ICAP metadata...")
                enriched = self.kg.enrich_with_icap_metadata(config.ICAP_SOURCE_DIR)
                if enriched > 0:
                    # 重新儲存圖譜
                    self.kg.save(path)

            self.engine = GraphRAGQueryEngine(self.kg)
            self.update_graph_status()
            self.statusBar().showMessage("圖譜載入完成")
        except Exception as e:
            QMessageBox.warning(self, "載入失敗", str(e))

    def update_database(self):
        """顯示 JSON 資料統計"""
        self.statusBar().showMessage("讀取資料統計中...")
        self.update_db_btn.setEnabled(False)

        try:
            json_store = CompetencyJSONStore(config.PARSED_JSON_V2_DIR)
            stats = json_store.get_statistics()

            result_msg = (
                f"JSON 資料庫統計：\n\n"
                f"• 職能基準: {stats.get('total_standards', 0)} 筆\n"
                f"• 工作任務: {stats.get('total_tasks', 0)}\n"
                f"• 知識項目: {stats.get('total_knowledge', 0)}\n"
                f"• 技能項目: {stats.get('total_skills', 0)}\n"
                f"• 態度項目: {stats.get('total_attitudes', 0)}\n"
                f"• 唯一職類別: {stats.get('unique_categories', 0)}\n"
                f"• 唯一職業別: {stats.get('unique_occupations', 0)}"
            )

            QMessageBox.information(self, "資料統計", result_msg)
        except Exception as e:
            QMessageBox.critical(self, "讀取失敗", f"無法讀取資料：\n{e}")

        self.statusBar().showMessage("就緒")
        self.update_db_btn.setEnabled(True)

    def init_embeddings(self, force_rebuild: bool = True):
        """
        重新建立 Embedding 向量索引（按鈕觸發，預設強制重建）

        Args:
            force_rebuild: 是否強制重建索引（預設 True，即「更新 Embedding」語義）
        """
        if not self.engine:
            QMessageBox.warning(self, "警告", "請先載入知識圖譜")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.init_embedding_btn.setEnabled(False)

        self.worker = EmbeddingInitWorker(self.engine, force_rebuild=force_rebuild)
        self.worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.finished.connect(self.on_embedding_init_finished)
        self.worker.start()

    def _auto_init_embeddings(self):
        """啟動時靜默自動載入已有的 Embedding 索引（不重建）"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.init_embedding_btn.setEnabled(False)
        self.statusBar().showMessage("自動載入 Embedding 索引...")

        self.worker = EmbeddingInitWorker(self.engine, force_rebuild=False)
        self.worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.finished.connect(self._on_auto_embedding_finished)
        self.worker.start()

    def _on_auto_embedding_finished(self, success: bool, msg: str):
        """自動 Embedding 載入完成（靜默，不彈窗）"""
        self.progress_bar.setVisible(False)
        self.init_embedding_btn.setEnabled(True)
        if success:
            self.embedding_status_label.setText("Embedding: 已載入 ✓")
            self.embedding_status_label.setStyleSheet("font-size: 9pt; color: green;")
            self.statusBar().showMessage("就緒（圖譜 + Embedding 已自動載入）")
        else:
            self.statusBar().showMessage("就緒（Embedding 自動載入失敗，請手動點擊「更新 Embedding」）")

    def on_embedding_init_finished(self, success: bool, msg: str):
        """Embedding 初始化完成"""
        self.progress_bar.setVisible(False)
        self.init_embedding_btn.setEnabled(True)

        if success:
            self.embedding_status_label.setText("Embedding: 已初始化 ✓")
            self.embedding_status_label.setStyleSheet("font-size: 9pt; color: green;")
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "失敗", msg)

        self.statusBar().showMessage("就緒")

    def init_llm(self):
        """初始化 LLM"""
        if not self.engine:
            QMessageBox.warning(self, "警告", "請先載入知識圖譜")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.init_llm_btn.setEnabled(False)
        self.statusBar().showMessage("載入 LLM 模型中（這可能需要一些時間）...")

        self.llm_worker = LLMInitWorker(self.engine)
        self.llm_worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.llm_worker.finished.connect(self.on_llm_init_finished)
        self.llm_worker.start()

    def on_llm_init_finished(self, success: bool, msg: str):
        """LLM 初始化完成"""
        self.progress_bar.setVisible(False)
        self.init_llm_btn.setEnabled(True)

        if success:
            self.llm_status_label.setText("LLM: 已初始化 ✓")
            self.llm_status_label.setStyleSheet("font-size: 9pt; color: green;")
            QMessageBox.information(self, "完成", msg)
        else:
            self.llm_status_label.setText("LLM: 初始化失敗")
            self.llm_status_label.setStyleSheet("font-size: 9pt; color: red;")
            QMessageBox.warning(self, "失敗", msg)

        self.statusBar().showMessage("就緒")

    def init_federated_search(self):
        """初始化聯邦搜索"""
        if not self.engine:
            QMessageBox.warning(self, "警告", "請先載入知識圖譜")
            return

        # 確保 Embedding 已初始化
        if self.engine.embedding_model is None:
            reply = QMessageBox.question(
                self, "初始化 Embedding",
                "聯邦搜索需要先初始化 Embedding，是否現在初始化？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.init_embeddings()
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage("初始化聯邦搜索系統...")

        self.fed_worker = FederatedSearchInitWorker(self.engine)
        self.fed_worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.fed_worker.finished.connect(self.on_federated_init_finished)
        self.fed_worker.start()

    def on_federated_init_finished(self, success: bool, msg: str):
        """聯邦搜索初始化完成"""
        self.progress_bar.setVisible(False)

        if success:
            # 取得統計資訊
            status_text = "聯邦搜索: 已初始化 ✓"
            if self.engine and self.engine.federated_manager:
                fm = self.engine.federated_manager
                cat_count = len(fm.category_sources)
                occ_count = len(fm.occupation_sources) if hasattr(fm, 'occupation_sources') else 0
                status_text = f"聯邦搜索: {cat_count}職類別, {occ_count}通俗職業 ✓"
            self.federated_status_label.setText(status_text)
            self.federated_status_label.setStyleSheet("color: green; font-size: 9pt;")
            QMessageBox.information(self, "完成", msg)
        else:
            self.federated_status_label.setText("聯邦搜索: 初始化失敗")
            self.federated_status_label.setStyleSheet("color: red; font-size: 9pt;")
            QMessageBox.warning(self, "失敗", msg)

        self.statusBar().showMessage("就緒")

    def show_category_list(self):
        """顯示職類別和通俗職業分類列表"""
        if not self.engine or not self.engine.is_federated_ready():
            QMessageBox.warning(
                self, "警告",
                "請先初始化聯邦搜索系統才能查看列表"
            )
            return

        content = ""

        # 通俗職業分類列表
        occupations = self.engine.get_occupation_list()
        if occupations:
            content += "# 通俗職業分類列表\n\n"
            content += f"共 {len(occupations)} 個通俗職業分類（更精細的路由）\n\n"
            content += "| 通俗職業分類 | 所屬職類別 | 職能基準數 | 節點數 |\n"
            content += "|-------------|-----------|-----------|--------|\n"

            for occ in occupations[:50]:  # 顯示前 50 個
                parent = occ.get('parent_category', '-')
                content += f"| {occ['name']} | {parent} | {occ['standards_count']} | {occ['node_count']} |\n"

            if len(occupations) > 50:
                content += f"\n*（僅顯示前 50 個，共 {len(occupations)} 個）*\n"
            content += "\n---\n\n"

        # 職類別列表
        categories = self.engine.get_category_list()
        if categories:
            content += "# 職類別列表\n\n"
            content += f"共 {len(categories)} 個職類別\n\n"
            content += "| 職類別 | 職能基準數 | 節點數 |\n"
            content += "|--------|-----------|--------|\n"

            for cat in categories:
                content += f"| {cat['name']} | {cat['standards_count']} | {cat['node_count']} |\n"

        if not content:
            QMessageBox.information(self, "列表", "沒有找到資料")
            return

        dialog = ResultDialog("聯邦搜索分類列表", content, self)
        self.result_dialogs.append(dialog)
        dialog.show()

    def update_graph_status(self):
        """更新圖譜狀態"""
        if self.kg:
            stats = self.kg.get_statistics()
            industry_count = stats.get('行業別數', 0)
            self.graph_status_label.setText(
                f"圖譜狀態: 已載入 ✓\n"
                f"  節點: {stats.get('節點總數', 0):,}\n"
                f"  邊: {stats.get('邊總數', 0):,}\n"
                f"  行業別: {industry_count}"
            )
            self.graph_status_label.setStyleSheet("font-size: 10pt; padding: 5px; color: green;")
            self.stats_btn.setEnabled(True)
            self.visualize_btn.setEnabled(True)
            self.community_viz_btn.setEnabled(True)

            # 更新職能基準列表
            self.update_standard_list()

    def show_graph_stats(self):
        """顯示圖譜統計"""
        if self.kg:
            stats = self.kg.get_statistics()
            dialog = GraphStatsDialog(stats, self)
            dialog.exec()

    def update_standard_list(self):
        """更新職能基準列表（載入所有項目）"""
        if not self.kg:
            return

        self.standard_list.clear()
        self.all_standard_items = []

        standards = self.kg.get_nodes_by_type("職能基準")
        for std_id in standards:
            node_data = self.kg.get_node_data(std_id)
            if node_data:
                name = node_data.get("name", std_id)
                self.all_standard_items.append((name, std_id))

        # 按名稱排序
        self.all_standard_items.sort(key=lambda x: x[0])

        # 顯示所有項目
        self.display_standard_items(self.all_standard_items)

    def filter_standard_list(self, filter_text: str):
        """根據關鍵字過濾職能基準列表"""
        filter_text = filter_text.strip().lower()

        if not filter_text:
            # 無過濾條件，顯示所有
            self.display_standard_items(self.all_standard_items)
        else:
            # 過濾匹配的項目
            filtered = [
                (name, std_id) for name, std_id in self.all_standard_items
                if filter_text in name.lower()
            ]
            self.display_standard_items(filtered)

    def display_standard_items(self, items: list):
        """顯示職能基準項目列表"""
        self.standard_list.clear()

        for name, std_id in items:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, std_id)
            self.standard_list.addItem(item)

        # 更新計數標籤
        total = len(self.all_standard_items)
        shown = len(items)
        if shown == total:
            self.standard_count_label.setText(f"共 {total} 項")
        else:
            self.standard_count_label.setText(f"顯示 {shown} / {total} 項")

    def on_standard_double_clicked(self, item: QListWidgetItem):
        """雙擊職能基準項目"""
        name = item.text()
        self.semantic_input.setText(name)
        self.query_tabs.setCurrentIndex(4)  # 切換到語義搜尋 Tab
        self.do_semantic_query()

    def visualize_graph(self):
        """視覺化知識圖譜"""
        if not self.kg:
            QMessageBox.warning(self, "警告", "請先載入知識圖譜")
            return

        # 取得選中的職能基準作為中心節點
        selected_items = self.standard_list.selectedItems()
        center_node = None
        if selected_items:
            center_node = selected_items[0].data(Qt.ItemDataRole.UserRole)
        else:
            QMessageBox.information(self, "提示", "請先從列表中選擇一個職能基準作為視覺化中心")
            return

        # 取得視覺化控制參數
        depth = self.viz_depth_spin.value()
        node_limit = self.viz_node_limit_spin.value()
        selected_types = self._get_selected_viz_types()

        if not selected_types:
            QMessageBox.warning(self, "警告", "請至少選擇一種節點類型")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.visualize_btn.setEnabled(False)

        output_path = config.OUTPUTS_DIR / "graph_visualization.html"
        self.worker = GraphVisualizeWorker(
            self.kg, output_path, center_node,
            depth=depth, node_limit=node_limit, selected_types=selected_types
        )
        self.worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.finished.connect(self.on_visualize_finished)
        self.worker.start()

    def on_visualize_finished(self, success: bool, msg: str, file_path: str):
        """視覺化完成"""
        self.progress_bar.setVisible(False)
        self.visualize_btn.setEnabled(True)

        if success:
            reply = QMessageBox.question(
                self, "視覺化完成",
                f"{msg}\n\n是否在瀏覽器中開啟？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
        else:
            QMessageBox.warning(self, "失敗", msg)

        self.statusBar().showMessage("就緒")

    def visualize_community(self):
        """社群視覺化（類似 Microsoft GraphRAG 風格）"""
        if not self.kg:
            QMessageBox.warning(self, "警告", "請先載入知識圖譜")
            return

        # 取得選中的職能基準作為中心節點
        selected_items = self.standard_list.selectedItems()
        center_node = None
        if selected_items:
            center_node = selected_items[0].data(Qt.ItemDataRole.UserRole)
        else:
            QMessageBox.information(self, "提示", "請先從列表中選擇一個職能基準作為視覺化中心")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.community_viz_btn.setEnabled(False)
        self.statusBar().showMessage("生成社群視覺化中...")

        # 使用背景執行緒
        self.community_worker = CommunityVisualizeWorker(
            self.kg,
            config.OUTPUTS_DIR / "community_visualization.html",
            center_node=center_node,
            max_nodes=self.viz_node_limit_spin.value()
        )
        self.community_worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.community_worker.finished.connect(self.on_community_visualize_finished)
        self.community_worker.start()

    def on_community_visualize_finished(self, success: bool, msg: str, file_path: str):
        """社群視覺化完成"""
        self.progress_bar.setVisible(False)
        self.community_viz_btn.setEnabled(True)

        if success:
            reply = QMessageBox.question(
                self, "社群視覺化完成",
                f"{msg}\n\n是否在瀏覽器中開啟？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
        else:
            QMessageBox.warning(self, "失敗", msg)

        self.statusBar().showMessage("就緒")

    # =============================
    # 查詢操作
    # =============================

    def check_ready(self) -> bool:
        """檢查系統是否就緒"""
        if not self.engine:
            QMessageBox.warning(self, "警告", "請先載入知識圖譜")
            return False
        return True

    def do_compare_query(self):
        """執行跨職業比較"""
        if not self.check_ready():
            return

        occ1 = self.compare_occ1_input.text().strip()
        occ2 = self.compare_occ2_input.text().strip()

        if not occ1 or not occ2:
            QMessageBox.warning(self, "警告", "請輸入兩個職業名稱")
            return

        self.execute_query("跨職業比較", occupation1=occ1, occupation2=occ2)

    def do_career_query(self):
        """執行職涯路徑查詢"""
        if not self.check_ready():
            return

        from_occ = self.career_from_input.text().strip()
        to_occ = self.career_to_input.text().strip()

        if not from_occ or not to_occ:
            QMessageBox.warning(self, "警告", "請輸入起點和目標職業")
            return

        self.execute_query("職涯路徑", from_occupation=from_occ, to_occupation=to_occ)

    def do_ability_query(self):
        """執行能力反查"""
        if not self.check_ready():
            return

        # 確保 Embedding 已初始化
        if self.engine.vector_index is None:
            reply = QMessageBox.question(
                self, "初始化 Embedding",
                "能力反查需要初始化 Embedding，是否現在初始化？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.init_embeddings()
            return

        ability = self.ability_input.text().strip()
        if not ability:
            QMessageBox.warning(self, "警告", "請輸入知識/技能關鍵字")
            return

        self.execute_query("能力反查", ability=ability)

    def do_stats_query(self):
        """執行聚合統計"""
        if not self.check_ready():
            return

        type_map = {"全部": "all", "知識": "knowledge", "技能": "skill"}
        ability_type = type_map.get(self.stats_type_combo.currentText(), "all")
        top_k = self.stats_topk_spin.value()

        self.execute_query("聚合統計", ability_type=ability_type, top_k=top_k)

    def do_semantic_query(self):
        """執行語義搜尋"""
        if not self.check_ready():
            return

        # 確保 Embedding 已初始化
        if self.engine.vector_index is None:
            reply = QMessageBox.question(
                self, "初始化 Embedding",
                "語義搜尋需要初始化 Embedding，是否現在初始化？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.init_embeddings()
            return

        query = self.semantic_input.text().strip()
        if not query:
            QMessageBox.warning(self, "警告", "請輸入查詢內容")
            return

        top_k = self.semantic_topk_spin.value()
        depth = self.semantic_depth_spin.value()

        # 檢查是否啟用聯邦搜索
        use_federated = self.federated_checkbox.isChecked()

        if use_federated:
            # 檢查聯邦搜索是否已初始化
            if not self.engine.is_federated_ready():
                reply = QMessageBox.question(
                    self, "初始化聯邦搜索",
                    "聯邦搜索尚未初始化，是否現在初始化？\n"
                    "(這可能需要一些時間)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.init_federated_search()
                return

            top_k_categories = self.federated_topk_spin.value()
            top_k_occupations = self.occupation_topk_spin.value()
            use_occupation_routing = self.occupation_routing_checkbox.isChecked()
            self.execute_query(
                "聯邦語義搜尋",
                query=query,
                top_k=top_k,
                expand_depth=depth,
                use_federated=True,
                top_k_categories=top_k_categories,
                top_k_occupations=top_k_occupations,
                use_occupation_routing=use_occupation_routing
            )
        else:
            self.execute_query("語義搜尋", query=query, top_k=top_k, expand_depth=depth)

    def execute_query(self, query_type: str, **kwargs):
        """執行查詢"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.worker = QueryWorker(self.engine, query_type, **kwargs)
        self.worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.finished.connect(self.on_query_finished)
        self.worker.start()

    def _copy_result(self):
        """複製結果到剪貼簿"""
        content = self.result_display.toPlainText().strip()
        if content:
            QApplication.clipboard().setText(content)
            self.statusBar().showMessage("已複製到剪貼簿", 2000)

    def on_query_finished(self, result: QueryResult, elapsed: float):
        """查詢完成"""
        self.progress_bar.setVisible(False)

        # 顯示結果並捲動到頂部
        self.result_display.setMarkdown(result.answer)
        self.result_display.moveCursor(QTextCursor.MoveOperation.Start)

        # 顯示查詢時間
        if elapsed >= 60:
            time_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        else:
            time_str = f"{elapsed:.1f}s"
        self.query_time_label.setText(f"查詢類型: {result.query_type} | 耗時: {time_str}")

        # 保存到歷史
        self.search_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": result.query,
            "query_type": result.query_type,
            "answer": result.answer,
            "elapsed": elapsed
        })

        self.statusBar().showMessage(f"查詢完成: {result.query_type}")

    def show_result_popup(self):
        """顯示結果彈出視窗"""
        content = self.result_display.toMarkdown()
        if content.strip():
            dialog = ResultDialog("查詢結果詳情", content, self)
            self.result_dialogs.append(dialog)
            dialog.show()


# =============================
# 主程式
# =============================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 設定字型
    font = QFont("Microsoft JhengHei", 10)
    app.setFont(font)

    window = GraphRAGMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
