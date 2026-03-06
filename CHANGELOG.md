# Changelog

所有重要變更均記錄於此文件。格式依據 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)。

---

## [Unreleased]

---

## [1.2.0] - 2026-03-06

### 新增
- 系統啟動時自動初始化聯邦搜索（從 `occupation_index.json` 快速載入）
- 系統啟動時自動初始化 LLM（從設定的預設路徑載入）
- 「更新職業索引」按鈕：重新建立 `occupation_index.json` 並重載聯邦搜索
- 「更改 LLM」按鈕：開啟檔案選擇器切換 GGUF 模型
- LLM 自動初始化期間顯示進度條動畫
- 自動初始化期間狀態標籤即時顯示「載入中...」

### 修復
- **PyTorch meta tensor 錯誤**：`accelerate` 函式庫污染 PyTorch 初始化 context 導致 `CategoryRouter` 被建立在 meta device 上；改用 `with torch.device('cpu'):` context manager 解決
- **LLM 回答亂生成**：`_build_context()` 未納入主要職責與工作任務內容；現在從 JSON store 取得完整任務資料並注入 prompt（最多 3 個職能基準 × 8 個任務）
- **搜尋大小寫不敏感**：`competency_store.py` 的 `search_standards()` 與 `graph_rag.py` 的 `search_indicators_by_keyword()` 修正為不區分大小寫比對
- **搜尋效能**：移除 `search_indicators_by_keyword()` 中 O(n²) 的重複結果檢查邏輯

---

## [1.1.0] - 2026-03-04

### 新增
- 預計算通俗職業分類索引 `occupation_index.json`（905 個職能基準、225 個職業、99 個類別），聯邦搜索載入時間從數秒降至 ~0.02 秒
- 擴充 LLM 提示詞的【重要術語定義】與【回答原則】為完整分類式指引

### 變更
- 全面重整 README，反映目前系統架構與安裝流程

### 移除
- 舊版 SQLite Schema 與查詢 SQL 檔案

---

## [1.0.0] - 2026-03-01

### 新增
- 初始版本：台灣職能基準 Graph RAG 系統
- PDF 解析器 `pdf_parser_v2.py`：支援 TF 與非 TF 前綴格式
- JSON 儲存與索引 `competency_store.py`
- 知識圖譜建立 `graph_builder.py`
- 聯邦搜索 `federated_search.py`：CategoryRouter 訓練 + FAISS 向量索引
- PyQt6 主介面 `graph_rag_ui.py`：四種搜尋模式（語意、關鍵字、混合、LLM）
- Graph RAG 查詢引擎 `graph_rag.py`

[Unreleased]: https://github.com/oneghostzhang/RAG_test/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/oneghostzhang/RAG_test/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/oneghostzhang/RAG_test/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/oneghostzhang/RAG_test/releases/tag/v1.0.0
