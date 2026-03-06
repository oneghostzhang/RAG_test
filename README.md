# 職能基準知識圖譜 Graph RAG 系統

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![UI](https://img.shields.io/badge/UI-PyQt6-41CD52?logo=qt&logoColor=white)
![Version](https://img.shields.io/badge/Version-v1.2.0-orange)

> 基於台灣 ICAP 職能基準的知識圖譜系統，整合 Graph RAG 技術，支援跨職業能力比較、職涯路徑規劃與自然語言查詢。

---

## 目錄

- [系統架構](#-系統架構)
- [核心功能](#-核心功能)
- [專案結構](#-專案結構)
- [快速開始](#-快速開始)
- [知識圖譜結構](#-知識圖譜結構)
- [查詢範例](#-查詢範例)
- [模組說明](#-模組說明)
- [效能](#-效能)
- [技術棧](#️-技術棧)
- [注意事項](#️-注意事項)
- [更新日誌](#-更新日誌)

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                    資料輸入層                            │
│   ICAP 職能基準 PDF  →  pdf_parser_v2.py  →  JSON       │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                    資料存取層                            │
│   competency_store.py (CompetencyJSONStore)              │
│   ├── parsed_json_v2/ (905+ JSON 檔)                    │
│   └── data/occupation_index.json (預計算索引)           │
└──────────────┬──────────────────┬────────────────────────┘
               │                  │
┌──────────────▼──────┐  ┌────────▼────────────────────────┐
│   知識圖譜層         │  │   向量索引層                    │
│  graph_builder.py   │  │   FAISS embedding               │
│  NetworkX 有向多重圖 │  │   sentence-transformers         │
│  29,130 節點        │  │   vectordb/                     │
│  283,562 邊         │  └────────────────────────────────┘
└──────────────┬──────┘           │
               │                  │
┌──────────────▼──────────────────▼────────────────────────┐
│                    查詢引擎層                             │
│   graph_rag.py  +  federated_search.py                   │
│   ├── 語義搜尋（FAISS 向量相似度）                       │
│   ├── 聯邦搜索（職類別 centroid 路由）                   │
│   └── 通俗職業路由（occupation_index.json 快速載入）     │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                    介面層                                │
│   graph_rag_ui.py (PyQt6 桌面應用程式)                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 核心功能

| 功能 | 說明 |
|------|------|
| 📄 **PDF 自動解析** | 支援所有職能基準代碼格式（`TFB5120-003v3`、`HBR2431-001v4` 等），批次處理 900+ PDF |
| 🕸️ **知識圖譜建構** | 11 種節點 × 13 種邊，自動推斷職涯晉升路徑與知識技能關聯 |
| 🔍 **三種搜索模式** | 語義搜尋 / 聯邦搜索（centroid 路由）/ 聯邦搜索（通俗職業路由） |
| 🖥️ **PyQt6 桌面 UI** | 一鍵解析、建圖、搜索、視覺化，含社群偵測功能 |
| ⚡ **快速索引載入** | 預計算 `occupation_index.json`，聯邦搜索索引載入從 ~2s 降至 ~0.02s |

---

## 📁 專案結構

```
Graph_RAG_test/
│
├── 🐍 核心模組
│   ├── config.py               # 系統配置（路徑、模型參數）
│   ├── pdf_parser_v2.py        # PDF 解析器（支援多種代碼格式）
│   ├── competency_store.py     # JSON 資料存取層
│   ├── graph_builder.py        # 知識圖譜建構
│   ├── graph_community.py      # 社群偵測
│   ├── graph_rag.py            # Graph RAG 查詢引擎
│   ├── federated_search.py     # 聯邦搜索（RAGRoute 式路由）
│   └── graph_rag_ui.py         # PyQt6 桌面 UI
│
├── 🛠️ 工具腳本
│   ├── quick_start.py          # 命令列快速啟動
│   └── batch_test.py           # 批次 PDF 解析測試
│
├── 📦 依賴
│   └── requirements.txt
│
├── 📂 data/
│   ├── occupation_index.json   # ⚡ 預計算職業分類索引（快速載入用）
│   ├── raw_pdf/                # 原始 PDF（不納入版控）
│   └── parsed_json_v2/         # 解析後 JSON（不納入版控）
│
├── 📂 lib/                     # 前端視覺化函式庫
│   ├── vis-9.1.2/              # vis-network 圖譜渲染
│   └── tom-select/             # 下拉選單元件
│
├── graph_db/                   # 知識圖譜存檔（不納入版控）
└── vectordb/                   # 向量索引（不納入版控）
```

---

## 🚀 快速開始

### 1. 下載專案

```bash
git clone https://github.com/oneghostzhang/RAG_test.git
cd RAG_test
```

> `data/occupation_index.json`（職業分類預計算索引）已包含在版控中，clone 後即可直接使用，**無需重新建立索引**。

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 準備資料

將 ICAP 職能基準 PDF 放入 `data/raw_pdf/` 資料夾：

```
RAG_test/
└── data/
    └── raw_pdf/
        ├── 西點麵包烘焙助理-職能基準.pdf
        ├── 保健食品產業行銷企劃師-職能基準.pdf
        └── ...
```

> PDF 可從 [ICAP 職能發展應用平台](https://icap.wda.gov.tw/) 下載。

### 4. 啟動桌面 UI（建議）

```bash
python graph_rag_ui.py
```

**操作流程：**

```
[解析 PDF] → 選擇 raw_pdf 資料夾 → 批次轉為 JSON
     ↓
[建構圖譜] → 從 JSON 建立知識圖譜 + Embedding 向量索引
     ↓
[搜索查詢] → 輸入問題 → 選擇搜索模式 → 取得答案
```

### 5. 命令列使用

```bash
# 測試：解析 5 個 PDF + 建構小型圖譜
python quick_start.py --mode test --limit 5

# 建構指定數量的圖譜
python quick_start.py --mode build --limit 100

# 互動式查詢
python quick_start.py --mode interactive

# 批次解析 PDF
python batch_test.py --input data/raw_pdf --output data/parsed_json_v2
```

### 6. 關於 occupation_index.json

`data/occupation_index.json` 是聯邦搜索的預計算索引，儲存了 905 個職能基準的職業分類對應關係，讓系統啟動時可跳過掃描所有 JSON 的步驟（載入時間從 ~2s 降至 ~0.02s）。

- **首次使用**：clone 後直接可用（已內含於版控）
- **新增 PDF 後**：UI 解析 PDF 完成時會自動重建，或手動執行：

```python
from competency_store import build_occupation_index_json
build_occupation_index_json()  # 輸出至 data/occupation_index.json
```

---

## 📊 知識圖譜結構

### 節點類型（11 種）

| 節點 | 代碼格式 | 範例 |
|------|---------|------|
| 職能基準 | `TFB5120-003v3` | 日式烹飪廚師 |
| 職類別 | `TFB` | 休閒與觀光旅遊／餐飲管理 |
| 職業別 | `5120` | 廚師 |
| 行業別 | `I5611` | 住宿及餐飲業／餐館 |
| 主要職責 | `T1` | 烘焙前置準備 |
| 工作任務 | `T1.1` | 整理環境與設備（級別 2） |
| 工作產出 | `O1.1.1` | 環境維護紀錄表 |
| 行為指標 | `P1.1.1` | 依食品安全衛生規範... |
| 知識 K | `K01` | 食品安全衛生相關規範 |
| 技能 S | `S01` | 器具選用及操作能力 |
| 態度 A | `A01` | 主動積極 |

### 邊類型（13 種）

```
職能基準 ──屬於職類──▶ 職類別
職能基準 ──屬於職業──▶ 職業別
職能基準 ──適用行業──▶ 行業別
職能基準 ──包含職責──▶ 主要職責 ──包含任務──▶ 工作任務
工作任務 ──產出──────▶ 工作產出
工作任務 ──要求行為──▶ 行為指標
工作任務 ──需要知識──▶ 知識
工作任務 ──需要技能──▶ 技能
職能基準 ──要求態度──▶ 態度
知識     ──相關於────▶ 知識
技能     ──前置於────▶ 技能
職能基準 ──可晉升至──▶ 職能基準
```

---

## 💬 查詢範例

```
# 跨職業比較
烘焙助理和餐飲服務人員有哪些共同技能？

# 職涯路徑
從餐飲服務人員晉升到賣場規劃人員需要補充什麼能力？

# 能力反查
具備食品安全衛生知識適合哪些職業？

# 主要職責查詢
LED光學設計工程師的主要職責有哪些？

# 知識/技能統計
最常被需要的知識技能 Top 10？

# 語義搜尋
找與品質管理相關的職能基準
```

---

## 🔧 模組說明

<details>
<summary><b>pdf_parser_v2.py</b> — PDF 解析器</summary>

- 使用 pdfplumber 提取文字與表格
- 支援所有代碼格式（TF、HBR、LED 等前綴）
- 動態偵測表格表頭位置（相容不同版面 PDF）
- 行業別多值自動拆分，儲存為 JSON 陣列
- 輸出含 `chunks_for_rag` 的結構化 JSON
</details>

<details>
<summary><b>competency_store.py</b> — JSON 資料存取層</summary>

- 直接讀取 parsed_json_v2 目錄，取代 SQLite
- 支援行業別多值欄位存取與全文索引
- `fix_industry_in_json_files()` — 批次修正行業代碼格式
- `build_occupation_index_json()` — 產生預計算索引檔
</details>

<details>
<summary><b>federated_search.py</b> — 聯邦搜索（RAGRoute 概念）</summary>

- 優先從 `occupation_index.json` 載入（~0.02s），不存在時掃描 JSON
- 兩種路由：職類別 centroid 路由 / 通俗職業名稱路由
- 縮小全量搜索範圍，提升查詢精度與速度
</details>

<details>
<summary><b>graph_rag_ui.py</b> — PyQt6 桌面 UI</summary>

- PDF 解析 / 圖譜建構 / 向量索引一鍵操作
- 搜索模式：語義搜尋 / 聯邦搜索 / 聯邦+通俗職業路由
- 知識圖譜 HTML 互動視覺化 + 社群偵測視覺化
- PDF 解析完成後自動重建 occupation_index.json
</details>

<details>
<summary><b>graph_builder.py</b> — 知識圖譜建構</summary>

- NetworkX 有向多重圖（29,130 節點 / 283,562 邊）
- 自動推斷職涯晉升路徑與知識技能關聯
- 圖譜序列化存檔至 graph_db/
</details>

<details>
<summary><b>graph_rag.py</b> — 查詢引擎</summary>

- FAISS 向量索引 + 圖譜遍歷混合檢索
- 完整職能術語定義 prompt（K/S/A/T/O/P 代碼說明）
- 19 條回答原則（代碼區分、格式規範、資料可信度）
- 可選用 llama-cpp-python 本地 LLM
</details>

---

## 📈 效能

| 操作 | 時間 | 備註 |
|------|------|------|
| 解析單一 PDF | 1–3 秒 | 依 PDF 複雜度 |
| 建構圖譜（100 個） | 3–5 分鐘 | 含 PDF 解析 |
| Embedding 初始化 | 30–60 秒 | 首次載入模型 |
| 聯邦索引載入（冷） | ~0.02 秒 | 從 occupation_index.json |
| 語義搜尋 | ~7 秒 | 已建立向量索引 |
| 聯邦搜索 | ~7–9 秒 | 依路由策略 |

---

## 🛠️ 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| PDF 解析 | pdfplumber | 文字與表格提取 |
| 知識圖譜 | NetworkX | 有向多重圖結構 |
| 向量檢索 | FAISS | 高效相似度搜尋 |
| Embedding | sentence-transformers | 文本向量化 |
| 桌面 UI | PyQt6 | 操作介面 |
| 圖譜視覺化 | vis-network 9.1.2 | 互動式圖譜渲染 |
| LLM（可選） | llama-cpp-python | 本地答案生成 |

---

## ⚠️ 注意事項

1. **PDF 解析精度**：依賴 PDF 版面一致性，複雜排版建議人工核對解析結果
2. **記憶體需求**：完整圖譜約 2–4 GB，Embedding 模型約 500 MB，建議系統記憶體 ≥ 8 GB
3. **資料來源**：職能基準資料來自 [ICAP 職能發展應用平台](https://icap.wda.gov.tw/)，僅供學習研究使用
4. **索引更新**：新增或重新解析 PDF 後，occupation_index.json 會自動重建；手動重建請執行 `build_occupation_index_json()`

---

## 📋 更新日誌

| 版本 | 日期 | 更新內容 |
|------|------|---------|
| v1.2.0 | 2026-03-06 | 自動初始化聯邦搜索與 LLM；修復 PyTorch meta tensor 錯誤；修正 LLM 回答亂生成（注入工作任務內容）；搜尋改為大小寫不敏感 |
| v1.1.0 | 2026-03-04 | 預計算 occupation_index.json（索引載入加速 100x）；移除舊版 pdf_parser / competency_db；擴充 LLM prompt 術語定義與回答原則 |
| v1.0.0 | 2026-01-07 | 初始版本：PDF 解析、知識圖譜建構、Graph RAG 查詢、PyQt6 UI、聯邦搜索 |

---

## 📄 授權

本專案採用 [MIT License](LICENSE) 授權。

---

**版本**: v1.2.0 　**最後更新**: 2026-03-06
