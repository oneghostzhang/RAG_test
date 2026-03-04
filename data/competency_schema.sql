-- ============================================================
-- 職能基準資料庫 DDL (修正版 v3)
-- ============================================================
--
-- 修正項目：
-- 1. 新增「補充說明」欄位（學歷經驗條件、名詞解釋）
-- 2. 移除多餘的 standard_attitude_map（態度直接查 competency_item）
-- 3. 支援多個職業別/行業別（用 JSON 陣列存儲，保持查詢簡單）
-- 4. 優化索引設計
--
-- FTS5 Tokenizer 選擇：trigram
-- 理由：unicode61 對中文無實際斷詞效果，trigram 將文字切成連續 3 字元片段，
--       能有效匹配中文子字串。需要 SQLite 3.34.0+ (2020-12)。
--       若環境版本較低，請將 tokenize='trigram' 改為 tokenize='unicode61'
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. 職能基準主表 (standard)
-- ============================================================
CREATE TABLE IF NOT EXISTS standard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_code TEXT NOT NULL UNIQUE,          -- 職能項目代碼 (如 TFB3433-001v3)
    standard_name TEXT NOT NULL,                 -- 職能項目名稱

    -- 所屬類別欄位（主要值，若有多個則取第一個）
    category_code TEXT,                          -- 職類別代碼
    category_name TEXT,                          -- 職類別名稱
    occupation_code TEXT,                        -- 職業別代碼 (如 5120)
    occupation_name TEXT,                        -- 職業別名稱 (如 廚師)
    industry_code TEXT,                          -- 行業別代碼 (如 I5611)
    industry_name TEXT,                          -- 行業別名稱 (如 住宿及餐飲業／餐館)

    -- 多值欄位（JSON 陣列格式，支援一個職能基準屬於多個分類）
    category_list TEXT,                          -- 職類別清單 JSON: [{"code":"xxx","name":"yyy"},...]
    occupation_list TEXT,                        -- 職業別清單 JSON
    industry_list TEXT,                          -- 行業別清單 JSON

    -- 基本資訊
    job_description TEXT,                        -- 工作描述
    level INTEGER,                               -- 基準級別 (1-6)
    dev_update_date TEXT,                        -- 發展更新日期

    -- 補充說明（新增）
    qualification_requirements TEXT,             -- 學歷經驗條件
    glossary TEXT,                               -- 名詞解釋（JSON 格式）
    other_notes TEXT,                            -- 其他補充說明

    -- 來源追溯
    source_file TEXT,                            -- 來源 PDF 檔案路徑
    source_url TEXT,                             -- 來源網址 (若有)

    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- standard 索引
CREATE INDEX IF NOT EXISTS idx_standard_category_code ON standard(category_code);
CREATE INDEX IF NOT EXISTS idx_standard_occupation_code ON standard(occupation_code);
CREATE INDEX IF NOT EXISTS idx_standard_industry_code ON standard(industry_code);
CREATE INDEX IF NOT EXISTS idx_standard_name ON standard(standard_name);
CREATE INDEX IF NOT EXISTS idx_standard_level ON standard(level);

-- ============================================================
-- 2. 主要職責表 (responsibility)
-- ============================================================
CREATE TABLE IF NOT EXISTS responsibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_id INTEGER NOT NULL,
    t_code TEXT NOT NULL,                        -- T1, T2, ...
    name TEXT NOT NULL,                          -- 職責名稱
    sort_order INTEGER NOT NULL DEFAULT 0,       -- PDF 原始順序

    -- 來源追溯
    source_page INTEGER,                         -- 來源 PDF 頁碼
    source_ref TEXT,                             -- 原始段落標記

    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (standard_id) REFERENCES standard(id) ON DELETE CASCADE,
    UNIQUE (standard_id, t_code)
);

CREATE INDEX IF NOT EXISTS idx_responsibility_standard ON responsibility(standard_id);
CREATE INDEX IF NOT EXISTS idx_responsibility_sort ON responsibility(standard_id, sort_order);

-- ============================================================
-- 3. 工作任務表 (task)
-- ============================================================
CREATE TABLE IF NOT EXISTS task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    responsibility_id INTEGER NOT NULL,
    task_code TEXT NOT NULL,                     -- T1.1, T1.2, ...
    name TEXT NOT NULL,                          -- 任務名稱
    competency_level INTEGER,                    -- 職能級別 (若有)
    sort_order INTEGER NOT NULL DEFAULT 0,       -- PDF 原始順序

    -- 來源追溯
    source_page INTEGER,                         -- 來源 PDF 頁碼
    source_ref TEXT,                             -- 原始段落標記

    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (responsibility_id) REFERENCES responsibility(id) ON DELETE CASCADE,
    UNIQUE (responsibility_id, task_code)
);

CREATE INDEX IF NOT EXISTS idx_task_responsibility ON task(responsibility_id);
CREATE INDEX IF NOT EXISTS idx_task_sort ON task(responsibility_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_task_code ON task(task_code);

-- ============================================================
-- 4. 工作產出表 (output)
-- ============================================================
-- 注意：SQLite UNIQUE 約束中，多個 NULL 值被視為不同，
-- 故 UNIQUE(task_id, o_code) 允許同一 task_id 有多筆 o_code=NULL 的記錄
CREATE TABLE IF NOT EXISTS output (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    o_code TEXT,                                 -- O1.1.1 (可為 NULL)
    name TEXT NOT NULL,                          -- 產出名稱
    sort_order INTEGER NOT NULL DEFAULT 0,       -- PDF 原始順序

    -- 來源追溯
    source_page INTEGER,                         -- 來源 PDF 頁碼

    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
    UNIQUE (task_id, o_code)
);

CREATE INDEX IF NOT EXISTS idx_output_task ON output(task_id);
CREATE INDEX IF NOT EXISTS idx_output_sort ON output(task_id, sort_order);

-- ============================================================
-- 5. 行為指標表 (indicator)
-- ============================================================
CREATE TABLE IF NOT EXISTS indicator (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    p_code TEXT NOT NULL,                        -- P1.1.1, P1.1.2, ...
    description TEXT NOT NULL,                   -- 完整行為指標文字
    sort_order INTEGER NOT NULL DEFAULT 0,       -- PDF 原始順序

    -- 來源追溯
    source_page INTEGER,                         -- 來源 PDF 頁碼
    source_ref TEXT,                             -- 原始段落標記

    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
    UNIQUE (task_id, p_code)
);

CREATE INDEX IF NOT EXISTS idx_indicator_task ON indicator(task_id);
CREATE INDEX IF NOT EXISTS idx_indicator_sort ON indicator(task_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_indicator_p_code ON indicator(p_code);

-- ============================================================
-- 6. 職能內涵表 (competency_item)
-- K=知識, S=技能, A=態度
-- 以 standard_id 為範圍，不全域共用（不同職能基準的 K01 內容可能不同）
-- ============================================================
CREATE TABLE IF NOT EXISTS competency_item (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_id INTEGER NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('K', 'S', 'A')),  -- K=知識, S=技能, A=態度
    code TEXT NOT NULL,                          -- K01, S03, A01, ...
    title TEXT NOT NULL,                         -- 名稱/標題
    description TEXT,                            -- 詳細描述 (若有)
    sort_order INTEGER NOT NULL DEFAULT 0,       -- 原始順序

    -- 來源追溯
    source_page INTEGER,                         -- 來源 PDF 頁碼

    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (standard_id) REFERENCES standard(id) ON DELETE CASCADE,
    UNIQUE (standard_id, type, code)
);

CREATE INDEX IF NOT EXISTS idx_competency_item_standard_type ON competency_item(standard_id, type);
CREATE INDEX IF NOT EXISTS idx_competency_item_code ON competency_item(code);

-- ============================================================
-- 7. 任務與 K/S/A 對應表 (task_competency_map)
-- 用於記錄每個任務需要哪些知識、技能、態度
-- ============================================================
CREATE TABLE IF NOT EXISTS task_competency_map (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    competency_item_id INTEGER NOT NULL,

    created_at TEXT DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
    FOREIGN KEY (competency_item_id) REFERENCES competency_item(id) ON DELETE CASCADE,
    UNIQUE (task_id, competency_item_id)
);

CREATE INDEX IF NOT EXISTS idx_task_competency_task ON task_competency_map(task_id);
CREATE INDEX IF NOT EXISTS idx_task_competency_item ON task_competency_map(competency_item_id);

-- ============================================================
-- 8. FTS5 全文檢索表 (indicator_fts)
-- 對行為指標文字進行全文檢索
-- ============================================================
CREATE VIRTUAL TABLE IF NOT EXISTS indicator_fts USING fts5(
    description,
    content='indicator',
    content_rowid='id',
    tokenize='trigram'
);

-- ============================================================
-- 9. FTS5 職能基準檢索表 (standard_fts)
-- 對職能基準名稱、工作描述等進行全文檢索
-- ============================================================
CREATE VIRTUAL TABLE IF NOT EXISTS standard_fts USING fts5(
    standard_name,
    job_description,
    qualification_requirements,
    content='standard',
    content_rowid='id',
    tokenize='trigram'
);

-- ============================================================
-- FTS5 同步觸發器 - indicator_fts
-- ============================================================
DROP TRIGGER IF EXISTS indicator_fts_ai;
CREATE TRIGGER indicator_fts_ai AFTER INSERT ON indicator BEGIN
    INSERT INTO indicator_fts(rowid, description) VALUES (new.id, new.description);
END;

DROP TRIGGER IF EXISTS indicator_fts_ad;
CREATE TRIGGER indicator_fts_ad AFTER DELETE ON indicator BEGIN
    INSERT INTO indicator_fts(indicator_fts, rowid, description) VALUES ('delete', old.id, old.description);
END;

DROP TRIGGER IF EXISTS indicator_fts_au;
CREATE TRIGGER indicator_fts_au AFTER UPDATE OF description ON indicator BEGIN
    INSERT INTO indicator_fts(indicator_fts, rowid, description) VALUES ('delete', old.id, old.description);
    INSERT INTO indicator_fts(rowid, description) VALUES (new.id, new.description);
END;

-- ============================================================
-- FTS5 同步觸發器 - standard_fts
-- ============================================================
DROP TRIGGER IF EXISTS standard_fts_ai;
CREATE TRIGGER standard_fts_ai AFTER INSERT ON standard BEGIN
    INSERT INTO standard_fts(rowid, standard_name, job_description, qualification_requirements)
    VALUES (new.id, new.standard_name, new.job_description, new.qualification_requirements);
END;

DROP TRIGGER IF EXISTS standard_fts_ad;
CREATE TRIGGER standard_fts_ad AFTER DELETE ON standard BEGIN
    INSERT INTO standard_fts(standard_fts, rowid, standard_name, job_description, qualification_requirements)
    VALUES ('delete', old.id, old.standard_name, old.job_description, old.qualification_requirements);
END;

DROP TRIGGER IF EXISTS standard_fts_au;
CREATE TRIGGER standard_fts_au AFTER UPDATE OF standard_name, job_description, qualification_requirements ON standard BEGIN
    INSERT INTO standard_fts(standard_fts, rowid, standard_name, job_description, qualification_requirements)
    VALUES ('delete', old.id, old.standard_name, old.job_description, old.qualification_requirements);
    INSERT INTO standard_fts(rowid, standard_name, job_description, qualification_requirements)
    VALUES (new.id, new.standard_name, new.job_description, new.qualification_requirements);
END;

-- ============================================================
-- updated_at 自動更新觸發器
-- ============================================================
DROP TRIGGER IF EXISTS standard_updated;
CREATE TRIGGER standard_updated AFTER UPDATE ON standard
FOR EACH ROW WHEN old.updated_at = new.updated_at BEGIN
    UPDATE standard SET updated_at = datetime('now', 'localtime') WHERE id = old.id;
END;

DROP TRIGGER IF EXISTS responsibility_updated;
CREATE TRIGGER responsibility_updated AFTER UPDATE ON responsibility
FOR EACH ROW WHEN old.updated_at = new.updated_at BEGIN
    UPDATE responsibility SET updated_at = datetime('now', 'localtime') WHERE id = old.id;
END;

DROP TRIGGER IF EXISTS task_updated;
CREATE TRIGGER task_updated AFTER UPDATE ON task
FOR EACH ROW WHEN old.updated_at = new.updated_at BEGIN
    UPDATE task SET updated_at = datetime('now', 'localtime') WHERE id = old.id;
END;

DROP TRIGGER IF EXISTS output_updated;
CREATE TRIGGER output_updated AFTER UPDATE ON output
FOR EACH ROW WHEN old.updated_at = new.updated_at BEGIN
    UPDATE output SET updated_at = datetime('now', 'localtime') WHERE id = old.id;
END;

DROP TRIGGER IF EXISTS indicator_updated;
CREATE TRIGGER indicator_updated AFTER UPDATE ON indicator
FOR EACH ROW WHEN old.updated_at = new.updated_at BEGIN
    UPDATE indicator SET updated_at = datetime('now', 'localtime') WHERE id = old.id;
END;

DROP TRIGGER IF EXISTS competency_item_updated;
CREATE TRIGGER competency_item_updated AFTER UPDATE ON competency_item
FOR EACH ROW WHEN old.updated_at = new.updated_at BEGIN
    UPDATE competency_item SET updated_at = datetime('now', 'localtime') WHERE id = old.id;
END;

-- ============================================================
-- 手動重建 FTS 索引程序
-- ============================================================
-- 使用方式（indicator_fts）：
-- DELETE FROM indicator_fts;
-- INSERT INTO indicator_fts(rowid, description) SELECT id, description FROM indicator;
-- INSERT INTO indicator_fts(indicator_fts) VALUES('optimize');
--
-- 使用方式（standard_fts）：
-- DELETE FROM standard_fts;
-- INSERT INTO standard_fts(rowid, standard_name, job_description, qualification_requirements)
--     SELECT id, standard_name, job_description, qualification_requirements FROM standard;
-- INSERT INTO standard_fts(standard_fts) VALUES('optimize');
