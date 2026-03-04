-- ============================================================
-- 職能基準資料庫 - 常用查詢範例
-- ============================================================

-- ============================================================
-- 查詢 1: 根據職類別代碼查詢所有職能基準
-- ============================================================
SELECT
    standard_code,
    standard_name,
    industry_name,
    level
FROM standard
WHERE category_code = :category_code
ORDER BY standard_code;

-- 範例: 查詢職類別代碼為 '3433' 的所有職能基準


-- ============================================================
-- 查詢 2: 根據行業別代碼查詢所有職能基準
-- ============================================================
SELECT
    standard_code,
    standard_name,
    occupation_name,
    level
FROM standard
WHERE industry_code = :industry_code
ORDER BY standard_code;

-- 範例: 查詢行業別代碼為 'I5611' (餐館) 的所有職能基準


-- ============================================================
-- 查詢 3: 根據職業別代碼查詢所有職能基準
-- ============================================================
SELECT
    standard_code,
    standard_name,
    industry_name,
    level
FROM standard
WHERE occupation_code = :occupation_code
ORDER BY standard_code;

-- 範例: 查詢職業別代碼為 '5120' (廚師) 的所有職能基準


-- ============================================================
-- 查詢 4: 給定 standard_code，列出 T → Task → P 完整路徑
-- （使用 sort_order 排序，保持 PDF 原始順序）
-- ============================================================
SELECT
    s.standard_code,
    s.standard_name,
    r.t_code,
    r.name AS responsibility_name,
    r.sort_order AS r_order,
    t.task_code,
    t.name AS task_name,
    t.competency_level,
    t.sort_order AS t_order,
    i.p_code,
    i.description AS indicator_description,
    i.sort_order AS i_order
FROM standard s
JOIN responsibility r ON r.standard_id = s.id
JOIN task t ON t.responsibility_id = r.id
JOIN indicator i ON i.task_id = t.id
WHERE s.standard_code = :standard_code
ORDER BY r.sort_order, t.sort_order, i.sort_order;

-- 範例: 查詢 'TFB3433-001v3' (中式烹飪廚師) 的完整結構


-- ============================================================
-- 查詢 5: FTS5 全文檢索行為指標（使用 bm25 排序）
-- ============================================================
SELECT
    s.standard_code,
    s.standard_name,
    t.task_code,
    i.p_code,
    highlight(indicator_fts, 0, '【', '】') AS highlighted_text,
    i.description,
    bm25(indicator_fts) AS relevance_score
FROM indicator_fts fts
JOIN indicator i ON i.id = fts.rowid
JOIN task t ON t.id = i.task_id
JOIN responsibility r ON r.id = t.responsibility_id
JOIN standard s ON s.id = r.standard_id
WHERE indicator_fts MATCH :keyword
ORDER BY bm25(indicator_fts)
LIMIT 20;

-- 範例: 搜尋「安全衛生」
-- 範例: 搜尋「故障」
-- 進階: 搜尋多個關鍵字「食品 安全」


-- ============================================================
-- 查詢 6: FTS5 全文檢索職能基準名稱/描述
-- ============================================================
SELECT
    standard_code,
    standard_name,
    job_description,
    bm25(standard_fts) AS relevance_score
FROM standard_fts fts
JOIN standard s ON s.id = fts.rowid
WHERE standard_fts MATCH :keyword
ORDER BY bm25(standard_fts)
LIMIT 20;

-- 範例: 搜尋「烹飪」
-- 範例: 搜尋「工程師」


-- ============================================================
-- 查詢 7: 查詢特定職能基準的所有知識項目
-- ============================================================
SELECT
    ci.code,
    ci.title,
    ci.description
FROM competency_item ci
JOIN standard s ON ci.standard_id = s.id
WHERE s.standard_code = :standard_code
  AND ci.type = 'K'
ORDER BY ci.sort_order;


-- ============================================================
-- 查詢 8: 查詢特定職能基準的所有技能項目
-- ============================================================
SELECT
    ci.code,
    ci.title,
    ci.description
FROM competency_item ci
JOIN standard s ON ci.standard_id = s.id
WHERE s.standard_code = :standard_code
  AND ci.type = 'S'
ORDER BY ci.sort_order;


-- ============================================================
-- 查詢 9: 查詢特定任務需要的知識和技能
-- ============================================================
SELECT
    ci.type,
    ci.code,
    ci.title
FROM task_competency_map tcm
JOIN competency_item ci ON tcm.competency_item_id = ci.id
JOIN task t ON tcm.task_id = t.id
JOIN responsibility r ON t.responsibility_id = r.id
JOIN standard s ON r.standard_id = s.id
WHERE s.standard_code = :standard_code
  AND t.task_code = :task_code
ORDER BY ci.type, ci.sort_order;


-- ============================================================
-- 查詢 10: 統計各行業別的職能基準數量
-- ============================================================
SELECT
    industry_code,
    industry_name,
    COUNT(*) AS standard_count
FROM standard
WHERE industry_code IS NOT NULL AND industry_code != ''
GROUP BY industry_code, industry_name
ORDER BY standard_count DESC
LIMIT 20;


-- ============================================================
-- 查詢 11: 統計各職業別的職能基準數量
-- ============================================================
SELECT
    occupation_code,
    occupation_name,
    COUNT(*) AS standard_count
FROM standard
WHERE occupation_code IS NOT NULL AND occupation_code != ''
GROUP BY occupation_code, occupation_name
ORDER BY standard_count DESC
LIMIT 20;


-- ============================================================
-- 查詢 12: 查詢特定等級的所有職能基準
-- ============================================================
SELECT
    standard_code,
    standard_name,
    industry_name,
    occupation_name
FROM standard
WHERE level = :level
ORDER BY standard_code;

-- 範例: 查詢等級 4 的所有職能基準
