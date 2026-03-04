"""
職能基準知識圖譜 Graph RAG 系統 - 快速啟動腳本
"""

import sys
from pathlib import Path

# 確保可以導入專案模組
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import get_config
from pdf_parser import CompetencyPDFParser
from graph_builder import CompetencyKnowledgeGraph
from graph_rag import GraphRAGQueryEngine

config = get_config()


def test_pdf_parser():
    """測試 PDF 解析器"""
    print("\n" + "=" * 60)
    print("測試 1: PDF 解析器")
    print("=" * 60)

    # 找一個 PDF 測試
    pdf_files = list(config.ICAP_SOURCE_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(f"✗ 在 {config.ICAP_SOURCE_DIR} 中找不到 PDF 檔案")
        return None

    print(f"找到 {len(pdf_files)} 個 PDF 檔案")

    # 解析第一個 PDF
    test_pdf = pdf_files[0]
    print(f"\n測試檔案: {test_pdf.name}")

    parser = CompetencyPDFParser()
    result = parser.parse_pdf(test_pdf)

    if result.parse_success:
        print("✓ 解析成功！")
        print(f"\n基本資訊:")
        for key, value in result.職能基準.items():
            if isinstance(value, dict):
                print(f"  {key}: {value.get('名稱', '')} ({value.get('代碼', '')})")
            else:
                print(f"  {key}: {value}")

        print(f"\n主要職責: {len(result.主要職責)} 個")
        for duty in result.主要職責[:2]:
            print(f"  - {duty['代碼']}: {duty['名稱']}")
            print(f"    工作任務: {len(duty.get('工作任務', []))} 個")

        print(f"\n知識清單: {len(result.知識清單)} 項")
        print(f"技能清單: {len(result.技能清單)} 項")
        print(f"態度清單: {len(result.態度清單)} 項")

        return result
    else:
        print(f"✗ 解析失敗: {result.parse_errors}")
        return None


def test_graph_builder(limit: int = 5):
    """測試知識圖譜建構"""
    print("\n" + "=" * 60)
    print("測試 2: 知識圖譜建構")
    print("=" * 60)

    print(f"\n從 PDF 建構知識圖譜（限制: {limit} 個）...")

    kg = CompetencyKnowledgeGraph()
    count = kg.build_from_pdf_directory(config.ICAP_SOURCE_DIR, limit=limit)

    if count > 0:
        print(f"✓ 成功建構 {count} 個職能基準")

        # 顯示統計
        stats = kg.get_statistics()
        print(f"\n圖譜統計:")
        print(f"  節點總數: {stats['節點總數']}")
        print(f"  邊總數: {stats['邊總數']}")

        print(f"\n節點類型:")
        for node_type, count in stats["節點類型統計"].items():
            print(f"    {node_type}: {count}")

        # 推斷關聯
        print("\n推斷職涯路徑和知識/技能關聯...")
        kg.infer_career_paths()
        kg.infer_related_knowledge_skills()

        # 儲存圖譜
        graph_path = config.GRAPH_DB_DIR / config.GRAPH_FILE
        kg.save(graph_path)
        print(f"\n✓ 圖譜已儲存至: {graph_path}")

        return kg
    else:
        print("✗ 建構失敗")
        return None


def test_graph_rag(kg: CompetencyKnowledgeGraph):
    """測試 Graph RAG 查詢"""
    print("\n" + "=" * 60)
    print("測試 3: Graph RAG 查詢")
    print("=" * 60)

    if kg is None:
        print("✗ 跳過（知識圖譜未建構）")
        return

    print("\n初始化查詢引擎...")
    engine = GraphRAGQueryEngine(kg)

    try:
        engine.initialize_embeddings()
        print("✓ Embedding 模型初始化完成")
    except Exception as e:
        print(f"✗ Embedding 初始化失敗: {e}")
        return

    # 測試查詢
    test_queries = [
        "中式烹飪廚師的行業別代碼為何？",
        "食品安全相關的職業",
    ]

    for query in test_queries:
        print(f"\n查詢: {query}")
        print("-" * 40)

        try:
            result = engine.query(query)
            print(f"查詢類型: {result.query_type}")
            print(f"答案:\n{result.answer[:500]}...")
        except Exception as e:
            print(f"✗ 查詢失敗: {e}")


def run_interactive_mode(kg: CompetencyKnowledgeGraph):
    """執行互動查詢模式"""
    print("\n" + "=" * 60)
    print("互動查詢模式")
    print("=" * 60)

    if kg is None:
        # 嘗試載入已存在的圖譜
        graph_path = config.GRAPH_DB_DIR / config.GRAPH_FILE
        if graph_path.exists():
            kg = CompetencyKnowledgeGraph()
            kg.load(graph_path)
        else:
            print("✗ 找不到知識圖譜，請先建構")
            return

    engine = GraphRAGQueryEngine(kg)
    engine.initialize_embeddings()

    print("\n輸入問題進行查詢，輸入 'quit' 退出")
    print("支援的查詢類型:")
    print("  1. 跨職業比較: 「烘焙助理和餐飲服務人員有哪些共同技能？」")
    print("  2. 職涯路徑: 「從餐飲服務人員晉升到主管需要什麼能力？」")
    print("  3. 能力反查: 「具備食品安全知識適合哪些職業？」")
    print("  4. 聚合統計: 「最常需要的技能 Top 10」")
    print("  5. 語義搜尋: 任意問題")
    print("-" * 60)

    while True:
        try:
            question = input("\n問題: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                break

            if not question:
                continue

            result = engine.query(question)

            print(f"\n[{result.query_type}]")
            print("-" * 40)
            print(result.answer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"✗ 錯誤: {e}")

    print("\n再見！")


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="職能基準 Graph RAG 快速啟動")
    parser.add_argument(
        "--mode", "-m",
        choices=["test", "build", "query", "interactive"],
        default="test",
        help="執行模式: test(測試), build(建構圖譜), query(單次查詢), interactive(互動)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="建構圖譜時的限制數量（測試用）"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="查詢問題"
    )

    args = parser.parse_args()

    # 設定日誌
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    print("\n" + "=" * 60)
    print("職能基準知識圖譜 Graph RAG 系統")
    print("=" * 60)
    print(f"專案目錄: {config.PROJECT_ROOT}")
    print(f"ICAP 資料: {config.ICAP_SOURCE_DIR}")

    if args.mode == "test":
        # 執行所有測試
        test_pdf_parser()
        kg = test_graph_builder(limit=args.limit)
        test_graph_rag(kg)

        print("\n" + "=" * 60)
        print("測試完成！")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 建構完整圖譜: python quick_start.py --mode build --limit 100")
        print("  2. 互動查詢: python quick_start.py --mode interactive")
        print("  3. 單次查詢: python quick_start.py --mode query -q '您的問題'")

    elif args.mode == "build":
        # 只建構圖譜
        kg = test_graph_builder(limit=args.limit)
        if kg:
            stats = kg.get_statistics()
            print(f"\n完成！圖譜包含 {stats['節點總數']} 個節點和 {stats['邊總數']} 條邊")

    elif args.mode == "query" and args.query:
        # 單次查詢
        graph_path = config.GRAPH_DB_DIR / config.GRAPH_FILE
        if not graph_path.exists():
            print("✗ 找不到知識圖譜，請先建構")
            return

        kg = CompetencyKnowledgeGraph()
        kg.load(graph_path)

        engine = GraphRAGQueryEngine(kg)
        engine.initialize_embeddings()

        result = engine.query(args.query)
        print(f"\n[{result.query_type}]")
        print("-" * 40)
        print(result.answer)

    elif args.mode == "interactive":
        # 互動模式
        run_interactive_mode(None)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
