"""
批次測試 PDF Parser（新版格式）
隨機選取 PDF 檔案進行解析測試，輸出新版 JSON 格式（含內建 RAG chunks）
"""

import json
import random
import sys
from pathlib import Path
from pdf_parser_v2 import parse_pdf_to_json

# 設定輸出編碼
sys.stdout.reconfigure(encoding='utf-8')

def batch_test(num_files=10, legacy_format=False):
    """
    批次測試 PDF 解析

    Args:
        num_files: 測試檔案數量
        legacy_format: 是否使用舊版格式
    """
    pdf_dir = Path(r"C:\Users\User\Graph_RAG_test\data\raw_pdf")
    output_dir = Path(r"C:\Users\User\Graph_RAG_test\batch_output")
    output_dir.mkdir(exist_ok=True)

    # 取得所有 PDF 檔案
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 個 PDF 檔案")

    # 隨機選取
    selected = random.sample(pdf_files, min(num_files, len(pdf_files)))
    print(f"隨機選取 {len(selected)} 個檔案進行測試")
    print(f"輸出格式: {'舊版（相容 competency_db）' if legacy_format else '新版（含內建 RAG chunks）'}\n")

    results = []

    for i, pdf_path in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] 處理: {pdf_path.name}")

        result = {
            "file": pdf_path.name,
            "success": False,
            "error": None,
            "stats": {}
        }

        try:
            # 解析 PDF
            json_path = output_dir / f"{pdf_path.stem}.json"
            data = parse_pdf_to_json(str(pdf_path), str(json_path), legacy_format=legacy_format)

            if data.get("parse_success"):
                result["success"] = True

                if legacy_format:
                    # 舊版格式統計
                    basic = data.get("職能基準", {})
                    result["stats"] = {
                        "代碼": basic.get("代碼", ""),
                        "名稱": basic.get("名稱", ""),
                        "基準級別": basic.get("基準級別", 0),
                        "主要職責數": len(data.get("主要職責", [])),
                        "工作任務數": sum(len(r.get("工作任務", [])) for r in data.get("主要職責", [])),
                        "知識項目數": len(data.get("知識清單", {})),
                        "技能項目數": len(data.get("技能清單", {})),
                        "態度項目數": len(data.get("態度清單", {})),
                    }
                else:
                    # 新版格式統計
                    metadata = data.get("metadata", {})
                    basic = data.get("basic_info", {})
                    result["stats"] = {
                        "代碼": metadata.get("code", ""),
                        "名稱": metadata.get("name", ""),
                        "版本": metadata.get("version", ""),
                        "基準級別": basic.get("level", 0),
                        "工作任務數": len(data.get("competency_tasks", [])),
                        "知識項目數": len(data.get("competency_knowledge", [])),
                        "技能項目數": len(data.get("competency_skills", [])),
                        "態度項目數": len(data.get("competency_attitudes", [])),
                        "RAG_chunks數": len(data.get("chunks_for_rag", [])),
                    }

                stats = result['stats']
                chunks_info = f" - {stats.get('RAG_chunks數', 'N/A')} chunks" if not legacy_format else ""
                print(f"    [OK] 成功: {stats['名稱']}{chunks_info}")
            else:
                result["error"] = data.get("parse_errors", ["解析失敗"])
                print(f"    [FAIL] 失敗: {result['error']}")

        except Exception as e:
            result["error"] = str(e)
            print(f"    [ERROR] 錯誤: {e}")

        results.append(result)

    # 輸出總結
    print("\n" + "="*60)
    print("批次處理總結")
    print("="*60)

    success_count = sum(1 for r in results if r["success"])
    print(f"成功: {success_count}/{len(results)}")

    print("\n各檔案詳細:")
    for r in results:
        if r["success"]:
            s = r["stats"]
            if legacy_format:
                print(f"  [OK] {s['名稱']} ({s['代碼']})")
                print(f"       級別:{s['基準級別']} | 職責:{s['主要職責數']} | 任務:{s['工作任務數']} | K:{s['知識項目數']} | S:{s['技能項目數']} | A:{s['態度項目數']}")
            else:
                print(f"  [OK] {s['名稱']} ({s['代碼']}) {s.get('版本', '')}")
                print(f"       級別:{s['基準級別']} | 任務:{s['工作任務數']} | K:{s['知識項目數']} | S:{s['技能項目數']} | A:{s['態度項目數']} | Chunks:{s['RAG_chunks數']}")
        else:
            print(f"  [FAIL] {r['file']}: {r['error']}")

    # 儲存結果報告
    report_path = output_dir / "batch_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n報告已儲存至: {report_path}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='批次測試 PDF Parser')
    parser.add_argument('-n', '--num', type=int, default=10, help='測試檔案數量')
    parser.add_argument('--legacy', action='store_true', help='使用舊版格式（相容 competency_db）')
    args = parser.parse_args()

    batch_test(args.num, legacy_format=args.legacy)
