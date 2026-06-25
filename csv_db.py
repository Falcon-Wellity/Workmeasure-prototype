import csv
import os
from datetime import datetime
import cv2  # 解析済み画像ファイルをローカルに保存するために使用
import pandas as pd

# ==============================================================================
# 【開発設計・備忘録】 将来的なシステム拡張へのロードマップ
# ------------------------------------------------------------------------------
# 1. データベース移行計画:
#    現在はプロトタイプ運用の利便性からCSVを使用していますが、複数人での同時アクセスや
#    検索速度の向上、セキュリティ強化のため、将来的にSQLデータベース（SQLite / PostgreSQL等）
#    へのリプレイスを想定しています。
#    - テーブル名候補: `posture_reports`
#
# 2. 画像ストレージ計画:
#    `image_path` 列は現在ローカル（サーバー内）のファイルパスを格納していますが、
#    クラウド運用（Streamlit Cloud等）ではコンテナ再起動時にローカルデータが消去される特性があります。
#    本番運用時は、AWS S3やGoogle Cloud Storage等の外部オブジェクトストレージへ画像をアップロードし、
#    その「アクセスURL（文字列）」をこの列に格納する設計に移行します。
# ==============================================================================

# データベースファイルの保存名定義（wm_main.py側での読み込みパスと厳密に一致させること）
CSV_FILE_PATH = "posture_report_database.csv"
# キャプチャした写真（ワイヤーフレーム描画済み）を保存するディレクトリ名
IMAGE_DIR = "captured_images"

def save_report_to_csv(work_env, work_posture, angles, issues, evaluation, advice, image_bgr):
    """
    確定した評価レポート、計測関節角度、環境実測データ、および解析プレビュー写真を
    ローカル環境（またはサーバー環境）に永続化して蓄積します。
    """
    file_exists = os.path.isfile(CSV_FILE_PATH)
    
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        
    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    report_id = f"RPT-{timestamp_str}"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    image_filename = f"{report_id}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    
    image_bgr_converted = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr_converted)

    report_data = {
        "report_id": report_id,
        "created_at": current_time,
        "work_env": work_env,
        "work_posture": work_posture,
        "measured_side": angles.get("side", "不明"),
        "angle_back": angles.get("back", 0),
        "angle_vertical": angles.get("vertical", 0),
        "angle_hip": angles.get("hip", 0),
        "angle_femoral": angles.get("femoral", 0),
        "angle_knee": angles.get("knee", 0),
        "height_chair": angles.get("height_chair", 0),
        "height_work": angles.get("height_work", 0),
        "selected_issues": "、".join(issues) if issues else "特になし",
        "comment_evaluation": evaluation,
        "comment_advice": advice,
        "image_path": image_path
    }
    
    fieldnames = list(report_data.keys())
    
    with open(CSV_FILE_PATH, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(report_data)
        
    return report_id

# ==============================================================================
# 今回新しく追加された削除用関数（wm_main.py から呼び出されます）
# ==============================================================================
def delete_report_by_id(report_id):
    """
    指定されたレポートIDのレコードをCSVから削除し、関連する画像ファイルも削除します。
    """
    if not os.path.isfile(CSV_FILE_PATH):
        return False
        
    try:
        # CSVを読み込み
        df = pd.read_csv(CSV_FILE_PATH, encoding="utf-8-sig")
        
        # 削除対象の行を特定して画像パスを取得
        target_row = df[df["report_id"] == report_id]
        if not target_row.empty:
            img_path = target_row.iloc[0]["image_path"]
            # 画像ファイルの削除（存在する場合）
            if os.path.exists(img_path):
                os.remove(img_path)
                
        # 対象IDを除外してCSVを上書き保存
        df_filtered = df[df["report_id"] != report_id]
        df_filtered.to_csv(CSV_FILE_PATH, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        print(f"Error deleting record: {e}")
        return False