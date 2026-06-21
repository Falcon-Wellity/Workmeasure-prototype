import csv
import os
from datetime import datetime
import cv2 # 画像保存のために追加

# ==========================================
# 【備忘録】将来的にSQLデータベースに移行する予定
# - テーブル名候補: `posture_reports`
# - image_path列は、将来クラウドストレージ（S3等）のURLを格納する予定
# ==========================================

CSV_FILE_PATH = "posture_report_database.csv"
IMAGE_DIR = "captured_images" # 画像を保存するフォルダ名

def save_report_to_csv(work_env, work_posture, angles, issues, evaluation, advice, image_bgr):
    """
    確定したレポート、計測データ、およびキャプチャー写真を保存・蓄積します。
    """
    # 既存ファイルの存在確認
    file_exists = os.path.isfile(CSV_FILE_PATH)
    
    # 画像保存用フォルダがなければ作成
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        
    # 一意のIDとタイムスタンプの生成
    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    report_id = f"RPT-{timestamp_str}"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 【新規】画像の保存処理 ---
    image_filename = f"{report_id}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    
    # Streamlit(RGB)からOpenCV(BGR)に変換して保存
    image_bgr_converted = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr_converted)
    # ----------------------------

    # 1. データの整形（image_path を追加）
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
        "image_path": image_path # CSVに画像の保存場所を記録
    }
    
    fieldnames = list(report_data.keys())
    
    with open(CSV_FILE_PATH, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(report_data)
        
    return report_id