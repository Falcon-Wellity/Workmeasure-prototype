import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps  # スマホ画像の回転補正用

# --- CSV蓄積モジュールをインポート ---
from csv_db import save_report_to_csv, delete_report_by_id

# ==========================================
# 1. MediaPipe ＆ 画像処理ライブラリの初期設定
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 視認性向上のためのカスタム描画スタイル定義（太さと半径を大きく変更）
CUSTOM_LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=5) # 赤い点
CUSTOM_CONNECTIONS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2) # 緑の線

# ==========================================
# 2. 姿勢分析のための幾何学計算関数
# ==========================================
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_horizontal_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.array([b[0], a[1]])
    return calculate_angle(b, a, c)

def calculate_vertical_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.array([b[0], a[1]])
    return calculate_angle(a, b, c)

# ==========================================
# 3. Streamlit UI 画面レイアウト定義
# ==========================================
st.set_page_config(page_title="WorkMeasure プロトタイプ", layout="wide")
st.title("WorkMeasure - 作業姿勢評価Prototype")

# サイドバーメニュー（項目を追加）
st.sidebar.header("メニュー")
app_mode = st.sidebar.selectbox("操作を選択", ["新規姿勢計測", "食事姿勢計測", "正面姿勢計測", "レポート履歴"])

# メモリ保持用セッションの初期化関数
def reset_measurement_session():
    st.session_state.captured_image = None      
    st.session_state.captured_angles = {}       
    st.session_state.show_report = False        

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None      
if "captured_angles" not in st.session_state:
    st.session_state.captured_angles = {}       
if "show_report" not in st.session_state:
    st.session_state.show_report = False        

# --- 各メニューに応じた画面描画 ---
if app_mode == "新規姿勢計測":
    st.subheader("⚠️ 横向きから、全身（耳・肩・股関節・膝・足）が写るように撮影・保存してください")

    # 明示的なクリアボタンを配置
    if st.session_state.captured_image is not None or st.session_state.show_report:
        if st.button("🔄 画面を初期化して、新しい計測を始める", type="secondary"):
            reset_measurement_session()
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        work_env = st.selectbox("作業環境を選択", ["デスクワーク", "軽作業（立ち仕事）", "重作業"])
    with col2:
        work_posture = st.selectbox("作業姿勢を選択", ["座位", "立位", "低位置"])

    display_col, result_col = st.columns([2, 1])

    # --- パターンA：まだ撮影/アップロードしていない（画像未確定状態） ---
    if st.session_state.captured_image is None:
        with display_col:
            input_method = st.radio("撮影方法を選択してください：", ["リアルタイムカメラで撮影", "保存済みの写真をアップロード"])
            st.write("---")
            
            if input_method == "リアルタイムカメラで撮影":
                run_camera = st.checkbox("カメラを起動する", value=False)
                FRAME_WINDOW = st.image([]) 
            else:
                run_camera = False
                uploaded_file = st.file_uploader("アライメント写真をアップロード (JPG/PNG)", type=["jpg", "jpeg", "png"])
            
        with result_col:
            st.write("### 📊 リアルタイム角度分析（参考値）")
            back_placeholder = st.empty()           
            trunk_vertical_placeholder = st.empty() 
            hip_placeholder = st.empty()            
            femoral_placeholder = st.empty()        
            knee_placeholder = st.empty()           
            
            st.write("---")
            if input_method == "リアルタイムカメラで撮影":
                capture_button = st.button("📸 この姿勢を記録する（シャッター）", use_container_width=True)
            else:
                capture_button = st.button("🔍 アップロードした写真を解析・保存", use_container_width=True, type="primary")

        # ------------------------------------------
        # A-1. リアルタイムカメラの処理ロジック
        # ------------------------------------------
        if input_method == "リアルタイムカメラで撮影" and run_camera:
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    raise Exception("カメラデバイスを開けませんでした。")
                    
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while run_camera:
                        ret, frame = camera.read()
                        if not ret:
                            st.error("カメラ映像を取得できませんでした。")
                            break
                            
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False 
                        results = pose.process(image)  
                        image.flags.writeable = True  
                        
                        current_angles = {}
                        
                        if results.pose_landmarks:
                            # カスタムスタイルを適用（線と点を太く）
                            mp_drawing.draw_landmarks(
                                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=CUSTOM_LANDMARK_STYLE,
                                connection_drawing_spec=CUSTOM_CONNECTIONS_STYLE
                            )
                            try:
                                landmarks = results.pose_landmarks.landmark
                                right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                                left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
                                
                                if right_shoulder_vis > left_shoulder_vis:
                                    ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                                    side_text = "右側"
                                else:
                                    ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                                    side_text = "左側"
                                
                                angle_back = 180.0 - calculate_angle(ear, shoulder, hip)
                                angle_trunk_vertical = calculate_vertical_angle(shoulder, hip)
                                angle_hip = 180.0 - calculate_angle(shoulder, hip, knee)
                                angle_femoral = calculate_horizontal_angle(hip, knee)
                                # 【修正】膝の屈曲角計算を 180度からの引き算に変更
                                angle_knee = 180.0 - calculate_angle(hip, knee, ankle)
                                
                                current_angles = {
                                    "back": int(angle_back), "vertical": int(angle_trunk_vertical),
                                    "hip": int(angle_hip), "femoral": int(angle_femoral),
                                    "knee": int(angle_knee), "side": side_text
                                }
                                
                                back_placeholder.metric(label="首の傾き", value=f"{current_angles['back']} 度")
                                trunk_vertical_placeholder.metric(label="体幹の前傾角度 (垂直線基準)", value=f"{current_angles['vertical']} 度")
                                hip_placeholder.metric(label="股関節の角度", value=f"{current_angles['hip']} 度")
                                femoral_placeholder.metric(label="大腿の傾斜 (水平線基準)", value=f"{current_angles['femoral']} 度")
                                knee_placeholder.metric(label="膝の角度", value=f"{current_angles['knee']} 度")
                            except:
                                pass
                        
                        FRAME_WINDOW.image(image)
                        
                        if capture_button and current_angles:
                            st.session_state.captured_image = image
                            st.session_state.captured_angles = current_angles
                            camera.release()
                            st.rerun()
                camera.release()
            except Exception as e:
                st.error("⚠️ 本環境ではリアルタイムカメラを起動できません。左側の選択肢から『保存済みの写真をアップロード』を選択して解析を行ってください。")

        # ------------------------------------------
        # A-2. 写真アップロードの処理ロジック
        # ------------------------------------------
        if input_method == "保存済みの写真をアップロード" and uploaded_file is not None:
            # スマホ撮影写真の回転（EXIF）をPILで正常位置に補正してからOpenCV形式に変換
            pil_image = Image.open(uploaded_file)
            pil_image = ImageOps.exif_transpose(pil_image)
            raw_image = np.array(pil_image)
            
            # RGBフォーマットに統一
            if raw_image.shape[2] == 4:  # RGBA対策
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGBA2RGB)
            image = raw_image.copy()
            
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results = pose.process(image)
                current_angles = {}
                
                if results.pose_landmarks:
                    # カスタムスタイルを適用（線と点を太く）
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=CUSTOM_LANDMARK_STYLE,
                        connection_drawing_spec=CUSTOM_CONNECTIONS_STYLE
                    )
                    try:
                        landmarks = results.pose_landmarks.landmark
                        right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                        left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
                        
                        if right_shoulder_vis > left_shoulder_vis:
                            ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                            side_text = "右側"
                        else:
                            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            side_text = "左側"
                        
                        angle_back = 180.0 - calculate_angle(ear, shoulder, hip)
                        angle_trunk_vertical = calculate_vertical_angle(shoulder, hip)
                        angle_hip = 180.0 - calculate_angle(shoulder, hip, knee)
                        angle_femoral = calculate_horizontal_angle(hip, knee)
                        # 【修正】膝の屈曲角計算を 180度からの引き算に変更
                        angle_knee = 180.0 - calculate_angle(hip, knee, ankle)
                        
                        current_angles = {
                            "back": int(angle_back), "vertical": int(angle_trunk_vertical),
                            "hip": int(angle_hip), "femoral": int(angle_femoral),
                            "knee": int(angle_knee), "side": side_text
                        }
                        
                        back_placeholder.metric(label="首の傾き", value=f"{current_angles['back']} 度")
                        trunk_vertical_placeholder.metric(label="体幹の前傾角度 (垂直線基準)", value=f"{current_angles['vertical']} 度")
                        hip_placeholder.metric(label="股関節の角度", value=f"{current_angles['hip']} 度")
                        femoral_placeholder.metric(label="大腿の傾斜 (水平線基準)", value=f"{current_angles['femoral']} 度")
                        knee_placeholder.metric(label="膝の角度", value=f"{current_angles['knee']} 度")
                    except:
                        pass
            
            with display_col:
                st.image(image, caption="解析プレビュー（線と点がハッキリ重なっているか確認してください）")
                
            if capture_button and current_angles:
                st.session_state.captured_image = image
                st.session_state.captured_angles = current_angles
                st.rerun()

    # --- パターンB：すでに写真が確定（撮影またはアップロード完了）した状態 ---
    else:
        with display_col:
            st.info("📌 以下の姿勢で写真を一時保存しました。")
            st.image(st.session_state.captured_image)  
            
        with result_col:
            st.write("### 🔒 計測角度データ（参考値）")
            angles = st.session_state.captured_angles
            st.write(f"・首の傾き: **{angles['back']} 度**")
            st.write(f"・体幹の前傾: **{angles['vertical']} 度**")
            st.write(f"・股関節角度: **{angles['hip']} 度**")
            st.write(f"・大腿の傾斜: **{angles['femoral']} 度**")
            st.write(f"・膝の角度: **{angles['knee']} 度**")
            
            st.write("---")
            st.write("### 📏 環境実測データの入力")
            height_chair = st.number_input("座面高さ (cm) ※実測", min_value=0, max_value=200, value=40)
            height_work = st.number_input("作業高さ (cm) ※実測", min_value=0, max_value=200, value=75)
            
            st.write("---")
            st.write("### 🩺 リハ専門職の目による問題箇所の選択")
            selected_issues = []
            if st.checkbox("首の傾き（または猫背）"): selected_issues.append("首の傾き")
            if st.checkbox("体幹の過度な前傾/後傾"): selected_issues.append("体幹前傾角度")
            if st.checkbox("股関節の不適切姿勢（仙骨座りなど）"): selected_issues.append("股関節の不適切姿勢")
            if st.checkbox("大腿の傾斜"): selected_issues.append("大腿の傾斜")
            if st.checkbox("膝関節の角度"): selected_issues.append("膝関節の角度")
            if st.checkbox("座面の高さ不適合"): selected_issues.append("座面高さ")
            if st.checkbox("作業台・手元の高さ不適合"): selected_issues.append("作業高さ")
            
            st.write("---")
            if st.button("🔄 撮り直す / 写真を選び直す", type="secondary", use_container_width=True):
                reset_measurement_session()
                st.rerun()
                
            if st.button("📄 この姿勢でBasicReportを作成する", type="primary", use_container_width=True):
                st.session_state.captured_angles["height_chair"] = height_chair
                st.session_state.captured_angles["height_work"] = height_work
                st.session_state.show_report = True
                st.rerun()

        # --- 画面最下部：BasicReport表示エリア ---
        if st.session_state.show_report:
            st.write("---")
            st.write("## 📝 作業環境評価レポート (BasicReport)")
            
            angles = st.session_state.captured_angles
            
            rep_col1, rep_col2 = st.columns(2)
            with rep_col1:
                st.markdown(f"**対象作業環境:** {work_env} / **想定姿勢:** {work_posture}")
                st.markdown(f"**計測データ一覧 ({angles['side']}: 参考値):**")
                st.text(f"・首の傾き: {angles['back']}度\n・体幹前傾: {angles['vertical']}度\n・股関節角: {angles['hip']}度\n・大腿角度: {angles['femoral']}度\n・膝の角度: {angles['knee']}度")
            with rep_col2:
                st.markdown(f"**環境実測データ:**")
                st.text(f"・座面高さ: {angles['height_chair']} cm\n- 作業高さ: {angles['height_work']} cm")
                
                if selected_issues:
                    issue_text = "、".join(selected_issues)
                    st.error(f"⚠️ **要注意箇所:** {issue_text}")
                else:
                    st.success("✅ **要注意箇所:** 特になし（良好なアライメントです）")
            
            st.write("### ✍️ 専門職コメント入力")
            
# --- ✍️ 専門職コメントの自動生成ロジック ---
            
            # 各チェックボックスに対応するテンプレート辞書（評価・助言）
            # ※文言は現場に合わせていつでも自由に変更・調整してください。
            templates = {
                "首の傾き": {
                    "eva": "・頭部・頸部が前方に突出しており、首や肩の筋肉や骨への持続的なストレスが懸念されます。",
                    "adv": "・モニターや手元の対象物の高さを上げ視線を高く保ち、首への負担や肩こりを予防しましょう。"
                },
                "体幹前傾角度": {
                    "eva": f"・体幹が過度に前傾しており、腰背部の筋肉や椎間板に大きな負担がかかっています。（計測値: {angles['vertical']}度）腰痛のリスクが高まっています。",
                    "adv": "・椅子の奥まで深く腰掛け、骨盤を立てて背もたれを活用できる作業距離に調整してください。"
                },
                "股関節の不適切姿勢": {
                    "eva": "・骨盤が後傾し、いわゆる『仙骨座り』の傾向が見られます。坐骨で体重を支持できておらず、腰痛のリスクが高まっています。",
                    "adv": "・クッションなどを活用して骨盤の後傾を防ぎ、坐骨結節でしっかりと座面を捉えられるよう座り方を指導してください。"
                },
                "大腿の傾斜": {
                    "eva": "・座面に対して大腿部が平行に保たれておらず、骨盤や股関節のアライメント不良、または座面の適合不全が疑われます。",
                    "adv": "・足の裏全体がしっかりと床（またはフットレスト）に接地するよう、足元の環境を見直してください。"
                },
                "膝関節の角度": {
                    "eva": f"・膝関節の屈曲角度が不適切（計測値: {angles['knee']}度）であり、下肢の血流阻害や、立ち上がり時の負担増につながる恐れがあります。",
                    "adv": "・足首や膝が概ね90度前後の楽な角度に保てるよう、椅子の奥行きや足元のスペースを確保してください。"
                },
                "座面高さ": {
                    "eva": f"・椅子の座面高さ（計測値: {angles['height_chair']}cm）を作業者の体格、または手元の作業台とうまく適合できていません。",
                    "adv": "・座面の高さを調整し、足裏が接地した状態で肘が90度屈曲して作業台に添えられる位置を基準にしてください。"
                },
                "作業高さ": {
                    "eva": f"：作業台・手元の高さ（計測値: {angles['height_work']}cm）が不適合です。手元が低すぎる、または高すぎることが原因で肩・腰・膝への負担が懸念されます。",
                    "adv": "・可能であれば昇降デスク等の活用、または肘の高さに合わせた手元台の設置などで、適切な作業高さを確保してください。"
                }
            }

            # 選択された項目に基づいてテキストを組み立てる
            if not selected_issues:
                # 【ご要望】チェックが1項目も無い場合
                default_eva = "姿勢や作業環境は身体への影響が少なく概ね良好です。"
                default_advice = "【環境調整案】\n・現在の良好な作業環境と姿勢を維持してください。\n\n【身体へのアプローチ】\n・定期的な休憩やストレッチを行い、同一姿勢の持続を防ぎましょう。\n・以下の運動を１日１回実施しましょう。"
            else:
                # チェックがある場合、項目ごとに文章を改行して結合
                eva_list = []
                adv_list = []
                for issue in selected_issues:
                    if issue in templates:
                        eva_list.append(templates[issue]["eva"])
                        adv_list.append(templates[issue]["adv"])
                
                default_eva = "【解析結果に基づく専門的評価】\n" + "\n".join(eva_list)
                default_advice = "【環境調整案・指導内容】\n" + "\n".join(adv_list) + "\n\n【身体へのアプローチ】\n・"

            # 組み立てたテンプレートをテキストエリアの初期値(value)としてセット
            comment_eva = st.text_area("■ 理学療法士の評価", value=default_eva, height=220)
            comment_advice = st.text_area("■ 理学療法士の助言", value=default_advice, height=220)

            if st.button("💾 レポート内容を確定（プロトタイプ版保存）", type="primary"):
                try:
                    saved_id = save_report_to_csv(
                        work_env=work_env,
                        work_posture=work_posture,
                        angles=angles,
                        issues=selected_issues,
                        evaluation=comment_eva,
                        advice=comment_advice,
                        image_bgr=st.session_state.captured_image
                    )
                    # 【修正】保存成功時にセッションデータを即座に初期化し、次回計測が真っ新に行えるようにする
                    reset_measurement_session()
                    st.session_state["save_success_msg"] = f"🎉 レポートが確定し、CSVデータベースに蓄積されました！ (ID: {saved_id})"
                    st.rerun()
                except Exception as e:
                    st.error(f"データの保存中にエラーが発生しました: {e}")

        if "save_success_msg" in st.session_state:
            st.success(st.session_state["save_success_msg"])
            del st.session_state["save_success_msg"]

# --- 将来用追加メニューのプレースホルダー ---
elif app_mode == "食事姿勢計測":
    st.subheader("🍽️ 食事姿勢計測・評価（言語聴覚士向け機能）")
    st.info("💡 【将来拡張予定】\n言語聴覚士（ST）による嚥下機能評価や、食事場面における頸部・体幹アライメントの分析・スクリーニングを行える専用モードを実装予定です。")

elif app_mode == "正面姿勢計測":
    st.subheader("🧍 正面姿勢計測・アライメント分析（理学療法士向け機能）")
    st.info("💡 【将来拡張予定】\n理学療法士（PT）による正面（前額面）からの姿勢分析機能です。肩の高さの左右差、骨盤の傾き、体幹の側方傾斜などの評価を可能にする予定です。")
                    
elif app_mode == "レポート履歴":
    st.subheader("📁 過去の作業環境評価レポート履歴")
    
    CSV_FILE_PATH = "posture_report_database.csv"
    
    if not os.path.isfile(CSV_FILE_PATH):
        st.info("まだ保存されたレポート履歴がありません。新規計測を行ってデータを登録してください。")
    else:
        try:
            df = pd.read_csv(CSV_FILE_PATH, encoding="utf-8-sig")
            
            # 【安全ガード】IDや日時が空っぽ(NaN)の壊れた古い行を自動的に除外する
            df = df.dropna(subset=["report_id", "created_at"])
            
            if df.empty:
                st.info("有効なレポート履歴がありません。新規計測を行ってデータを登録してください。")
            else:
                df_sorted = df.iloc[::-1].reset_index(drop=True)
                st.write("### 🕒 レポート一覧（選択すると詳細が表示されます）")
                
                df_summary = df_sorted[["report_id", "created_at", "work_env", "work_posture", "selected_issues"]]
                df_summary.columns = ["レポートID", "計測日時", "作業環境", "作業姿勢", "要注意箇所"]
                
                selected_row = st.radio("確認したいレポートを選択してください：", range(len(df_summary)), 
                                       format_func=lambda x: f"{df_summary.iloc[x]['計測日時']} | {df_summary.iloc[x]['作業環境']} ({df_summary.iloc[x]['作業姿勢']})")
                
                st.write("---")
                
                if selected_row is not None:
                    full_data = df_sorted.iloc[selected_row]
                    target_report_id = full_data['report_id']
                    st.write(f"## 📝 レポート詳細 : {target_report_id}")
                    st.caption(f"計測日時: {full_data['created_at']}")
                    
                    hist_col1, hist_col2 = st.columns([1, 1])
                    with hist_col1:
                        st.write("### 📸 計測時のアライメント写真")
                        img_path = full_data["image_path"]
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.warning("⚠️ 写真ファイルが見つかりません。")
                            
                    with hist_col2:
                        st.write("### 📊 計測数値データ")
                        st.text(f"・首の傾き: {full_data['angle_back']} 度\n"
                                f"・体幹前傾: {full_data['angle_vertical']} 度\n"
                                f"・股関節角: {full_data['angle_hip']} 度\n"
                                f"・大腿傾斜: {full_data['angle_femoral']} 度\n"
                                f"・膝の角度: {full_data['angle_knee']} 度\n\n"
                                f"・座面高さ: {full_data['height_chair']} cm\n"
                                f"・作業高さ: {full_data['height_work']} cm")
                        
                        st.write("---")
                        if full_data["selected_issues"] != "特になし":
                            st.error(f"⚠️ **要注意箇所:** {full_data['selected_issues']}")
                        else:
                            st.success("✅ **要注意箇所:** 特になし（良好）")
                    
                    st.write("---")
                    st.write("### 🩺 専門職による評価・指導内容")
                    st.info(f"**■ リハ専門職の評価**\n\n{full_data['comment_evaluation']}")
                    st.success(f"**■ リハ専門職の助言・環境調整案**\n\n{full_data['comment_advice']}")
                    
                    # ------------------------------------------
                    # 【追加】不要な履歴の消去機能
                    # ------------------------------------------
                    st.write("---")
                    st.write("### 🗑️ レポートの管理")
                    confirm_delete = st.checkbox("このレポートを削除することを確認します。", key=f"del_chk_{target_report_id}")
                    
                    if st.button("🚨 このレポートを削除する", type="primary", disabled=not confirm_delete, key=f"del_btn_{target_report_id}"):
                        if delete_report_by_id(target_report_id):
                            st.success(f"レポート {target_report_id} を削除しました。")
                            st.rerun()
                        else:
                            st.error("削除処理に失敗しました。")
                    
        except Exception as e:
            st.error(f"履歴の読み込み中にエラーが発生しました: {e}")