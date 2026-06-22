import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# --- CSV蓄積モジュールをインポート ---
from csv_db import save_report_to_csv

# ==========================================
# 1. MediaPipe ＆ 画像処理ライブラリの初期設定
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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
st.title("WorkMeasure (ワークメジャー) - 姿勢計測プロトタイプ")

# サイドバーメニュー
st.sidebar.header("メニュー")
app_mode = st.sidebar.selectbox("操作を選択", ["新規姿勢計測", "レポート履歴"])

if app_mode == "新規姿勢計測":
    st.subheader("⚠️ 横向きから、全身（耳・肩・股関節・膝・足）が写るように撮影・保存してください")

    col1, col2 = st.columns(2)
    with col1:
        work_env = st.selectbox("作業環境を選択", ["デスクワーク", "軽作業（立ち仕事）", "重作業"])
    with col2:
        work_posture = st.selectbox("作業姿勢を選択", ["座位", "立位"])

    display_col, result_col = st.columns([2, 1])

    # メモリ保持用セッションの初期化
    if "captured_image" not in st.session_state:
        st.session_state.captured_image = None      
    if "captured_angles" not in st.session_state:
        st.session_state.captured_angles = {}       
    if "show_report" not in st.session_state:
        st.session_state.show_report = False        

    # --- パターンA：まだ撮影/アップロードしていない（画像未確定状態） ---
    if st.session_state.captured_image is None:
        with display_col:
            # 入力方法の選択（ラジオボタンで並行化）
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
        # A-1. リアルタイムカメラの処理ロジック（Web用安全ガード付き）
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
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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
                                angle_knee = calculate_angle(hip, knee, ankle)
                                
                                current_angles = {
                                    "back": int(angle_back), "vertical": int(angle_trunk_vertical),
                                    "hip": int(angle_hip), "femoral": int(angle_femoral),
                                    "knee": int(angle_knee), "side": side_text
                                }
                                
                                back_placeholder.metric(label="首の傾き", value=f"{current_angles['back']} 度")
                                trunk_vertical_placeholder.metric(label="体幹の前傾角度 (垂直線基準)", value=f"{current_angles['vertical']} 度")
                                hip_placeholder.metric(label="股関節の角度", value=f"{current_angles['hip']} 度")
                                femoral_placeholder.metric(label="大腿の角度 (水平線基準)", value=f"{current_angles['femoral']} 度")
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
                st.error("⚠️ 本環境（Webサーバー等）ではリアルタイムカメラを起動できません。左側のメニューから『保存済みの写真をアップロード』を選択して解析を行ってください。")

        # ------------------------------------------
        # A-2. 写真アップロードの処理ロジック
        # ------------------------------------------
        if input_method == "保存済みの写真をアップロード" and uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results = pose.process(image)
                current_angles = {}
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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
                        angle_knee = calculate_angle(hip, knee, ankle)
                        
                        current_angles = {
                            "back": int(angle_back), "vertical": int(angle_trunk_vertical),
                            "hip": int(angle_hip), "femoral": int(angle_femoral),
                            "knee": int(angle_knee), "side": side_text
                        }
                        
                        back_placeholder.metric(label="首の傾き", value=f"{current_angles['back']} 度")
                        trunk_vertical_placeholder.metric(label="体幹の前傾角度 (垂直線基準)", value=f"{current_angles['vertical']} 度")
                        hip_placeholder.metric(label="股関節の角度", value=f"{current_angles['hip']} 度")
                        femoral_placeholder.metric(label="大腿の角度 (水平線基準)", value=f"{current_angles['femoral']} 度")
                        knee_placeholder.metric(label="膝の角度", value=f"{current_angles['knee']} 度")
                    except:
                        pass
            
            with display_col:
                st.image(image, caption="解析プレビュー（棒人間が重なっているか確認してください）")
                
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
            st.write(f"・大腿の角度: **{angles['femoral']} 度**")
            st.write(f"・膝の角度: **{angles['knee']} 度**")
            
            st.write("---")
            st.write("### 📏 環境実測データの入力")
            height_chair = st.number_input("座面高さ (cm) ※実測", min_value=0, max_value=200, value=40)
            height_work = st.number_input("作業高さ (cm) ※実測", min_value=0, max_value=200, value=75)
            
            st.write("---")
            st.write("### 🩺 理学療法士の目による問題箇所の選択")
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
                st.session_state.captured_image = None
                st.session_state.captured_angles = {}
                st.session_state.show_report = False
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
            
            default_eva = "やや首が前方に傾いた姿勢です。" if "首の傾き" in selected_issues else "全体的なアライメントは概ね良好です。"
            if "座面高さ" in selected_issues:
                default_eva += f"椅子の高さが{angles['height_chair']}cmと実測され、作業環境とのミスマッチによる負担が懸念されます。"
                
            comment_eva = st.text_area("■ 理学療法士の評価", value=default_eva, height=200)
            comment_advice = st.text_area("■ 理学療法士の助言", value="【環境調整案】\n・作業高さを調整し、頸部屈角（首の傾き）が強くならない視線を確保しましょう。\n\n【身体へのアプローチ】\n・", height=200)
            
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
                    st.session_state["save_success_msg"] = f"🎉 レポートが確定し、CSVデータベースに蓄積されました！ (ID: {saved_id})"
                    st.rerun()
                except Exception as e:
                    st.error(f"データの保存中にエラーが発生しました: {e}")

        if "save_success_msg" in st.session_state:
            st.success(st.session_state["save_success_msg"])
            del st.session_state["save_success_msg"]
                    
elif app_mode == "レポート履歴":
    st.subheader("📁 過去の作業環境評価レポート履歴")
    
    import os
    import pandas as pd
    
    CSV_FILE_PATH = "posture_report_database.csv"
    
    if not os.path.isfile(CSV_FILE_PATH):
        st.info("まだ保存されたレポート履歴がありません。新規計測を行ってデータを確定させてください。")
    else:
        try:
            df = pd.read_csv(CSV_FILE_PATH, encoding="utf-8-sig")
            if df.empty:
                st.info("データが空です。")
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
                    st.write(f"## 📝 レポート詳細 : {full_data['report_id']}")
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
                                f"・大腿角度: {full_data['angle_femoral']} 度\n"
                                f"・膝の角度: {full_data['knee']} 度\n\n"
                                f"・座面高さ: {full_data['height_chair']} cm\n"
                                f"・作業高さ: {full_data['height_work']} cm")
                        
                        st.write("---")
                        if full_data["selected_issues"] != "特になし":
                            st.error(f"⚠️ **要注意箇所:** {full_data['selected_issues']}")
                        else:
                            st.success("✅ **要注意箇所:** 特になし（良好）")
                    
                    st.write("---")
                    st.write("### 🩺 専門職による評価・指導内容")
                    st.info(f"**■ 理学療法士の評価**\n\n{full_data['comment_evaluation']}")
                    st.success(f"**■ 理学療法士の助言・環境調整案**\n\n{full_data['comment_advice']}")
                    
        except Exception as e:
            st.error(f"履歴の読み込み中にエラーが発生しました: {e}")