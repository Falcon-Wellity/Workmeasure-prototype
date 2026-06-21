import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd # 履歴の読み込みを簡単にするため追加

# 【重要】WebRTC（ブラウザカメラ）用部品を追加
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av

# CSV蓄積モジュールをインポート
from csv_db import save_report_to_csv, CSV_FILE_PATH

# ==========================================
# 1. MediaPipe ＆ 画像処理ライブラリの初期設定
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 2. 姿勢分析のための幾何学計算関数（幾何学ロジック）
# ==========================================
def calculate_angle(a, b, c):
    """3つの座標点から、点bを頂点とする関節角度（度数法：0〜180度）を計算する関数"""
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
    """股関節(a)と膝(b)を結ぶ線分が、水平線からどれだけ傾いているかを計算する（大腿角用）"""
    a = np.array(a)
    b = np.array(b)
    c = np.array([b[0], a[1]])
    return calculate_angle(b, a, c)

def calculate_vertical_angle(a, b):
    """股関節(b)と肩(a)を結ぶ線分が、垂直線からどれだけ傾いているかを計算する（体幹前傾角用）"""
    a = np.array(a)
    b = np.array(b)
    c = np.array([b[0], a[1]])
    return calculate_angle(a, b, c)

# ==========================================
# 【新規】ブラウザから送られてくる映像をリアルタイム処理するクラス
# ==========================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. ブラウザから届いた映像をnumpy配列（RGB）に変換
        image = frame.to_ndarray(format="rgb24")
        
        # 2. MediaPipeで骨格分析
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        
        current_angles = {}
        
        if results.pose_landmarks:
            # 画面上に骨格を描画
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
                    "back": int(angle_back),
                    "vertical": int(angle_trunk_vertical),
                    "hip": int(angle_hip),
                    "femoral": int(angle_femoral),
                    "knee": int(angle_knee),
                    "side": side_text
                }
                
                # グローバル変数等を経由せず、セッション経由でUIとデータを同期
                st.session_state["live_angles"] = current_angles
                st.session_state["live_image"] = image
                
            except:
                pass
        
        return av.VideoFrame.from_ndarray(image, format="rgb24")

# ==========================================
# 3. Streamlit UI 画面レイアウト定義
# ==========================================
st.set_page_config(page_title="WorkMeasure プロトタイプ", layout="wide")
st.title("WorkMeasure (ワークメジャー) - 姿勢計測プロトタイプ")

# サイドバーメニュー
st.sidebar.header("メニュー")
app_mode = st.sidebar.selectbox("操作を選択", ["新規姿勢計測", "レポート履歴"])

if app_mode == "新規姿勢計測":
    st.subheader("⚠️ 横向きから、全身（耳・肩・股関節・膝・足）が写るように撮影してください")

    col1, col2 = st.columns(2)
    with col1:
        work_env = st.selectbox("作業環境を選択", ["デスクワーク", "軽作業（立ち仕事）", "重作業"])
    with col2:
        work_posture = st.selectbox("作業姿勢を選択", ["座位", "立位"])

    # メイン表示エリアの分割
    display_col, result_col = st.columns([2, 1])

    # メモリ保持用 st.session_state の初期化
    if "captured_image" not in st.session_state:
        st.session_state.captured_image = None      # 撮影写真
    if "captured_angles" not in st.session_state:
        st.session_state.captured_angles = {}       # 撮影時の角度データ
    if "show_report" not in st.session_state:
        st.session_state.show_report = False        # レポート表示フラグ
    if "live_angles" not in st.session_state:
        st.session_state.live_angles = {}           # リアルタイム角度の一時保持
    if "live_image" not in st.session_state:
        st.session_state.live_image = None          # リアルタイム映像の一時保持

    # --- パターンA：まだ撮影していない（リアルタイムカメラ起動状態） ---
    if st.session_state.captured_image is None:
        with display_col:
            st.write("### 🎥 カメラ映像")
            # WebRTC経由でブラウザのカメラを安全に起動
            ctx = webrtc_streamer(
                key="pose-detector",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=PoseProcessor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
        with result_col:
            st.write("### 📊 リアルタイム角度分析（参考値）")
            
            # セッションから最新の値を取り出して表示
            live = st.session_state.live_angles
            st.metric(label="首の傾き", value=f"{live.get('back', 0)} 度")
            st.metric(label="体幹の前傾角度 (垂直線基準)", value=f"{live.get('vertical', 0)} 度")
            st.metric(label="股関節の角度", value=f"{live.get('hip', 0)} 度")
            st.metric(label="大腿の角度 (水平線基準)", value=f"{live.get('femoral', 0)} 度")
            st.metric(label="膝の角度", value=f"{live.get('knee', 0)} 度")
            
            st.write("---")
            capture_button = st.button("📸 この姿勢を記録する（シャッター）", use_container_width=True)

            # シャッターボタンが押されたら現在のライブコマを固定保存
            if capture_button and st.session_state.live_image is not None:
                st.session_state.captured_image = st.session_state.live_image
                st.session_state.captured_angles = st.session_state.live_angles
                st.rerun()

    # --- パターンB：すでに写真を撮影した状態（静止画確認 ＆ セルフチェック ＆ レポート生成） ---
    else:
        with display_col:
            st.info("📌 以下の姿勢で写真を一時保存しました。")
            st.image(st.session_state.captured_image)  
            
        with result_col:
            st.write("### 🔒 計測角度データ（参考値）")
            angles = st.session_state.captured_angles
            st.write(f"・首の傾き: **{angles.get('back', 0)} 度**")
            st.write(f"・体幹の前傾: **{angles.get('vertical', 0)} 度**")
            st.write(f"・股関節角度: **{angles.get('hip', 0)} 度**")
            st.write(f"・大腿の角度: **{angles.get('femoral', 0)} 度**")
            st.write(f"・膝の角度: **{angles.get('knee', 0)} 度**")
            
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
            if st.button("🔄 撮り直す（データを破棄）", type="secondary", use_container_width=True):
                st.session_state.captured_image = None
                st.session_state.captured_angles = {}
                st.session_state.show_report = False
                st.session_state.live_image = None
                st.session_state.live_angles = {}
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
                st.markdown(f"**計測データ一覧 ({angles.get('side', '不明')}: 参考値):**")
                st.text(f"・首の傾き: {angles.get('back', 0)}度\n・体幹前傾: {angles.get('vertical', 0)}度\n・股関節角: {angles.get('hip', 0)}度\n・大腿角度: {angles.get('femoral', 0)}度\n・膝の角度: {angles.get('knee', 0)}度")
            with rep_col2:
                st.markdown(f"**環境実測データ:**")
                st.text(f"・座面高さ: {angles.get('height_chair', 0)} cm\n・作業高さ: {angles.get('height_work', 0)} cm")
                
                if selected_issues:
                    issue_text = "、".join(selected_issues)
                    st.error(f"⚠️ **要注意箇所:** {issue_text}")
                else:
                    st.success("✅ **要注意箇所:** 特になし（良好なアライメントです）")
            
            st.write("### ✍️ 専門職コメント入力（ここから自由にレポートを編集できます）")
            
            default_eva = "やや首が前方に傾いた姿勢です。" if "首の傾き" in selected_issues else "全体的なアライメントは概ね良好です。"
            if "座面高さ" in selected_issues:
                default_eva += f"椅子の高さが{angles.get('height_chair', 0)}cmと実測され、作業環境とのミスマッチによる負担が懸念されます。"
                
            comment_eva = st.text_area("■ 理学療法士の評価", value=default_eva, height=200)
            comment_advice = st.text_area("■ 理学療法士の助言", value="【環境調整案】\n・作業高さを調整し、頸部屈曲角（首の傾き）が強くならない視線を確保しましょう。\n・\n\n【身体へのアプローチ】\n・", height=200)
            
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
                    st.success(f"🎉 レポートが確定し、写真付きでCSVデータベースに蓄積されました！ (ID: {saved_id})")
                except Exception as e:
                    st.error(f"データの保存中にエラーが発生しました: {e}")

# ==========================================
# 4. レポート履歴機能
# ==========================================
elif app_mode == "レポート履歴":
    st.subheader("📋 過去の作業環境評価レポート履歴")
    
    if not os.path.isfile(CSV_FILE_PATH):
        st.info("まだ確定されたレポート履歴がありません。「新規姿勢計測」からレポートを作成してください。")
    else:
        try:
            df = pd.read_csv(CSV_FILE_PATH, encoding="utf-8-sig")
            df = df.iloc[::-1]
            
            report_options = df.apply(lambda row: f"{row['report_id']} ({row['created_at']}) - {row['work_env']}", axis=1).tolist()
            selected_option = st.selectbox("確認したいレポートを選択してください", report_options)
            
            selected_index = report_options.index(selected_option)
            report_data = df.iloc[selected_index]
            
            st.write("---")
            st.write(f"### 📄 レポート詳細: {report_data['report_id']}")
            st.write(f"**計測日時:** {report_data['created_at']}")
            
            hist_col1, hist_col2 = st.columns([1, 1])
            
            with hist_col1:
                st.write("#### 📸 撮影時の姿勢アライメント")
                img_path = report_data['image_path']
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning("⚠️ このレポートに対応する画像ファイルが見つかりません。")
            
            with hist_col2:
                st.write("#### 📊 計測数値データ")
                
                metrics_text = f"""・対象環境 / 姿勢: {report_data['work_env']} / {report_data['work_posture']}
・計測方向: {report_data['measured_side']}
・首の傾き: {report_data['angle_back']} 度
・体幹前傾: {report_data['angle_vertical']} 度
・股関節角: {report_data['angle_hip']} 度
・大腿角度: {report_data['angle_femoral']} 度
・膝の角度: {report_data['angle_knee']} 度
・座面高さ: {report_data['height_chair']} cm
・作業高さ: {report_data['height_work']} cm"""

                st.text_area(
                    label="計測値一覧", 
                    value=metrics_text, 
                    height=240, 
                    disabled=True, 
                    label_visibility="collapsed", 
                    key=f"hist_metrics_{report_data['report_id']}"
                )
                
                issues_val = str(report_data['selected_issues'])
                if issues_val != "特なし" and issues_val != "特になし" and issues_val != "":
                    st.error(f"⚠️ **要注意箇所:** {issues_val}")
                else:
                    st.success("✅ **要注意箇所:** 特になし（良好）")
                    
            st.write("---")
            st.write("#### 🩺 理学療法士の評価・コメント")
            
            st.text_area(
                "■ 理学療法士の評価（過去の記録）", 
                value=report_data['comment_evaluation'], 
                height=150, 
                key=f"hist_eva_{report_data['report_id']}"
            )
            
            st.text_area(
                "■ 理学療法士の助言（過去の記録）", 
                value=report_data['comment_advice'], 
                height=150, 
                key=f"hist_adv_{report_data['report_id']}"
            )
            
        except Exception as e:
            st.error(f"履歴の読み込み中にエラーが発生しました: {e}")