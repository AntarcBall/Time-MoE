import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
import time

st.set_page_config(page_title="Time-MoE AD Monitor", layout="wide")

st.title("🎓 Time-MoE 50M Anomaly Detection 실시간 브리핑")

# 사이드바: 시스템 상태
st.sidebar.header("🖥️ System Status")
try:
    gpu_info = os.popen('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits').read()
    if gpu_info:
        util, mem_used, mem_total = gpu_info.strip().split(',')
        st.sidebar.metric("GPU Utilization", f"{util}%")
        st.sidebar.progress(int(util)/100)
        st.sidebar.metric("VRAM Usage", f"{mem_used}MB / {mem_total}MB")
except:
    st.sidebar.error("Could not fetch GPU info.")

# 1. 체크포인트 현황
st.header("📂 Saved Checkpoints")
ckpt_dir = "Time-MoE/checkpoints"

# 로그 데이터 먼저 파싱 (체크포인트와 매칭하기 위함)
log_path = "Time-MoE/run_base.log"
log_df = pd.DataFrame()
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        log_content = f.read()
    pattern = r"\|\s+(\d+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|"
    matches = re.findall(pattern, log_content)
    if matches:
        log_df = pd.DataFrame(matches, columns=['Step', 'Loss', 'Gating', 'F1-L1', 'F1-L2', 'F1-Total'])
        for col in log_df.columns:
            log_df[col] = pd.to_numeric(log_df[col])

if os.path.exists(ckpt_dir):
    ckpts = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
    
    ckpt_list = []
    for c in ckpts:
        path = os.path.join(ckpt_dir, c)
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
        try:
            size = os.popen(f"du -sh {path}").read().split()[0]
        except:
            size = "N/A"
        
        # 폴더명에서 스텝 번호 추출
        step_match = re.search(r'step-(\d+)', c)
        step_val = int(step_match.group(1)) if step_match else -1
        
        # 로그 데이터에서 해당 스텝의 Loss 찾기
        ckpt_loss = -1.0 # 기본값 float
        if step_val != -1 and not log_df.empty:
            matched_row = log_df[log_df['Step'] == step_val]
            if not matched_row.empty:
                try:
                    ckpt_loss = float(matched_row.iloc[0]['Loss'])
                except:
                    ckpt_loss = -1.0
        
        ckpt_list.append({
            "Checkpoint": c, 
            "Step": step_val, 
            "Loss": ckpt_loss, # 컬럼명 단순화 및 float 보장
            "Saved At": mtime, 
            "Size": size
        })
    
    if ckpt_list:
        st.table(pd.DataFrame(ckpt_list))
    else:
        st.info("No checkpoint folders found yet.")
else:
    st.warning("Checkpoints directory not found.")

# 2. 에이전트 리포트 분석 (F1 Score 추이)
st.header("📊 Agent Performance Report")
if not log_df.empty:
    df = log_df
    # Summary Metrics
    latest = df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Step", int(latest['Step']))
    m2.metric("Latest Loss", f"{latest['Loss']:.4f}")
    m3.metric("F1-Total", f"{latest['F1-Total']:.4f}")
    m4.metric("Gating Balance", f"{latest['Gating']:.4f}")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Loss Convergence")
        st.line_chart(df.set_index('Step')['Loss'])
    with col2:
        st.subheader("📈 F1 Score Progress")
        st.line_chart(df.set_index('Step')[['F1-L1', 'F1-L2', 'F1-Total']])
        
    st.subheader("📋 Historical Agent Reports")
    # use_container_width deprecated -> width
    st.dataframe(df.sort_values('Step', ascending=False))
else:
    if os.path.exists(log_path):
        st.info("Agent reports (Step | Loss | F1...) not found in log yet. Waiting for first 3h evaluation...")
    else:
        st.error(f"Log file '{log_path}' not found. Training might not have started correctly.")

# 3. Deep Diagnosis (심층 분석)
st.divider()
st.header("🔬 Deep Diagnosis")

if os.path.exists(ckpt_dir):
    ckpts = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
    
    if ckpts:
        selected_ckpt = st.selectbox("Select Checkpoint for Analysis", ckpts)
        
        if st.button("🚀 Run Deep Analysis (Warning: Consumes VRAM)"):
            with st.spinner("Analyzing... This may take 1-2 minutes..."):
                # Run external script
                # Note: Assuming training is NOT running or VRAM is shared
                import subprocess
                ckpt_path = os.path.join(ckpt_dir, selected_ckpt)
                output_dir = os.path.join("Time-MoE/analysis_results", selected_ckpt)
                
                # Check VRAM safety (simple check)
                # If training is running, this might fail or slow down training
                # We add a warning
                st.warning("⚠️ Running analysis while training is active may cause OOM. Pause training if needed.")
                
                cmd = ["python3", "Time-MoE/run_deep_analysis.py", ckpt_path, output_dir]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode == 0:
                    st.success("Analysis Completed!")
                else:
                    st.error(f"Analysis Failed:\n{process.stderr}")

        # Display Results if available
        # 버튼을 누르지 않아도, 이미 분석된 결과가 있으면 보여주도록 로직 수정
        output_dir = os.path.join("Time-MoE/analysis_results", selected_ckpt) if 'selected_ckpt' in locals() else None
        
        if output_dir and os.path.isdir(output_dir):
            # Check if any images exist
            images = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            if images:
                st.subheader(f"Results for {selected_ckpt}")
                tab1, tab2, tab3, tab4 = st.tabs(["Score Dist", "Expert Heatmap", "FFT Spectrum", "PR Curve"])
                
                def show_plot(filename):
                    full_path = os.path.join(output_dir, filename)
                    if os.path.exists(full_path):
                        st.image(full_path, caption=filename)
                    else:
                        st.warning(f"File {filename} not found.")

                with tab1: show_plot("1_score_distribution.png")
                with tab2: show_plot("2_expert_heatmap.png")
                with tab3: show_plot("3_fft_spectrum.png")
                with tab4: show_plot("5_pr_curve.png")
            else:
                st.info("Analysis folder exists but contains no images. Run analysis to generate plots.")

# Auto-refresh logic
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Auto-refreshing every 60s.")
time.sleep(60)
st.rerun()
