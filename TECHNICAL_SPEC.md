# [종합 기술 명세서] Time-MoE 기반 단일 클래스 이상 탐지 파이프라인

본 명세서는 RTX 3090 환경에서 구동 중인 이상 탐지 시스템의 재현성(Reproducibility)을 95% 이상 보장하기 위한 공학적 설계와 워크플로우를 기술합니다.

---

## 1. 데이터 거버넌스 및 전처리 (Data Lifecycle)

### **채널 구성 및 마스킹 전략**
- **Channel Independence**: 3상 전류 및 1상 자속 데이터를 다변량 시퀀스가 아닌 독립된 단변량(Univariate) 시계열들의 집합으로 처리. 모델은 각 채널의 고유 주파수 특성을 개별 학습함.
- **Sensor Masking Simulation**: 자속 센서(CH4)의 부재를 가정하여 해당 채널의 모든 데이터를 0(Zero-masking)으로 고정. 모델이 전류 데이터에만 의존하여 이상을 탐지하도록 강제함.

### **수치적 안정화 (Preprocessing Safety)**
- **Safe Z-Score 스케일링**: 
    - 표준편차가 극도로 작거나 0인 경우(Flat-line) 발생하는 NaN을 방지하기 위해 분모에 Epsilon(`1e-6`)을 추가.
    - 입력값에 대해 `nan_to_num`을 다단계(로딩, 정규화, 입력 직전)로 적용하여 수치적 폭주를 원천 차단함.
- **Data Augmentation**: 윈도우 추출 시 `random_offset`을 `[0, min(stride-1, total_points - window_size)]` 범위 내에서 부여하여 신호의 위상(Phase) 변화에 대한 불변성을 확보함.
- **바이너리 직렬화**: 대용량 데이터를 처리하기 위해 NPY 포맷을 거쳐 최종적으로 고정 크기 블록의 바이너리 파일로 병합하고, 인덱스 메타데이터를 통해 랜덤 액세스 성능을 극대화함.

---

## 2. 모델 아키텍처 및 설정 (Architecture Options)

### **모델 규모 선정 (Sizing Rationale)**
- **Target Architecture**: Time-MoE Base (약 50M Parameters).
- **선정 근거**: 24GB VRAM 자원과 1M 포인트 규모의 데이터셋에 최적화된 "적정 기술" 규모. 과적합을 방지하면서도 MoE의 전문가 분산 학습 효과를 충분히 누릴 수 있는 규모임.

### **세부 구조 (Structural Features)**
- **Backbone**: Llama 스타일의 Decoder-only Transformer.
    - **Normalization**: RMSNorm 적용 (Epsilon `1e-5`로 상향하여 수치 안정성 강화).
    - **Positioning**: Rotary Positional Embedding (RoPE) 사용.
    - **Activation**: SwiGLU 기반 MLP 블록.
- **Sparse MoE Mechanism**:
    - **Top-K Routing**: 토큰(포인트) 단위로 상위 2개의 전문가를 동적으로 선택.
    - **Logit Scaling**: 소프트맥스 전 최대값 차감(Max-subtraction)을 통해 지수 폭발 방지.
    - **Shared Expert**: 시그모이드 게이팅 기반의 공유 전문가를 전문가 그룹과 병렬 배치하여 기초적인 공통 특징을 보존함.
- **Multi-resolution Heads**: {1, 8, 32, 64} 해상도의 출력 투영층을 배치하여 다양한 시간 단위의 이상을 동시 포착.

---

## 3. 하이퍼파라미터 및 학습 전략 (Training Strategy)

### **핵심 파라미터 및 최적화**
- **Learning Rate**: 1e-4 (표준 Transformer 스크래치 학습 수준). 안정적 안착을 위해 초기 선형 웜업(Warmup) 스케줄링 적용.
- **Effective Batch Size**: 128 (물리 배치 8 × 그래디언트 누적 16). MoE 라우팅의 통계적 안정성을 보장할 만큼 충분히 큰 배치를 형성함.
- **Loss Function**: Huber Loss (오차 민감도 조절용).
- **Optimizer**: 8-bit AdamW 또는 Pure AdamW (GPU-bound 최적화).

### **수치적 고도화**
- **Router Weight Re-initialization**: 게이트 선형층 가중치를 매우 작은 표준편차(`0.01`)로 초기화하여 학습 초기에 특정 전문가로 출력이 쏠리는 현상(Expert Collapse)을 원천 방지함.
- **Aux Loss Warmup**: 로드 밸런싱 보조 손실 가중치를 0에서 목표치까지 선형적으로 증가시켜, 모델이 "예측 능력"을 먼저 확보한 뒤 전문가 학습을 수행하도록 유도함.

---

## 4. 자동화 에이전트 및 평가 메커니즘 (Agent Workflow)

### **3시간 주기 하트비트 루틴**
학습 에이전트는 실시간 경과 시간을 감시하며 다음 루틴을 수행함:
1. **정상 평균 프로파일링**: 학습 데이터(Normal)의 은닉 상태를 샘플링하여 "정상 상태 평균 벡터"를 실시간 갱신.
2. **Hybrid Anomaly Score 계산**:
    - **L1 (MSE Score)**: 다해상도 예측 오차의 가중합.
    - **L2 (Latent Distance)**: 은닉 상태와 정상 데이터 평균 벡터 간의 유클리드 거리.
    - **Final Score**: `L1 + 0.1 * L2` 가중 결합을 통해 물리적/통계적 이상치를 동시 탐지.
3. **지표 최적화 (F-0.5 Score)**: Precision(오탐 방지)에 가중치를 둔 F-beta 지표를 최대화하는 최적 임계치를 자동 검색.
4. **Dynamic Tuning**: F1 점수가 목표치(0.950) 미달 시 Huber Loss의 Delta 값을 점진적으로 감쇠(`0.98배`)시켜 오차에 대한 민감도를 강화함.

---

## 5. 실행 및 성능 최적화 (System Optimization)

- **VRAM 관리**: BF16 정밀도와 Flash-Attention 2를 결합하여 RTX 3090의 24GB 내에서 고속 연산 수행.
- **체크포인트 전략**: 주기적 평가 시점마다 성능 수치를 파일명에 포함하여 영구 저장하여 최적 시점 복원 지원.
- **로깅**: Python Unbuffered 모드를 통해 실시간 스트리밍 로그를 확보하여 원격 모니터링 편의성 제공.
