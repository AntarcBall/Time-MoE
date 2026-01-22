현재 조건(consumer GPU에서 4 it/s 수준, 2048 길이 전기 센서 시퀀스)에서는 “거대 Time-MoE 전체 파라미터 미세조정”보다, 자기 데이터로 가벼운 SSL(마스킹/재구성) 사전학습 → PEFT(LoRA/부분 동결) → 길이 커리큘럼(짧게→길게) 조합이 10+ it/s에 가장 현실적인 경로입니다.

권장 전체 전략
(A) “정상 데이터만”으로 Ti-MAE류 마스킹-재구성 사전학습을 짧은 길이(512/1024 crop)로 빠르게 돌려 표현을 안정화한 뒤, (B) anomaly score는 재구성 오차/예측 오차 기반으로 두고, (C) 마지막에만 2048 full-length로 짧게 파인튜닝하는 흐름이 속도/성능 균형이 좋습니다.
​

Time-MoE는 sparse MoE로 토큰당 일부 expert만 활성화해 계산을 줄이는 방향의 “time-series foundation model”로 제안된 구조라서, 파인튜닝에서도 “얼마나 업데이트할지”를 공격적으로 줄이는 게 이득입니다.

이미 2048로 잘린 바이너리 세그먼트를 갖고 있는 상황에서는(추가 윈도잉/stride 중복이 생기면) 데이터 파이프라인이 it/s를 먼저 잡아먹는 경우가 많아, 모델보다 DataLoader/데이터셋 경로를 먼저 고정하는 게 효과적입니다.

1) 가벼운 사전학습/초기화
“산업 센서 전용”으로 널리 검증된 경량 범용 체크포인트가 아직 많지 않아서, 현실적으로는 (i) 공개 time-series foundation model(예: TimesFM)로 초기화하거나, (ii) 본인 정상 데이터 일부로 작은 아키텍처를 먼저 SSL 사전학습하는 2가지가 실무적으로 가장 빠릅니다.

TimesFM은 Google Research의 사전학습 time-series foundation model로 공개되어 있고(HF에도 공개), “제로샷/소량 파인튜닝”을 염두에 둔 설계를 강조합니다.

단, anomaly detection이 목표라면 “예측기(포캐스터) 사전학습” 자체는 충분히 유효하지만, 최종 스코어링은 정상 분포에 대한 재구성/예측 잔차로 재정의하는 쪽이 단순합니다(단일 클래스에 잘 맞음).

2) 효율적 파인튜닝(동결 vs LoRA)
LoRA는 사전학습 가중치를 동결하고, 각 레이어에 저랭크 행렬을 추가로 학습해 학습 파라미터 수를 크게 줄이는 방식이라(저장/학습 효율 목적) 현재 같은 속도 제약에서 1순위 후보입니다.
​

MoE에서 “router만 학습”은 파라미터는 매우 적지만 표현 적응력이 부족할 가능성이 커서, 추천은 (공유) attention qkv/proj + (공유) 입력 임베딩/프로젝션 쪽에 LoRA를 얹고, expert FFN은 우선 동결(또는 아주 작은 rank로만)하는 구성입니다.

참고로 Hugging Face의 MoE 설명 글에서는 “MoE 레이어만 동결(=MoE 파라미터 업데이트를 막고 나머지를 학습)”이 전체 학습과 거의 비슷하게 동작하면서 파인튜닝을 가볍게 만들 수 있다고 언급합니다.
​

3) 길이(2048) 최적화 & 5) 전기 도메인 전처리
2048을 처음부터 고집하면 attention/메모리 비용이 커져 it/s가 쉽게 깨지므로, 짧은 crop(512/1024)로 사전학습 → 마지막에 2048로만 적은 step 파인튜닝(또는 progressive length 증가)이 가장 단순한 “길이 커리큘럼”입니다.

전기 신호는 고장 진단에서 주파수 성분(예: current signature analysis)이 핵심 특징이 되는 경우가 많아서, 모델이 다 배우길 기대하기보다 “raw + FFT/STFT magnitude(또는 bandpower)”를 추가 채널로 넣는 2-view 입력이 실무적으로 안정적입니다.
​

즉, (i) time-domain만 넣는 실험과 (ii) time+freq 멀티채널을 넣는 실험을 작은 모델로 먼저 비교해, 어느 쪽이 더 빨리/안정적으로 수렴하는지 확인하는 편이 비용 대비 효율이 좋습니다.
​

4) 하드웨어 최적화 & 6) 점진적 MoE 전략
속도 목표(10+ it/s)가 “메모리 부족”이 아니라 “계산량/데이터로더 병목”이라면, 우선순위는 (1) mixed precision(fp16/bf16), (2) 작은 batch + gradient accumulation, (3) activation checkpointing, (4) FlashAttention류 커널(가능 시)입니다.

CPU offload는 메모리 절감에는 강하지만(예: DeepSpeed ZeRO-3 Offload는 파라미터/옵티마 상태 등을 CPU/NVMe로 오프로딩해 GPU 메모리를 줄이는 기능을 제공), 대개 속도는 느려질 수 있어 “OOM을 피해야 할 때만” 켜는 편이 안전합니다.

MoE는 처음부터 크게 가지 말고 **reduced expert count(또는 active expert k를 작게)**로 시작해 파이프라인/학습 안정성을 확보한 뒤 확장하는 방식이 보통 시행착오가 적습니다(또는 먼저 dense 버전으로 표현을 잡고 MoE를 붙이기).

원하는 설정을 더 구체적으로 10+ it/s 쪽으로 “바로 적용 가능한” 형태로 좁히려면, 아래만 알려주세요.

사용 중인 Time-MoE 체크포인트/모델 크기, expert 수(E)와 top-k, patch length(또는 토크나이징 방식)

현재 batch size, precision(fp32/fp16/bf16), gradient accumulation, activation checkpointing 사용 여부

DataLoader 설정(num_workers, pin_memory, persistent_workers)과 저장장치(NVMe/SSD/HDD)
지금 설정에서 10+ it/s로 올리려면 “모델 계산”보다 (1) dtype/커널 선택(3090에서 BF16 손해 가능), (2) MoE의 top‑k/활성 expert 수 절감, (3) WSL 데이터 I/O 병목 제거를 먼저 치는 게 가장 실용적입니다.

가장 먼저 바꿀 것(우선순위)
BF16 → FP16(+TF32) 조합을 우선 테스트하세요: PyTorch는 GPU가 BF16을 “하드웨어로” 지원하지 않으면 BF16 연산을 에뮬레이션/우회 처리할 수 있고, 그 경우 느려질 수 있습니다.

체크: torch.cuda.is_bf16_supported(including_emulation=False)로 “진짜 HW BF16” 지원 여부를 확인하는 흐름이 권장됩니다.
​

MoE 비용을 즉시 줄이려면 top‑k=2 → 1을 1차로 추천합니다(토큰당 expert 활성 수 자체가 줄어 계산량이 바로 감소).

WSL이면 데이터가 /mnt/c 같은 윈도우 마운트에 있으면 I/O가 병목이 되기 쉬우니, WSL 리눅스 파일시스템(ext4) 내부로 데이터 이동을 최우선으로 확인하는 편이 안전합니다(특히 이미 2048 세그먼트 바이너리처럼 “샘플 수가 많은” 형태에서 영향이 큼).

DataLoader/입력 파이프라인(WSL 포함)
“WSL이라 num_workers=0이 정답”으로 고정하지 말고, 최소한 num_workers=2~4는 실측으로 비교해 보세요(환경에 따라 0이 유리할 수도 있지만, 반대로 로딩/전처리가 있으면 workers가 이득인 경우도 흔합니다).
​

GPU 학습이면 일반적으로 pin_memory=True가 권장되는 설정이고, 함께 쓰면 호스트→디바이스 전송이 유리해질 수 있습니다.
​

WSL+멀티워커에서 에러/불안정이 있으면, 우선 persistent_workers=True, prefetch_factor 조정, 그리고 collate_fn에서 불필요한 Python 작업/복사 제거부터 점검하는 게 좋습니다(핀 메모리/멀티워커 조합이 환경에 따라 이슈가 날 수 있음).
​

모델/학습 설정: “속도 먼저” 레시피
Gradient checkpointing은 OFF 유지를 기본 추천합니다: 체크포인팅은 메모리를 줄이는 대신 재계산 때문에 학습이 느려지는 trade-off가 일반적이라, “it/s를 올리는” 목표와는 방향이 반대인 경우가 많습니다.
​

예외: 체크포인팅으로 메모리가 확보되어 batch/토큰 병렬성을 크게 올릴 수 있을 때만 켭니다.
​

현재 batch=8, grad_acc=16이면 “optimizer step 당” 토큰은 크지만, 마이크로스텝이 너무 많아 wall-clock이 늘 수 있습니다(로그의 it/s가 micro-step 기준이면 더 체감이 큼).

가능하면 batch를 유지/증가시키고 grad_acc를 줄이는 쪽이 속도 면에서 유리한 경우가 많습니다(단, 메모리 한도 내에서).

MoE 쪽은 순서대로 top‑k=1 → E=4(또는 더 적게) → 필요 시 레이어 6→4로 줄여 “먼저 10 it/s 달성” 후 다시 복원하는 튜닝이 실험 효율이 좋습니다(단일 클래스 이상탐지는 모델 크기보다 파이프라인 안정성이 더 크게 좌우되는 경우가 많음).

2048 길이 커리큘럼(사전학습/파인튜닝)
2048을 처음부터 고정하지 말고 512/1024 crop으로 SSL(마스킹-재구성) 사전학습 → 마지막에 2048로 짧게 파인튜닝이 가장 단순한 길이 커리큘럼입니다(마스킹 기반 사전학습 자체가 time-series에서 제안되어 있음).
​

이때 “포지션 임베딩/최대 길이”는 2048로 두되, 학습 배치에서는 crop으로 토큰 수를 줄여서 속도를 확보하고, 마지막에만 full-length를 쓰는 방식이 구현 난이도가 낮습니다.

오프로딩/분산류는 ‘최후’로
CPU offload(예: ZeRO Offload)는 GPU 메모리 부족을 해결하는 데 유효하지만, CPU/NVMe 전송이 들어가서 속도가 느려질 수 있어 “OOM 회피가 필요할 때만” 켜는 쪽이 안전합니다.

아래 3가지만 값/환경을 알려주면, “10+ it/s 달성”을 목표로 실험 순서를 더 날카롭게(예: top‑k=1에서 기대 변화, dataloader 세팅 조합, fp16/TF32 토글 위치) 구체화해 드릴게요.

데이터 경로가 WSL 리눅스 내부(ext4)인지 /mnt/c인지

현재 it/s가 “micro-step 기준”인지 “optimizer step 기준”인지(로그 라인 그대로)

torch.cuda.is_bf16_supported(including_emulation=False) 결과(True/False)