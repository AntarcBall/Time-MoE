말씀하신 증상(3090 단일 GPU에서 2.4B Time‑MoE, 1.3h/step + step1 이후 loss=0/grad_norm=nan)은 “(A) 시스템적으로는 offload/accumulation 때문에 처리량이 바닥이고, (B) 수치적으로는 라우터/aux loss/분산 설정(특히 DeepSpeed ZeRO-2 조합)에서 초기 라우팅 붕괴 또는 NaN 전파가 난 케이스”로 보는 게 가장 합리적입니다.
​
아래는 초기 붕괴 방지(Initialization), 처리량 최적화(3090 한계 포함), NaN/0 loss 디버깅 루틴, 스케일 다운 권장 파라미터 범위, 그리고 바로 붙여 쓸 설정 스니펫(ZeRO-2 / LoRA 대안) 순서로 정리합니다.
​

MoE 초기화(붕괴 방지)
MoE에서 “loss가 즉시 0으로 가거나(실제로는 underflow/NaN가 0처럼 로깅되는 경우 포함), 특정 expert로 라우팅이 몰리면서 학습이 멈추는” 문제는 라우터의 초기 로짓 스케일/바이어스/보조 손실이 초반에 너무 강하면 자주 발생합니다.
​
또한 Time‑MoE류는 라우팅 collapse를 막기 위해 main loss 외에 load-balancing용 aux loss를 추가하는 설계를 강조합니다.
​

라우터(게이트) 가중치 초기화

원칙: 초기에는 “거의 균등 라우팅 + 높은 엔트로피”가 되도록 라우터 로짓의 분산을 작게 시작합니다.
​

실전 팁: 라우터의 마지막 linear( 
W
g
W 
g
  )를 “아주 작은 표준편차(예: 1e-3~1e-2)”로 초기화하고 bias는 0으로 두어 초반에 softmax가 한 expert로 포화되지 않게 합니다.
​

Top‑2 routing이면, 초기 로짓 스케일이 크면 곧바로 arg-top2가 고정되기 쉬우므로(학습 초반 탐색이 사라짐) 라우터 로짓을 작게 시작하는 쪽이 안전합니다.
​

Expert MLP 초기화

Expert들은 일반 transformer MLP 초기화(Xavier/Kaiming 계열 + residual 스케일링)로 두되, “expert 간 출력 스케일 편차”가 크면 라우터가 특정 expert를 선호하는 피드백 루프가 생길 수 있어, expert weight init 분산을 통일하는 것이 중요합니다.
​

특히 SwiGLU/게이트드 MLP는 두 linear의 스케일이 엇갈리면 초반 출력 분산이 커질 수 있어, 라우터보다 expert 쪽을 과도하게 크게 초기화하지 않는 것이 안정적입니다.
​

Aux loss 스케줄링(매우 중요)

load-balancing auxiliary loss는 크면 “interference gradient”를 만들어 본 손실을 망가뜨릴 수 있다는 지적이 있으며, 초반에 특히 민감합니다.
​

전략: router_aux_loss_factor를 warmup(예: 0 → target까지 1k~10k step)시키거나, 초반에는 0으로 두고 라우터가 기본 표현을 어느 정도 잡은 뒤 켜는 방식이 붕괴 방지에 유리합니다.
​

처리량(1.3h/step) 현실성 & 가속 포인트
3090(24GB)에서 2.4B, seq=2048, micro-batch=2, grad_accum=64 조합은 “계산 자체 + activation 저장 + 통신/CPU offload + optimizer 업데이트”가 모두 겹쳐서 global step 기준 시간이 비정상적으로 길어지기 쉽습니다.
​
게다가 ZeRO-2 CPU offload는 GPU 메모리를 아끼는 대신 PCIe 전송/CPU 연산 대기로 속도가 크게 떨어질 수 있어, “OOM은 피하지만 step 시간이 폭증”하는 전형적인 트레이드오프입니다.
​

global step 기준 1시간이 ‘정상’인가?

“불가능한 수치”는 아니지만, 정상적인 GPU‑bound 학습이라기보다 offload/accumulation 병목이 지배하는 상태일 가능성이 큽니다.
​

특히 gradient_accumulation_steps=64이면 1 global step이 곧 64번의 forward/backward 누적이므로, log에 찍히는 “step” 정의부터 재확인이 필요합니다(= ‘optimizer step’인지 ‘micro step’인지).
​

3090에서 현실적으로 줄일 수 있는 것

(1) accum을 줄이고 microbatch를 늘리는 방향이 보통 더 빠릅니다(가능하면). 다만 2.4B/2048에서는 VRAM이 안 나올 수 있어, 그 경우 모델 축소가 유일한 답일 때가 많습니다.
​

(2) CPU offload는 속도를 갉아먹으므로, “optimizer offload만” 최소로 쓰고 나머지는 GPU에 두는 방향이 일반적으로 낫습니다(3090은 PCIe라 더 치명적).
​

(3) torch.compile은 모델/커널 패턴에 따라 이득이 갈리지만, 오버헤드가 큰 구간에선 mode='reduce-overhead'가 도움이 될 수 있습니다(대신 디버깅은 어려워짐). (이 항목은 원리 제안이며, 실제 효과는 프로파일링으로 확인 필요)

loss=0 / grad_norm=nan 원인 가설과 디버깅 루틴
DeepSpeed ZeRO-2 조합에서 특정 버전에서 loss가 0으로 보이고 grad_norm이 nan으로 찍히는 이슈 리포트가 존재하며, 설정/버전에 따라 재현된 사례가 보고되어 있습니다.
​
따라서 “모델이 진짜로 loss 0으로 수렴”이라기보다 (a) NaN이 생겼는데 로깅/집계가 0처럼 보이거나, (b) ZeRO-2 쪽 grad norm 계산/통신 경로에서 NaN이 섞이는 케이스를 먼저 의심하는 게 안전합니다.
​

1) DeepSpeed/Trainer 조합 이슈 여부 먼저 배제

동일 코드에서 DeepSpeed를 끄고(순수 DDP 또는 단일 GPU) 아주 짧게(예: 10 step) 돌려서 loss/grad가 정상인지 확인합니다.
​

DeepSpeed 사용 시에는 문서에 나온 ZeRO-2 기본 옵션(버킷/overlap/reduce_scatter 등)부터 시작해 점진적으로 켜는 게 안정적입니다.
​

2) “BF16인데 왜 0?”

BF16은 FP32와 같은 exponent 범위를 갖지만 가수 비트가 적어(정밀도 낮음) 연산/정규화/softmax 조합에서 NaN/Inf가 생기면 빠르게 전파될 수 있습니다.
​

특히 MoE 라우터 softmax가 포화(매우 큰/작은 로짓)되면 선택 확률이 0/1로 쏠리고, aux loss/정규화와 얽혀 NaN이 생기기 쉽습니다.
​

3) HuberLoss delta / aux loss factor 체크

Time‑MoE 계열 구현에서 main loss로 Huber를 쓰고 aux loss를 더하는 설계가 소개되어 있고, aux loss가 라우팅 붕괴를 막지만 크면 학습을 방해할 수 있습니다.
​

따라서 Huber(delta)가 너무 작아서(대부분이 L1 구간) 기울기가 이상해진다기보다, 우선순위는 router_aux_loss_factor를 0으로 내려서 “main loss만”으로 NaN이 재현되는지 확인하는 것입니다.
​

4) 메모리 안 깨고 forward 값 검사

권장 순서:

torch.autograd.set_detect_anomaly(True)는 느리지만 1~2 step에서 NaN 원인을 잡는 데 유용합니다.
​

라우터 로짓/softmax 출력/Top‑k 결과에 대해 torch.isfinite() 체크를 하고, NaN/Inf가 나오면 해당 텐서의 min/max/mean만 로깅합니다(전체 텐서 print 금지).
​

backward hook으로 “어느 레이어에서 grad가 NaN이 되는지”를 좁힐 수 있습니다.
​

DeepSpeed 쪽 버그 의심 시 CUDA sanitizer 같은 접근이 언급되기도 합니다.
​

2.4B가 과한가? (1M 포인트 기준)
Time‑MoE 논문/요약 자료는 2.4B 모델이 “Time‑300B(300B time points)” 같은 초대형 코퍼스에서 스케일링을 검증했다고 밝힙니다.
​
반대로 질문의 데이터는 약 1M time points로, from-scratch로 2.4B를 돌리면 “데이터 대비 파라미터 과대 + 단일 GPU 학습 시간 폭증”이 동시에 발생할 가능성이 큽니다.
​

단일(또는 거의 단일) univariate TS anomaly detection에서 “<1주 수렴”이 목표라면, 2.4B는 구조 연구 목적이 아니라면 비현실적일 확률이 높고, 보통은 수십~수백M급(또는 더 작게)로도 충분히 실험/수렴을 볼 수 있는 경우가 많습니다.
​

특히 seq=2048은 학습 비용을 크게 올리므로, 먼저 짧은 컨텍스트(예: 256~1024)로 안정 수렴/NaN 원인 제거 후에 늘리는 절차가 운영상 안전합니다.
​

바로 쓰는 설정 스니펫(ZeRO-2 / LoRA 대안)
Hugging Face 문서의 ZeRO-2 예시 설정은 아래 항목들을 포함합니다(optimizer CPU offload, bucket 설정, reduce_scatter/overlap_comm 등).
​
3090에서는 이 구성을 “출발점”으로 쓰되, 속도가 너무 느리면 offload를 최소화하거나(가능하면) 모델 크기/seq를 낮추는 쪽이 더 효과적일 때가 많습니다.
​

DeepSpeed ZeRO-2 (기본형)
json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
LoRA를 “대안”으로 쓰는 경우(현실적 플랜)
Time‑MoE를 from scratch로 2.4B 학습하는 대신, (1) 더 작은 dense/소형 MoE를 scratch로 학습해 방법 검증 후 (2) 큰 모델에는 LoRA/adapter로 제한된 모듈만 업데이트하는 방식이 단일 3090 환경에서 더 현실적일 때가 많습니다(특히 1M 포인트 규모).
​
다만 MoE를 scratch로 “라우터+expert까지 전부 새로” 만들 거라면, LoRA는 ‘학습 안정화’보다 ‘메모리 절감’에 더 초점이 있으므로(라우터 붕괴 자체는 init/aux 스케줄링이 더 핵심) 라우터/aux 설계를 먼저 안정화하는 것을 권합니다.
​

원하는 경우, 아래 6가지만 알려주면 “원인 분기(DeepSpeed 버그/라우터 NaN/손실 스케일/데이터 스케일)”를 거의 확정적으로 좁혀서, 그에 맞는 최소 변경 패치(초기화 코드/aux warmup/ZeRO-2 옵션/compile 적용 범위)를 더 구체적인 코드 레벨로 제시할 수 있습니다.

DeepSpeed 버전, torch 버전, transformers 버전

ZeRO-2에서 overlap_comm/reduce_scatter/contiguous_gradients 등 현재 켜둔 값

라우터 로짓의 초기 분산(현재 init 방식)

router_aux_loss_factor 값과 warmup 여부

입력 시계열 정규화 방식(평균0/표준화/클리핑)

“loss=0.0”가 찍히는 시점의 실제 loss_tensor가 finite인지(torch.isfinite(loss)) 여부