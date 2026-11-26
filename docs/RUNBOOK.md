# 서버 실행 순서

GitHub 저장소: https://github.com/EricYang544/korean-emotion-tts
로컬 origin은 해당 저장소에 연결한다. 원격 push 완료 여부는 실제 Git 출력으로 확인한다.
설치는 서버에서 실행한다. 맥은 편집·검토·업로드에 사용한다.
코드 변경 시 작은 테스트를 먼저 실행하고, 학습 중에는 checkout/pull하지 않는다.

## 1. 코드와 데이터

GitHub Private 저장소에 코드가 올라간 뒤 **Vast.ai 터미널**에서:

```bash
cd /workspace
git clone git@github.com:EricYang544/korean-emotion-tts.git
cd korean-emotion-tts
cat /workspace/AGENTS.md
mkdir -p /workspace/korean-emotion-tts-data/raw /workspace/korean-emotion-tts-runs
```

Private repo를 읽을 수 있는 SSH 인증이 필요하다. 키는 저장소에 넣지 않는다.

**맥 터미널**에서 데이터 전송:

```bash
scp /Users/suchan/workplace/PSP/teacher_emotion_0031/data/teacher_0031_source_kvw3c00n.zip \
    /Users/suchan/workplace/PSP/teacher_emotion_0031/data/teacher_0031_source_kvw3c00n.zip.sha256 \
    vast-ai-23039:/workspace/korean-emotion-tts-data/raw/
```

이후 명령은 모두 **Vast.ai 서버, 저장소 루트**에서 실행한다.

```bash
export KT_DATA=/workspace/korean-emotion-tts-data
export KT_RUNS=/workspace/korean-emotion-tts-runs
```

## 2. 학습 환경

서버 기본 `/venv/main`은 변경하지 않는다. GPU 연산 테스트가 먼저다.
아래 버전은 계획한 고정 버전이며 실제 설치 검증 전이다. 오류가 나면 로그를 보존하고 그 단계에서 멈춘다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e .
python scripts/check_environment.py
python scripts/test_runtime.py
python -m pip freeze > "$KT_RUNS/environment-lock.txt"
```

WORLD 빌드에 컴파일러가 필요할 수 있다. `pip check`와 수치 테스트가 통과하기 전 학습하지 않는다.
환경·설정·소스를 고정한 다음 복원 검사를 진행한다. 코드를 고치면 동일 코드로 복원/파일럿을 다시 확인한다.

## 3. 외부 모델 준비

### MFA

Miniforge/Conda가 준비된 상태에서 독립 환경을 만든다. 공식 설치 문서는 SOURCES.md 참고.

```bash
conda create -p /workspace/korean-emotion-tts-mfa -c conda-forge montreal-forced-aligner=3.4.3 -y
export KT_MFA=/workspace/korean-emotion-tts-mfa/bin/mfa
"$KT_MFA" version
"$KT_MFA" align --help
"$KT_MFA" g2p --help
```

MFA 공식 model release의 **Korean MFA v3.0.0** acoustic ZIP, dictionary, G2P ZIP을 각각 준비한다.
서로 다른 phone set의 korean_jamo/다른 버전을 섞지 않는다. 파일은 아래 위치로 보관한다.

```text
$KT_DATA/assets/korean_mfa_acoustic.zip
$KT_DATA/assets/korean_mfa.dict
$KT_DATA/assets/korean_mfa_g2p.zip
```

각 파일의 원본 URL, 버전, SHA-256을 함께 기록한다. 해당 모델은 CC BY 4.0이므로 발표/문서에 출처를 남긴다.

### HiFi-GAN

공식 저장소의 README에 연결된 pretrained models에서 **UNIVERSAL_V1**의 generator와 config를 받는다.
학습에 사용된 교수님 보코더를 복사하지 않는다. 다운로드한 config를 사용하며 임의 설정을 끼워 넣지 않는다.

```bash
mkdir -p "$KT_DATA/assets"
git clone https://github.com/jik876/hifi-gan.git "$KT_DATA/assets/hifi-gan"
git -C "$KT_DATA/assets/hifi-gan" rev-parse HEAD
```

확인한 commit을 유지한다. generator를 `universal_v1_generator`, config를 `universal_v1_config.json`으로 보관한다.
공식 출처를 확인한 첫 다운로드 해시를 기록하고, 이후 전송/재사용 때 비교한다. 자기 파일에서 계산한 해시만으로 출처가 증명되는 것은 아니다.

```bash
sha256sum "$KT_DATA/assets/universal_v1_generator" > "$KT_DATA/assets/universal_v1_generator.sha256"
ketts register-vocoder --repo "$KT_DATA/assets/hifi-gan" \
  --checkpoint "$KT_DATA/assets/universal_v1_generator" \
  --config "$KT_DATA/assets/universal_v1_config.json" \
  --sha256-file "$KT_DATA/assets/universal_v1_generator.sha256" \
  --output "$KT_DATA/assets/vocoder.json"
```

공식 소스에 수정이 필요하면 임의로 변경하고 넘어가지 않는다. 호환성 수정과 출처 기록을 먼저 검토한다.

## 4. 원본 준비와 OOV

```bash
ketts prepare \
  --archive "$KT_DATA/raw/teacher_0031_source_kvw3c00n.zip" \
  --sha256-file "$KT_DATA/raw/teacher_0031_source_kvw3c00n.zip.sha256" \
  --lexicon "$KT_DATA/assets/korean_mfa.dict" --output "$KT_DATA/source_review"
```

`source_review/oov.txt`가 비어 있지 않으면 같은 G2P로 생성한다. 자동 생성 발음은 일부를 검토한다.

```bash
"$KT_MFA" g2p "$KT_DATA/source_review/oov.txt" \
  "$KT_DATA/assets/korean_mfa_g2p.zip" "$KT_DATA/assets/oov.dict" --num_pronunciations 1
```

그 뒤 같은 prepare 명령에 `--extra-lexicon "$KT_DATA/assets/oov.dict"`를 추가하고
`--output "$KT_DATA/source"`로 다시 실행한다. OOV가 원래 0이면 extra-lexicon 없이 source를 준비한다.
기존 output은 덮어쓰지 않는다. `source/contract.json`의 oov_count가 0이어야 정렬할 수 있다.

## 5. 보코더 복원과 정렬 검사

`source/corpus/0031`에서 감정별 WAV를 선택해 원본과 복원을 듣는다. WAV 경로는 실제 파일명으로 지정한다.

```bash
ketts reconstruct --assets "$KT_DATA/assets/vocoder.json" \
  --wav "$KT_DATA/source/corpus/0031/실제파일명.wav" \
  --output "$KT_RUNS/reconstruction"
ketts align --source "$KT_DATA/source" --mfa "$KT_MFA" \
  --acoustic-model "$KT_DATA/assets/korean_mfa_acoustic.zip" \
  --output "$KT_DATA/alignment_sample" --per-emotion 10
```

Praat 등에서 sample TextGrid와 WAV 경계를 확인한다. 샘플 정렬은 확인용이며 전체 특징 추출에 사용하지 않는다.
복원이 심하게 왜곡되거나 정렬이 틀리면 다음 단계로 넘어가지 않는다.

```bash
ketts align --source "$KT_DATA/source" --mfa "$KT_MFA" \
  --acoustic-model "$KT_DATA/assets/korean_mfa_acoustic.zip" --output "$KT_DATA/aligned"
ketts features --source "$KT_DATA/source" --alignments "$KT_DATA/aligned" --output "$KT_DATA/features"
```

features는 동일 입력·코드면 완료된 개별 배열을 재사용한다. 다른 조건이면 새 output이 필요하다.
`features/report.json`과 `rejected.json`을 보고 감정별 손실이 편향되지 않았는지 확인한다.

## 6. 학습

아래 청취 확인 플래그는 실제로 복원 음성을 들은 뒤 지정한다.

```bash
ketts train --mode smoke --features "$KT_DATA/features" --assets "$KT_DATA/assets/vocoder.json" \
  --reconstruction "$KT_RUNS/reconstruction" --audio-reviewed --output "$KT_RUNS/smoke"
ketts train --mode overfit --features "$KT_DATA/features" --assets "$KT_DATA/assets/vocoder.json" \
  --reconstruction "$KT_RUNS/reconstruction" --audio-reviewed --output "$KT_RUNS/overfit"
```

smoke는 실행 경로, overfit는 작은 데이터 학습 여부를 검사한다. 수치 보고서의 passed만으로 발음 성공을 뜻하지 않는다.
100 step/2,000 step 학습의 속도와 VRAM을 확인하고 원본/합성 음성을 듣는다.
메모리 부족 시 configs/base.json을 복사해 batch_size=8, accumulation=2로 바꾸고 새 실험 경로에서 다시 시작한다.
본 학습을 장시간 실행할 때는 tmux 세션을 이용한다.

```bash
ketts train --mode full --features "$KT_DATA/features" --assets "$KT_DATA/assets/vocoder.json" \
  --reconstruction "$KT_RUNS/reconstruction" --audio-reviewed \
  --pilot-report "$KT_RUNS/overfit/run_report.json" --pilot-audio-reviewed --output "$KT_RUNS/full"
```

같은 명령에 `--resume`을 붙이면 last.pt부터 재개한다. 코드·데이터·설정이 달라지면 재개를 차단한다.
시간 제한은 재개 시 초기화하지 않는다. 24시간 연장은 별도 계획 없이 자동으로 하지 않는다.
Ctrl-C는 마지막 완료된 checkpoint를 보존한다. 50/500 step 저장 전 중단되면 재개 파일이 없을 수 있다.
NaN/OOM은 성공으로 숨기지 않으며 마지막 원자적 checkpoint는 남는다.
최근 복구 체크포인트 2개, best/last, 5,000 step milestone을 보관한다.

## 7. 합성과 평가

```bash
ketts synthesize --checkpoint "$KT_RUNS/full/best.pt" --assets "$KT_DATA/assets/vocoder.json" \
  --text '오늘은 집에 일찍 돌아왔어요.' --emotion Happy --output "$KT_RUNS/demo/happy.wav"
ketts evaluate --checkpoint "$KT_RUNS/full/best.pt" --assets "$KT_DATA/assets/vocoder.json" \
  --features "$KT_DATA/features" --texts configs/evaluation_sentences.txt --output "$KT_RUNS/evaluation"
```

평가 문장은 train/validation과 겹치면 거부한다. 최종 학습 전에 10문장을 확정해 변경하지 않는다.
새 단어 OOV는 같은 G2P로 생성한 사전을 `--extra-lexicon`으로 전달한다. 기존 발음을 덮어쓰지 않으며
훈련 vocab에 없는 음소는 허용하지 않는다. 단어 사전 방식의 문맥 발음 한계는 보고서에 명시한다.

`blind_audio` 30개는 답안 파일 없이 평가자에게 전달한다. `listening.csv`에 지각 감정과 1~5점을 기록한다.
평균 embedding 비교 10개, acoustic_metrics.csv, f0_comparison.svg도 생성된다.
validation은 모델 선택용, test 점수는 최종 보고용이며 test로 학습 설정을 조절하지 않는다.

## 8. 백업

매일 및 학습 종료 후 **맥 터미널**에서 checkpoint/설정/로그/평가 결과를 별도 보관한다.

```bash
rsync -av --partial vast-ai-23039:/workspace/korean-emotion-tts-runs/ /맥의/백업폴더/
```

전송 완료와 checkpoint 크기/해시를 확인하기 전 서버를 삭제하지 않는다. 원본 데이터도 삭제하지 않는다.
