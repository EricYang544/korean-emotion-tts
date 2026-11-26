# 코드 읽는 순서

## 1. 데이터 한 줄

`inspect_data.py`의 manifest에는 WAV 경로, 문장, 화자, 감정, 길이가 들어 있다.
Happy=0, Sad=1, Angry=2다. source ZIP은 22,050Hz mono PCM16 변환까지만 끝난 자료다.
`prepare.py`는 음성을 바꾸지 않고 한글 전사·발음 사전·split을 준비한다.
문장 그룹을 통째로 분할해서 같은 문장이 train/validation에 중복되지 않게 한다.

## 2. 발음과 정렬

`text.py`는 단어를 음소 목록으로 바꾼다. 정렬과 추론에서 같은 사전을 사용한다.
사전에 없는 단어(OOV)는 무시하지 않고 MFA G2P를 통해 보충해야 한다.

`alignment.py`는 TextGrid의 단어/음소 시간 구간을 읽는다.
문장 시작·단어 사이·문장 끝에 `<pause>`를 둔다. pause 길이는 0일 수도 있다.
음소별 duration은 mel 프레임 개수다. 정렬 결과의 음소가 사전과 다르면 제외한다.
음소를 임의로 추가하거나 모든 음소에 동일 길이를 주지 않는다.

## 3. 특징 배열

`audio.py`, `features.py`를 읽는다. 한 발화의 배열은 아래와 같다.

| 배열 | shape | 의미 |
|---|---|---|
| phones | `[N]` | 음소 ID, 0은 padding, 1은 pause |
| durations | `[N]` | 각 음소의 mel 프레임 수 |
| pitch / voiced / energy | `[N]` | log-F0, 유성 여부, log-RMS |
| mel | `[T,80]` | 시간별 80개 mel 주파수 성분 |

`sum(durations) == T`가 반드시 성립해야 한다.
피치는 유성 프레임만 평균하고, 무성 여부를 별도 배열로 유지한다.
피치와 에너지의 평균·표준편차는 train에서만 계산한다. mel에는 별도 z-score를 적용하지 않는다.

## 4. 모델

`model.py`의 순서대로 읽는다.

1. `FFTBlock`: attention으로 음소 사이 문맥을 보고 convolution으로 주변 정보를 처리한다.
2. `Predictor`: 각 음소의 duration/pitch/voicing/energy를 예측한다.
3. `length_regulate`: 예를 들어 `[A,B,C]`, 길이 `[2,0,1]`을 `[A,A,C]`로 바꾼다.
4. `EmotionFastSpeech2`: 위 구성 요소를 연결한다.
5. `loss_terms`: 정답과 예측을 비교하고 padding은 제외한다.

배치 입력 phones `[B,N]`은 embedding을 거쳐 `[B,N,256]`이 된다.
감정 embedding `[B,256]`을 음소 시간축에 반복해 더한다.
Length Regulator를 통과하면 `[B,T,256]`이고, mel 출력은 `[B,T,80]`이다.

학습에서는 정답 duration/pitch/energy를 사용해 mel을 만들면서 예측기도 함께 학습한다.
합성에서는 정답 특징 없이 예측기로 길이와 운율을 만든다. 따라서 validation의 정답 조건 mel loss와
실제 자유 합성 품질은 다르다. 두 가지를 모두 확인해야 한다.
이 구현은 음소 단위 운율 예측을 사용하는 간소화 FastSpeech2이며 논문의 CWT 피치 방식 전체 재현이 아니다.

## 5. 학습과 재개

`dataset.py`는 길이가 다른 배열을 padding하고 길이별 batch를 만든다.
`train.py`는 loss → backward → gradient clipping → optimizer step을 명시적으로 실행한다.
체크포인트에는 모델, optimizer, 난수 상태, epoch/batch 위치, 누적 시간, 설정과 데이터 식별자가 들어간다.
학습률은 step으로 다시 계산하므로 scheduler 위치도 보존된다.

## 6. 합성과 평가

`synthesize.py`는 문장과 감정만 입력받는다. `vocoder.py`는 mel을 WAV로 바꾼다.
`evaluate.py`는 새 문장 10개를 세 감정과 평균 embedding으로 생성한다.
평균 embedding은 따로 학습한 무감정 모델이 아니다. 수치 차이가 곧 감정 인지 성공은 아니므로
blind_audio를 듣고 자연스러움·명료도·지각 감정을 기록한다.
