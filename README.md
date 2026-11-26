# Korean Emotion TTS

화자 0031의 한국어 음성 5,983개(약 6.98시간)로 기쁨·슬픔·화남을 제어하는 TTS 프로젝트.
FastSpeech2 기반 음향 모델과 감정 embedding을 직접 구현했다. 음향 모델은 처음부터 학습하고,
공개 MFA 한국어 정렬 모델과 공식 HiFi-GAN Universal 보코더는 고정해서 재사용한다.
교수님 코드와 이전 Colab 프로젝트를 import하지 않는다.

**현재 상태: 로컬 코드 작성 완료, GPU 실행 검증 전.** 학습된 체크포인트나 성공한 합성 결과는 아직 없다.

- [서버 실행 순서](docs/RUNBOOK.md): 설치부터 평가까지, 맥/서버 명령 구분
- [코드 읽는 순서](docs/LEARNING.md): 텐서 shape와 각 단계의 역할
- [검증과 알려진 한계](docs/STATUS.md)
- [외부 자료 출처](docs/SOURCES.md)

## 흐름

```text
ZIP + manifest → 검증/분할 → MFA 정렬 → duration/mel/F0/energy
                                            ↓
음소 → Encoder + 감정 embedding → 운율 예측 → Length Regulator → Decoder → mel
                                                                          ↓
                                                                   고정 HiFi-GAN → WAV
```

## 명령

```bash
PYTHONPATH=src python3 -m korean_emotion_tts --help
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

패키지를 설치하면 `ketts` 명령으로도 실행할 수 있다. 단계별 명령은
`prepare`, `align`, `features`, `register-vocoder`, `reconstruct`, `train`, `synthesize`, `evaluate`다.
어떤 명령도 암묵적으로 모델을 내려받거나 장시간 학습을 시작하지 않는다.

학습은 `smoke`(100 step), `overfit`(최대 2,000 step), `full`(최대 50,000 step/24시간)로 구분한다.
본 학습은 작은 실험의 가중치를 이어 쓰지 않고 새로 시작한다.

## 보관

코드만 GitHub에 올리고 데이터·가중치·체크포인트·키는 Git 외부에 보관한다.
기본 서버 경로는 코드 `/workspace/korean-emotion-tts`, 데이터 `/workspace/korean-emotion-tts-data`,
결과 `/workspace/korean-emotion-tts-runs`다. 모든 코드에서 데이터 경로는 명령행 인자로 받는다.

기존 source 검사:

```bash
PYTHONPATH=src python3 -m korean_emotion_tts.inspect_data \
  --archive /데이터/teacher_0031_source_kvw3c00n.zip \
  --sha256-file /데이터/teacher_0031_source_kvw3c00n.zip.sha256 \
  --output reports/source_check_new.json
```
