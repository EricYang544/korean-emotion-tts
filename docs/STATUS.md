# 구현·검증 상태

## 구현됨

- 데이터 ZIP 해시/WAV/라벨 검사, 문장 그룹 분할, OOV 보고와 단일 발음 사전
- 별도 MFA 실행 연동, TextGrid duration, train 전용 피치/에너지 통계와 개별 특징 캐시
- 직접 구현한 감정 FastSpeech2, masked losses, BF16 학습, 작은 실험/본 학습 분리
- 원자적 checkpoint와 optimizer/RNG/batch cursor/누적시간 재개
- 고정된 공식 보코더 연동, 복원 검사, 자유 합성, 동일 문장/평균 embedding/청취 평가 자료

## 실제 확인한 것

- source ZIP 5,983개, 약 6.977시간: 해시, WAV 형식/길이, 라벨-전사 대응 검사 통과.
- 전사 정규화 5,983개 전부 통과(아직 실제 MFA 사전 OOV 검사는 아님).
- 정렬 전 분할: train 5,385 / val 299 / test 299. 정렬 후에는 감소할 수 있음.
- 표준 라이브러리 테스트 9개 통과: 경로/해시, 전사, 사전, 그룹 분할, duration 규칙.
- 수치 테스트 6개 작성, 로컬 torch/numpy/librosa/soundfile 미설치로 skip.
- Python 문법 컴파일과 CLI help 검사 통과.

## 아직 검증하지 못한 것

이 로컬 환경에서는 pip가 필요한 패키지 버전을 찾지 못해 설치되지 않았다.
GPU 서버 SSH도 앞서 Operation not permitted로 차단됐으며, 이 작업에서 서버를 변경하지 않았다.
따라서 PyTorch forward/backward, WORLD, MFA 실제 정렬, HiFi-GAN 가중치 로딩, CUDA 학습과
자유 합성의 성공을 아직 주장할 수 없다. 서버에서 scripts/test_runtime.py는 의존성이 없거나
하나라도 skip이면 실패하도록 작성했다. 통과 후 복원→소수 정렬→파일럿 순서로 검증한다.

## 품질상 한계

- 세 감정은 서로 다른 문장으로 녹음되어 텍스트 내용과 감정의 상관관계가 존재한다.
- 단어별 사전 발음은 문장 문맥의 모든 한국어 음운 변화를 처리하지 못할 수 있다.
- MFA 정렬 모델과 Universal 보코더의 이번 한국어 감정 데이터 품질은 직접 청취가 필요하다.
- 피치 범위 50~600Hz 및 정렬의 엄격한 음소 검사가 일부 감정을 더 많이 제외할 수 있으므로 보고서를 확인한다.
- first run은 최대 24시간이며 그 안에 자연스러운 발음/감정 품질 달성을 보장하지 않는다.

사용자가 GitHub 저장소 EricYang544/korean-emotion-tts를 생성했고, 로컬 origin을 연결했다.
원격 연결은 현재 도구 환경의 github.com DNS 오류로 차단되어 push 완료 여부를 확인할 수 없다.
외부 asset 다운로드와 서버 설치·학습은 수행하지 않았다.
