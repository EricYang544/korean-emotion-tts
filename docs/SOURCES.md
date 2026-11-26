# 출처와 재사용 범위

- FastSpeech2 논문: https://arxiv.org/abs/2006.04558
  - 구조 개념을 참고하여 음향 모델·학습·데이터 코드를 직접 작성.
  - 음소 단위 log-F0/energy 사용, CWT와 postnet 생략; 논문 전체 재현이 아님.
- HiFi-GAN 공식 구현 및 UNIVERSAL_V1: https://github.com/jik876/hifi-gan (MIT)
  - 외부 checkout의 Generator와 사전학습 generator 가중치를 inference에 사용.
  - 원본 repo LICENSE를 보존. repo commit, source file hashes, config/checkpoint hashes 기록.
  - mel padding/filter/log 규칙을 호환되게 구현. waveform별 peak normalization은 적용하지 않음.
- MFA: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
  - 한국어 모델·사전·G2P는 같은 phone set의 공식 v3.0.0 release를 사용 (CC BY 4.0).
  - acoustic: https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-korean_mfa-v3.0.0
  - dictionary: https://github.com/MontrealCorpusTools/mfa-models/tree/main/dictionary/korean/mfa
  - G2P: https://github.com/MontrealCorpusTools/mfa-models/releases/tag/g2p-korean_mfa-v3.0.0
  - G2P CLI: https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/dictionary_generating.html
- WORLD: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
  - DIO/StoneMask 피치 추출에 사용.
- AI Hub 원본 감성 및 발화 스타일별 음성합성 데이터
  - 데이터 이용 조건에 따라 보관/사용. 저장소에 원본 음성·라벨을 배포하지 않음.

교수님 코드·가중치, 이전 MMS/NeMo/Colab 실행 코드는 포함하지 않았다.
