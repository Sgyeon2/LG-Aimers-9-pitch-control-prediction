# model/

학습 산출물이 생성되는 위치입니다. **학습된 모델 파일(`*.pkl`)은 이 저장소에 포함하지 않습니다.**

번들 안에 2019~2024 원본으로 만든 **투수 ID 별 이력표**가 들어 있어 대회 데이터 파생물에 해당하고, 파일 크기도 160MB 를 넘기 때문입니다. 번들을 만드는 명령과 전체 설정값은 저장소 루트 [README.md](../README.md) 에 있습니다.

| 산출물 | 생성 명령 |
| --- | --- |
| `base.pkl` — 학습 번들 (Pipeline + 피처 순서 + 고정 이력표 + 신인 서브모델) | `python train.py ... --out model/base.pkl` |
| `model.pkl` — 후처리 상수를 주입한 배포 번들 | `python patch_post.py --src model/base.pkl --out model/model.pkl ...` |
| `train_meta.json` — 학습 파라미터·패키지 버전 기록 | `train.py` 가 자동 생성 |
