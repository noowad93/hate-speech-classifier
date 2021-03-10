# Hate Speech Classifer
## Requirements

```shell
pip install -r requirements.txt -r requirements-dev.txt
```
## Data

* 본 레포에서 사용된 데이터는 [Link](https://github.com/kocohub/korean-hate-speech)에서 다운받으실 수 있습니다.
## Model
* Huggingface S3에 업로드되어있는 KoElectra를 이용합니다.
* Bias Classifier와 Hate Classifier를 joint하게 학습합니다.
## To-do

  + 모델 저장이 오래걸리는 현상 트러블슈팅
  + ml-flow로 로깅
  + Semi-Supervision
