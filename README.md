
## pro-AAE-real
変形したARマーカの姿勢推定を行うリポジトリです

## 動作環境
- ubuntu 16.04 (OS)
- python 3.5.2 (言語)
- pytorch 1.3.1
- GeForce GTX 1060 (GPU)

## ファイル説明

- real-augumented-autoencoder :変形ARマーカから平面ARマーカを復元（潜在変数の取得）するネットワークの学習と学習したAEより潜在変数を取得する
- cos_sim　:「real-augumented-autoencoder」で得た潜在変数を用いて姿勢推定を行う



