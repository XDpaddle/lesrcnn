# Lesrcnn_paddle

https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

Paddle 复现版本

## 训练步骤
### train sr
```bash
python train.py -opt config/train/train_lesrcnn_xx.yml
```
多卡仅需测试步骤
```bash
python test.py -opt config/test/test_lesrcnn_xx.yml
```