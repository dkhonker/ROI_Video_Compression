# DVC: An End-to-end Deep Video Compression Framework

训练/测试操作和原始代码均保持一致。

## 使用前安装
由于liteflownet3 模块用到了<a href=https://pypi.org/project/spatial-correlation-sampler/>spatial-correlation-sampler</a>，所以需要安装。

```
pip install spatial-correlation-sampler
```

## 增加的新特性
- 增加可替换的liteflownet3模块

使用方法：
直接在net.py内修改self.opticFlow

- 增加新的test.py

一次性测试不同的$\lambda$，并将结果提交到wandb，减轻画图工作量。

使用方法：
和原始代码一致，不过需要将main.py换成test.py