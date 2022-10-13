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
