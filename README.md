# **meerstack**
[![codecov](https://codecov.io/gh/zhaotingchen/meerstack/graph/badge.svg?token=BEE45774LQ)](https://codecov.io/gh/zhaotingchen/meerstack)

`meerstack` deals with single dish stacking with 21cm intensity mapping.

## Installation
Clone this repo:
```
git clone https://github.com/zhaotingchen/meerstack
```

And run
```
cd meerstack/
pip install -e .
```

It uses [hiimtool](https://github.com/zhaotingchen/hiimtool) as dependency which needs to be installed separately as well.

If you are using it on ilifu and don't want to create your own environment for it, you can simply clone the repo and do:

```python
import sys
sys.path.append('path/to/hiimtool/src/')
sys.path.append('path/to/meerstack/src/')
```
## Development
If you want to contribute to the developing of `meerstack`, make sure you check [requirements](DEVELOPING.md).
