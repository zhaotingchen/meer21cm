# **meer21cm**
[![codecov](https://codecov.io/gh/zhaotingchen/meer21cm/graph/badge.svg?token=BEE45774LQ)](https://codecov.io/gh/zhaotingchen/meer21cm)

`meer21cm` deals with single dish 21cm intensity mapping using MeerKAT.

## Installation
Clone this repo:
```
git clone https://github.com/zhaotingchen/meer21cm
```

And run
```
cd meer21cm/
pip install -e .
```

It uses [hiimtool](https://github.com/zhaotingchen/hiimtool) for some functionalities.

If you are using it on ilifu and don't want to create your own environment for it, you can simply clone the repo and do:

```python
import sys
sys.path.append('path/to/hiimtool/src/')
sys.path.append('path/to/meer21cm/src/')
```
## Development
If you want to contribute to the developing of `meer21cm`, make sure you check [the workflow](DEVELOPING.md).
