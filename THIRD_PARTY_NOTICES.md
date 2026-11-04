# Third-party notices

The original code in this repository is released under the MIT License (see `LICENSE`).
The components below are adapted from other projects and remain under their own licenses.
Where a license requires it, the copyright and permission notice is reproduced here and the
license text is included in the `LICENSES/` folder.

| Component in this repository | Origin | License |
|---|---|---|
| `*/model/EEGNet.py`, `*/model/EEGNet_features.py` (constructor arguments and docstring) | EEGNet, Lawhern et al., *J. Neural Eng.* 2018; reference implementation <https://github.com/vlawhern/arl-eegmodels> | CC0 1.0 (see the upstream `LICENSE.txt`) |
| `Entropy()` in `*/utils.py` | SHOT, Liang et al., *ICML* 2020 (<https://github.com/tim-learn/SHOT>), as used in DeepTransferEEG | MIT, Copyright (c) 2020 |
| `openbmi/OpenBMI/datasets/get_cross_subject_data.py`: channel table and `.mat` reading routine | MIN2Net, Autthasan et al., *IEEE TBME* 2022; <https://github.com/IoBT-VISTEC/MIN2Net> | Apache-2.0 (`LICENSES/Apache-2.0.txt`); the modifications are listed in the file header |

## Backbones of the comparison table that are not included

The ShallowConvNet ("ConvNet"), LMDA-Net and EEG-Conformer backbones used for the
backbone comparison in the paper were adapted from the public code of the original
authors (input of 3 channels × 1000 samples, two classes):

- ShallowConvNet and LMDA-Net: <https://github.com/MiaoZhengQing/LMDA-Code> (Miao et al., *NeuroImage* 2023; ShallowConvNet after Schirrmeister et al., *Hum. Brain Mapp.* 2017).
- EEG-Conformer: <https://github.com/eeyhsong/EEG-Conformer> (Song et al., *IEEE TNSRE* 2023).

These files are not redistributed here; please obtain the networks from the repositories above.

## MIT License text

Applies to the MIT-licensed components listed above, with the copyright lines given in the table.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets

The datasets are not redistributed with this repository.

- **BCI Competition IV, datasets 2a and 2b** (Graz University of Technology). Downloads
  require accepting the data-use terms at <https://www.bbci.de/competition/iv/>. The true
  labels of the evaluation sessions are distributed separately on the competition results
  page. Please cite Brunner et al. (2008) for 2a, Leeb et al. (2008) for 2b, and Tangermann
  et al., *Front. Neurosci.* 2012 for the competition.
- **OpenBMI motor-imagery dataset** (Lee et al., *GigaScience* 2019), GigaDB dataset
  100542, <http://gigadb.org/dataset/100542> (DOI 10.5524/100542). Use is subject to the
  terms stated on the dataset page.
