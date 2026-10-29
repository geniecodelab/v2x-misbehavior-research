# VeReMi NextGen Pilot Dataset Provenance

## Source

- Dataset: VeReMi NextGen v1
- DOI: `10.5281/zenodo.19665762`
- Official landing page: <https://veremi-dataset.github.io/veremi-nextgen>
- Official archive: <https://zenodo.org/records/19665762>
- Retrieved: 2026-08-15
- Local location: `data/raw/veremi_nextgen/archives/`

## License and attribution

The Zenodo dataset package's `LICENSE.txt` is Creative Commons Attribution 4.0 International (CC BY 4.0). The local license copy is retained next to the raw archives and has MD5 `2ab724713fdaf49e4523c4503bfd068d`.

The dataset and resulting reports must attribute:

> Hermann, A., Remmers, J. N., Eisermann, D., Erb, B., and Kargl, F. VeReMi NextGen: A Dataset for Evaluating Misbehavior Detection Systems in VANETs. IEEE VNC 2026.

The generator/source repository is separately marked Apache-2.0. Dataset redistribution and derived publications should follow the packaged CC BY 4.0 terms.

## Downloaded subset

| File | Bytes | MD5 | Purpose |
| --- | ---: | --- | --- |
| `InTAS_highway_2.zip` | 19,335,264 | `d5fefb873c5e766b702bd036f18a326d` | Benign baseline |
| `InTAS_highway_2_constantPositionOffset.zip` | 20,872,394 | `4b8aa6b684a093de218256bae74c1a6e` | Fixed position spoofing |
| `InTAS_highway_2_randomPositionOffset.zip` | 20,894,253 | `0b72ffd770351817deb18fa3ec630a61` | Random position spoofing |
| `InTAS_highway_2_constantSpeedOffset.zip` | 20,859,029 | `27a7d170fd937b65736d06054ba97bc4` | Fixed speed spoofing |
| `InTAS_highway_2_randomSpeedOffset.zip` | 20,874,226 | `092c7f5d150f613414de0b0797430968` | Random speed spoofing |
| `LICENSE.txt` | 18,706 | `2ab724713fdaf49e4523c4503bfd068d` | Dataset license |
| `README.md` | 5,923 | `b2a121a1ab2f362283fb60fffaaf31e2` | Publisher documentation |

The five data archives total 102,835,166 bytes. The full 36.2 GB release was not downloaded.

## Split and usage policy

- Preserve the publisher's nested `Train`, `Validation`, and `Test` directories.
- Do not perform a random message-row split.
- Fit the RandomForest only on Train.
- Fit the probability calibrator and choose ML/Hybrid thresholds only on Validation.
- Evaluate final metrics once on Test.
- Keep raw archives and processed features out of Git.
- Do not claim that these position/speed archives contain the required camera-confirmed Ghost Vehicle scenario. Ghost remains a separate synthetic sensor/Ground Truth mismatch evaluation.

## Integrity note

The Zenodo HTML documentation exposed a stale MD5 for the constant-position archive during retrieval. The actual file size and MD5 matched the current Zenodo REST API record exactly. All other archive hashes matched both the published listing and downloaded files.
