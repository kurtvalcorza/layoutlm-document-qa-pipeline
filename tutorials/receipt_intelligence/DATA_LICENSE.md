# Data licence and attribution — receipt intelligence capstone

**Dataset:** CORD v2 (`naver-clova-ix/cord-v2`, revision `7f0115a4b758a71d6473b8d085751692da2fef98`), a public
sample of Indonesian receipts published by NAVER CLOVA. **Licence:** Creative Commons Attribution 4.0
International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>.

**Citation:** Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., & Lee, H. (2019). *CORD: A consolidated
receipt dataset for post-OCR parsing*. Document Intelligence Workshop at NeurIPS. <https://github.com/clovaai/cord>

**Changes made by the notebook (none are relabelling):** EXIF orientation is applied (all 1,000 source images
record orientation 1); annotation quadrilaterals are converted to axis-aligned pixel boxes and clipped to the
page; amount strings are normalised by the declared `cord_mixed_v1` grammar into canonical decimal strings;
two within-train exact-pixel duplicates are excluded (`cord-v2:train:0109`, `cord-v2:train:0285`). Source photos
and annotations are CORD's, not original DIMER data. The default results ZIP omits receipt photographs and
full-page OCR transcripts.

**Other components (separate licences):** `impira/layoutlm-document-qa` model weights (MIT); Tesseract 5.5.0 and
`tessdata_fast` language data (Apache-2.0); conda-forge OCR dependencies (licences recorded per package in
`ocr_manifest.json`); capstone code (repository licence).
