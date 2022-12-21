# File structure
```bash

├── common
│   ├── engine.py
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── render.py
│   └── utils.py
├── config.py
├── dataset
│   ├── data_loader.py
│   ├── __init__.py
│   └── sampler.py
├── extracted_frames
├── extracted_audio
├── evalai_test
│   ├── ttm-best.pth # need to download
│   ├── checkpoint
│   │   └── tmp.txt
│   └── output
│       └── tmp
├── model
│   ├── __init__.py
│   ├── model.py
│   ├── resnet.py
│   └── resse.py
├── preprocess
│   ├── dataset
│   │   ├── data_loader.py
│   │   ├── __init__.py
│   │   ├── sampler.py
│   │   └── split
│   │       ├── train.list
│   │       └── val.list
│   ├── extract_audio.py
│   ├── extract_frames.py
│   ├── run.py
│   └── video_crop.py
├── README.md
├── requirements.txt
├── run.py
├── scripts
│   ├── download_clips.py
│   ├── get_json.py
│   ├── get_lam_result.py
│   ├── get_ttm_result.py
│   ├── merge.py
│   └── preprocessing.py
└── setup.py
```

# installation
- pip install -r requirements.txt
- install ttm-best.pth at [checkpoint](https://drive.google.com/drive/folders/1MGrhm3J1dKoWPSL3RvC3qb3QeiIqe9vi?usp=sharing)
