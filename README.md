# DLCV Final Project ( Talking to me )

# How to run your code?
* TODO: Please provide the scripts for TAs to reproduce your results, including training and inference. For example, 
```shell script=
bash train.sh <Path to videos folder> <Path to seg folder> <Path to bbox folder>
bash inference.sh <Path to videos folder> <Path to seg folder> <Path to bbox folder> <Path to output csv file>
```

# Usage
To start working on this final project, you should clone this repository into your local machine by the following command:

    git clone https://github.com/ntudlcv/DLCV-Fall-2022-Final-1-<team name>.git
  
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1Y-gwBmucYgbWLLk-u6coHi7LybFLXgA9gV8KiOiKShI/edit?usp=sharing) to view the slides of Final Project - Talking to me. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

# Submission Rules
### Deadline
111/12/29 (Thur.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion

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
