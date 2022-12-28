# DLCV Final Project ( Talking to me )
# Important note:
I have softlinks to these directories: **extracted_frames, extracted_audio, data**
Please pay attention to them when fetching this version

# Installation
```bash script=
$ pip install -r requirements.txt
```
- install ttm-best.pth at [checkpoint](https://drive.google.com/drive/folders/1MGrhm3J1dKoWPSL3RvC3qb3QeiIqe9vi?usp=sharing)

# Preprocess
*For Training videos*
```bash script=
$ python run.py --preprocess 
```

*For Testing videos*
```bash script=
$ python run.py --preprocess --eval
```


# How to inference 
To inference a csv, you need to run the following command:
```bash script= 
$ python3 run.py --eval --checkpoint <checkpoint path> --num_worker <option> --device_id <option>
```
Output csv file will be located in the following **./evalai_test/output/<exp_name>/results/pred.csv**
To specify the threshold, you may check this file **./evalai_test/threhold.py**

# Submission Rules
### Deadline
111/12/29 (Thur.) 23:59 (GMT+8)
For more details, please click [this link](https://docs.google.com/presentation/d/1Y-gwBmucYgbWLLk-u6coHi7LybFLXgA9gV8KiOiKShI/edit?usp=sharing) to view the slides of Final Project - Talking to me. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion

# File structure
```bash
├── common
│   ├── engine.py
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── render.py
│   └── utils.py
├── config.py
├── data 
├── dataset
│   ├── data_loader.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── sampler.py
│   └── split
│       ├── test.list
│       ├── train.list
│       └── val.list
├── environment.yml
├── evalai_test
│   ├── checkpoint
│   │   └── tmp.txt
│   ├── output
│   │   ├── BaselineLSTM
│   │   │   ├── gt.csv.rank.0
│   │   │   ├── pred.csv.rank.0
│   │   │   └── result
│   │   │       ├── gt.csv
│   │   │       └── pred.csv
│   │   ├── checkpoint
│   │   │   └── tmp.txt
│   │   ├── log
│   │   ├── result
│   │   │   ├── gt.csv
│   │   │   ├── pred.csv
│   │   │   └── threshold.py 
│   │   ├── tmp
│   │   │   └── pred_0.csv
│   │   └── ViViT
│   │       └── result
│   │           ├── gt.csv
│   │           └── pred.csv
│   └── ttm-best.pth
├── extracted_audio 
├── extracted_frames 
├── model
│   ├── __init__.py
│   ├── model.py
│   ├── module.py
│   ├── __pycache__
│   ├── resnet.py
│   └── resse.py
├── preprocess
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── sampler.py
│   │   └── split
│   │       ├── test.list
│   │       ├── train.list
│   │       └── val.list
│   ├── extract_audio.py
│   ├── extracted_audio 
│   ├── extracted_frames 
│   ├── extract_frames.py
│   ├── run.py
│   └── video_crop.py
├── __pycache__
├── README.md
├── requirements.txt
├── run.py
├── scripts
│   ├── download_clips.py
│   ├── get_json.py
│   ├── get_lam_result.py
│   ├── get_ttm_result.py
│   ├── merge.py
│   └── preprocessing.py
├── setup.py
└── test.py
```
# How to run your code?
* TODO: Please provide the scripts for TAs to reproduce your results, including training and inference. For example, 

**Need to preprocess first before inference and train**
```shell script=
bash preprocess.sh
```

## Training
```shell script=
bash train.sh 
```
## Inferencing
```shell script=
bash inference.sh <Path to checkpoint>
```
