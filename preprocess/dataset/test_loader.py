import glob
import json
import logging
import os
import random
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import pandas as pd
import soundfile
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
from moviepy.editor import VideoFileClip
import torchaudio

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def helper():
    return defaultdict(OrderedDict)


def get_md5(s):
    md5hash = hashlib.md5(s.encode(encoding="UTF-8"))
    md5 = md5hash.hexdigest()
    return md5


def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples


def get_bbox(bbox_path):
    bboxes = {}
    bbox_csv = pd.read_csv(bbox_path)
    for idx, bbox in bbox_csv.iterrows():
        # for frame in frames:
        frameid = int(bbox["frame_id"])
        personid = int(bbox["person_id"])
        bbox = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        identifier = str(frameid) + ":" + str(personid)
        bboxes[identifier] = bbox

    return bboxes


def makeFileList(filepath):
    with open(filepath, "r") as f:
        videos = f.readlines()
    return [uid.strip() for uid in videos]


def make_dataset(file_list, data_path, maxframe=None, minframe=10, mode="eval"):
    # file list is a list of training or validation file names

    face_crop = {}
    segments = []

    for uid in tqdm(file_list):
        seg_path = os.path.join(data_path, "seg", uid + "_seg.csv")
        bbox_path = os.path.join(data_path, "bbox", uid + "_bbox.csv")
        uid = uid.strip()
        face_crop[uid] = get_bbox(bbox_path)
        seg = pd.read_csv(seg_path)

        for idx, gt in seg.iterrows():
            personid = gt["person_id"]
            start_frame = int(gt["start_frame"])
            end_frame = int(gt["end_frame"])
            seg_length = end_frame - start_frame + 1
            save_id = f"{uid}_{personid}_{start_frame}_{end_frame}"

            ##### for setting maximum frame size and minimum frame size
            if (
                (mode == "train" and minframe != None and seg_length < minframe)
                or (seg_length <= 1)
                or (personid == 0)
            ):
                continue
            elif maxframe != None and seg_length > maxframe:
                it = int(seg_length / maxframe)
                for i in range(it):
                    sub_start = start_frame + i * maxframe
                    sub_end = min(end_frame, sub_start + maxframe)
                    sub_length = sub_end - sub_start + 1
                    if minframe != None and sub_length < minframe:
                        continue
                    segments.append([uid, personid, sub_start, sub_end, save_id])
            else:
                segments.append([uid, personid, start_frame, end_frame, save_id])

    return segments, face_crop


class test_ImagerLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        audio_path,
        video_path,
        file_path,
        seg_info,
        mode="eval",
        transform=None,
    ):

        self.data_path = data_path
        self.audio_path = audio_path
        self.video_path = video_path
        self.file_path = file_path
        self.seg_info = json.load(open(seg_info))
        self.file_list = makeFileList(
            self.file_path,
        )
        print(f"{mode} file with length: {str(len(self.file_list))}")

        print("start making dataset")
        self.segments, self.face_crop = make_dataset(self.file_list, self.data_path)
        print("finish making dataset")
        self.transform = transform
        self.mode = mode

    def __getitem__(self, indices):
        source_audio = self._get_audio(indices)
        source_video = self._get_video(indices)
        sid = self.segments[indices][4]
        return source_video, source_audio, sid

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        video_size = 128
        uid, personid, start_frame, end_frame, _ = self.segments[index]
        cap = cv2.VideoCapture(os.path.join(self.video_path, f"{uid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = []
        for i in tqdm(range(start_frame, end_frame + 1)):
            key = str(i) + ":" + str(personid)
            # print("key: ", key)
            # print("face: ", dict(list(self.face_crop[uid].items())[:3]))
            if key in self.face_crop[uid].keys():
                bbox = self.face_crop[uid][key]
                if os.path.isfile(
                    f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png"
                ):
                    img = cv2.imread(
                        f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png"
                    )
                    face = cv2.resize(img, (video_size, video_size))
                else:
                    ret, img = cap.read()

                    if not ret:
                        print("not ret")
                        video.append(
                            np.zeros((1, video_size, video_size, 3), dtype=np.uint8)
                        )
                        continue

                    if not os.path.isdir(f"./extracted_frames/{uid}"):
                        os.mkdir(f"./extracted_frames/{uid}")
                    x1, y1, x2, y2 = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )

                    face = img[y1:y2, x1:x2, :]
                    if face.size != 0:
                        print(f"{uid}/write: {i:05d}_{personid}")
                        cv2.imwrite(
                            f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png", face
                        )
                try:
                    face = cv2.resize(face, (video_size, video_size))
                except:
                    # bad bbox
                    face = np.zeros((video_size, video_size, 3), dtype=np.uint8)

                if debug:
                    import matplotlib.pyplot as plt

                    plt.imshow(face)
                    plt.show()

                video.append(np.expand_dims(face, axis=0))
            else:
                print("not in face crop")
                video.append(np.zeros((1, video_size, video_size, 3), dtype=np.uint8))
                continue
        cap.release()
        video = np.concatenate(video, axis=0)
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
        print("[get video] video shape: ", video.shape)
        return video

    def _get_audio(self, index):
        uid, _, start_frame, end_frame, _ = self.segments[index]
        if not os.path.isfile(os.path.join(self.audio_path, f"{uid}.wav")):
            video = VideoFileClip(os.path.join(self.video_path, f"{uid}.mp4"))
            audio = video.audio
            audio.write_audiofile(os.path.join(self.audio_path, f"{uid}.wav"))

        audio, sample_rate = torchaudio.load(
            f"{self.audio_path}/{uid}.wav", normalize=True
        )

        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = transform(audio)
        audio = torch.mean(audio, dim=0)

        onset = int(start_frame / 30 * 16000)
        offset = int(end_frame / 30 * 16000)
        crop_audio = audio[onset:offset]
        return crop_audio.to(torch.float32)

    def _get_target(self, index):
        if self.mode == "train":
            return torch.LongTensor([self.segments[index][2]])
        else:
            return self.segments[index]
