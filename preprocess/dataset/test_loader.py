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


# def make_dataset(data_path):

#     logger.info('load: ' + data_path)
#     vid_path_list = []
#     aud_path_list = []
#     sid_list = []
#     sid2fid_list = {}

#     for vid_id in tqdm(os.listdir(data_path)):
#         for seg_id in os.listdir(os.path.join(data_path, vid_id)):
#             sid = vid_id + ':' + seg_id
#             sid_list.append(sid)
#             sid2fid_list[sid] = []
#             aud_path_list.append(os.path.join(data_path, vid_id, seg_id, 'audio', 'aud.wav'))
#             if os.path.exists(os.path.join(data_path, vid_id, seg_id, 'face')):
#                 vid_path_list.append(os.path.join(data_path, vid_id, seg_id, 'face'))
#                 for img_path in os.listdir(os.path.join(data_path, vid_id, seg_id, 'face')):
#                     fid = img_path.split('_')[1].split('.')[0]
#                     sid2fid_list[sid].append(fid)
#             else:
#                 vid_path_list.append('None')
#     return vid_path_list, aud_path_list, sid_list, sid2fid_list


def get_bbox(bbox_path):
    bboxes = {}
    bbox_csv = pd.read_csv(bbox_path)
    for idx, bbox in bbox_csv.iterrows():

        # check the bbox, interpolate when necessary
        # frames = check(frames)

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


def make_dataset(file_list, aud_dir, vid_dir, seg_info, min_frames=15, max_frames=150):
    segments = []
    for sid in tqdm(file_list):
        aud_path = os.path.join(aud_dir, f"{sid}.wav")
        vid_path = os.path.join(vid_dir, f"{sid}")
        seg_length = seg_info[sid]["frame_num"]
        start_frame = 0
        end_frame = seg_length - 1
        if seg_length > max_frames:
            it = int(seg_length / max_frames)
            for i in range(it):
                sub_start = start_frame + i * max_frames
                sub_end = min(end_frame, sub_start + max_frames)
                sub_length = sub_end - sub_start + 1
                if sub_length < min_frames:
                    continue
                segments.append([sid, aud_path, vid_path, sub_start, sub_end])
        else:
            segments.append([sid, aud_path, vid_path, start_frame, end_frame])
    return segments


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

        self.segments = make_dataset(
            self.file_list, self.audio_path, self.video_path, self.seg_info
        )
        print("finish making dataset")
        self.transform = transform
        self.mode = mode

    def __getitem__(self, indices):
        source_audio = self._get_audio(indices)
        source_video = self._get_video(indices)
        sid = self.segments[indices][0]
        return source_video, source_audio, sid, self.seg_info[sid]["frame_list"]

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        video_size = 128
        uid, personid, _, start_frame, end_frame, _ = self.segments[index]
        cap = cv2.VideoCapture(os.path.join(self.video_path, f"{uid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = []
        for i in range(start_frame, end_frame + 1):
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
        uid, _, _, start_frame, end_frame, _ = self.segments[index]
        if not os.path.isfile(os.path.join(self.audio_path, f"{uid}.wav")):
            audiovideo = VideoFileClip(os.path.join(self.video_path, f"{uid}.mp4"))
            audio = video.audio
            audio.write_audiofile(os.path.join(self.audio_path, f"{uid}.wav"))

        audio, sample_rate = torchaudio.load(
            f"{self.audio_path}/{uid}.wav", normalize=True
        )

        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = transform(audio)
        # transform = torchaudio.transforms.DownmixMono(channels_first=True)
        # audio = transform(audio)
        audio = torch.mean(audio, dim=0)

        onset = int(start_frame / 30 * 16000)
        offset = int(end_frame / 30 * 16000)
        crop_audio = audio[onset:offset]

        # print("[get audio] crop audio shape", crop_audio.shape)
        # if self.mode == 'eval':
        # l = offset - onset
        # crop_audio = np.zeros(l)
        #     index = random.randint(0, len(self.segments)-1)
        #     uid, _, _, _, _, _ = self.segments[index]
        #     audio, sample_rate = soundfile.read(f'{self.audio_path}/{uid}.wav')
        #     crop_audio = normalize(audio[onset: offset])
        # else:
        #     crop_audio = normalize(audio[onset: offset])
        # return torch.tensor(crop_audio, dtype=torch.float32)
        print("[get audio] audio shape: ", audio.shape)
        return crop_audio.to(torch.float32)

    def _get_target(self, index):
        if self.mode == "train":
            return torch.LongTensor([self.segments[index][2]])
        else:
            return self.segments[index]
