import cv2
import os
import numpy as np
from timeit import default_timer as timer


def vide2frames(path: str):
    cap = cv2.VideoCapture(path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('frames/' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def frames2video(file_path, _frames, fps=30):
    def img_to_cv_format(img):
        return np.array(img)[:, :, ::-1]

    _, h, w, _ = _frames.shape
    # _fourcc = cv2.VideoWriter_fourcc(*'mp4v',)
    writer = cv2.VideoWriter(file_path, -1, fps, (w, h))

    for frame in _frames:
        writer.write(img_to_cv_format(frame))

    writer.release()


def extract_clips(video_ranges, path_to_frames, path_to_storage):
    for i, clip_range in enumerate(video_ranges):
        t_start = timer()

        frames = np.array([cv2.imread(f'{path_to_frames}/{frame}.jpg') for frame in range(clip_range[0], clip_range[-1])])
        frames2video(f'{path_to_storage}/clip{i}.mp4', frames)

        t_finish = timer()
        print(f'the {i}st clip got ready in {t_finish - t_start} ms.')


def extract_posters(_single_frames, path_to_frames, path_to_storage):
    for _i, image in enumerate(_single_frames):
        img = cv2.imread(f'{path_to_frames}/{image}.jpg', 0)
        cv2.imwrite(f'{path_to_storage}/{_i}.jpg', img)

#  \/  \/ uncomment to extract frames \/  \/
# vide2frames(os.path.abspath('TED Micheal Levin .mp4'))
# print('done extracting frames.')


# manually selected ranges  -> index_single_frames & index_video_ranges
# todo: get user input for these values

index_single_frames = [
    2183,
    2901,
    35327,

]

index_video_ranges = [
    [5703, 6800],
    [25449, 25589],
    [25590, 25880],
    [25881, 26002],
    [26003, 26124],
    [26125, 26365],
    [26367, 26602],
    [26602, 26734],
    [26735, 27198],
    [27489, 27529],
    [27530, 27599],
    [27600, 27924],
    [27925, 28046],
    [28047, 28167],
    [28169, 28409],
    [28411, 28710],
]

extract_clips(index_video_ranges, "frames", "clips")
extract_posters(index_single_frames, "frames", "posters")

