import sys
import os
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d","--dir", default='./', type=str,
                    help='The directory path storing video frame image.')
parser.add_argument("-o","--output", default='video.avi', type=str,
                    help='Output video name.')
parser.add_argument("-f","--fps", default=30, type=int,
                    help='The fps of video.')
parser.add_argument("--width", default=960, type=int,
                    help='The width of video frame image.')
parser.add_argument("--height", default=536, type=int,
                    help='The height of video frame image.')

args, unknown = parser.parse_known_args()

Frame_Dir = args.dir
Output = args.output
FPS = args.fps
Width = args.width
Height = args.height
codec = cv2.VideoWriter_fourcc(*'XVID')
out_vid = cv2.VideoWriter(Output, codec, FPS, (Width, Height))

if __name__ == "__main__":
    if(os.path.isdir(Frame_Dir)):

        frame_list = os.listdir(Frame_Dir)
        frame_list = sorted(frame_list)
        for f in frame_list:
            img_path = os.path.join(Frame_Dir,f)
            img = cv2.imread(img_path)
            out_vid.write(img)
            print(f)
        out_vid.release()
    else:
        print("There is not a directory : ",Frame_Dir)