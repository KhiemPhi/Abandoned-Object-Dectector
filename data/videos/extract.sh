#!/bin/sh
ffmpeg -i ./ABODA/video1.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video1/frm_%05d.jpg
ffmpeg -i ./ABODA/video2.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video2/frm_%05d.jpg
ffmpeg -i ./ABODA/video3.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video3/frm_%05d.jpg
ffmpeg -i ./ABODA/video4.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video4/frm_%05d.jpg
ffmpeg -i ./ABODA/video5.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video5/frm_%05d.jpg
ffmpeg -i ./ABODA/video6.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video6/frm_%05d.jpg
ffmpeg -i ./ABODA/video7.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video7/frm_%05d.jpg
ffmpeg -i ./ABODA/video8.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video8/frm_%05d.jpg
ffmpeg -i ./ABODA/video9.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video9/frm_%05d.jpg
ffmpeg -i ./ABODA/video10.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video10/frm_%05d.jpg
ffmpeg -i ./ABODA/video11.avi -vf "scale=320:240,fps=25" -q:v 2 ../frames/ABODA/video11/frm_%05d.jpg