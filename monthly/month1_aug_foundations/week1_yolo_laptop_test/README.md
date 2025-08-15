
# created python venv and activated it
python -m venv mlrobotics
source mlrobotics/bin/activate

# what to install
pip install ultralytics opencv-python

# to check if installed properly
python -c "import cv2; import ultralytics; print('OpenCV & YOLO ready')"