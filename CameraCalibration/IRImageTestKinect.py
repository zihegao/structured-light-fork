from primesense import openni2
import numpy as np
import cv2

# initialize OpenNI2 with the path to the Tools folder
openni2.initialize("C:\Program Files\OpenNI2\Tools")

# connect to the first available device
dev = openni2.Device.open_any()

# create infrared and color video streams
ir_stream = dev.create_ir_stream()
color_stream = dev.create_color_stream()

# set resolution, format, and frame rate for both streams
color_stream.set_video_mode(openni2.VideoMode(openni2.PIXEL_FORMAT_RGB888, 640, 480, 10))
ir_stream.set_video_mode(openni2.VideoMode(openni2.PIXEL_FORMAT_GRAY8, 640, 480, 5))

# sync depth and color streams (even though depth isn't used here)
dev.set_depth_color_sync_enabled(True)

# start both streams
ir_stream.start()
color_stream.start()

# main loop - run until the user presses 'q'
captureNum = 0
k=None
while k != ord('q'):

    # read one frame from each stream
    frame_color = color_stream.read_frame()
    frame_ir = ir_stream.read_frame()

    # convert raw buffers into NumPy arrays with correct shapes
    ir_img = np.array(frame_ir.get_buffer_as_uint8()).reshape(frame_ir.height, frame_ir.width)
    color_img = np.array(frame_color.get_buffer_as_triplet()).reshape(frame_color.height, frame_color.width, -1)

    # show both images
    cv2.imshow("depth",ir_img)
    cv2.imshow("color",color_img)

    # wait for 30 ms for a keypress
    k = cv2.waitKey(30)

# stop the streams when finished
ir_stream.stop()
color_stream.stop()
#openni2.unload()
