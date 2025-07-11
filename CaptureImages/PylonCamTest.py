from interface import IDSinterface
import cv2

def main():
    # initialize and start the camera up
    my_interface = IDSinterface()
    my_interface.select_and_start_device()

    # sets exposure time to 50ms and gain to 2.0 to upper contrast and make image usable 
    my_interface.set_exposure_time(1000 * 50)  # microseconds
    my_interface.set_gain(2.0)

    # captures the image
    image = my_interface.capture()

    # then saves the image using OpenCV 
    filename = "captured_image.png"
    cv2.imwrite(filename, image)

    print(f"Image saved as: {filename}")

    # clean up
    my_interface.stop_acquisition()
    del my_interface

if __name__ == "__main__":
    main()
