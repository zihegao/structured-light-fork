import cv2
import cv2.aruco as aruco

# Define parameters
squares_Nx = 7             # number of chessboard squares in X direction
squares_Ny = 9              # number of chessboard squares in Y direction
square_length = 30         # square length in mm
marker_length = 21         # marker side length, in mm, (must be < square_length)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) # dictionary of Aruco markers, 4x4 resolution, 50 unqiue markers

# Create the Charuco board
board = aruco.CharucoBoard(
    (squares_Nx, squares_Ny),
    square_length,     
    marker_length,     
    dictionary=dictionary
)

if  __name__ == "__main__":
    # Generate the Charuco board image and save to png file for printing
    # Image is generated with the desired size (in pixels). If the printer uses the same DPI setting, the physical size will be correct.
    
    # Specify DPI
    DPI = 600 # Dots per inch, 600 is common for high-quality printing
    DPMM = DPI / 25.4  # Dots per millimeter

    # Calculate output image size in pixels
    out_width = int(squares_Nx*square_length*DPMM)
    out_height = int(squares_Ny*square_length*DPMM)
    image = board.generateImage(
        outSize=(out_width, out_height),   # output size in pixels
        marginSize=0,                     # extra margin in pixels
        borderBits=1
    )

    # Print instructions:
    print(f"To print the Charuco board:")
    print(f"- Use any paper size as long as large enough to fit the board ({squares_Nx*square_length}mm x {squares_Ny*square_length}mm)")
    print(f"- Set printer DPI to {DPI}")
    print(f"- Print at 100% scale (no fit-to-page or scaling)")
    print(f"- Verify the printed board size with a ruler")
    print(f"- Example printing software: use IrfanView, set Image->Information->DPI, then print")

    # Save to file
    cv2.imwrite("charuco_board.png", image)