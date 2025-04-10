import os
import numpy as np
import colour
import argparse


def Clip3(x, y, z):
    """
    Clips value z between x and y
    - Returns x if z < x
    - Returns y if z > y
    - Returns z otherwise
    """
    return np.where(z < x, x, np.where(z > y, y, z))


# Constants
bitdepth_y = 10
bitdepth_c = 10
bitdepth_rgb = 10
bitdepth_yuv = 8
range_y = 219  # Difference between max and min Y
range_uv = 224  # Difference between max and min UV
range_rgb_max = np.left_shift(1, (bitdepth_c - 2)) - 1
range_ycgco_max = np.left_shift(1, (bitdepth_c)) - 1
bitshift_offset = np.left_shift(1, (bitdepth_c - 1))  # 512 here


def load_yuv444p_frame(file_handle, width, height):
    """
    Load a single 8-bit YUV444p frame from an open file handle.

    Parameters:
    file_handle: Open file handle positioned at the start of a frame
    width (int): Frame width
    height (int): Frame height

    Returns:
    np.ndarray: YUV frame with shape (height, width, 3) or None if end of file
    """
    pixels_per_frame = width * height
    bytes_per_frame = pixels_per_frame * 3

    # Read one frame
    frame_data = file_handle.read(bytes_per_frame)
    if len(frame_data) < bytes_per_frame:
        return None  # End of file or incomplete frame

    # Initialize frame array
    yuv_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Extract Y, U, V planes
    y = np.frombuffer(frame_data, dtype=np.uint8, count=pixels_per_frame)
    u = np.frombuffer(frame_data, dtype=np.uint8,
                      count=pixels_per_frame, offset=pixels_per_frame)
    v = np.frombuffer(frame_data, dtype=np.uint8,
                      count=pixels_per_frame, offset=2 * pixels_per_frame)

    # Reshape and store in yuv_frame
    yuv_frame[:, :, 0] = y.reshape(height, width)
    yuv_frame[:, :, 1] = u.reshape(height, width)
    yuv_frame[:, :, 2] = v.reshape(height, width)

    return yuv_frame


def convert_yuv_frame_to_ycgco_re(yuv_frame):
    """
    Convert a single 8-bit YUV frame to 10-bit YCgCo-RE.
    8-bit YUV to 8-bit RGB conversion is done using the colour package.
    The RGB values are then transformed to YCgCo-RE format.

    Parameters:
    yuv_frame (np.ndarray): YUV frame with shape (height, width, 3)

    Returns:
    np.ndarray: YCgCo-RE frame with shape (height, width, 3)
    """
    height, width, _ = yuv_frame.shape

    # Initialize output array
    ycgco_frame = np.zeros((height, width, 3), dtype=np.uint16)
    rgb_frame = np.zeros((height, width, 3), dtype=np.uint16)

    # Use colour package to convert to RGB 10 bit full range
    rgb_frame = colour.YCbCr_to_RGB(
        yuv_frame,
        in_bits=8,
        in_int=True,
        in_legal=True,
        out_int=True,
        out_bits=8,
        out_legal=False)

    frame_r = rgb_frame[:, :, 0]
    frame_g = rgb_frame[:, :, 1]
    frame_b = rgb_frame[:, :, 2]

    # Apply YCgCo-RE transformation
    frame_co = frame_r - frame_b
    const_t = frame_b + np.right_shift((frame_co), 1)
    frame_cg = frame_g - const_t
    frame_y = const_t + np.right_shift(frame_cg, 1)

    ycgco_frame[:, :, 0] = frame_y
    ycgco_frame[:, :, 1] = np.clip(
        frame_cg + bitshift_offset, 0, range_ycgco_max)
    ycgco_frame[:, :, 2] = np.clip(
        frame_co + bitshift_offset, 0, range_ycgco_max)

    return ycgco_frame, rgb_frame


def convert_ycgco_re_frame_to_rgb(ycgco_frame):
    """
    Convert a single 10-bit YCgCo-RE frame to 8/10-bit RGB.

    Parameters:
    ycgco_frame (np.ndarray): YCgCo-RE frame with shape (height, width, 3)

    Returns:
    np.ndarray: RGB frame with shape (height, width, 3)
    """
    height, width, _ = ycgco_frame.shape

    # Initialize output array
    target_dtype = np.uint8
    rgb_frame = np.zeros((height, width, 3), dtype=target_dtype)

    # Extract YCgCo channels
    frame_y = ycgco_frame[:, :, 0].astype(np.int32)
    frame_cg = ycgco_frame[:, :, 1].astype(np.int32)
    frame_co = ycgco_frame[:, :, 2].astype(np.int32)

    # Remove lifting offset
    frame_cg = frame_cg - bitshift_offset
    frame_co = frame_co - bitshift_offset

    # Apply inverse YCgCo-RE transformation
    const_t = frame_y - np.right_shift(frame_cg, 1)
    frame_g = const_t + frame_cg
    frame_b = const_t - np.right_shift(frame_co, 1)
    frame_r = frame_b + frame_co

    # Clip to valid RGB Range if there are any outlier
    rgb_frame[:, :, 0] = np.clip(frame_r, 0, range_rgb_max)
    rgb_frame[:, :, 1] = np.clip(frame_g, 0, range_rgb_max)
    rgb_frame[:, :, 2] = np.clip(frame_b, 0, range_rgb_max)

    return rgb_frame


def process_yuv444p_to_ycgco_re(
        input_file, width, height, start_frame=0, num_frames=None):
    """
    Process YUV444p to YCgCo-RE sequentially, returning frames one at a time.

    Parameters:
    input_file (str): Path to input YUV444p file
    width (int): Frame width
    height (int): Frame height
    start_frame (int): First frame to process
    num_frames (int): Number of frames to process (None for all remaining)

    Returns:
    Generator: Yields each converted YCgCo-RE frame
    """
    pixels_per_frame = width * height
    bytes_per_frame = pixels_per_frame * 3
    processed_frames = 0

    # Calculate total frames
    file_size = os.path.getsize(input_file)
    max_frames = file_size // bytes_per_frame

    if start_frame >= max_frames:
        raise ValueError(
            f"Start frame {start_frame} exceeds total frames {max_frames}")

    if num_frames is None:
        num_frames = max_frames - start_frame
    else:
        num_frames = min(num_frames, max_frames - start_frame)

    print(f"Processing {num_frames} frames starting from frame {start_frame}")

    with open(input_file, 'rb') as f_in:
        # Skip to start frame
        f_in.seek(start_frame * bytes_per_frame)

        # Process each frame sequentially
        for frame_idx in range(num_frames):
            # Load a single YUV frame
            yuv_frame = load_yuv444p_frame(f_in, width, height)

            if yuv_frame is None:
                break  # End of file

            # Convert to YCgCo-RE
            ycgco_frame, rgb_frame_buffer = convert_yuv_frame_to_ycgco_re(
                yuv_frame)
            rgb_frame = convert_ycgco_re_frame_to_rgb(
                ycgco_frame)
            # Check the round-trip conversion
            round_trip_diff = np.unique(rgb_frame_buffer - rgb_frame)
            if len(round_trip_diff) > 1:
                print(
                    "Warning: Round-trip conversion mismatch!",
                    round_trip_diff)
            else:
                print("Round-trip conversion successful!", round_trip_diff)
            processed_frames += 1
            if (processed_frames % 10 == 0) or (
                    processed_frames == num_frames):
                print(
                    f"Processed frame {start_frame + processed_frames}/{start_frame + num_frames}")

            # Yield the frame for further processing or writing
            yield ycgco_frame

    print(f"Conversion complete. Processed {processed_frames} frames.")


def save_ycgco_frames_planar(ycgco_frames, output_file):
    """
    Save YCgCo-RE frames to a file in planar format.

    Parameters:
    ycgco_frames: Iterable of YCgCo-RE frames with shape (height, width, 3)
    output_file (str): Path to output file
    """
    with open(output_file, 'wb') as f_out:
        frame_count = 0
        for frame in ycgco_frames:
            # Write planar data (Y plane, then Cg plane, then Co plane)
            frame[:, :, 0].astype(np.uint16).tofile(f_out)  # Y plane
            frame[:, :, 1].astype(np.uint16).tofile(f_out)  # Cg plane
            frame[:, :, 2].astype(np.uint16).tofile(f_out)  # Co plane
            frame_count += 1

    print(f"Saved {frame_count} frames to {output_file}")


# Example usage:
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Convert YUV444p video to YCgCo-RE format')
    parser.add_argument('--input_file', '-i', help='Input YUV444p file path')
    parser.add_argument(
        '--output_file',
        '-o',
        nargs='?',
        help='Output YCgCo-RE file path (default: input_file + "_ycgco_re.yuv")')
    parser.add_argument(
        '--width',
        type=int,
        required=True,
        help='Frame width in pixels')
    parser.add_argument(
        '--height',
        type=int,
        required=True,
        help='Frame height in pixels')
    parser.add_argument(
        '--frames',
        '-f',
        type=int,
        default=5,
        help='Number of frames to process (default: 5 frames)')
    parser.add_argument(
        '--start',
        '-s',
        type=int,
        default=0,
        help='Starting frame (default: 0)')

    args = parser.parse_args()

    # Process arguments
    input_file = args.input_file
    output_file = args.output_file if args.output_file else input_file + '_ycgco_re.yuv'
    width, height = args.width, args.height
    frame_count = args.frames
    start_frame = args.start

    print(f"Processing {input_file} ({width}x{height}) to {output_file}")

    ycgco_frames = process_yuv444p_to_ycgco_re(
        input_file, width, height, start_frame=0, num_frames=frame_count)
    save_ycgco_frames_planar(ycgco_frames, output_file)
