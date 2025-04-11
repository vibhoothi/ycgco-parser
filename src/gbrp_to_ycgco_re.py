import os
import numpy as np
import argparse

# Constants
bitdepth_y = 10
bitdepth_c = 10
bitdepth_rgb = 10
bitdepth_yuv = 8
range_y = 219  # Difference between max and min Y
range_uv = 224  # Difference between max and min UV
range_rgb_max = np.left_shift(1, (bitdepth_c - 2)) - 1  # 255 here
range_ycgco_max = np.left_shift(1, (bitdepth_c)) - 1  # 1023 here
bitshift_offset = np.left_shift(1, (bitdepth_c - 1))  # 512 here


def load_gbrp_frame(file_handle, width, height):
    """
    Load a single 8-bit GBR Planar frame from an open file handle.
    """
    pixels_per_frame = width * height
    frame_data = file_handle.read(pixels_per_frame * 3)

    if len(frame_data) < pixels_per_frame * 3:
        return None  # End of file or incomplete frame

    # Create gbr_data from buffer
    gbr_data = np.frombuffer(frame_data, dtype=np.uint8)

    # Extract G, B, R planes
    g = gbr_data[:pixels_per_frame].reshape(height, width)
    b = gbr_data[pixels_per_frame:2 * pixels_per_frame].reshape(height, width)
    r = gbr_data[2 * pixels_per_frame:].reshape(height, width)

    # Stack into RGB order
    return np.stack([r, g, b], axis=2)


def convert_rgb_frame_to_ycgco_re(rgb_frame):
    """
    Convert a single 8-bit RGB frame to 10-bit YCgCo-RE.
    """
    height, width, _ = rgb_frame.shape

    # Initialize output array
    ycgco_frame = np.zeros((height, width, 3), dtype=np.uint16)

    # We need to have this in int32 to avoid overflow in transformations
    rgb_frame = rgb_frame.astype(np.int32)

    # Extract RGB channels
    frame_r = rgb_frame[:, :, 0]
    frame_g = rgb_frame[:, :, 1]
    frame_b = rgb_frame[:, :, 2]

    # Apply YCgCo-RE transformation
    frame_co = frame_r - frame_b
    const_t = frame_b + np.right_shift((frame_co), 1)
    frame_cg = frame_g - const_t
    frame_y = const_t + np.right_shift(frame_cg, 1)

    # Clip to valid YCgCo-RE Range if there are any outliers
    ycgco_frame[:, :, 0] = frame_y
    ycgco_frame[:, :, 1] = np.clip(
        frame_cg + bitshift_offset, 0, range_ycgco_max)
    ycgco_frame[:, :, 2] = np.clip(
        frame_co + bitshift_offset, 0, range_ycgco_max)

    return ycgco_frame


def convert_ycgco_re_frame_to_rgb(ycgco_frame, return_gbr=False):
    """
    Convert a single 10-bit YCgCo-RE frame to 8/10-bit RGB.
    """
    height, width, _ = ycgco_frame.shape

    # Initialize output array
    rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)

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

    if return_gbr:
        rgb_frame = rgb_frame[:, :, [1, 2, 0]]

    return rgb_frame


def process_gbrp_to_ycgco_re(
        input_file, width, height, start_frame=0, num_frames=None, decode_only=False):
    """
    Process GBRP (8 Bit) to YCgCo-RE.
    """
    pixels_per_frame = width * height
    bytes_per_frame = pixels_per_frame * 3
    # We dealing 10 bit here
    if decode_only:
        bytes_per_frame = bytes_per_frame * 2

    # Calculate frames
    file_size = os.path.getsize(input_file)
    total_frames = file_size // bytes_per_frame

    if start_frame >= total_frames:
        raise ValueError(
            f"Start frame {start_frame} exceeds total frames {total_frames}")

    if num_frames is None:
        num_frames = total_frames - start_frame
    else:
        num_frames = min(num_frames, total_frames - start_frame)

    print(
        f"Processing {num_frames} frames starting from frame {start_frame}; Total Frames in Video {total_frames}")

    with open(input_file, 'rb') as f_in:
        # Skip to start frame
        f_in.seek(start_frame * bytes_per_frame)

        # Process frames
        for i in range(num_frames):
            if decode_only:
                # Read the frame data and reshape into YCgCo planes
                ycgco_data = np.frombuffer(
                    f_in.read(bytes_per_frame), dtype=np.uint16)
                # Extract the planes and reshape
                y = ycgco_data[:pixels_per_frame].reshape(height, width)
                cg = ycgco_data[pixels_per_frame:2 *
                                pixels_per_frame].reshape(height, width)
                co = ycgco_data[2 * pixels_per_frame:].reshape(height, width)
                bit_mask = 0x03FF  # 10-bit mask (1023)
                ycgco_frame = np.stack(
                    [y & bit_mask, cg & bit_mask, co & bit_mask], axis=2)
                gbr_frame_buffer = convert_ycgco_re_frame_to_rgb(
                    ycgco_frame, return_gbr=True)
                if i == 0 or i == num_frames - 1 or i % 10 == 0:
                    print(
                        f"Processed frame {start_frame + i + 1}/{start_frame + num_frames}")
                yield gbr_frame_buffer
            else:
                rgb_frame = load_gbrp_frame(f_in, width, height)
                if rgb_frame is None:
                    break
                ycgco_frame = convert_rgb_frame_to_ycgco_re(rgb_frame)
                rgb_frame_buffer = convert_ycgco_re_frame_to_rgb(
                    ycgco_frame)
                round_trip_diff = np.unique(rgb_frame_buffer - rgb_frame)
                # Print progress occasionally
                if i == 0 or i == num_frames - 1 or i % 10 == 0:
                    print(
                        f"Processed frame {start_frame + i + 1}/{start_frame + num_frames}")
                    if len(round_trip_diff) > 1:
                        print(
                            "Warning: Round-trip conversion mismatch!",
                            round_trip_diff)
                    else:
                        print("Round-trip conversion successful.")

                yield ycgco_frame

    print(f"Conversion complete.")


def save_ycgco_frames_planar(ycgco_frames, output_file, decode_only=False):
    """
    Save YCgCo-RE frames to a file in planar format.
    """
    target_dtype = np.uint8 if decode_only else np.uint16
    with open(output_file, 'wb') as f_out:
        frame_count = 0
        for frame in ycgco_frames:
            # Write planar data (Y plane, then Cg plane, then Co plane)
            frame[:, :, 0].astype(target_dtype).tofile(f_out)  # Y plane
            frame[:, :, 1].astype(target_dtype).tofile(f_out)  # Cg plane
            frame[:, :, 2].astype(target_dtype).tofile(f_out)  # Co plane
            frame_count += 1

    print(f"Saved {frame_count} frames to {output_file}")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Convert GBRP video to YCgCo-RE format')
    parser.add_argument('--input_file', '-i', help='Input GBRP video path')
    parser.add_argument(
        '--output_file', '-o', nargs='?',
        help='Output YCgCo-RE file path (default: input_file + "_ycgco_re.yuv")')
    parser.add_argument(
        '--width', type=int, default=1280, help='Frame width in pixels')
    parser.add_argument(
        '--height', type=int, default=720, help='Frame height in pixels')
    parser.add_argument(
        '--frames', '-f', type=int, default=5, help='Number of frames to process (default: 5 frames)')
    parser.add_argument(
        '--start', '-s', type=int, default=0, help='Starting frame (default: 0)')
    parser.add_argument(
        "--decode", '-d', action='store_true', help='Decode the YCgCo-RE file back to RGB (default: False)')

    args = parser.parse_args()

    # Process arguments
    input_file = args.input_file if args.input_file else 'ducks_take_off_444_720p50_5f_rgb.rgb'
    if args.decode:
        output_file = args.output_file if args.output_file else input_file + '_decoded.rgb'
    else:
        output_file = args.output_file if args.output_file else input_file + '_ycgco_re.yuv'
    width, height = args.width, args.height
    frame_count = args.frames
    start_frame = args.start

    print(f"Processing {input_file} ({width}x{height}) to {output_file}")

    ycgco_frames = process_gbrp_to_ycgco_re(
        input_file, width, height, start_frame=0, num_frames=frame_count, decode_only=args.decode)
    save_ycgco_frames_planar(
        ycgco_frames,
        output_file,
        decode_only=args.decode)
    input_file = args.input_file if args.input_file else ''
