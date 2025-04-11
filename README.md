# Sample Repository to convert GBR Planar to YCgCo-RE (Even)


This is based on [ITU-T H.273 (07/2024)](https://www.itu.int/rec/T-REC-H.273).


## Workflow

Input (GBRP 8bit) -> YCgCo-RE (10bit) -> Raw YUV with YCgCo-RE data.

Tested with Round-trip conversion to RGB to the source to verified reversiblity.

## FFmpeg commands to make the Input sample

1. Convert input to Raw RGB stream if they are not GBRP
   > `ffmpeg -i $INPUT.y4m -pix_fmt gbrp $INPUT_GBPRP.rgb` 
2. Run the Program `gbrp_to_ycgco_re.py` with input and specifying the dimensions
3. Convert the output YCgCo-RE to whatever you want
   > `ffmpeg -r 24 -pix_fmt yuv444p10 -s 2048x858 -i $INPUT_ycgco_re.yuv -y -strict -1 $OUTPUT.Y4M`


## CLI Help

```bash
python3 src/gbrp_to_ycgco_re.py --help
usage: gbrp_to_ycgco_re.py [-h] [--input_file INPUT_FILE] [--output_file [OUTPUT_FILE]] [--width WIDTH] [--height HEIGHT] [--frames FRAMES] [--start START]
                           [--decode]

Convert GBRP video to YCgCo-RE format

options:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Input GBRP video path
  --output_file [OUTPUT_FILE], -o [OUTPUT_FILE]
                        Output YCgCo-RE file path (default: input_file + "_ycgco_re.yuv")
  --width WIDTH         Frame width in pixels
  --height HEIGHT       Frame height in pixels
  --frames FRAMES, -f FRAMES
                        Number of frames to process (default: 5 frames)
  --start START, -s START
                        Starting frame (default: 0)
  --decode, -d          Decode the YCgCo-RE file back to RGB (default: False)
```


## Other References

1. Libavif Implemetantion: Handled with `AVIF_REFORMAT_MODE_YCGCO_RE` flag. Can
   be found in [reformat.c](https://github.com/AOMediaCodec/libavif/blob/main/src/reformat.c#L360)
2. SparkYUV: Archived repo, but has YCgCo-RE [conversions](https://github.com/awxkee/sparkyuv/blob/1d31a63b17ac0273b7aa4579449b18e325e4eca9/src/YCgCoR-inl.h#L48)