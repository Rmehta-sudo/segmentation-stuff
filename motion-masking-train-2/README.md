# text-motion-mask

## Speed notes

The current teacher (patch correlation / block matching) has a finite search radius.
For a lookahead of `x_frames=4`, a displacement cap of `max_displacement=5` means it can
reliably track roughly up to $5/4 \approx 1.25$ px/frame along a given axis.

In practice, once the true pixel speed gets high (roughly ~3.5 px/frame and above),
the teacher often fails due to search-range limits and aliasing.

So, in `generate-vids.py` we cap `SPEED_RANGE_PX_PER_FRAME` to `<= 3.5` by default.
If you want higher-speed videos, you should also increase the teacher's
`--max-displacement` (and expect extraction to get slower), or move to a more robust
coarse-to-fine / multi-scale teacher.
