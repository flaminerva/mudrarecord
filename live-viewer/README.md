Live plot of a `mudrarecord stream-csv` stream read from stdin:

    mudrarecord stream-csv | live-viewer/live-viewer --width=5s

Panels: SNC raw (-32768..32767), SNC RMS (0..32768), acc (±2 g), gyro
(±250 deg/s), fixed y axes, on one time axis in seconds before the newest
row. `--width` sets the time window (default 5s). SNC RMS is smoothed over
the last 6 packets (~100 ms); `--rms-packets 1` draws each packet's values
as upstream, `--rms-packets N` sets the window:

    mudrarecord stream-csv | live-viewer/live-viewer --width=5s --rms-packets 1

Upstream's options also work: one panel per `--group` (column numbers like
`4-9`, a pattern on the column names like `a*`, or a channel token like
`imu` selecting rows by their stream column). As upstream, each line is
scaled to the panel height on its own (y numbers hidden), except in
`--shared-range-groups` panels, which share one real y range:

    mudrarecord stream-csv | live-viewer/live-viewer --width=5s \
      --group='snc*' --group='rms*' --group='a*' --shared-range-groups=1,2,3

Ondřej Bojar and Claude Opus 4.8
MIT License
