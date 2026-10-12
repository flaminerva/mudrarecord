"""live-viewer: sample placement, no filling of empty cells, line breaks."""
import importlib.machinery
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[1] / "live-viewer" / "live-viewer"
_loader = importlib.machinery.SourceFileLoader("live_viewer", str(_path))
lv = importlib.util.module_from_spec(importlib.util.spec_from_loader("live_viewer", _loader))
_loader.exec_module(lv)

HEADER = ("timestamp_ns,timestamp_iso,stream,snc1,snc2,snc3,rms1,rms2,rms3,"
          "ax,ay,az,gx,gy,gz,device_ts").split(",")


def _row(t_ms, stream, values, device_ts=None):
    cells = [""] * 13
    start = {"snc": 0, "imu": 6, "gyro": 9}[stream]
    for i, v in enumerate(values):
        cells[start + i] = str(v)
    if device_ts is not None:
        cells[12] = str(device_ts)
    return ",".join([str(int(t_ms * 1e6)), "", stream] + cells) + "\n"


def _read(lines, rms_packets=lv.RMS_PACKETS):
    signals = lv.make_signals(100.0, rms_packets)
    r = lv.Reader(io.StringIO("".join(lines)), HEADER, signals)
    r.run()
    return r, signals


def test_rows_of_one_packet_spread_over_the_interval():
    lines = []
    for p in range(4):                       # one acc record (8 rows) every 8 ms
        lines += [_row(8 * p, "imu", [16384 * k, 0, 0]) for k in range(8)]
    _, s = _read(lines)
    t, v = s["acc"].arrays()
    assert t[:8].tolist() == [0.0] * 8       # first packet: no interval yet
    assert np.allclose(np.diff(t[8:]), 0.001)  # then 1 ms apart, last at arrival
    assert t[-1] == pytest.approx(0.024)
    assert v[:8, 0].tolist() == list(range(8))  # scaled to g


def test_after_a_gap_the_median_interval_is_used():
    lines = []
    for p in [0, 1, 2, 3, 10]:               # packet 10 arrives after a gap
        lines += [_row(8 * p, "imu", [0, 0, 1]) for _ in range(8)]
    _, s = _read(lines)
    t, _ = s["acc"].arrays()
    last = t[-8:]
    assert np.allclose(np.diff(last), 0.001) and last[-1] == pytest.approx(0.080)
    conn = lv.connect_mask(t)
    assert not conn[-9] and conn[-8:-1].all()   # broken only across the gap


def test_rms_default_is_each_packets_value_as_upstream():
    rms = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0]
    lines = [_row(10 * p, "snc", [1, 2, 3, r, 5, float("nan")])
             for p, r in enumerate(rms) for _ in range(18)]
    s = lv.make_signals(100.0)
    lv.Reader(io.StringIO("".join(lines)), HEADER, s).run()
    t, v = s["snc"].arrays()
    assert np.allclose(t, 0.01 * np.arange(8))                 # one point per packet
    assert v[:, 0].tolist() == rms                             # unsmoothed
    assert np.allclose(v[:, 1], 5) and np.isnan(v[:, 2]).all()


def test_rms_one_point_per_packet_moving_rms_of_6_packets():
    rms = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0]
    lines = [_row(10 * p, "snc", [1, 2, 3, r, 5, float("nan")])
             for p, r in enumerate(rms) for _ in range(18)]
    _, s = _read(lines, rms_packets=6)
    t, v = s["snc"].arrays()
    assert np.allclose(t, 0.01 * np.arange(8))                 # one point per packet
    want = [np.sqrt(np.mean(np.square(rms[max(0, p - 5):p + 1]))) for p in range(8)]
    assert np.allclose(v[:, 0], want)                          # RMS over the last 6
    assert np.allclose(v[:, 1], 5) and np.isnan(v[:, 2]).all()


@pytest.mark.parametrize("n", [1, 3])
def test_rms_packets_sets_the_moving_rms_window(n):
    rms = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0]
    lines = [_row(10 * p, "snc", [1, 2, 3, r, 5, float("nan")])
             for p, r in enumerate(rms) for _ in range(18)]
    _, s = _read(lines, rms_packets=n)
    t, v = s["snc"].arrays()
    assert np.allclose(t, 0.01 * np.arange(8))                 # one point per packet
    want = [np.sqrt(np.mean(np.square(rms[max(0, p - n + 1):p + 1]))) for p in range(8)]
    assert np.allclose(v[:, 0], want)                          # RMS over the last n
    if n == 1:
        assert v[:, 0].tolist() == rms                         # each packet's value
    assert np.allclose(v[:, 1], 5) and np.isnan(v[:, 2]).all()


@pytest.mark.parametrize("argv", [["--rms-packets", "0"], ["--rms-packets", "-1"],
                                  ["--rms-packets", "1.5"],
                                  ["--rms-packets", "1", "--group", "snc*"],
                                  ["--rms-packets", "6", "--group", "snc*"]])
def test_rms_packets_rejected(argv):
    with pytest.raises(SystemExit):
        lv.main(argv)


def _imu_on_device_clock(ts_list, arrival_ms):
    lines = []
    for ts, a in zip(ts_list, arrival_ms):
        lines += [_row(a, "imu", [k, 0, 0], ts % (1 << 32)) for k in range(8)]
    return _read(lines)


def test_acc_on_device_clock_mapped_by_the_fit():
    # records every 8000 ticks; arrival jitters by 0-3 ms, device time does not
    n = 400
    ts = [1_000_000 + 8000 * k for k in range(n)]
    jitter = [3 * ((k * 7) % 4) / 3 for k in range(n)]
    _, s = _imu_on_device_clock(ts, [8 * k + j for k, j in enumerate(jitter)])
    t, v = s["acc"].arrays()
    d = np.diff(t[8:])
    assert np.allclose(d, d.mean(), rtol=1e-6)                 # evenly spaced, no jitter
    assert d.mean() == pytest.approx(0.001, rel=1e-3)          # 8000 ticks / 8, ~1 us ticks
    assert np.allclose(v[-8:, 0] * 16384, range(8))
    assert t[-1] == pytest.approx(8e-3 * (n - 1) + np.mean(jitter) / 1e3, abs=1e-4)


def test_device_clock_unwraps_and_breaks_at_gaps():
    start = (1 << 32) - 8000 * 150
    ts = [start + 8000 * k for k in range(300)] + [start + 8000 * 310]
    _, s = _imu_on_device_clock(ts, [8 * k for k in range(300)] + [8 * 310])
    t, _ = s["acc"].arrays()
    assert np.all(np.diff(t[8:]) > 0)                          # monotonic across the wrap
    conn = lv.connect_mask(t)
    assert not conn[-9] and conn[-8:-1].all()                  # gap: only broken there


def test_without_device_ts_acc_falls_back_to_arrival_time():
    _, s = _read([_row(8 * p, "imu", [0, 0, 1]) for p in range(3) for _ in range(8)])
    t, _ = s["acc"].arrays()
    assert t[-1] == pytest.approx(0.016) and np.allclose(np.diff(t[8:]), 0.001)


def test_signal_only_at_rows_with_values():
    lines = [_row(0, "imu", [1, 2, 3]), _row(1, "snc", [1, 2, 3, 4, 5, 6]),
             _row(2, "gyro", [131.072, 0, 0]), _row(3, "imu", [1, 2, 3])]
    r, s = _read(lines)
    assert len(s["acc"].arrays()[0]) == 2
    assert len(s["gyro"].arrays()[0]) == 1 and len(s["snc"].arrays()[0]) == 1
    assert s["gyro"].arrays()[1][0, 0] == pytest.approx(1.0)   # deg/s
    assert r.latest == pytest.approx(0.003)


def test_upstream_csv_without_gyro_columns():
    header = HEADER[:12]
    signals = lv.make_signals(10.0)
    r = lv.Reader(io.StringIO(""), header, signals)
    assert set(r.cols) == {"raw", "snc", "acc"}


def test_railed_snc_read_from_raw_hex():
    import struct
    vals = [100 + i for i in range(54)]
    vals[3 * 5 + 0], vals[3 * 5 + 2] = 32767, -32768            # row 5: ch1 / ch3 railed
    raw = struct.pack("<54h", *vals) + bytes(4)
    header = HEADER[:3] + ["raw_hex"] + HEADER[3:]
    lines = []
    for k in range(18):
        snc = ["nan" if v in (32767, -32768) else str(v) for v in vals[3 * k:3 * k + 3]]
        lines.append(",".join(["1000000", "", "snc", raw.hex()] + snc + ["1", "2", "3"]
                              + [""] * 7) + "\n")
    signals = lv.make_signals(100.0)
    lv.Reader(io.StringIO("".join(lines)), header, signals).run()
    _, v = signals["raw"].arrays()
    assert v[5].tolist() == [32767, 116, -32768]                 # at the edges, right row
    assert v[4].tolist() == [112, 113, 114] and v[6].tolist() == [118, 119, 120]
    no_hex = [",".join(ln.split(",")[:3] + ln.split(",")[4:]) for ln in lines]
    signals = lv.make_signals(100.0)
    lv.Reader(io.StringIO("".join(no_hex)), HEADER, signals).run()
    assert np.isnan(signals["raw"].arrays()[1][5, [0, 2]]).all()  # gap without raw_hex


def test_connect_mask_breaks_above_twice_the_median():
    t = np.array([0.0, 1, 2, 3, 6, 7, 8])
    assert lv.connect_mask(t).tolist() == [True, True, True, False, True, True, False]


@pytest.mark.parametrize("text, s", [("5s", 5.0), ("500ms", 0.5), ("2m", 120.0), ("3", 3.0)])
def test_parse_duration(text, s):
    assert lv.parse_duration(text) == pytest.approx(s)


# --- --group mode (upstream's options) ----------------------------------------

FULL_HEADER = HEADER[:3] + ["raw_hex"] + HEADER[3:]


def test_group_columns_as_upstream():
    def cols(spec):
        c, token = lv.group_columns(spec, FULL_HEADER)
        assert token is None
        return c
    assert cols("snc*") == ["snc1", "snc2", "snc3"]
    assert cols("rms*") == ["rms1", "rms2", "rms3"]
    assert cols("a*") == ["ax", "ay", "az"]
    assert cols("g?") == ["gx", "gy", "gz"]
    assert cols("4-6") == ["raw_hex", "snc1", "snc2"]      # 1-based, any column (as upstream)
    assert cols("11,13") == ["ax", "az"]


def test_channel_token_selects_rows_by_stream_as_upstream():
    lines = [_row(8 * p, "imu", [16384, 0, 0], 1000 + 8000 * p) for p in range(3) for _ in range(8)]
    lines += [_row(8 * p + 1, "gyro", [5, 6, 7], 1000 + 8000 * p) for p in range(3) for _ in range(8)]
    lines.sort(key=lambda ln: int(ln.split(",")[0]))
    rows = [ln.rstrip("\n").split(",") for ln in lines]
    # columns: those with data in imu rows of the sample (device_ts too, as upstream)
    assert lv.group_columns("imu", HEADER, rows) == (["ax", "ay", "az", "device_ts"], "imu")
    assert lv.group_columns("ax", HEADER, rows)[1] == "ax"         # a plain token, even a column name
    assert lv.group_columns("ax", HEADER, rows)[0] == [h for h in HEADER[3:]]   # no rows: all columns
    signals, panels = lv.group_layout(HEADER, ["imu", "gyro", "g*"], 100.0, rows)
    assert not any(p.get("lazy") for p in panels)                  # every --group panel shown at once
    assert [c[2] for c in panels[1]["curves"]] == ["gx", "gy", "gz", "device_ts"]
    assert [c[2] for c in panels[0]["curves"]] == ["ax", "ay", "az", "device_ts"]
    lv.Reader(io.StringIO("".join(lines)), HEADER, signals).run()
    assert len(signals["imu:ax"].arrays()[0]) == 24                # imu rows only
    assert len(signals["gx"].arrays()[0]) == 24 and signals["gx"].arrays()[1][0, 0] == 5


def test_readme_example_layout_and_shared_ranges():
    signals, panels = lv.group_layout(FULL_HEADER, ["snc*", "rms*", "a*"], 10.0)
    assert [p["title"] for p in panels] == ["snc*", "rms*", "a*"]
    assert [[c[2] for c in p["curves"]] for p in panels] == [
        ["snc1", "snc2", "snc3"], ["rms1", "rms2", "rms3"], ["ax", "ay", "az"]]
    assert all(p["y_range"] is None for p in panels)               # autoscaled
    assert lv.shared_groups("1,2,3", 3) == {0, 1, 2}
    with pytest.raises(SystemExit):
        lv.shared_groups("4", 3)
    data = [[np.array([0.0, 5.0])], [np.array([-2.0, 1.0])], [np.array([np.nan, 100.0])]]
    assert lv.y_ranges(data, {0, 1}) == [(-2.0, 5.0), (-2.0, 5.0), (99.0, 101.0)]
    assert lv.y_ranges([[np.array([np.nan])]], set()) == [None]
    # non-shared panels, as upstream: each line on its own 0-1 scale; flat at 0.5
    assert lv.normalize(np.array([10.0, 20.0, 15.0])).tolist() == [0.0, 1.0, 0.5]
    assert lv.normalize(np.array([7.0, 7.0, np.nan]))[:2].tolist() == [0.5, 0.5]


def test_group_mode_keeps_raw_values_and_new_placement():
    # per column: raw CSV values (no unit scaling, no RMS smoothing),
    # rms one point per packet, acc on the device clock, railed snc from raw_hex
    import struct
    vals = [100 + i for i in range(54)]
    vals[3 * 5] = 32767                                            # row 5: snc1 railed
    raw = struct.pack("<54h", *vals) + bytes(4)
    lines = []
    for p in range(3):
        for k in range(18):
            snc = ["nan" if 3 * k == 15 else str(v) for v in vals[3 * k:3 * k + 3]]
            lines.append(",".join([str(10_000_000 * p), "", "snc", raw.hex()] + snc
                                  + [str(100 + p), "2", "3"] + [""] * 7) + "\n")
    for r in range(30):                                            # acc records every 8000 ticks
        for k in range(8):
            lines.append(",".join([str(8_000_000 * r + 3_000_000 * (r % 2)), "", "imu", "ab"]
                                  + [""] * 6 + [str(16384), "0", "0"] + [""] * 3
                                  + [str(1_000_000 + 8000 * r)]) + "\n")
    lines.sort(key=lambda ln: int(ln.split(",")[0]))
    signals, _ = lv.group_layout(FULL_HEADER, ["snc*", "rms*", "a*"], 100.0)
    lv.Reader(io.StringIO("".join(lines)), FULL_HEADER, signals).run()
    _, s1 = signals["snc1"].arrays()
    assert s1[5, 0] == 32767 and s1[4, 0] == 112                   # railed value restored
    _, r1 = signals["rms1"].arrays()
    assert r1[:, 0].tolist() == [100, 101, 102]                    # one per packet, unsmoothed
    t, ax = signals["ax"].arrays()
    assert ax[0, 0] == 16384                                       # raw LSB
    assert np.allclose(np.diff(t[8:]), np.diff(t[8:]).mean(), rtol=1e-3)   # device clock: no jitter
