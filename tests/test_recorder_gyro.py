"""Gyro rows, the appended CSV columns and the Mudra-Gyro LSL outlet."""
import sys
import types

import pytest

from mudrarecord import protocol
from mudrarecord.recorder import CSVSink, LSLSink, decode_imu, decode_samples

from test_imu_records import ACC_ACC, ACC_GYRO, GYRO_ACC, GYRO_GYRO

RECORDS = protocol.IMU_LAYOUT_RECORDS
LEGACY = protocol.IMU_LAYOUT_LEGACY


@pytest.mark.parametrize("packet, streams", [
    (ACC_GYRO, ["imu"] * 8 + ["gyro"] * 8),
    (GYRO_ACC, ["gyro"] * 8 + ["imu"] * 8),
    (ACC_ACC, ["imu"] * 16),
    (GYRO_GYRO, ["gyro"] * 16),
])
def test_decode_imu_records(packet, streams):
    rows = decode_imu(packet, RECORDS)
    assert [r[0] for r in rows] == streams
    recs = protocol.split_imu_records(packet)
    assert [r[2] for r in rows] == [recs[0].device_ts] * 8 + [recs[1].device_ts] * 8
    assert rows[0][1] == list(recs[0].samples[0])


def test_imu_rows_are_accelerometer_in_both_framings():
    # "imu" rows are acc in either framing (legacy: nothing / gyro).
    for packet in (ACC_GYRO, GYRO_ACC):
        acc = [r[1] for r in decode_imu(packet, RECORDS) if r[0] == "imu"]
        assert all(abs(sum(v * v for v in s) ** 0.5 / 16384 - 1) < 0.05 for s in acc)


def test_decode_imu_legacy_matches_decode_samples():
    for packet in (ACC_GYRO, GYRO_ACC):
        rows = decode_imu(packet, LEGACY)
        assert [r[1] for r in rows] == decode_samples("imu", packet)
        assert all(r[0] == "imu" and r[2] is None for r in rows)


def test_imu_layout_option_and_unknown_tags():
    assert protocol.resolve_imu_layout("auto", "6.0.9.10") == RECORDS
    assert protocol.resolve_imu_layout("auto", "6.0.11.5") == LEGACY
    assert protocol.resolve_imu_layout("records", "6.0.11.5") == RECORDS   # forced
    assert protocol.resolve_imu_layout("legacy", "6.0.9.10") == LEGACY
    pkt = bytearray(ACC_GYRO)
    pkt[53] = 0x33
    unknown = []
    assert [r[0] for r in decode_imu(bytes(pkt), RECORDS, unknown)] == ["imu"] * 8
    assert unknown == [0x33]                     # reported for the warning, not mislabelled


def test_csv_gyro_row_and_device_ts(tmp_path):
    path = tmp_path / "g.csv"
    sink = CSVSink(str(path), flush_every=1)
    sink.open()
    rows = decode_imu(GYRO_ACC, RECORDS)
    for stream, ch, ts in rows:
        sink.write(stream, 5, GYRO_ACC, ch, device_ts=ts)
    sink.close()
    lines = path.read_text().strip().splitlines()
    assert lines[0].split(",")[12:] == ["gx", "gy", "gz", "device_ts"]
    first = lines[1].split(",")
    assert first[2] == "gyro"
    assert first[3:12] == [""] * 9                       # snc, rms, acc empty
    assert first[12:15] == [str(v) for v in rows[0][1]]  # gx, gy, gz
    assert first[15] == str(rows[0][2])                  # raw u32
    acc = lines[9].split(",")
    assert acc[2] == "imu" and acc[12:15] == ["", "", ""] and acc[15] == str(rows[8][2])
    assert sink.count == 16


class _FakeOutlet:
    def __init__(self, info):
        self.info = info
        self.samples = []

    def push_sample(self, sample, ts):
        self.samples.append((sample, ts))


class _FakeDesc:
    def __init__(self):
        self.values, self.children = {}, []

    def append_child(self, name):
        c = _FakeDesc()
        self.children.append((name, c))
        return c

    def append_child_value(self, key, value):
        self.values[key] = value
        return self


class _FakeInfo:
    def __init__(self, name, type, channel_count, nominal_srate, channel_format, source_id):
        self.name, self.type, self.channel_count = name, type, channel_count
        self._desc = _FakeDesc()

    def desc(self):
        return self._desc


@pytest.fixture
def fake_pylsl(monkeypatch):
    mod = types.SimpleNamespace(StreamInfo=_FakeInfo, StreamOutlet=_FakeOutlet,
                                IRREGULAR_RATE=0.0, cf_float32="float32")
    monkeypatch.setitem(sys.modules, "pylsl", mod)
    return mod


def test_lsl_gyro_is_a_separate_outlet(fake_pylsl):
    sink = LSLSink("dev")
    sink.open()
    for stream, ch, ts in decode_imu(ACC_GYRO, RECORDS):
        sink.write(stream, 1_000_000_000, ACC_GYRO, ch, device_ts=ts)
    imu, gyro = sink._outlets["imu"], sink._outlets["gyro"]
    assert imu.info.name == "Mudra-IMU" and imu.info.channel_count == 3
    assert gyro.info.name == "Mudra-Gyro" and gyro.info.channel_count == 3
    labels = [c.values["label"] for _, c in gyro.info.desc().children[0][1].children]
    assert labels == ["gx", "gy", "gz"]
    # same host timestamp on both streams -> directly alignable in LSL
    assert {s[1] for s in imu.samples} == {s[1] for s in gyro.samples} == {1.0}
    assert len(imu.samples) == len(gyro.samples) == 8
