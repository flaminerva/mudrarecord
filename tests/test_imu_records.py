"""IMU record layout (firmware 6.0.9.10), tested on packets captured from a
Mudra Link lying still (one packet per record combination)."""
import pytest

from mudrarecord import protocol

# acc + gyro enabled, framing (acc, gyro): both records share the timestamp.
ACC_GYRO = bytes.fromhex(
    "0a703fb0052cfc703fe80570fca03fb40558fca03f8c0538fcb43f640538fc1c40700564fc"
    "04407c0558fc7c3f300554fc3ae5c800046500eafffdff880005000300af000200f9ff9400"
    "e1ffdfff6c00d5ff00005500e1fff4ff4d00020015005500e8ff02003ae5c800")
# acc + gyro enabled, framing (gyro, acc): the pair is split across packets.
GYRO_ACC = bytes.fromhex(
    "047200fdff07005f0004000e005c000200e9ff8200e3ffe7ff7c000200f4ff7000feff0500"
    "5200fdff1f006500f6ffffff390f16010a803f3805b0fd743f3405a0fda03f1c050cfec83f"
    "3405f0fdcc3f2005ccfdc83f2805d0fdb03f0805b0fdbc3fec04dcfdba2a1601")
# acc only.
ACC_ACC = bytes.fromhex(
    "0a303ff0058cfb5c3fa805d4fb603fdc0504fc983fe005fcfbc43fc405acfb743fc4058cfb"
    "983fac05d0fb683f0406c0fbbffb4f010a443f3406f0fb9c3f04061cfc843f0806a4fbc43f"
    "100694fbe83ff4051cfcac3fac0548fc983fb00500fcb43ff005dcfb34175001")
# gyro only.
GYRO_GYRO = bytes.fromhex(
    "043f00dafffcff5d00ecff1f006800eeff25007000f1fffdff6100fdfff6ff650003000e00"
    "6800eeffedff7300d4fff1ff94c96c01046b00dcffedff5600e9ff10006c000b002f006200"
    "21004700760023000b006600170002007600000003007400f1fff4ff48e56c01")


def _norm(s):
    return sum(v * v for v in s) ** 0.5


@pytest.mark.parametrize("packet, kinds", [
    (ACC_GYRO, ["acc", "gyro"]),
    (GYRO_ACC, ["gyro", "acc"]),
    (ACC_ACC, ["acc", "acc"]),
    (GYRO_GYRO, ["gyro", "gyro"]),
])
def test_split_real_packets(packet, kinds):
    recs = protocol.split_imu_records(packet)
    assert [r.kind for r in recs] == kinds
    for r in recs:
        assert len(r.samples) == 8
        if r.kind == "acc":     # still: |a| ≈ 1 g ≈ 16384 LSB
            assert all(abs(_norm(s) / 16384 - 1) < 0.05 for s in r.samples)
        else:                   # still: near zero (bias ~ 100 LSB)
            assert all(_norm(s) < 300 for s in r.samples)


def test_record_fields_and_timestamps():
    a, g = protocol.split_imu_records(ACC_GYRO)
    assert (a.tag, g.tag) == (0x0A, 0x04)
    assert a.samples[0] == (0x3F70, 0x05B0, -980)
    assert a.device_ts == g.device_ts == 0x00C8E53A
    first, second = protocol.split_imu_records(ACC_ACC)
    assert 0 < second.device_ts - first.device_ts < 20000   # consecutive acc records


def test_legacy_decoder_mislabels_gyro_on_this_firmware():
    # Documents the bug the record layout fixes: the legacy decoder ignores
    # 0x0a-first packets and returns gyro samples as "acc" for 0x04-first ones.
    assert protocol.decode_imu_packet(ACC_GYRO) == []
    legacy = protocol.decode_imu_packet(GYRO_ACC)
    gyro_record = protocol.split_imu_records(GYRO_ACC)[0]
    assert [tuple(s) for s in legacy] == list(gyro_record.samples)


@pytest.mark.parametrize("n", [0, 1, 52, 54, 105, 107])
def test_split_rejects_bad_lengths(n):
    assert protocol.split_imu_records(bytes(n)) == []


def test_unknown_tag_is_reported_not_mislabelled():
    pkt = bytearray(ACC_GYRO)
    pkt[53] = 0x33
    assert [r.kind for r in protocol.split_imu_records(bytes(pkt))] == ["acc", "unknown"]


def test_layout_for_firmware():
    assert protocol.imu_layout_for_firmware("6.0.9.10") == protocol.IMU_LAYOUT_RECORDS
    assert protocol.imu_layout_for_firmware("6.0.9.10\x00") == protocol.IMU_LAYOUT_RECORDS
    assert protocol.imu_layout_for_firmware("6.0.11.5") == protocol.IMU_LAYOUT_LEGACY
    assert protocol.imu_layout_for_firmware("") == protocol.IMU_LAYOUT_LEGACY

