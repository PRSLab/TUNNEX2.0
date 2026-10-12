# --- Test: test_tunnex_input.py. Unit tests for tunnex_input.py
# Run with: pytest test_tunnex_input.py -v ---


# --- Modules ---
from pathlib import Path
from unittest.mock import MagicMock


# --- Module to test ---
from tunnex_2.irc_computations.ancillary_functions.tunnex_input import ( # type: ignore
    write_input_head,
    writing_el_energy,
    writing_zpve_energy)


# --- Helpers and shared data ---
EL_ENERGY = [ (-1.0, -153.10), (-0.5, -153.08), (0.0, -153.05), (0.5, -153.12), (1.0, -153.15)]
ZPVE_FWD = [(0.5, 0.03), (1.0, 0.04)]
ZPVE_REV = [(-1.0, 0.02), (-0.5, 0.025)]
TS_ZPVE = (0.0, 0.05)


def _make_settings(**kwargs):
    """ Returns a TunnexInputSettings-like MagicMock with sensible defaults """
    defaults = dict(freq = 1234.56, T = 300.0, E0 = -153.123456, ZPVE0 = 0.054321,
        potential_scaling_factor = 1.0, number_of_levels = 5,
        T_arrhenius_min = 200, T_arrhenius_max = 400, T_step = 10,
        prob_aver_mode_qmt = "finite_sum")
    defaults.update(kwargs)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


# ===========================================================================
# --- write_input_head - normal behaviour and input validation ---
# ===========================================================================
class TestWriteInputHead:

    def test_returns_path_with_correct_stem(self, tmp_path):
        """ Verify the generated file has the correct filename stem """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        result = write_input_head("_forward", src, _make_settings())
        expected = "mol_forward"
        assert result.stem == expected, f"expected {expected}, got {result.stem}"

    def test_returns_txt_suffix(self, tmp_path):
        """ Verify the generated file has the .txt suffix """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        result = write_input_head("_backward", src, _make_settings())
        expected = ".txt"
        assert result.suffix == ".txt", f"expected {expected}, got {result.suffix}"

    def test_file_is_created(self, tmp_path):
        """ Verify the generated file is created successfully """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        result = write_input_head("_forward", src, _make_settings())
        assert result.is_file(), f"expected {True}, got {result.is_file()}"

    def test_header_contains_freq(self, tmp_path):
        """ Verify the header contains the specified frequency """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        expected = 999.99
        s = _make_settings(freq=expected)
        result = write_input_head("_forward", src, s)
        assert str(expected) in result.read_text(encoding="utf-8"), f"expected {str(expected)} in {result}"

    def test_header_contains_temperature(self, tmp_path):
        """ Verify the header contains the specified temperature """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        expected = 250.0
        s = _make_settings(T=expected)
        result = write_input_head("_forward", src, s)
        assert str(expected) in result.read_text(encoding="utf-8"), f"expected {str(expected)} in {result}"

    def test_header_contains_E0(self, tmp_path):
        """ Verify the header contains the specified E0 value """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        expected = -123.456789
        s = _make_settings(E0=expected)
        result = write_input_head("_forward", src, s)
        assert str(expected) in result.read_text(encoding="utf-8"), f"expected {str(expected)} in {result}"

    def test_header_contains_ZPVE0(self, tmp_path):
        """ Verify the header contains the specified ZPVE0 value """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        expected = 0.012345
        s = _make_settings(ZPVE0=expected)
        result = write_input_head("_forward", src, s)
        assert str(expected) in result.read_text(encoding="utf-8"), f"expected {str(expected)} in {result}"

    def test_header_contains_scaling_factor(self, tmp_path):
        """ Verify the header contains the specified scaling factor and number of levels """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        expected = [1.23, "N=7", "integral"]
        s = _make_settings(potential_scaling_factor=expected[0],
                           number_of_levels=float(expected[1][-1]), prob_aver_mode_qmt="integral")
        result = write_input_head("_forward", src, s)
        for val in expected:
            assert str(val) in result.read_text(encoding="utf-8"), f"expected {str(val)} in {result}"

    def test_header_contains_temperature_range(self, tmp_path):
        """ Verify the header contains the specified temperature range """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        expected = [150, 500, 25]
        s = _make_settings(T_arrhenius_min=expected[0], T_arrhenius_max=expected[1], T_step=expected[2])
        result = write_input_head("_forward", src, s)
        text = result.read_text(encoding="utf-8")
        for val in expected:
            assert str(val) in text, f"expected {str(val)} in {result}"

    def test_different_directions_produce_different_files(self, tmp_path):
        """ Verify different directions produce different files and the function returns a Path object """
        src = tmp_path / "mol.gjf"
        src.write_text("")
        fwd = write_input_head("_forward", src, _make_settings())
        bwd = write_input_head("_backward", src, _make_settings())
        result = [fwd, bwd]
        assert fwd != bwd, f"expected {fwd} != {bwd}; test #1"
        for val in result:
            assert val.is_file(), f"expected {True}, got {val.is_file()}; test #2"
            assert isinstance(val, Path), f"expected {Path}, got {type(val)}; test #3"


# ===========================================================================
# --- writing_el_energy - normal behaviour and input validation ---
# ===========================================================================
class TestWritingElEnergy:

    def test_appends_irc_el_section_header(self, tmp_path):
        """ Verify the (IRC, E) section header is appended """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_el_energy(f, EL_ENERGY)
        expected = "IRC; E"
        assert expected in f.read_text(encoding="utf-8"), f"expected {expected} in {f}"

    def test_appends_end_el_marker(self, tmp_path):
        """ Verify the (IRC, E) end marker is appended """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_el_energy(f, EL_ENERGY)
        expected = "END; E"
        assert expected in f.read_text(encoding="utf-8"), f"expected {expected} in {f}"

    def test_all_el_irc_points_written(self, tmp_path):
        """ Verify all IRC points and energies are written """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_el_energy(f, EL_ENERGY)
        text = f.read_text(encoding="utf-8")
        for irc, energy in EL_ENERGY:
            assert f"{abs(irc):.6f}" in text, f"expected {abs(irc):.6f} in {f}; test #1"
            assert f"{energy:.6f}" in text, f"expected {energy:.6f} in {f}; test #2"

    def test_forward_el_irc_sign_positive(self, tmp_path):
        """ Verify forward IRC values retain their sign """
        f = tmp_path / "out.txt"
        f.write_text("")
        expected = [-1.0, "-1.000000"]
        writing_el_energy(f, [(expected[0], -153.10)], reverse_key=False)
        text = f.read_text(encoding="utf-8")
        # IRC = -1.0, sign factor = +1, the function writes -1.000000
        assert expected[1] in text, f"expected {expected[1]} in {f}"

    def test_reverse_el_irc_sign_flipped(self, tmp_path):
        """ Verify reverse IRC values have their sign flipped """
        f = tmp_path / "out.txt"
        f.write_text("")
        expected = [-1.0, "1.000000"]
        writing_el_energy(f, [(expected[0], -153.10)], reverse_key=True)
        text = f.read_text(encoding="utf-8")
        # IRC = -1.0, sign factor = -1, the function writes 1.000000
        assert expected[1] in text, f"expected {expected[1]} in {f}"

    def test_el_forward_sorted_ascending(self, tmp_path):
        """ Verify forward IRC points are sorted in ascending order """
        f = tmp_path / "out.txt"
        f.write_text("")
        shuffled = [(1.0, -153.09), (-0.7, -153.10), (0.0, -153.05)]
        writing_el_energy(f, shuffled, reverse_key=False)
        lines = [l for l in f.read_text().splitlines() if ";" in l and "IRC" not in l and "END" not in l]
        ircs = [float(l.split(";")[0]) for l in lines]
        assert ircs == sorted(ircs), f"expected {sorted(ircs)}, got {ircs}"

    def test_el_reverse_sorted_ascending(self, tmp_path):
        """ Verify reverse IRC points are sorted in ascending order """
        f = tmp_path / "out.txt"
        f.write_text("")
        shuffled = [(1.0, -153.09), (-0.7, -153.10), (0.0, -153.05)]
        writing_el_energy(f, shuffled, reverse_key=True)
        lines = [l for l in f.read_text().splitlines() if ";" in l and "IRC" not in l and "END" not in l]
        ircs = [float(l.split(";")[0]) for l in lines]
        assert ircs == sorted(ircs, reverse=False), f"expected {sorted(ircs, reverse=True)}, got {ircs}"

    def test_el_appends_to_existing_content(self, tmp_path):
        """ Verify existing file content is preserved and the function returns None """
        f = tmp_path / "out.txt"
        f.write_text("HEADER\n")
        retn = writing_el_energy(f, EL_ENERGY)
        result = f.read_text().startswith("HEADER")
        assert retn is None, f"expected {None}, got {result}; test #1"
        assert result, f"expected {True}, got {result}; test #2"


# ===========================================================================
# --- writing_zpve_energy - normal behaviour and input validation ---
# ===========================================================================
class TestWritingZpveEnergy:

    def test_appends_irc_zpve_section_header(self, tmp_path):
        """ Verify the (IRC, ZPVE) section header is appended """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, TS_ZPVE)
        expected = "IRC; ZPVE"
        assert expected in f.read_text(encoding="utf-8"), f"expected {expected} in {f}"

    def test_appends_end_zpve_marker(self, tmp_path):
        """ Verify the (IRC, ZPVE) end marker is appended """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, TS_ZPVE)
        expected = "END; ZPVE"
        assert expected in f.read_text(encoding="utf-8"), f"expected {expected} in {f}"

    def test_none_zpve_writes_zeros(self, tmp_path):
        """ Verify zero ZPVE values are written when ZPVE data is missing """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, None, None, TS_ZPVE)
        text = f.read_text(encoding="utf-8")
        # Should contain two lines with (IRC_1, 0.000000); (IRC_2, 0.000000)
        zero_lines = [l for l in text.splitlines() if ";" in l and "0.000000" in l and "IRC" not in l and "END" not in l]
        expected = 2
        assert len(zero_lines) == expected, f"expected {expected}, got {len(zero_lines)}"

    def test_none_zpve_uses_el_energy_endpoints(self, tmp_path):
        """ Verify electronic energy endpoints are used when ZPVE data is missing """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, None, None, TS_ZPVE)
        text = f.read_text(encoding="utf-8")
        expected = ["-1.000000", "1.000000"]
        # Endpoints of electronic energy are -1.0 and 1.0
        for val in expected:
            assert val in text, f"expected {val} in {f}"

    def test_ts_zpve_included_in_output(self, tmp_path):
        """ Verify the transition-state ZPVE value is included in the output """
        f = tmp_path / "out.txt"
        f.write_text("")
        ts_zpve = (0.0, 0.099)
        expected = str(ts_zpve[1])
        writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, ts_zpve)
        assert expected in f.read_text(encoding="utf-8"), f"expected {expected} in {f}"

    def test_all_zpve_forward_points_written(self, tmp_path):
        """ Verify all forward ZPVE points are written """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, TS_ZPVE)
        text = f.read_text(encoding="utf-8")
        for _, zpve in ZPVE_FWD:
            assert f"{zpve:.6f}" in text, f"expected {zpve:.6f} in {text}"

    def test_all_zpve_reverse_points_written(self, tmp_path):
        """ Verify all reverse ZPVE points are written """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, TS_ZPVE)
        text = f.read_text(encoding="utf-8")
        for _, zpve in ZPVE_REV:
            assert f"{zpve:.6f}" in text, f"expected {zpve:.6f} in {text}"

    def test_forward_zpve_irc_sign_positive(self, tmp_path):
        """ Verify forward IRC values retain their sign """
        f = tmp_path / "out.txt"
        f.write_text("")
        zpve_fwd = [(1.0, 0.04)]
        zpve_rev = [(-1.0, 0.02)]
        expected = ["1.000000", "-1.000000"]
        writing_zpve_energy(f, EL_ENERGY, zpve_fwd, zpve_rev, (0.0, 0.05), reverse_key=False)
        text = f.read_text(encoding="utf-8")
        # IRC = -1.0, sign factor = +1, the function writes -1.000000
        for val in expected:
            assert val in text, f"expected {val} in {f}"

    def test_reverse_zpve_irc_sign_flipped(self, tmp_path):
        """ Verify reverse IRC values have their sign flipped """
        f = tmp_path / "out.txt"
        f.write_text("")
        zpve_fwd = [(1.0, 0.04)]
        zpve_rev = [(-1.0, 0.02)]
        expected = ["1.000000", "-1.000000"]
        writing_zpve_energy(f, EL_ENERGY, zpve_fwd, zpve_rev, (0.0, 0.05), reverse_key=True)
        text = f.read_text(encoding="utf-8")
        # IRC = -1.0, sign factor = -1, the function writes 1.000000
        for val in expected:
            assert val in text, f"expected {val} in {f}"

    def test_forward_sorted_ascending(self, tmp_path):
        """ Verify forward IRC points are sorted in ascending order """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, TS_ZPVE, reverse_key=False)
        data_lines = [l for l in f.read_text().splitlines() if ";" in l and "IRC" not in l and "END" not in l]
        ircs = [float(l.split(";")[0]) for l in data_lines]
        assert ircs == sorted(ircs), f"expected {sorted(ircs)}, got {ircs}"

    def test_reverse_sorted_ascending(self, tmp_path):
        """ Verify reverse IRC points are sorted in ascending order """
        f = tmp_path / "out.txt"
        f.write_text("")
        writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, TS_ZPVE, reverse_key=True)
        data_lines = [l for l in f.read_text().splitlines() if ";" in l and "IRC" not in l and "END" not in l]
        ircs = [float(l.split(";")[0]) for l in data_lines]
        assert ircs == sorted(ircs, reverse=False), f"expected {sorted(ircs, reverse=True)}, got {ircs}"

    def test_appends_to_existing_content(self, tmp_path):
        """ Verify existing file content is preserved and the function returns None """
        f = tmp_path / "out.txt"
        f.write_text("EXISTING\n")
        retn = writing_zpve_energy(f, EL_ENERGY, ZPVE_FWD, ZPVE_REV, TS_ZPVE)
        result = f.read_text().startswith("EXISTING")
        assert retn is None, f"expected {None}, got {result}; test #1"
        assert result, f"expected {True}, got {result}; test #2"