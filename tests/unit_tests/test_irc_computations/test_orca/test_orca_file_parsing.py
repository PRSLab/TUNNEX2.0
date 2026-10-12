# --- Test: test_orca_file_parsing. Unit tests for .\gaussian\gaussian_orca_parsing.py
# Run with: pytest test_gaussian_orca_parsing.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch

import numpy as np

from tunnex_2.constants_and_dataclasses.dataclasses import IRCData # type: ignore


# --- Module to test ---
from tunnex_2.irc_computations.orca.orca_file_parsing import ( # type: ignore
    _orca_mass_extraction,
    orca_extract_geom_from_opt_file,
    orca_extract_geom_from_irc_point,
    _orca_struct_extraction_irc,
    reading_orca_irc_tunnex,
    _orca_collect_freq_and_modes_tunnex,
    reading_orca_struct_file,
    orca_mode_coordinate_reader)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.irc_computations.orca.orca_file_parsing"
ORCA_PATTERNS = "tunnex_2.irc_computations.orca.orca_file_parsing.orca_patterns"


def _make_file(tmp_path, name, content=""):
    """ Should make a dummy file for tests """
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


# ===========================================================================
# --- _orca_mass_extraction - normal behaviour ---
# ===========================================================================
class TestOrcaMassExtractionNormal:

    def _run(self, tmp_path, masses, filename="mol.out"):
        """ Should create the input file for the function and run it """
        f = tmp_path / filename    
        mass_lines = "\n".join(f"{i} C 6.0000 0 {mass} 0.000000 0.990460 0.000000"
        for i, mass in enumerate(masses))
        text = ("CARTESIAN COORDINATES (A.U.)\n"
                "----------------------------\n"
                "NO LB      ZA    FRAG     MASS         X           Y           Z\n"
                f"{mass_lines}\n")
        f.write_text(text)
        return _orca_mass_extraction(f)

    def test_string_suffix_lower_case_check(self, tmp_path):
        """ Tests accepting an uppercase .OUT file suffix """
        result = self._run(tmp_path, [12.0, 15.999, 1.008], filename="mol.OUT")
        assert result[0] == 12.0, f"expected {12.0}, got {result[0]}" 
    
    def test_returns_numpy_array(self, tmp_path):
        """ Tests returning the extracted atomic masses as a NumPy array """
        result = self._run(tmp_path, [12.0, 1.0])
        assert isinstance(result, np.ndarray), f"expected {np.ndarray}, got {type(result)}" 

    def test_correct_values_extracted(self, tmp_path):
        """ Tests correctly extracting all atomic masses from the ORCA output """
        expected = [12.0, 1.008, 15.999]
        result = self._run(tmp_path, expected)
        assert np.allclose(result, expected), f"expected {expected}, got {result}"

    def test_reads_from_cache_file_if_exists(self, tmp_path):
        """ Tests reading atomic masses from an existing cache file """
        cache = tmp_path / "orca_atom_masses.out"
        cache_text = (
            "CARTESIAN COORDINATES (A.U.)\n"
            "----------------------------\n"
            "NO LB      ZA    FRAG     MASS         X           Y           Z\n"
            "0 C     6.0000    0    12.011    0.000000    0.990460    0.000000\n"
            "1 O     8.0000    0    15.999   -2.310787    0.121564    0.000000\n")
        cache.write_text(cache_text)
        mol = tmp_path / "mol.out"
        mol.write_text("irrelevant")
        result = _orca_mass_extraction(mol)
        assert np.allclose(result, [12.011, 15.999]), f"expected {[12.011, 15.999]}, got {result}"

    def test_writes_cache_when_missing(self, tmp_path):
        """ Tests creating the cache file when it does not exist """
        mol = tmp_path / "mol.out"
        mol.write_text(
            "CARTESIAN COORDINATES (A.U.)\n"
            "----------------------------\n"
            "NO LB      ZA    FRAG     MASS         X           Y           Z\n"
            "0 C     6.0000    0    12.000    0.000000    0.990460    0.000000\n")
        _orca_mass_extraction(mol)
        cache = tmp_path / "orca_atom_masses.out"
        assert cache.is_file(), f"expected {True}, got {cache.is_file()}"

    def test_does_not_overwrite_existing_cache(self, tmp_path):
        """ Tests that an existing cache file is not overwritten """
        cache = tmp_path / "orca_atom_masses.out"
        cache_content = (
            "CARTESIAN COORDINATES (A.U.)\n"
            "----------------------------\n"
            "NO LB      ZA    FRAG     MASS         X           Y           Z\n"
            "0 C     6.0000    0     12.000    0.000000    0.990460    0.000000\n")
        cache.write_text(cache_content)
        mol = tmp_path / "mol.out"
        mol.write_text("irrelevant")
        result = _orca_mass_extraction(mol)
        read_data = cache.read_text()
        assert np.allclose(result, [12.000]), f"expected {[12.000]}, got {result}; test #1"
        assert read_data == cache_content, f"expected {cache_content}, got {read_data}; test #2"


# ===========================================================================
# --- _orca_mass_extraction - input validation ---
# ===========================================================================
class TestOrcaMassExtractionValidation:

    def test_non_out_suffix_raises(self, tmp_path):
        """ Tests raising an error when the input file does not have an .out suffix. Expecting a ValueError """
        f = tmp_path / "mol.log"
        f.write_text("")
        with pytest.raises(ValueError, match=".out file"):
            _orca_mass_extraction(f)

    def test_no_mass_block_raises(self, tmp_path):
        """ Tests raising an error when the atomic mass block cannot be found. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("no masses here")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_MASS_BLOCK.search.return_value = None
            with pytest.raises(ValueError, match="Cannot find the atomic mass block"):
                _orca_mass_extraction(f)

    def test_empty_mass_values_raises(self, tmp_path):
        """ Tests raising an error when the mass block contains no atomic masses. Expecting a ValueError """
        f = tmp_path / "mol.out"
        f.write_text("placeholder")
        match_mock = MagicMock()
        match_mock.group.return_value = ""
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_MASS_BLOCK.search.return_value = match_mock
            with pytest.raises(ValueError, match="no atomic masses"):
                _orca_mass_extraction(f)

    def test_zero_mass_raises(self, tmp_path):
        """ Tests raising an error when an atomic mass is zero. Expecting a ValueError """
        f = tmp_path / "mol.out"
        text = ("CARTESIAN COORDINATES (A.U.)\n"
                "----------------------------\n"
                "NO LB      ZA    FRAG     MASS         X           Y           Z\n"
                "0 C     6.0000    0     0.000    0.000000    0.990460    0.000000\n"
                "1 O     8.0000    0    15.999   -2.310787    0.121564    0.000000")
        f.write_text(text)
        with pytest.raises(ValueError, match="positive"):
            _orca_mass_extraction(f)

    def test_negative_mass_raises(self, tmp_path):
        """ Tests raising an error when an atomic mass is negative. Expecting a ValueError """
        f = tmp_path / "mol.out"
        text = ("CARTESIAN COORDINATES (A.U.)\n"
                "----------------------------\n"
                "NO LB      ZA    FRAG     MASS         X           Y           Z\n"
                "0 C     6.0000    0    -1.000    0.000000    0.990460    0.000000\n"
                "1 O     8.0000    0    15.999   -2.310787    0.121564    0.000000")
        f.write_text(text)
        with pytest.raises(ValueError, match="positive"):
            _orca_mass_extraction(f)


# ===============================================================================
# --- orca_extract_geom_from_opt_file - normal behaviour and input validation ---
# ===============================================================================
class TestOrcaExtractGeomFromOptFile:

    def test_returns_string_by_default(self, tmp_path):
        """ Tests returning the geometry as a string by default """
        f = _make_file(tmp_path, "mol.out", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK.findall.return_value = ["block"]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = [("C", "0.0", "0.0", "0.0")]
            result = orca_extract_geom_from_opt_file(f, as_array=False)
        assert isinstance(result, str), f"expected {str}, got {type(result)}; test #1"
        assert "C" in result, f"expected {"C"} in {result}; test #2"

    def test_returns_ndarray_when_requested(self, tmp_path):
        """ Tests returning the geometry as a NumPy array when requested """
        f = _make_file(tmp_path, "mol.out", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK.findall.return_value = ["block"]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = [
                ("C", "1.0", "2.0", "3.0"),
                ("H", "4.0", "5.0", "6.0")]
            result = orca_extract_geom_from_opt_file(f, as_array=True)
        assert isinstance(result, np.ndarray), f"expected {np.ndarray}, got {type(result)}; test #1"
        assert result.shape == (2, 3), f"expected {(2, 3)}, got {result.shape}; test #2"

    def test_uses_last_geometry_block(self, tmp_path):
        """ Tests using the last available geometry block """
        f = _make_file(tmp_path, "mol.out", "blocks")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK.findall.return_value = [
            "first_block", "last_block"]
            call_n = [0]
            def findall_side(block):
                call_n[0] += 1
                if "last" in block:
                    return [("C", "9.0", "9.0", "9.0")]
                return [("C", "0.0", "0.0", "0.0")]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.side_effect = findall_side
            result = orca_extract_geom_from_opt_file(f, as_array=False)
        assert "9.0" in result, f"expected {"9.0"} in {result}"

    def test_no_geometry_block_raises(self, tmp_path):
        """ Tests raising an error when no geometry block is found. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "no geometry")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK.findall.return_value = []
            with pytest.raises(ValueError, match="The geometry of the"):
                orca_extract_geom_from_opt_file(f)

    def test_no_atom_coordinates_in_block_raises(self, tmp_path):
        """
        Tests raising an error when no atom coordinates
        are found in the geometry block. Expecting a ValueError
        """
        f = _make_file(tmp_path, "mol.out", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK.findall.return_value = ["block"]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = []
            with pytest.raises(ValueError, match="No atom coordinates"):
                orca_extract_geom_from_opt_file(f)

    def test_invalid_coords_raises(self, tmp_path):
        """ Tests raising an error when atom coordinates are not numeric. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_OPTIMIZED_SPECIES_GEOM_BLOCK.findall.return_value = ["block"]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = [("C", "not", "a", "number")]
            with pytest.raises(ValueError, match="Invalid coordinates"):
                orca_extract_geom_from_opt_file(f, as_array=True)


# ================================================================================
# --- orca_extract_geom_from_irc_point - normal behaviour and input validation ---
# ================================================================================
class TestOrcaExtractGeomFromIRCPoint:

    def test_returns_string(self, tmp_path):
        """ Tests returning the geometry as a string """
        f = _make_file(tmp_path, "irc.xyz", "coords")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = [
                ("C", "0.0", "0.0", "0.0"),
                ("H", "1.0", "0.0", "0.0")]
            result = orca_extract_geom_from_irc_point(f)
        assert isinstance(result, str), f"expected {str}, got {type(result)}; test #1"
        assert "C 0.0 0.0 0.0" in result, f"expected {"C 0.0 0.0 0.0"} in {result}; test #2"
        assert "H 1.0 0.0 0.0" in result, f"expected {"H 1.0 0.0 0.0"} in {result}; test #3"

    def test_no_geometry_raises(self, tmp_path):
        """ Tests raising an error when no atom coordinates are found. Expecting a ValueError """
        f = _make_file(tmp_path, "irc.xyz", "no coords")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = []
            with pytest.raises(ValueError, match="was not found"):
                orca_extract_geom_from_irc_point(f)


# ===========================================================================
# --- _orca_struct_extraction_irc - normal behaviour ---
# ===========================================================================
class TestOrcaStructExtractionIRCNormal:

    def _run(self, tmp_path, blocks):
        """ Should creates a test IRC file, mocks geometry extraction, and run the function """
        f = _make_file(tmp_path, "irc.xyz", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_IRC_COORDINATES_BLOCK.findall.return_value = blocks
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.side_effect = [
                [("C", "0.0", "0.0", "0.0"), ("H", "1.0", "0.0", "0.0")]
                for _ in blocks]
            return _orca_struct_extraction_irc(f)

    def test_returns_three_items(self, tmp_path):
        """ Tests that the function returns energies, geometries, and geometry texts """
        result = self._run(tmp_path, [("-153.0", "g")])
        assert len(result) == 3, f"expected {3}, got {len(result)}"

    def test_energies_are_ndarray(self, tmp_path):
        """ Tests that extracted energies are returned as a NumPy array """
        energies, _, _ = self._run(tmp_path, [("-153.0", "g"), ("-152.9", "g")])
        assert isinstance(energies, np.ndarray), f"expected {np.ndarray}, got {type(energies)}"

    def test_geometries_array_shape(self, tmp_path):
        """Tests that geometries are returned with shape (n_points, n_atoms, 3) """
        _, geoms, _ = self._run(tmp_path, [("-153.0", "g"), ("-152.9", "g")])
        # 2 IRC points, 2 atoms, 3 coords
        assert geoms.shape == (2, 2, 3), f"expected {(2, 2, 3)}, got {geoms.shape}"

    def test_geometry_texts_is_list_of_strings(self, tmp_path):
        """ Tests that geometry texts are returned as a list of strings """
        _, _, texts = self._run(tmp_path, [("-153.0", "g")])
        assert isinstance(texts, list), f"expected {list}, got {type(texts)}; test #1"
        assert isinstance(texts[0], str), f"expected {str}, got {type(texts[0])}; test #2"

    def test_energies_converted_to_float(self, tmp_path):
        """ Tests that extracted energy values are converted to floating-point numbers """
        energies, _, _ = self._run(tmp_path, [("-153.123456", "g")])
        assert isinstance(energies[0], float), f"expected {float}, "
        f"got {energies[0]}; test #2"
        assert np.isclose(energies[0], -153.123456), f"expected {-153.123456}, "
        f"got {energies[0]}; test #2"

    def test_multiple_blocks_all_parsed(self, tmp_path):
        """ Tests that all IRC geometry blocks are parsed and returned """
        n = 5
        blocks = [(f"-153.{i}", "g") for i in range(n)]
        energies, geoms, texts = self._run(tmp_path, blocks)
        assert len(energies) == n, f"expected {n}, got {len(energies)}; test #1"
        assert geoms.shape[0] == n, f"expected {n}, got {geoms.shape[0]}; test #2"
        assert len(texts) == n, f"expected {n}, got {len(texts)}; test #3"


# ================================================================================
# --- _orca_struct_extraction_irc - input validation ---
# ================================================================================
class TestOrcaStructExtractionIRCValidation:

    def test_no_geometry_blocks_raises(self, tmp_path):
        """
        Tests raising an error when no IRC geometry blocks are found.
        Expecting a ValueError
        """
        f = _make_file(tmp_path, "irc.xyz", "no blocks")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_IRC_COORDINATES_BLOCK.findall.return_value = []
            with pytest.raises(ValueError, match="No geometries were found"):
                _orca_struct_extraction_irc(f)

    def test_empty_coordinates_for_block_raises(self, tmp_path):
        """
        Tests raising an error when an IRC geometry block
        contains no atomic coordinates. Expecting a ValueError
        """
        f = _make_file(tmp_path, "irc.xyz", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_IRC_COORDINATES_BLOCK.findall.return_value = [("-153.0", "geom")]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = []
            with pytest.raises(ValueError, match="No atomic coordinates"):
                _orca_struct_extraction_irc(f)

    def test_not_a_number_coordinates_raises(self, tmp_path):
        """
        Tests raising an error when an IRC geometry contains
        non-numeric coordinates. Expecting a ValueError
        """
        f = _make_file(tmp_path, "irc.xyz", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_IRC_COORDINATES_BLOCK.findall.return_value = [("-153.0", "geom")]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = [("C", "not", "a", "number")]
            with pytest.raises(ValueError, match="Invalid Cartesian coordinates type"):
                _orca_struct_extraction_irc(f)

    def test_invalid_number_of_coordinates_raises(self, tmp_path):
        """
        Tests raising an error when an atom does not have
        exactly three Cartesian coordinates. Expecting a ValueError
        """
        f = _make_file(tmp_path, "irc.xyz", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_IRC_COORDINATES_BLOCK.findall.return_value = [("-153.0", "geom")]
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.return_value = [("C", "0.0", "0.0")]
            with pytest.raises(ValueError, match="Invalid Cartesian coordinates length"):
                _orca_struct_extraction_irc(f)
    
    def test_number_of_coordinates_mismatch_raises(self, tmp_path):
        """
        Tests raising an error when IRC geometries contain
        different numbers of atoms. Expecting a ValueError
        """
        f = _make_file(tmp_path, "irc.xyz", "block")
        with patch(ORCA_PATTERNS) as op:
            op.PATTERN_IRC_COORDINATES_BLOCK.findall.return_value = [("-153.0", "geom1"),
                                                                     ("-153.2", "geom2")]
            geom = (("C", "0.0", "0.0", "0.0"), ("H", "1.0", "1.0", "1.0"))
            op.PATTERN_GEOMETRY_LINE_ORCA.findall.side_effect = [list(geom),[geom[0]]]
            with pytest.raises(ValueError, match="Inconsistent number of atoms between"):
                _orca_struct_extraction_irc(f)


# ================================================================================
# --- reading_orca_irc_tunnex - normal behaviour and input validation ---
# ================================================================================
class TestReadingOrcaIRCTunnex:

    def _make_irc_data(self, n=3):
        """ Should make the IRC data for the function """
        energies = np.array([-153.0 - 0.01 * i for i in range(n)])
        geoms = np.zeros((n, 2, 3))
        texts = [f"C 0 0 {i}\nH 1 0 {i}" for i in range(n)]
        return energies, geoms, texts

    def test_proj_freq_false_returns_none_zpves(self, tmp_path):
        """ Tests that ZPVE data is None when projected frequencies are disabled """
        irc = _make_file(tmp_path, "irc.xyz", "")
        ts = _make_file(tmp_path, "ts.out", "")
        tsg = _make_file(tmp_path, "ts.gjf", "")
        cmd = _make_file(tmp_path, "cmd.txt", "")
        energies, geoms, texts = self._make_irc_data()
        with patch(FUNC_PATH+".file_check", side_effect=[irc, ts, tsg]), \
             patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])) as ome, \
             patch(FUNC_PATH+"._orca_struct_extraction_irc", return_value=(energies, geoms, texts)) as ose, \
             patch(FUNC_PATH+".irc_steps_comp", return_value=np.array([0.1, 0.1])) as isc:
             result = reading_orca_irc_tunnex(irc, ts, tsg, cmd, proj_freq=False)
        assert isinstance(result, IRCData), f"expected {IRCData}, got {type(result)}; test #1"
        assert result.zpve_energies_forward is None, f"expected {None}, "
        f"got {result.zpve_energies_forward}; test #2" 
        assert result.zpve_energies_reverse is None, f"expected {None}, "
        f"got {result.zpve_energies_reverse}; test #3"
        ome.assert_called_once() # it should be called once
        ose.assert_called_once() # it should be called once
        isc.assert_called_once() # it should be called once

    def test_proj_freq_true_calls_orca_collect(self, tmp_path):
        """ Tests that projected ZPVE collection is called when projected frequencies are enabled """
        irc = _make_file(tmp_path, "irc.xyz", "")
        ts = _make_file(tmp_path, "ts.out", "")
        tsg = _make_file(tmp_path, "ts.gjf", "")
        cmd = _make_file(tmp_path, "cmd.txt", "")
        energies, geoms, texts = self._make_irc_data(3)
        zpves = np.array([0.03, 0.04, 0.035])
        with patch(FUNC_PATH+".file_check", side_effect=[irc, ts, tsg]) as fc, \
             patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])), \
             patch(FUNC_PATH+"._orca_struct_extraction_irc", return_value=(energies, geoms, texts)), \
             patch(FUNC_PATH+".irc_steps_comp", return_value=np.array([0.1, 0.1])), \
             patch(FUNC_PATH+".orca_collect_proj_zpves", return_value=zpves) as mock_collect:
             result = reading_orca_irc_tunnex(irc, ts, tsg, cmd, proj_freq=True)
        counter = fc.call_count
        assert isinstance(result, IRCData), f"expected {IRCData}, got {type(result)}; test #1"
        assert result.zpve_energies_forward is not None, f"expected not {None}, "
        f"got {result.zpve_energies_forward}; test #2" 
        assert result.zpve_energies_reverse is not None, f"expected not {None}, "
        f"got {result.zpve_energies_reverse}; test #3"
        assert counter == 3, f"expected {3}, got {counter}; test #4"
        mock_collect.assert_called_once() # it should be called once
    
    def test_ts_at_irc_zero_after_shift(self, tmp_path):
        """ Tests that the transition state is shifted to IRC = 0.0 """
        irc = _make_file(tmp_path, "irc.xyz", "")
        ts = _make_file(tmp_path, "ts.out",  "")
        tsg = _make_file(tmp_path, "ts.gjf",  "")
        cmd = _make_file(tmp_path, "cmd.txt",  "")
        energies = np.array([-153.10, -153.05, -153.15])
        geoms = np.zeros((3, 2, 3))
        texts = ["g"] * 3
        with patch(FUNC_PATH+".file_check", side_effect=[irc, ts, tsg]), \
             patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])), \
             patch(FUNC_PATH+"._orca_struct_extraction_irc", return_value=(energies, geoms, texts)), \
             patch(FUNC_PATH+".irc_steps_comp", return_value=np.array([0.5, 0.5])):
             result = reading_orca_irc_tunnex(irc, ts, tsg, cmd, proj_freq=False)
        max_irc = max(result.electronic_energies, key=lambda x: x[1])[0]
        assert np.isclose(max_irc, 0.0, atol=1e-10), f"expected {0.0}, got {max_irc}"

    def test_zpve_energies_are_split_by_irc_sign(self, tmp_path):
        """ Tests that ZPVE data is split into forward and reverse IRC branches by IRC sign """
        irc = _make_file(tmp_path, "irc.xyz", "")
        ts = _make_file(tmp_path, "ts.out", "")
        tsg = _make_file(tmp_path, "ts.gjf", "")
        cmd = _make_file(tmp_path, "cmd.txt", "")
        energies, geoms, texts = self._make_irc_data(3)
        zpves = np.array([0.03, 0.04, 0.035])
        with patch(FUNC_PATH+".file_check", side_effect=[irc, ts, tsg]), \
            patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])), \
            patch(FUNC_PATH+"._orca_struct_extraction_irc", return_value=(energies, geoms, texts)), \
            patch(FUNC_PATH+".irc_steps_comp", return_value=np.array([0.1, 0.1])), \
            patch(FUNC_PATH+".orca_collect_proj_zpves", return_value=zpves):
            result = reading_orca_irc_tunnex(irc, ts, tsg, cmd, proj_freq=True)
        forward_ircs = [irc for irc, _ in result.zpve_energies_forward]
        reverse_ircs = [irc for irc, _ in result.zpve_energies_reverse]
        assert all(irc > 0 for irc in forward_ircs), \
            f"expected only positive IRC values, got {forward_ircs}; test #1"
        assert all(irc < 0 for irc in reverse_ircs), \
            f"expected only negative IRC values, got {reverse_ircs}; test #2"


# ===================================================================================
# --- _orca_collect_freq_and_modes_tunnex - normal behaviour and input validation ---
# ===================================================================================
class TestOrcaCollectFreqAndModesTunnex:

    def test_returns_four_values(self, tmp_path):
        """ Tests that the function returns ZPVE, frequencies, modes, and atom masses """
        opt = _make_file(tmp_path, "mol.out", "")
        hess = _make_file(tmp_path, "mol.hess", "")
        atom_masses = np.array([12.0, 1.0])
        structure = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]])
        hessian = np.eye(6)
        zpve = 0.03
        vibs = np.array([100.0, 200.0])
        modes = np.array([[1.0, 0.0, 0.0]])
        with patch(FUNC_PATH+"._orca_mass_extraction", return_value=atom_masses), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value=structure), \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=hessian), \
             patch(FUNC_PATH+".freq_and_modes_extraction_tunnex", return_value=(zpve, vibs, modes)):
             result = _orca_collect_freq_and_modes_tunnex(opt, hess)
        assert len(result) == 4, f"expected {4}, got {len(result)}"

    def test_returns_expected_values(self, tmp_path):
        """ Tests that the function returns the values produced by the extraction functions """
        opt = _make_file(tmp_path, "mol.out", "")
        hess = _make_file(tmp_path, "mol.hess", "")
        atom_masses = np.array([12.0, 1.0])
        structure = np.zeros((2, 3))
        hessian = np.eye(6)
        zpve = 0.03
        vibs = np.array([100.0, 200.0])
        modes = np.array([[1.0, 0.0, 0.0]])
        with patch(FUNC_PATH+"._orca_mass_extraction", return_value=atom_masses), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value=structure), \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=hessian), \
             patch(FUNC_PATH+".freq_and_modes_extraction_tunnex",
             return_value=(zpve, vibs, modes)):
             result = _orca_collect_freq_and_modes_tunnex(opt, hess)
        result_zpve, result_vibs, result_modes, result_masses = result
        assert result_zpve == zpve, f"expected {zpve}, got {result_zpve}; test #1"
        assert np.array_equal(result_vibs, vibs), \
            f"expected {vibs}, got {result_vibs}; test #2"
        assert np.array_equal(result_modes, modes), \
            f"expected {modes}, got {result_modes}; test #3"
        assert np.array_equal(result_masses, atom_masses), \
            f"expected {atom_masses}, got {result_masses}; test #4"

    def test_mass_extraction_called_with_opt_file(self, tmp_path):
        """ Tests that atom masses are extracted from the optimization output file """
        opt = _make_file(tmp_path, "mol.out", "")
        hess = _make_file(tmp_path, "mol.hess", "")
        with patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])) as mock_mass, \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value=np.zeros((2, 3))), \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=np.eye(6)), \
             patch(FUNC_PATH+".freq_and_modes_extraction_tunnex",
             return_value=(0.03, np.array([100.0]), np.array([1.0]))):
             _orca_collect_freq_and_modes_tunnex(opt, hess)
        mock_mass.assert_called_once_with(opt) # it should be called once

    def test_geometry_extraction_called_with_opt_file_species_and_array_flag(self, tmp_path):
        """ Tests that the optimized geometry is extracted with the requested species and array output """
        opt = _make_file(tmp_path, "mol.out", "")
        hess = _make_file(tmp_path, "mol.hess", "")
        with patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file",
             return_value=np.zeros((2, 3))) as mock_geom, \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=np.eye(6)), \
             patch(FUNC_PATH+".freq_and_modes_extraction_tunnex",
             return_value=(0.03, np.array([100.0]), np.array([1.0]))):
             _orca_collect_freq_and_modes_tunnex(opt, hess, species="transition_state")
        mock_geom.assert_called_once_with(opt, "transition_state", True)
        # it should be called once

    def test_hessian_reader_called_with_hess_file(self, tmp_path):
        """ Tests that the Hessian matrix is read from the specified Hessian file """
        opt = _make_file(tmp_path, "mol.out", "")
        hess = _make_file(tmp_path, "mol.hess", "")
        with patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value=np.zeros((2, 3))), \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=np.eye(6)) as mock_hess, \
             patch(FUNC_PATH+".freq_and_modes_extraction_tunnex",
             return_value=(0.03, np.array([100.0]), np.array([1.0]))):
             _orca_collect_freq_and_modes_tunnex(opt, hess)
        mock_hess.assert_called_once_with(hess) # it should be called once

    def test_freq_and_modes_extraction_called_with_correct_arguments(self, tmp_path):
        """ Tests that frequency and mode extraction receives the structure, masses, and Hessian """
        opt = _make_file(tmp_path, "mol.out", "")
        hess = _make_file(tmp_path, "mol.hess", "")
        atom_masses = np.array([12.0, 1.0])
        structure = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]])
        hessian = np.eye(6)
        with patch(FUNC_PATH+"._orca_mass_extraction", return_value=atom_masses), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file", return_value=structure), \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=hessian), \
             patch(FUNC_PATH+".freq_and_modes_extraction_tunnex",
             return_value=(0.03, np.array([100.0]), np.array([1.0]))) as mock_freq:
             _orca_collect_freq_and_modes_tunnex(opt, hess)
        mock_freq.assert_called_once_with(structure, atom_masses, hessian)
        # it should be called once

    def test_default_species_is_species(self, tmp_path):
        """ Tests that the default species argument is passed to the geometry extractor """
        opt = _make_file(tmp_path, "mol.out", "")
        hess = _make_file(tmp_path, "mol.hess", "")
        with patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])), \
             patch(FUNC_PATH+".orca_extract_geom_from_opt_file",
             return_value=np.zeros((2, 3))) as mock_geom, \
             patch(FUNC_PATH+".orca_hessian_reader", return_value=np.eye(6)), \
             patch(FUNC_PATH+".freq_and_modes_extraction_tunnex",
             return_value=(0.03, np.array([100.0]), np.array([1.0]))):
             _orca_collect_freq_and_modes_tunnex(opt, hess)
        mock_geom.assert_called_once_with(opt, "species", True)
        # it should be called once


# ===========================================================================
# --- reading_orca_struct_file - normal behaviour and input validation ---
# ===========================================================================
class TestReadingOrcaStructFile:

    def test_ts_position_is_zero(self, tmp_path):
        """ Tests that the transition state position is set to zero """
        f = _make_file(tmp_path, "mol.out", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT.findall.return_value = [("-153.0", "0.05")]
             el, zpve = reading_orca_struct_file(f, mode="ts")
        assert isinstance(el, tuple), f"expected {tuple}, got {type(el)}; test #1"
        assert isinstance(zpve, tuple), f"expected {tuple}, got {type(zpve)}; test #2"
        assert el[0] == 0.0, f"expected {0.0}, got {el[0]}; test #3"

    def test_react_position_is_minus_inf(self, tmp_path):
        """ Tests that the reactant position is set to negative infinity """
        f = _make_file(tmp_path, "mol.out", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT.findall.return_value = [("-153.0", "0.05")]
             el, _ = reading_orca_struct_file(f, mode="react")
        result = np.isinf(el[0]) and el[0] < 0
        assert result, f"expected {True}, got {result}"

    def test_prod_position_is_plus_inf(self, tmp_path):
        """ Tests that the product position is set to positive infinity """
        f = _make_file(tmp_path, "mol.out", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(ORCA_PATTERNS) as op:
            op.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT.findall.return_value = [("-153.0", "0.05")]
            el, _ = reading_orca_struct_file(f, mode="prod")
        result = np.isinf(el[0]) and el[0] > 0
        assert result, f"expected {True}, got {result}"

    def test_uses_last_match(self, tmp_path):
        """ Tests that the last energy and ZPVE match is used """
        f = _make_file(tmp_path, "mol.out", "")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT.findall.return_value = [
             ("-100.0", "0.01"), ("-153.0", "0.05")]
             el, zpve = reading_orca_struct_file(f, mode="ts")
        assert np.isclose(el[1], -153.0), f"expected {-153.0}, got {el[1]}; test #1"
        assert np.isclose(zpve[1], 0.05), f"expected {0.05}, got {zpve[1]}; test #2"

    def test_hess_parsing_uses_hessian_zpve(self, tmp_path):
        """ Tests using the ZPVE obtained from Hessian parsing """
        f = _make_file(tmp_path, "mol.out", "")
        hess = tmp_path / "mol.hess"
        hess.write_text("")
        with patch(FUNC_PATH+".file_check", side_effect=[f, hess]), \
             patch(ORCA_PATTERNS) as op, \
             patch(FUNC_PATH+"._orca_collect_freq_and_modes_tunnex",
             return_value=(0.099, None, None, None)) as hess_mock:
             op.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT.findall.return_value = [("-153.0", "0.05")]
             _, zpve = reading_orca_struct_file(f, mode="ts", hess_parsing=True)
        assert np.isclose(zpve[1], 0.099), f"expected {0.099}, got {zpve[1]}"
        hess_mock.assert_called_once() # it should be called once

    def test_invalid_mode_raises(self, tmp_path):
        """ Tests raising an error for an invalid structure mode. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "")
        with patch(FUNC_PATH+".file_check", return_value=f):
            with pytest.raises(ValueError, match="incorrect"):
                reading_orca_struct_file(f, mode="saddle")

    def test_no_el_energy_raises(self, tmp_path):
        """ Tests raising an error when the electronic energy is not found. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "no energy here")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(ORCA_PATTERNS) as op:
            op.PATTERN_EL_ENERGY_ZPVE_SINGLE_POINT.findall.return_value = []
            with pytest.raises(ValueError, match="was not found"):
                reading_orca_struct_file(f, mode="ts")


# ===========================================================================
# --- orca_mode_coordinate_reader - normal behaviour and input validation ---
# ===========================================================================
class TestOrcaModeCoordinateReader:

    def test_returns_freq_and_modes_and_masses(self, tmp_path):
        """ Tests returning vibrational frequencies, normal modes, and atom masses """
        f = _make_file(tmp_path, "mol.out", "block")
        mode_text = "    \n0  1\n0  0.1  0.2\n1  0.3  0.4\n"
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0, 1.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = ["1000.0", "2000.0"]
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = [mode_text]
             freq_and_modes, masses = orca_mode_coordinate_reader(f)
        expected = [1000.0, 0.1, 0.3]
        assert isinstance(masses, tuple), f"expected {tuple}, got {type(masses)}; test #1"
        assert masses == (12.0, 1.0), f"expected {(12.0, 1.0)}, got {masses}; test #2"
        assert isinstance(freq_and_modes, list), f"expected {list}, "
        f"got {type(freq_and_modes)}; test #3"
        assert isinstance(freq_and_modes[0], list), f"expected {list}, "
        f"got {type(freq_and_modes[0])}; test #4"
        for i, val in enumerate(freq_and_modes[0]):
            assert np.isclose(val, expected[i]), f"expected {expected[i]}, got {val}; test #{i+4}"

    def test_freq_tol_filters_small_frequencies(self, tmp_path):
        """Tests filtering out frequencies below the specified frequency tolerance """
        f = _make_file(tmp_path, "mol.out", "block")
        mode_text = "0  1\n0  0.1  0.2\n1  0.3  0.4\n"
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = ["0.005", "1000.0"]
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = [mode_text]
             freq_and_modes, _ = orca_mode_coordinate_reader(f, freq_tol=1e-2)
        freqs = [fm[0] for fm in freq_and_modes]
        assert 0.005 not in freqs, f"expected {0.005} not in {freqs}"

    def test_uses_last_vibr_block(self, tmp_path):
        """ Tests using the last vibrational block found in the output file """
        f = _make_file(tmp_path, "mol.out", "block")
        mode_text = "0\n0  0.1\n"
        with patch(FUNC_PATH+".file_check", return_value=f) as fc, \
             patch(FUNC_PATH+"._orca_mass_extraction", return_value=np.array([12.0])) as ome, \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["first_block", "last_block"]
             freq_calls = [0]
             def freq_side(block):
                freq_calls[0] += 1
                return ["999.0"] if "last" in block else ["1.0"]
             op.PATTERN_VIBR_FREQUENCIES.findall.side_effect = freq_side
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = [mode_text]
             freq_and_modes, _ = orca_mode_coordinate_reader(f)
        result = freq_and_modes[0][0]
        assert np.isclose(result, 999.0), f"expected {999.0}, got {result}"
        fc.assert_called_once() # it should be called once
        ome.assert_called_once() # it should be called once

    def test_hess_parsing_returns_freq_and_modes(self, tmp_path):
        """
        Tests returning frequencies, normal modes,
        and atom masses when parsing the Hessian file
        """
        f = _make_file(tmp_path, "mol.out", "")
        hess = tmp_path / "mol.hess"
        frequencies = np.array([100.0, 200.0])
        modes = np.array([[0.1, 0.2], [0.3, 0.4]])
        masses = np.array([12.0, 1.0])
        expected = [[100.0, 0.1, 0.2], [200.0, 0.3, 0.4]]
        with patch(FUNC_PATH + ".file_check", side_effect=[f, hess]), \
             patch(FUNC_PATH + "._orca_collect_freq_and_modes_tunnex",
             return_value=(0.05, frequencies, modes, masses)) as collect, \
             patch(FUNC_PATH + ".vibrations_format_transform",
             return_value=expected) as transform:
             result, result_masses = orca_mode_coordinate_reader(f, hess_parsing=True)
        assert result == expected, f"expected {expected}, got {result}; test #1"
        assert result_masses == (12.0, 1.0), f"expected {(12.0, 1.0)}, got {result_masses}; test #2"
        collect.assert_called_once_with(f, hess)
        # it should be called once
        transform.assert_called_once_with(hess, frequencies, modes, masses, 1e-2)
        # it should be called once
    
    def test_hess_parsing_skips_standard_parsing(self, tmp_path):
        """
        Test skipping standard frequency
        and mass parsing when Hessian parsing is enabled
        """
        f = _make_file(tmp_path, "mol.out", "")
        hess = tmp_path / "mol.hess"
        frequencies = np.array([100.0])
        modes = np.array([[0.1, 0.2]])
        masses = np.array([12.0])
        expected = [[100.0, 0.1, 0.2]]
        with patch(FUNC_PATH + ".file_check", side_effect=[f, hess]), \
             patch(FUNC_PATH + "._orca_collect_freq_and_modes_tunnex",
             return_value=(0.05, frequencies, modes, masses)), \
             patch(FUNC_PATH + ".vibrations_format_transform", return_value=expected), \
             patch(FUNC_PATH + "._orca_mass_extraction") as mass_extraction:
             result, _ = orca_mode_coordinate_reader(f, hess_parsing=True)
        assert result == expected, f"expected {expected}, got {result}"
        mass_extraction.assert_not_called() # it should not be called
    
    def test_no_vibr_blocks_raises(self, tmp_path):
        """ Raises an error when no vibrational blocks are found. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "no vibrational blocks")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction",
             return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = []
             with pytest.raises(ValueError, match="Frequencies were not found"):
                orca_mode_coordinate_reader(f)

    def test_no_frequencies_in_block_raises(self, tmp_path):
        """
        Raises an error when no vibrational frequencies
        are found in the block. Expecting a ValueError
        """
        f = _make_file(tmp_path, "mol.out", "block")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction",
             return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = []
             with pytest.raises(ValueError, match="Vibrational frequencies were not found"):
                orca_mode_coordinate_reader(f)

    def test_no_normal_modes_raises(self, tmp_path):
        """ Raises an error when the normal modes block is not found. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "block")
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction",
             return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = ["1000.0"]
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = []
             with pytest.raises(ValueError, match="Normal modes block was not found"):
                orca_mode_coordinate_reader(f)

    def test_not_a_number_in_normal_modes_raises(self, tmp_path):
        """
        Raises an error when a normal mode vector
        contains a non-numeric value. Expecting a ValueError
        """
        f = _make_file(tmp_path, "mol.out", "block")
        modes = ["0\n0  not_a_number\n"]
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction",
             return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = ["1000.0"]
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = modes
             with pytest.raises(ValueError, match="Invalid normal mode vector value"):
                orca_mode_coordinate_reader(f)

    def test_number_of_colums_and_values_raises(self, tmp_path):
        """
        Raises an error when the number of mode columns
        does not match the number of values. Expecting a ValueError
        """
        f = _make_file(tmp_path, "mol.out", "block")
        modes = ["0\n0  0.1  0.2\n"]
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction",
             return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = ["1000.0"]
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = modes
             with pytest.raises(ValueError, match="Numbers of columns and normal mode vectors"):
                orca_mode_coordinate_reader(f)

    def test_no_colums_headers_raises(self, tmp_path):
        """ Raises an error when no normal mode column headersare found. Expecting a ValueError """
        f = _make_file(tmp_path, "mol.out", "block")
        modes = ["0  0.1  0.2\n"]
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction",
             return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = ["1000.0"]
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = modes
             with pytest.raises(ValueError, match="Normal modes were not found"):
                orca_mode_coordinate_reader(f)

    def test_frequencies_and_modes_mismatch_raises(self, tmp_path):
        """
        Raises an error when the number of frequencies
        does not match the number of normal mode vectors. Expecting a ValueError
        """
        f = _make_file(tmp_path, "mol.out", "block")
        modes = ["0\n0  0.1\n"]
        with patch(FUNC_PATH+".file_check", return_value=f), \
             patch(FUNC_PATH+"._orca_mass_extraction",
             return_value=np.array([12.0])), \
             patch(ORCA_PATTERNS) as op:
             op.PATTERN_VIBR_BLOCK.findall.return_value = ["block"]
             op.PATTERN_VIBR_FREQUENCIES.findall.return_value = ["1000.0", "2000.0"]
             op.PATTERN_NORMAL_MODES_BLOCK.findall.return_value = modes
             with pytest.raises(ValueError, match="Number of frequencies and normal mode vectors"):
                orca_mode_coordinate_reader(f)