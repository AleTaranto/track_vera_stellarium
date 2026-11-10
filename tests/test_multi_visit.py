import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
import tvs
import astropy.units as u
from astropy.coordinates import Angle
from astropy.time import Time


def test_clear_and_active_layers(tmp_path):
    # Mock scripts dir
    with patch("tvs.get_stellarium_scripts_dir", return_value=tmp_path):
        assert tvs.get_active_layers() == set()
        tvs.record_active_layer("layer_1")
        tvs.record_active_layer("layer_2")
        assert tvs.get_active_layers() == {"layer_1", "layer_2"}

        tvs.unrecord_active_layer("layer_1")
        assert tvs.get_active_layers() == {"layer_2"}

        with patch("requests.post") as mock_post:
            tvs.clear_sky_image("http://localhost:8090", layer_id="layer_2")
            mock_post.assert_called_once()
            code = mock_post.call_args[1]["data"]["code"]
            assert 'core.removeSkyImage("layer_2");' in code
            assert tvs.get_active_layers() == set()

        tvs.record_active_layer("layer_a")
        tvs.record_active_layer("layer_b")
        with patch("requests.post") as mock_post:
            tvs.clear_sky_image("http://localhost:8090")
            mock_post.assert_called_once()
            code = mock_post.call_args[1]["data"]["code"]
            assert 'core.removeSkyImage("layer_a");' in code
            assert 'core.removeSkyImage("layer_b");' in code
            assert 'core.removeSkyImage("rubin_focal_plane");' in code
            assert tvs.get_active_layers() == set()


def test_visit_single_id():
    runner = CliRunner()
    with patch("tvs.ConsDB") as mock_consdb_cls, \
         patch("tvs.download_rubin_mosaic") as mock_download, \
         patch("tvs.prepare_rubin_sky_image") as mock_prep, \
         patch("tvs.plot_sky_image_to") as mock_plot, \
         patch("tvs.set_stellarium_time") as mock_time, \
         patch("tvs.set_view_to_camera") as mock_view, \
         patch("tvs.print_state") as mock_state, \
         patch("tvs.clear_sky_image") as mock_clear, \
         patch("tvs.QTable.read", side_effect=FileNotFoundError):

        mock_db = MagicMock()
        mock_consdb_cls.return_value = mock_db
        mock_db.query.return_value = [{
            "s_ra": 100.0,
            "s_dec": -30.0,
            "sky_rotation": 45.0,
            "exp_midpt_mjd": 60000.0,
        }]
        mock_download.return_value = "dummy.jpg"
        mock_prep.return_value = ("rubin_sky_2026041100613.png", "/path/to/png")

        result = runner.invoke(tvs.cli, ["visit", "2026041100613"])
        assert result.exit_code == 0
        mock_clear.assert_called_once()
        mock_download.assert_called_once_with("2026041100613", rsp_server=tvs.DEFAULT_RSP_SERVER, cookies_path=None)
        mock_prep.assert_called_once()
        mock_plot.assert_called_once()
        args, kwargs = mock_plot.call_args
        assert kwargs["layer_id"] == "rubin_focal_plane"
        assert kwargs["update_mosaic_camera"] is True


def test_visit_multiple_ids():
    runner = CliRunner()
    with patch("tvs.ConsDB") as mock_consdb_cls, \
         patch("tvs.download_rubin_mosaic") as mock_download, \
         patch("tvs.prepare_rubin_sky_image") as mock_prep, \
         patch("tvs.plot_sky_image_to") as mock_plot, \
         patch("tvs.set_stellarium_time") as mock_time, \
         patch("tvs.set_view_to_camera") as mock_view, \
         patch("tvs.print_state") as mock_state, \
         patch("tvs.clear_sky_image") as mock_clear, \
         patch("tvs.QTable.read", side_effect=FileNotFoundError):

        mock_db = MagicMock()
        mock_consdb_cls.return_value = mock_db
        mock_db.query.side_effect = [
            [{
                "s_ra": 100.0,
                "s_dec": -30.0,
                "sky_rotation": 45.0,
                "exp_midpt_mjd": 60000.0,
            }],
            [{
                "s_ra": 105.0,
                "s_dec": -28.0,
                "sky_rotation": 50.0,
                "exp_midpt_mjd": 60000.01,
            }]
        ]
        mock_download.side_effect = ["dummy1.jpg", "dummy2.jpg"]
        mock_prep.side_effect = [
            ("rubin_sky_2026041100613.png", "/path/1"),
            ("rubin_sky_2026041100614.png", "/path/2"),
        ]

        result = runner.invoke(tvs.cli, ["visit", "2026041100613, 2026041100614"])
        assert result.exit_code == 0
        assert mock_download.call_count == 2
        assert mock_prep.call_count == 2
        assert mock_plot.call_count == 2

        # First call has update_mosaic_camera=False and distinct layer_id
        first_call = mock_plot.call_args_list[0]
        assert first_call[1]["layer_id"] == "rubin_focal_plane_2026041100613"
        assert first_call[1]["update_mosaic_camera"] is False

        # Second call has update_mosaic_camera=True and distinct layer_id
        second_call = mock_plot.call_args_list[1]
        assert second_call[1]["layer_id"] == "rubin_focal_plane_2026041100614"
        assert second_call[1]["update_mosaic_camera"] is True


def test_visit_range():
    runner = CliRunner()
    with patch("tvs.ConsDB") as mock_consdb_cls, \
         patch("tvs.download_rubin_mosaic") as mock_download, \
         patch("tvs.prepare_rubin_sky_image") as mock_prep, \
         patch("tvs.plot_sky_image_to") as mock_plot, \
         patch("tvs.set_stellarium_time") as mock_time, \
         patch("tvs.set_view_to_camera") as mock_view, \
         patch("tvs.print_state") as mock_state, \
         patch("tvs.clear_sky_image") as mock_clear, \
         patch("tvs.QTable.read", side_effect=FileNotFoundError):

        mock_db = MagicMock()
        mock_consdb_cls.return_value = mock_db
        mock_db.query.side_effect = [
            [{
                "s_ra": 100.0,
                "s_dec": -30.0,
                "sky_rotation": 45.0,
                "exp_midpt_mjd": 60000.0,
            }],
            [{
                "s_ra": 101.0,
                "s_dec": -29.0,
                "sky_rotation": 46.0,
                "exp_midpt_mjd": 60000.01,
            }],
            [{
                "s_ra": 102.0,
                "s_dec": -28.0,
                "sky_rotation": 47.0,
                "exp_midpt_mjd": 60000.02,
            }]
        ]
        mock_download.side_effect = ["dummy1.jpg", "dummy2.jpg", "dummy3.jpg"]
        mock_prep.side_effect = [
            ("rubin_sky_2026041100613.png", "/path/1"),
            ("rubin_sky_2026041100614.png", "/path/2"),
            ("rubin_sky_2026041100615.png", "/path/3"),
        ]

        result = runner.invoke(tvs.cli, ["visit", "2026041100613-2026041100615"])
        assert result.exit_code == 0
        assert mock_download.call_count == 3
        assert mock_prep.call_count == 3
        assert mock_plot.call_count == 3


def test_visit_invalid_id():
    runner = CliRunner()
    result = runner.invoke(tvs.cli, ["visit", ""])
    assert result.exit_code != 0

    result2 = runner.invoke(tvs.cli, ["visit", "abc,def"])
    assert result2.exit_code != 0

    result3 = runner.invoke(tvs.cli, ["visit", "2026041100615-2026041100613"])
    assert result3.exit_code != 0
