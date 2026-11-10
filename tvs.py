import os
import re
import subprocess
import sys
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from time import sleep

import astropy.units as u
import click
import numpy as np
import requests
from astropy.coordinates import ICRS, AltAz, Angle, EarthLocation, SkyCoord
from astropy.table import QTable, Table
from astropy.time import Time
from PIL import Image, ImageOps

try:
    from colorama import init

    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

DEFAULT_STELLARIUM_URL = "http://localhost:8090"
CAMERA_CHOICES = ["LSSTCam", "ComCam"]
DEFAULT_RSP_SERVER = "https://usdf-rsp.slac.stanford.edu"
DEFAULT_RSP_TOKEN_PATH = "~/.lsst/rsp_token"
DEFAULT_EFD_CLIENT = "usdf_efd"
DEFAULT_RUBINTV_MOSAIC_PATH = (
    "/rubintv/event_image/summit-usdf/lsstcam/focal_plane_mosaic"
)
ALT_RUBINTV_MOSAIC_PATH = (
    "/rubintv/streamer-s3/usdf_Butler/lsstcam/focal_plane_mosaic"
)
DEFAULT_COOKIES_PATH = "cookies.txt"
DEFAULT_IMAGE_DIR = "images_rubin"
DEFAULT_SKY_LAYER_ID = "rubin_focal_plane"
ACTIVE_LAYERS_FILE = ".rubin_active_layers.txt"
DEFAULT_IMAGE_OPACITY = 0.55  # Opacity of the plotted Rubin sky image (0.0 to 1.0, e.g. 0.25 = 25%)
# Exact half-spans of LSSTCam focal plane in radians from Stellarium's LSSTCam.json
LSSTCAM_HALF_X_RAD = 0.03069328
LSSTCAM_HALF_Y_RAD = 0.03064734
# Using Stellarium values for Cerro Pachón
RUBIN_OBSERVATORY = EarthLocation(
    lat=Angle("-30d14m27.00s"),
    lon=Angle("-70d44m11.99s"),
    height=2722 * u.m,
)


# Cribbing some items from https://github.com/lsst-sims/rubin_nights
def get_access_token(tokenfile: str | Path | None = None) -> str:
    """Retrieve RSP access token.

    Parameters
    ----------
    tokenfile : `str`, `Path`, or None
        Path to token file.
        Default None will fall back to environment variable,
        ACCESS_TOKEN and then try lsst.rsp.get_access_token().

    Returns
    -------
    token : `str`
        Token value.
    """
    token = None
    try:
        # Try using lsst-rsp first
        import lsst.rsp.get_access_token as rsp_get_access_token

        token = rsp_get_access_token(tokenfile=tokenfile)
    except ImportError:
        # No lsst-rsp available
        token_path = (
            Path(tokenfile) if tokenfile is not None else Path(DEFAULT_RSP_TOKEN_PATH)
        ).expanduser()
        if token_path.exists():
            with open(token_path, "r") as f:
                token = f.read().strip()
        if token is None:
            # Try using environment variable
            token = os.environ.get("ACCESS_TOKEN", None)

    if token is None:
        print("No RSP token available.")
        token = ""
    return token


class ConsDB:
    """Class to query ConsDB.

    Uses the ConsDbClient if available, otherwise falls back to
    using the REST API.

    Parameters
    ----------
    server : str
        ConsDB server URL.
    token : str
        RSP access token.
    """

    def __init__(self, server, token):
        self.server = server
        self.token = token
        try:
            from lsst.summit.utils import ConsDbClient

            server = server.replace("https://", f"https://user:{token}@") + "/consdb"
            self.client = ConsDbClient(server)
        except ImportError:
            self.client = None

    def query(self, query):
        """Query ConsDB.

        Parameters
        ----------
        query : str
            SQL query to execute.

        Returns
        -------
        astropy.table.Table
            Table with results of query.
        """
        if self.client is not None:
            return self.client.query(query)
        else:
            auth = "user", self.token
            params = {
                "query": query,
            }
            response = requests.post(
                self.server + "/consdb/query", auth=auth, json=params
            )
            columns = response.json()["columns"]
            data = response.json()["data"]
            table = Table(names=columns, data=np.array(data))
            return table


class EFD:
    """Class to query EFD.

    Uses lsst.efd.client.EFDClient if available, otherwise falls back to
    using the REST API.

    Parameters
    ----------
    server : str
        EFD server URL.
    """

    def __init__(self, server):
        if "usdf" in server:
            site = "usdf_efd"
        else:
            site = "summit_efd"
        try:
            from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore
            from lsst_efd_client import EfdClient

            self.client = EfdClient(site)
        except ImportError:
            self.client = None
            creds_service = f"https://roundtable.lsst.codes/segwarides/creds/{site}"
            efd_creds = requests.get(creds_service).json()
            self.auth = efd_creds["username"], efd_creds["password"]
            self.url = f"https://{efd_creds['host']}{efd_creds['path']}query"

    def get_most_recent_row_before(self, topic, time):
        if self.client is not None:
            from aiohttp.client_exceptions import ServerDisconnectedError
            from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore

            try:
                out = getMostRecentRowWithDataBefore(
                    self.client, topic, timeToLookBefore=time, maxSearchNMinutes=5
                )
            except ServerDisconnectedError:
                raise ValueError()
            return out
        else:
            import pandas as pd

            # Stealing from summit_utils and rubin_nights
            earliest = Time("2019-01-01")
            if time < earliest:
                raise ValueError(f"Time {time} is before EFD start time.")
            earliest = max(earliest, time - 5 * u.min)
            query = f'select * from "{topic}" '
            query += f"where time > '{earliest.isot}Z' and time <= '{time.isot}Z'"
            params = {"db": "efd", "q": query}
            while True:
                try:
                    response = requests.get(
                        self.url, auth=self.auth, params=params
                    ).json()
                except ValueError:
                    print("Waiting for EFD...")
                    sleep(1.0)
                    continue
                else:
                    break

            statement = response["results"][0]
            if "series" not in statement:
                raise ValueError(f"No data found for topic {topic} at time {time}.")
            series = statement["series"][0]
            result = pd.DataFrame(series.get("values", []), columns=series["columns"])
            if "time" in result:
                result = result.set_index(pd.to_datetime(result["time"]))
                result = result.drop("time", axis=1)
            return result.iloc[-1]


def parallactic_angle(coord, time):
    """Calculate the parallactic angle.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Coordinates of the object.  Must have alt and az attributes.
    time : astropy.time.Time
        Time of the observation.

    Returns
    -------
    astropy.coordinates.Angle
        Parallactic angle.
    """
    loc = RUBIN_OBSERVATORY
    coord = coord.transform_to(AltAz(obstime=time, location=loc))

    ha, dec = coord.hadec.ha, coord.hadec.dec
    lat = loc.lat

    sinq = np.sin(ha)
    cosq = np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(ha)
    return Angle(np.arctan2(sinq, cosq))


def set_camera(url, camera, visible=True):
    """Set the camera in Stellarium.

    Parameters
    ----------
    url : str
        URL of Stellarium API.
    camera : str
        Camera to set.
    visible : bool, optional
        Whether the MosaicCamera detector wireframe grid is visible.
    """
    url = f"{url}/api/stelproperty/set"
    requests.post(url, data={"id": "MosaicCamera.currentCamera", "value": camera})
    requests.post(
        url,
        data={"id": "MosaicCamera.visible", "value": "True" if visible else "False"},
    )


def get_stellarium_scripts_dir() -> Path:
    """Return Stellarium's user scripts directory where core.loadSkyImage looks up files."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = (
            Path(appdata) / "Stellarium"
            if appdata
            else Path.home() / "AppData" / "Roaming" / "Stellarium"
        )
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Stellarium"
    else:
        base = Path.home() / ".stellarium"
    scripts_dir = base / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir


def resolve_cookies_path(cookies_path: str | Path | None = None) -> Path:
    """Locate cookies.txt in current working dir, script dir, or Rubin workspace directory."""
    if cookies_path is not None:
        return Path(cookies_path).expanduser()
    candidate_paths = [
        Path.cwd() / DEFAULT_COOKIES_PATH,
        Path(__file__).resolve().parent / DEFAULT_COOKIES_PATH,
        Path(r"\\wsl.localhost\Ubuntu\home\atara\rubin") / DEFAULT_COOKIES_PATH,
        Path.home() / "rubin" / DEFAULT_COOKIES_PATH,
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    return candidate_paths[0]


def download_rubin_mosaic(
    visit_id: int | str,
    rsp_server: str = DEFAULT_RSP_SERVER,
    cookies_path: str | Path | None = None,
    out_dir: str | Path = DEFAULT_IMAGE_DIR,
) -> Path:
    """Download RubinTV focal_plane_mosaic image matching the download-by-id logic.

    Parameters
    ----------
    visit_id : int or str
        Visit or exposure ID (e.g. 2026110400264, 20261104000264, or 2026-11-04_000264).
    rsp_server : str
        Base URL of the RSP server.
    cookies_path : str, Path, or None
        Path to cookies.txt (defaults to cookies.txt in current or rubin folder).
    out_dir : str or Path
        Directory where downloaded images are saved.

    Returns
    -------
    Path
        Path to the downloaded or cached JPEG image.
    """
    clean_id = re.sub(r"\D", "", str(visit_id).strip())
    if len(clean_id) < 9:
        raise ValueError(
            f"Invalid Rubin visit/exposure ID: '{visit_id}'. Expected format: YYYYMMDD + sequence number (e.g. 2026110400264 or 2026-11-04_000264)."
        )

    year = clean_id[0:4]
    month = clean_id[4:6]
    day = clean_id[6:8]
    date_str = f"{year}-{month}-{day}"

    img_num = int(clean_id[8:], 10)
    num_str = f"{img_num:06d}"

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    outfile = out_dir_path / f"image_{clean_id}.jpg"

    # Check cache in common locations
    candidate_cache = [
        outfile,
        out_dir_path / f"image_{clean_id[:13]}.jpg",
        out_dir_path / f"image_{num_str}.jpg",
        Path(f"images_{date_str}") / f"image_{num_str}.jpg",
        Path(r"\\wsl.localhost\Ubuntu\home\atara\rubin") / f"images_{date_str}" / f"image_{num_str}.jpg",
    ]
    for cached in candidate_cache:
        if cached.exists() and cached.stat().st_size > 1024:
            try:
                with open(cached, "rb") as f:
                    head = f.read(512).lower()
                if b"<html" not in head and b"<!doctype" not in head:
                    print(f"Using cached image: {cached}")
                    return cached
            except Exception:
                pass

    cookies_file = resolve_cookies_path(cookies_path)
    prefix = f"lsstcam_focal_plane_mosaic_{date_str}_"
    filename = f"{prefix}{num_str}.jpg"

    server = rsp_server.rstrip("/")
    primary_url = f"{server}{DEFAULT_RUBINTV_MOSAIC_PATH}/{filename}"
    alt_url = f"{server}{ALT_RUBINTV_MOSAIC_PATH}/{filename}"

    print(f"Downloading RubinTV image for {clean_id} (Date: {date_str}, Sequence: {num_str})")

    def _fetch_url(target_url: str) -> bool:
        try:
            cmd = ["curl", "-s", "-L", "-o", str(outfile)]
            if cookies_file.exists():
                cmd.extend(["-b", str(cookies_file)])
            cmd.append(target_url)
            subprocess.run(cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            session = requests.Session()
            if cookies_file.exists():
                jar = MozillaCookieJar(str(cookies_file))
                jar.load(ignore_discard=True, ignore_expires=True)
                session.cookies = jar
            resp = session.get(target_url, allow_redirects=True, timeout=30)
            if resp.status_code != 200:
                return False
            with open(outfile, "wb") as f:
                f.write(resp.content)

        if not outfile.exists() or outfile.stat().st_size <= 1024:
            return False
        with open(outfile, "rb") as f:
            head = f.read(512).lower()
        if b"<html" in head or b"<!doctype" in head:
            return False
        return True

    success = _fetch_url(primary_url)
    if not success:
        print(f"Primary endpoint did not return valid image, trying alternate: {alt_url}")
        success = _fetch_url(alt_url)

    if not success:
        if outfile.exists():
            with open(outfile, "rb") as f:
                head = f.read(512).lower()
            if b"<html" in head or b"<!doctype" in head:
                print("  ⚠ Authentication failed or image not found (received HTML)")
                raise RuntimeError(
                    f"Authentication failed or image not found when downloading {primary_url} (received HTML). "
                    f"Check {cookies_file}."
                )
        raise RuntimeError(
            f"Failed to download image for visit {clean_id} from {primary_url} and {alt_url}."
        )

    return outfile


def prepare_rubin_sky_image(
    image_path: str | Path,
    visit_id: int | str | None = None,
    transparent_corners: bool = True,
    opacity: float = DEFAULT_IMAGE_OPACITY,
) -> tuple[str, Path]:
    """Crop, orient, and mask a RubinTV focal_plane_mosaic image for Stellarium sky projection.

    The RubinTV focal_plane_mosaic JPEG is a 1200x1200 matplotlib figure with title,
    axes ticks, and a colorbar on the right. This function:
      1. Detects and crops the interior of the focal plane square box.
      2. Rotates 90 degrees clockwise (`ROTATE_270`) so the detector axes match
         Stellarium's `core.loadSkyImage` and `MosaicCamera` (`LSSTCam.json`) convention
         at `rotation = RotSkyPos`.
      3. Applies the requested `opacity` (default `DEFAULT_IMAGE_OPACITY = 0.25`, i.e. 25%)
         and masks the 4 empty white corner cutouts outside the rafts + corner wavefront
         sensors with RGBA = (0, 0, 0, 0) so OpenGL additive blending renders them
         completely transparent on the sky.
      4. Saves the PNG inside Stellarium's user `scripts/` folder for `core.loadSkyImage`.

    Returns
    -------
    script_rel_name : str
        Filename relative to Stellarium's `scripts/` directory.
    dest_path : Path
        Full path to the prepared PNG file.
    """
    image_path = Path(image_path)
    tag = (
        re.sub(r"\D", "", str(visit_id))[:13]
        if visit_id is not None
        else re.sub(r"[^\w\-.]", "_", image_path.stem)
    )
    script_rel_name = f"rubin_sky_{tag}.png"
    dest_path = get_stellarium_scripts_dir() / script_rel_name

    im = Image.open(image_path).convert("RGBA")
    arr = np.array(im)
    h, w, _ = arr.shape

    # If this is a standard 1200x1200 RubinTV matplotlib plot, crop to the inner focal plane frame
    gray = arr[:, :, :3].mean(axis=2)
    horiz = np.where((gray[:, int(w * 0.08) : int(w * 0.75)] < 30).mean(axis=1) > 0.90)[
        0
    ]
    vert = np.where((gray[int(h * 0.12) : int(h * 0.88), :] < 30).mean(axis=0) > 0.90)[
        0
    ]
    if len(horiz) >= 2 and len(vert) >= 2:
        y0 = int(horiz[0]) + 1
        y1 = int(horiz[-1])
        x0 = int(vert[0]) + 1
        # Exclude colorbar vertical border at x > 0.905 * w
        valid_vert = vert[vert < int(w * 0.905)]
        x1 = int(valid_vert[-1]) if len(valid_vert) >= 2 else int(vert[1])
        if (x1 - x0) > int(w * 0.5) and (y1 - y0) > int(h * 0.5):
            im = im.crop((x0, y0, x1, y1))

    # Rotate 90 deg clockwise so focal plane axes match Stellarium's core.loadSkyImage & LSSTCam.json
    oriented = im.transpose(Image.Transpose.ROTATE_270).convert("RGBA")
    m_arr = np.array(oriented)

    # Apply opacity scaling (0.0 to 1.0) across RGB and Alpha channels for Stellarium additive blending
    alpha_scale = float(np.clip(opacity, 0.0, 1.0))
    if alpha_scale < 1.0:
        m_arr = (m_arr.astype(np.float32) * alpha_scale).astype(np.uint8)

    if transparent_corners:
        mh, mw, _ = m_arr.shape
        yn = np.linspace(1.0, -1.0, mh)[:, None]
        xn = np.linspace(-1.0, 1.0, mw)[None, :]

        # Exact geometric footprint of LSSTCam science rafts (3x5 + 5x3 cross)
        in_science_cross = (np.abs(xn) <= 0.5985) | (np.abs(yn) <= 0.5985)

        # Exact geometric footprint of the 4 corner wavefront sensor pairs (ITL_WF in LSSTCam.json)
        in_wf_tl = (
            (xn >= -0.744)
            & (xn <= -0.610)
            & (yn >= 0.606)
            & (yn <= 0.748)
            & ~((xn >= -0.680) & (xn <= -0.672))
        )
        in_wf_tr = (
            (xn >= 0.606)
            & (xn <= 0.748)
            & (yn >= 0.610)
            & (yn <= 0.744)
            & ~((yn >= 0.672) & (yn <= 0.680))
        )
        in_wf_br = (
            (xn >= 0.610)
            & (xn <= 0.744)
            & (yn >= -0.748)
            & (yn <= -0.606)
            & ~((xn >= 0.672) & (xn <= 0.680))
        )
        in_wf_bl = (
            (xn >= -0.748)
            & (xn <= -0.606)
            & (yn >= -0.744)
            & (yn <= -0.610)
            & ~((yn >= -0.680) & (yn <= -0.672))
        )

        valid_footprint = in_science_cross | in_wf_tl | in_wf_tr | in_wf_br | in_wf_bl
        # Zero both RGB and Alpha so Stellarium's additive sky blending renders corners transparent
        m_arr[~valid_footprint] = 0

    out_im = Image.fromarray(m_arr)
    out_im.save(dest_path, format="PNG")
    return script_rel_name, dest_path


def get_active_layers_path() -> Path:
    """Return path to file tracking active sky layer IDs."""
    return get_stellarium_scripts_dir() / ACTIVE_LAYERS_FILE


def record_active_layer(layer_id: str):
    """Record an active sky layer ID."""
    p = get_active_layers_path()
    layers = get_active_layers()
    layers.add(layer_id)
    try:
        p.write_text("\n".join(sorted(layers)))
    except Exception:
        pass


def unrecord_active_layer(layer_id: str):
    """Remove a layer ID from tracked active sky layer IDs."""
    p = get_active_layers_path()
    layers = get_active_layers()
    if layer_id in layers:
        layers.remove(layer_id)
        try:
            p.write_text("\n".join(sorted(layers)))
        except Exception:
            pass


def get_active_layers() -> set[str]:
    """Retrieve set of currently tracked sky layer IDs."""
    p = get_active_layers_path()
    if p.exists():
        try:
            lines = p.read_text().splitlines()
            return {line.strip() for line in lines if line.strip()}
        except Exception:
            pass
    return set()


def clear_sky_image(url: str, layer_id: str | None = None):
    """Remove plotted Rubin sky image(s) from Stellarium.

    If `layer_id` is provided, remove only that specific layer.
    If `layer_id` is None, remove all tracked active sky layers as well as
    DEFAULT_SKY_LAYER_ID.
    """
    if layer_id is not None:
        layers_to_remove = {layer_id}
        unrecord_active_layer(layer_id)
    else:
        layers_to_remove = get_active_layers() | {DEFAULT_SKY_LAYER_ID}
        p = get_active_layers_path()
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    remove_statements = [f'core.removeSkyImage("{lid}");' for lid in sorted(layers_to_remove)]
    if remove_statements:
        code = "\n".join(remove_statements)
        requests.post(
            f"{url}/api/scripts/direct", data={"code": code, "useIncludes": "false"}
        )


def plot_sky_image_to(
    url: str,
    camera: str,
    ra: Angle,
    dec: Angle,
    rsp: Angle,
    script_image_name: str,
    layer_id: str = DEFAULT_SKY_LAYER_ID,
    max_bright: float = 8.0,
    show_grid: bool = False,
    update_mosaic_camera: bool = True,
):
    """Plot the prepared Rubin focal plane image onto Stellarium's celestial sphere.

    Updates MosaicCamera's position if `update_mosaic_camera=True` (so `status` and
    `actionSetViewToCamera` continue to work identically), hides the empty blue/cyan
    detector wireframe grid unless `show_grid=True`, and loads the real camera image
    at the exact sky coordinates and RotSkyPos rotation.
    """
    if not np.isfinite(ra):
        cprint(f"ra is not finite: {ra}", "bright_red")
    if not np.isfinite(dec):
        cprint(f"dec is not finite: {dec}", "bright_red")
    if not np.isfinite(rsp):
        cprint(f"rsp is not finite: {rsp}", "bright_red")

    prop_url = f"{url}/api/stelproperty/set"
    if update_mosaic_camera:
        for data in [
            {"id": "MosaicCamera.currentCamera", "value": camera},
            {"id": "MosaicCamera.visible", "value": "True" if show_grid else "False"},
            {"id": "MosaicCamera.ra", "value": ra.deg},
            {"id": "MosaicCamera.dec", "value": dec.deg},
            {"id": "MosaicCamera.rotation", "value": rsp.deg},
            {"id": "StelSkyLayerMgr.flagShow", "value": "True"},
        ]:
            requests.post(prop_url, data=data)
    else:
        requests.post(prop_url, data={"id": "StelSkyLayerMgr.flagShow", "value": "True"})

    ang_size_arcmin = float(
        np.degrees(LSSTCAM_HALF_X_RAD + LSSTCAM_HALF_Y_RAD) * 60.0
    )
    rot_deg = float(rsp.deg % 360.0)

    record_active_layer(layer_id)

    code = (
        f'core.removeSkyImage("{layer_id}");\n'
        f'core.loadSkyImage("{layer_id}", "{script_image_name}", '
        f"{ra.deg:.8f}, {dec.deg:.8f}, {ang_size_arcmin:.4f}, {rot_deg:.6f}, "
        f'100.0, {max_bright:.2f}, true, "EqJ2000", false);'
    )
    requests.post(
        f"{url}/api/scripts/direct", data={"code": code, "useIncludes": "false"}
    )


def slew_to(url, camera, ra, dec, rsp, visible=True):
    """Send camera repositioning command to Stellarium.

    Parameters
    ----------
    url : str
        URL of Stellarium API.
    camera : str
        Camera to slew.
    ra : Angle
        Right ascension.
    dec : Angle
        Declination.
    rsp : Angle
        Rotator sky position (E of N).
    visible : bool, optional
        Whether the MosaicCamera detector wireframe grid should be visible.
    """
    if not np.isfinite(ra):
        cprint(f"ra is not finite: {ra}", "bright_red")
    if not np.isfinite(dec):
        cprint(f"dec is not finite: {dec}", "bright_red")
    if not np.isfinite(rsp):
        cprint(f"rsp is not finite: {rsp}", "bright_red")

    url = f"{url}/api/stelproperty/set"
    for data in [
        {"id": "MosaicCamera.currentCamera", "value": camera},
        {"id": "MosaicCamera.visible", "value": "True" if visible else "False"},
        {"id": "MosaicCamera.ra", "value": ra.deg},
        {"id": "MosaicCamera.dec", "value": dec.deg},
        {"id": "MosaicCamera.rotation", "value": rsp.deg},
    ]:
        requests.post(url, data=data)


def parse_angle(value, unit=u.deg):
    """Parse an angle with units.

    Parameters
    ----------
    value : str
        Angle string.
    unit : astropy.units.Unit, optional
        Unit of the angle if not inferable from value.  Default is degrees.

    Returns
    -------
    astropy.coordinates.Angle
        Angle object.
    """
    try:
        angle = Angle(value)
    except u.UnitsError:
        angle = Angle(value, unit=unit)
    return angle


def get_stellarium_attributes(url):
    """Get particular Stellarium attributes of interest.

    Parameters
    ----------
    url : str
        URL of Stellarium API.

    Returns
    -------
    dict
        Dictionary with structure:
            time: astropy.time.Time
                Time in Stellarium.
            last: astropy.coordinates.Angle
                Local apparent sidereal time at Rubin Observatory.
            current_camera: str
                Current camera in Stellarium.
            az: astropy.coordinates.Angle
                Azimuth of current camera.
            alt: astropy.coordinates.Angle
                Altitude of current camera.
            rsp: astropy.coordinates.Angle
                Sky rotation (RotSkyPos) of current camera.
            q: astropy.coordinates.Angle
                Parallactic angle of current camera.
            ra: astropy.coordinates.Angle
                Right ascension of current camera.
            dec: astropy.coordinates.Angle
                Declination of current camera.
            rtp: astropy.coordinates.Angle
                Rotator position (RotTelPos) of current camera.
            ha: astropy.coordinates.Angle
                Hour angle of current camera.
    """
    time = Time(
        requests.get(f"{url}/api/main/status").json()["time"]["jday"], format="jd"
    )
    properties = requests.get(f"{url}/api/stelproperty/list").json()
    ra = properties["MosaicCamera.ra"]["value"]
    dec = properties["MosaicCamera.dec"]["value"]
    rsp = properties["MosaicCamera.rotation"]["value"]
    current_camera = properties["MosaicCamera.currentCamera"]["value"]
    while any(x is None for x in [ra, dec, rsp, current_camera]):
        print("Waiting for Stellarium...")
        sleep(0.5)
        properties = requests.get(f"{url}/api/stelproperty/list").json()
        ra = properties["MosaicCamera.ra"]["value"]
        dec = properties["MosaicCamera.dec"]["value"]
        rsp = properties["MosaicCamera.rotation"]["value"]
        current_camera = properties["MosaicCamera.currentCamera"]["value"]

    ra = Angle(ra * u.deg).wrap_at("360d")
    dec = Angle(dec * u.deg)
    rsp = Angle(rsp, unit=u.deg).wrap_at("360d")
    coord = SkyCoord(ra=ra, dec=dec)
    aa = coord.transform_to(AltAz(obstime=time, location=RUBIN_OBSERVATORY))
    last = Time(time, location=RUBIN_OBSERVATORY).sidereal_time("apparent")
    ha = last - ra
    alt = aa.alt
    az = aa.az.wrap_at("180d")
    q = parallactic_angle(coord, time).wrap_at("360d")
    rtp = q - rsp - 90 * u.deg
    rtp = rtp.wrap_at("90d")  # Should always be between -90 and +90 degrees
    return dict(
        time=time,
        last=last,
        current_camera=current_camera,
        az=az,
        alt=alt,
        rsp=rsp,
        q=q,
        ra=ra,
        dec=dec,
        rtp=rtp,
        ha=ha,
    )


def cprint(text, color=None):
    try:
        if HAS_COLORAMA:
            from colorama import Fore, Style

            # All color names
            colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]

            # Generate the color dictionary programmatically
            color_dict = {color: getattr(Fore, color.upper()) for color in colors}
            color_dict.update(
                {
                    f"bright_{color}": getattr(Fore, f"LIGHT{color.upper()}_EX")
                    for color in colors
                }
            )

            # Get the color code from the dictionary
            color_code = color_dict.get(color.lower(), "") if color else ""
            print(f"{color_code}{text}{Style.RESET_ALL}")
        else:
            print(text)
    except UnicodeEncodeError:
        safe_text = text.replace("\u2032", "'").replace("\u2033", '"')
        print(safe_text)


def print_state(api_url):
    data = get_stellarium_attributes(api_url)

    to_string_kwargs = dict(
        unit=u.deg,
        precision=1,
        pad=True,
        format="unicode",
        alwayssign=True,
    )

    print(f"Camera: {data['current_camera']}")
    cprint(
        f"Time: {data['time'].iso} UTC            "
        f"LAST: {data['last'].hour:6.3f} hours",
        "yellow",
    )
    cprint(
        f"HA:   {data['ha'].hour:6.3f} hours              "
        f"parallactic angle: {data['q'].deg:8.3f} deg",
        "bright_magenta",
    )

    az_str = data["az"].to_string(**to_string_kwargs)
    alt_str = data["alt"].to_string(**to_string_kwargs)
    rtp_str = f"{data['rtp'].deg:8.3f} deg"
    cprint(
        f"Az/Alt/RotTelPos:   {az_str:>13s}   {alt_str:>12s}   {rtp_str:>11s}",
        "bright_green",
    )

    ra_str = data["ra"].to_string(unit="hour", precision=1, pad=True)
    dec_str = data["dec"].to_string(**to_string_kwargs)
    rsp_str = f"{data['rsp'].deg:8.3f} deg"
    cprint(
        f"RA/Dec/RotSkyPos:   {ra_str:>13s}   {dec_str:>12s}   {rsp_str:>11s}",
        "bright_cyan",
    )
    rot_speed = (
        Angle("360d")
        / u.day
        * np.cos(RUBIN_OBSERVATORY.lat)
        * np.cos(data["az"])
        / np.cos(data["alt"])
    )
    alt_speed = (
        Angle("360d") / u.day * np.cos(RUBIN_OBSERVATORY.lat) * np.sin(data["az"])
    )
    az_speed = (
        Angle("360d")
        / u.day
        * (
            np.sin(RUBIN_OBSERVATORY.lat)
            - np.cos(RUBIN_OBSERVATORY.lat) * np.tan(data["alt"]) * np.cos(data["az"])
        )
    )

    rot_vel_str = f"{rot_speed.to(u.arcsec/u.s).value:>7.2f}"
    alt_vel_str = f"{alt_speed.to(u.arcsec/u.s).value:>7.2f}"
    az_vel_str = f"{az_speed.to(u.arcsec/u.s).value:>7.2f}"
    cprint(
        f'Az/Alt/Rot rate [" / s]: '
        f"{rot_vel_str:7s}        {alt_vel_str:7s}     {az_vel_str:7s}",
        "bright_red",
    )


def set_view_to_camera(api_url):
    """Set Stellarium view to camera and zoom to a sensible FOV if currently wide-field.

    Parameters
    ----------
    api_url : str
        URL of Stellarium API.
    """
    requests.post(f"{api_url}/api/stelaction/do", data={"id": "actionSetViewToCamera"})
    try:
        fov = requests.get(f"{api_url}/api/main/status", timeout=2).json()["view"]["fov"]
        if fov > 3.0:
            requests.post(
                f"{api_url}/api/scripts/direct",
                data={"code": "StelMovementMgr.zoomTo(5.0, 0.5);", "useIncludes": "false"},
                timeout=2,
            )
    except Exception:
        pass


def set_stellarium_time(api_url, time, timerate=None):
    """Set Stellarium time.

    Parameters
    ----------
    api_url : str
        URL of Stellarium API.
    time : astropy.time.Time
        Time to set.
    timerate : float, optional
        Timerate to set in days per second.  Default is None, which does not set the
        timerate.
    """
    data = {"time": time.mjd + 2400000.5}
    if timerate is not None:
        data["timerate"] = timerate
    requests.post(f"{api_url}/api/main/time", data=data)


@click.group()
@click.option(
    "--url",
    default=DEFAULT_STELLARIUM_URL,
    help=f"URL of Stellarium API. [default: {DEFAULT_STELLARIUM_URL}]",
)
@click.pass_context
def cli(ctx, url):
    """Track Vera Rubin Observatory in Stellarium!

    See subcommand help for details.
    """
    ctx.ensure_object(dict)
    ctx.obj["URL"] = url


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("lon", type=str)
@click.argument("lat", type=str)
@click.argument("rot", required=False, type=str)
@click.option(
    "--time",
    type=str,
    default=None,
    help="Explicitly set time in stellarium to this.  Accepts any format that"
    " astropy.time.Time can parse together with the --timeformat option."
    " [default: None]",
)
@click.option(
    "--timeformat",
    type=str,
    default=None,
    help="Time format to use.  Passed as the format argument to astropy.time.Time when"
    " explicitly setting the time.",
)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam]",
    default="LSSTCam",
)
@click.option(
    "--horizon",
    is_flag=True,
    help="Use horizon coordinates (Az/Alt/RotTelPos) instead of equitorial "
    "(RA/Dec/RotSkyPos).",
)
@click.option(
    "--rtp",
    is_flag=True,
    help="Interpret ROT as RotTelPos regardless of --horizon.",
)
@click.option(
    "--rsp",
    is_flag=True,
    help="Interpret ROT as RotSkyPos regardless of --horizon.",
)
@click.option(
    "--ha",
    is_flag=True,
    help="Interpret LON as Hour Angle instead of Right Ascension.  "
    "Only valid when --horizon is not set.",
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.option(
    "--visit-image",
    type=int,
    default=None,
    help="13-digit visit ID whose RubinTV focal plane image should be plotted instead of the detector grid.",
)
@click.option(
    "--image-file",
    type=click.Path(exists=True),
    default=None,
    help="Local Rubin focal plane image (.jpg/.png) to plot on the sky instead of the detector grid.",
)
@click.option(
    "--cookies",
    type=str,
    default=None,
    help=f"Path to cookies.txt for RubinTV download. [default: {DEFAULT_COOKIES_PATH}]",
)
@click.option(
    "--rsp-server",
    type=str,
    default=DEFAULT_RSP_SERVER,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
)
@click.option(
    "--show-grid",
    is_flag=True,
    help="Also keep the blue/cyan detector wireframe grid visible on top of the plotted image.",
)
@click.option(
    "--max-bright",
    type=float,
    default=8.0,
    help="Stellarium surface brightness parameter for the plotted sky image. [default: 8.0]",
)
@click.option(
    "--opacity",
    type=float,
    default=DEFAULT_IMAGE_OPACITY,
    help=f"Opacity of the plotted Rubin sky image between 0.0 and 1.0. [default: {DEFAULT_IMAGE_OPACITY}]",
)
@click.pass_context
def slew(
    ctx,
    lon,
    lat,
    rot,
    time,
    timeformat,
    camera,
    horizon,
    rtp,
    rsp,
    ha,
    no_follow,
    visit_image,
    image_file,
    cookies,
    rsp_server,
    show_grid,
    max_bright,
    opacity,
):
    """Slew mosaic to given position and rotation.

    LON is the longitudinal angle (HA, RA, or Az).

    LAT is the latitudinal angle (Dec or Alt).

    ROT is the rotation angle (RotSkyPos (E of N) or RotTelPos (CCW from +alt)).

    These may be specified in any format that astropy.units.Angle can parse.

    Note that LON without explicit units is interpreted as degrees when --horizon is
    set (i.e., as Az), and as hours otherwise (i.e., as RA or HA).

    Some examples:

        python tvs.py slew 12:34:56.7 -12:34:56.7 45

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --horizon

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --horizon --rsp

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --rtp

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --rtp --ha

        python tvs.py slew 12h34m56.7 -12d34m56.7 45deg

        python tvs.py slew 12.5h -12.5d 45deg

        python tvs.py slew "185 deg" "12 deg" "0.5 rad" --camera LATISS

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --no-follow

        python tvs.py slew 0 0 0 --time 2024-04-08T00:00:00

        python tvs.py slew 0 0 0 --time 60408.0 --timeformat mjd

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --visit-image 2026041100613
    """
    api_url = ctx.obj["URL"]
    extra_args = ctx.args
    if extra_args:
        lon = extra_args[0] if len(extra_args) > 0 else lon
        lat = extra_args[1] if len(extra_args) > 1 else lat
        rot = extra_args[2] if len(extra_args) > 2 else rot

    camera = "LSSTCam" if camera == "ComCam" else camera

    if time is not None:
        t = Time(time, format=timeformat)
        set_stellarium_time(api_url, t)

    use_image = (visit_image is not None) or (image_file is not None)
    set_camera(api_url, camera, visible=not use_image or show_grid)
    data = get_stellarium_attributes(api_url)
    if horizon:
        # Interpret lon/lat/rot as Az/Alt/RotTelPos
        alt = parse_angle(lat)
        az = parse_angle(lon)
        coord = SkyCoord(
            alt=alt,
            az=az,
            frame=AltAz(obstime=data["time"], location=RUBIN_OBSERVATORY),
        )
        q = parallactic_angle(coord, data["time"])
        ra = coord.icrs.ra
        dec = coord.icrs.dec
        if rot is None:  # Keep current rtp
            rotTelPos = data["rtp"]
        else:
            if rsp:
                rotTelPos = q - parse_angle(rot) - 90 * u.deg
            else:
                rotTelPos = parse_angle(rot)
        rotSkyPos = q - rotTelPos - 90 * u.deg
    else:
        # Interpret lon/lat/rot as (RA or HA)/Dec/RotSkyPos
        if ha:
            ra = data["last"] - parse_angle(lon, unit=u.hour)
        else:
            ra = parse_angle(lon, unit=u.hour)
        dec = parse_angle(lat)
        if rot is None:  # Keep current rsp
            rotSkyPos = data["rsp"]
        else:
            if rtp:
                coord = SkyCoord(ra=ra, dec=dec)
                q = parallactic_angle(coord, data["time"])
                rotSkyPos = q - parse_angle(rot) - 90 * u.deg
            else:
                rotSkyPos = parse_angle(rot)

    if use_image:
        raw_img = (
            Path(image_file)
            if image_file is not None
            else download_rubin_mosaic(
                visit_image, rsp_server=rsp_server, cookies_path=cookies
            )
        )
        script_img_name, _ = prepare_rubin_sky_image(
            raw_img, visit_id=visit_image, opacity=opacity
        )
        plot_sky_image_to(
            api_url,
            camera,
            ra,
            dec,
            rotSkyPos,
            script_img_name,
            max_bright=max_bright,
            show_grid=show_grid,
        )
    else:
        clear_sky_image(api_url)
        slew_to(api_url, camera, ra, dec, rotSkyPos)

    if not no_follow:
        set_view_to_camera(api_url)

    print_state(api_url)


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("name", type=str)
@click.argument("rot", type=str, required=False)
@click.option(
    "--time",
    type=str,
    default=None,
    help="Explicitly set time in stellarium to this.  Accepts any format that"
    " astropy.time.Time can parse together with the --timeformat option."
    " [default: None]",
)
@click.option(
    "--timeformat",
    type=str,
    default=None,
    help="Time format to use.  Passed as the format argument to astropy.time.Time when"
    " explicitly setting the time.  [default: None]",
)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam]",
    default="LSSTCam",
)
@click.option(
    "--horizon",
    is_flag=True,
    help="Use horizon rot (RotTelPos) instead of equatorial rot (RotSkyPos).",
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.option(
    "--visit-image",
    type=int,
    default=None,
    help="13-digit visit ID whose RubinTV focal plane image should be plotted instead of the detector grid.",
)
@click.option(
    "--image-file",
    type=click.Path(exists=True),
    default=None,
    help="Local Rubin focal plane image (.jpg/.png) to plot on the sky instead of the detector grid.",
)
@click.option(
    "--cookies",
    type=str,
    default=None,
    help=f"Path to cookies.txt for RubinTV download. [default: {DEFAULT_COOKIES_PATH}]",
)
@click.option(
    "--rsp-server",
    type=str,
    default=DEFAULT_RSP_SERVER,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
)
@click.option(
    "--show-grid",
    is_flag=True,
    help="Also keep the blue/cyan detector wireframe grid visible on top of the plotted image.",
)
@click.option(
    "--max-bright",
    type=float,
    default=8.0,
    help="Stellarium surface brightness parameter for the plotted sky image. [default: 8.0]",
)
@click.option(
    "--opacity",
    type=float,
    default=DEFAULT_IMAGE_OPACITY,
    help=f"Opacity of the plotted Rubin sky image between 0.0 and 1.0. [default: {DEFAULT_IMAGE_OPACITY}]",
)
@click.pass_context
def target(
    ctx,
    name,
    rot,
    time,
    timeformat,
    camera,
    horizon,
    no_follow,
    visit_image,
    image_file,
    cookies,
    rsp_server,
    show_grid,
    max_bright,
    opacity,
):
    """Slew to target by name.

    NAME is the name of the target (e.g., 'M20' or 'Trifid Nebula').

    ROT is the rotation angle (RotSkyPos (E of N) or RotTelPos (CCW from +alt)).

    The rotation angle may be specified in any format that astropy.units.Angle can
    parse.

    Some examples:

        python tvs.py target M20

        python tvs.py target M20 45

        python tvs.py target Trifid_Nebula 45 --horizon

        python tvs.py target M20 20deg

        python tvs.py target M20 "20 deg"

        python tvs.py target "Trifid Nebula" 45deg --camera LATISS

        python tvs.py target M20 45 --no-follow

        python tvs.py target M20 --time 2024-04-08T00:00:00

        python tvs.py target M20 --time 60408.0 --timeformat mjd
    """
    from astroquery.simbad import Simbad

    api_url = ctx.obj["URL"]
    extra_args = ctx.args
    if extra_args:
        name = extra_args[0] if len(extra_args) > 0 else name
        rot = extra_args[1] if len(extra_args) > 1 else rot

    if time is not None:
        t = Time(time, format=timeformat)
        set_stellarium_time(api_url, t)

    simbad = Simbad()
    simbad.add_votable_fields("ra", "dec")
    result = simbad.query_object(name)
    if result is None:
        raise ValueError(f"Object {name} not found in Simbad.")
    ra = Angle(result["ra"].data[0] * u.deg)
    dec = Angle(result["dec"].data[0] * u.deg)

    if horizon:
        data = get_stellarium_attributes(api_url)
        time = data["time"]
        coord = SkyCoord(ra=ra, dec=dec, frame=ICRS)
        q = parallactic_angle(coord, time)
        if rot is None:  # Keep current rtp
            rtp = data["rtp"]
        else:
            rtp = parse_angle(rot)
        rsp = q - rtp - 90 * u.deg
    else:
        if rot is None:  # Keep current rsp
            rsp = get_stellarium_attributes(api_url)["rsp"]
        else:
            rsp = parse_angle(rot)

    stel_cam = "LSSTCam" if camera == "ComCam" else camera
    use_image = (visit_image is not None) or (image_file is not None)
    if use_image:
        raw_img = (
            Path(image_file)
            if image_file is not None
            else download_rubin_mosaic(
                visit_image, rsp_server=rsp_server, cookies_path=cookies
            )
        )
        script_img_name, _ = prepare_rubin_sky_image(
            raw_img, visit_id=visit_image, opacity=opacity
        )
        plot_sky_image_to(
            api_url,
            stel_cam,
            ra,
            dec,
            rsp,
            script_img_name,
            max_bright=max_bright,
            show_grid=show_grid,
        )
    else:
        clear_sky_image(api_url)
        slew_to(api_url, stel_cam, ra, dec, rsp)

    if not no_follow:
        set_view_to_camera(api_url)
    print_state(api_url)


@cli.command()
@click.argument("visit", type=str)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam or inferred from date]",
    default=None,
)
@click.option(
    "--force_consdb",
    is_flag=True,
    help="Force use of ConsDB for coordinates.",
)
@click.option(
    "--rsp-token",
    type=str,
    help=f"Path to RSP token file. [default: {DEFAULT_RSP_TOKEN_PATH}]",
    default=DEFAULT_RSP_TOKEN_PATH,
)
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.option(
    "--rot-offset",
    type=float,
    help="Offset to apply to the RotSkyPos (E of N) angle.",
    default=None,
)
@click.option(
    "--grid-only",
    is_flag=True,
    help="Plot only the blue/cyan detector wireframe grid instead of downloading/plotting the real Rubin image.",
)
@click.option(
    "--show-grid",
    is_flag=True,
    help="Overlay the blue/cyan detector wireframe grid on top of the real Rubin image.",
)
@click.option(
    "--image-file",
    type=click.Path(exists=True),
    default=None,
    help="Use a specific local Rubin image file (.jpg/.png) instead of downloading from RubinTV. (Applicable when a single visit ID is provided).",
)
@click.option(
    "--cookies",
    type=str,
    default=None,
    help=f"Path to cookies.txt for RubinTV download. [default: {DEFAULT_COOKIES_PATH}]",
)
@click.option(
    "--max-bright",
    type=float,
    default=8.0,
    help="Stellarium surface brightness parameter for the plotted sky image. [default: 8.0]",
)
@click.option(
    "--opacity",
    type=float,
    default=DEFAULT_IMAGE_OPACITY,
    help=f"Opacity of the plotted Rubin sky image between 0.0 and 1.0. [default: {DEFAULT_IMAGE_OPACITY}]",
)
@click.pass_context
def visit(
    ctx,
    visit,
    camera,
    force_consdb,
    rsp_token,
    rsp_server,
    no_follow,
    rot_offset,
    grid_only,
    show_grid,
    image_file,
    cookies,
    max_bright,
    opacity,
):
    """Slew to match given visit ID(s) and plot the observed Rubin focal plane image(s) on the sky.

    VISIT is a visit ID or a comma-separated list of visit IDs to download and plot.

    By default, for LSSTCam visits, downloads (or loads from cache) the RubinTV
    focal_plane_mosaic image using cookies.txt (same logic as download_fromID.sh),
    crops and aligns it to the celestial sphere with the exact RotSkyPos rotation,
    and plots the real image in Stellarium in place of the empty detector grid.
    When multiple comma-separated visit IDs are passed, each image is downloaded and
    projected simultaneously onto Stellarium's celestial sphere as an independent layer.

    Some examples:

        # Plot real LSSTCam image for a visit on Stellarium's sky
        python tvs.py visit 2026041100613

        # Plot multiple Rubin focal plane images at the same time
        python tvs.py visit 2026041100613,2026041100614,2026041100615

        # Use a custom cookies.txt path or local image file
        python tvs.py visit 2026041100613 --cookies cookies.txt
        python tvs.py visit 2025110200011 --image-file image02_11.jpg

        # Show both the real image and the blue/cyan detector grid overlay
        python tvs.py visit 2026041100613 --show-grid

        # Plot only the original empty detector grid
        python tvs.py visit 2026041100613 --grid-only
    """
    api_url = ctx.obj["URL"]

    raw_items = [v.strip() for v in str(visit).split(",") if v.strip()]
    if not raw_items:
        raise ValueError(f"Invalid visit ID: '{visit}'")

    visit_list = []
    for item in raw_items:
        v_clean = re.sub(r"\D", "", item)
        if not v_clean:
            raise ValueError(f"Invalid visit ID: '{item}'")
        visit_list.append((item, v_clean, int(v_clean)))

    token = get_access_token(rsp_token)
    consdb = ConsDB(rsp_server, token)

    visits_table = None
    if not force_consdb:
        try:
            visits_table = QTable.read("visits.ecsv")
        except FileNotFoundError:
            pass

    # Clear previously plotted sky layers if not appending, or clear all for fresh plot
    clear_sky_image(api_url)

    last_info = None

    for idx, (raw_v, visit_clean, visit_int) in enumerate(visit_list):
        cam = camera
        # Override camera by date if possible:
        if cam is None:
            if visit_int < 2024102300000:
                print(f"[{raw_v}] Inferring camera is LATISS")
                cam = "LATISS"
            elif visit_int < 2024121200000:
                print(f"[{raw_v}] Inferring camera is ComCam")
                cam = "ComCam"
            elif visit_int < 2025041500000:
                print(f"[{raw_v}] Inferring camera is LATISS")
                cam = "LATISS"
            else:
                print(f"[{raw_v}] Inferring camera is LSSTCam")
                cam = "LSSTCam"

        if cam == "ComCam":
            database = "cdb_lsstcomcam"
        elif cam == "LSSTCam":
            database = "cdb_lsstcam"
        elif cam == "LATISS":
            database = "cdb_latiss"
        else:
            raise ValueError(f"Unknown camera: {cam}")

        v_rot_offset = rot_offset
        if v_rot_offset is None:
            if 2025041500000 < visit_int < 2025042200000:
                v_rot_offset = Angle("-90d")
            else:
                v_rot_offset = Angle("0d")
        v_rot_offset = parse_angle(v_rot_offset, unit=u.deg)

        ra = None
        dec = None
        rsp = None
        v_time = None
        if visits_table is not None and visit_int in visits_table["visit"]:
            print(f"[{raw_v}] Using visits.ecsv for coordinates")
            row = visits_table[visits_table["visit"] == visit_int][0]
            ra = row["ra"]
            dec = row["dec"]
            rsp = row["rsp"]
            v_time = row["time"]

        if ra is None:
            print(f"[{raw_v}] Using ConsDB for coordinates")
            query = f"select * from {database}.visit1 where visit_id = {visit_int}"
            data = consdb.query(query)[0]
            ra = Angle(data["s_ra"] * u.deg)
            dec = Angle(data["s_dec"] * u.deg)
            rsp = Angle(data["sky_rotation"] * u.deg)
            v_time = Time(data["exp_midpt_mjd"], format="mjd", scale="tai")

        rsp = rsp + v_rot_offset
        stel_cam = "LSSTCam" if cam == "ComCam" else cam

        # Determine layer id for this visit
        layer_id = f"rubin_focal_plane_{visit_clean}" if len(visit_list) > 1 else DEFAULT_SKY_LAYER_ID
        is_last = (idx == len(visit_list) - 1)

        v_image_file = image_file if (len(visit_list) == 1 or image_file is not None) else None

        if not grid_only and (cam == "LSSTCam" or v_image_file is not None):
            try:
                raw_img = (
                    Path(v_image_file)
                    if v_image_file is not None
                    else download_rubin_mosaic(
                        raw_v, rsp_server=rsp_server, cookies_path=cookies
                    )
                )
                script_img_name, _ = prepare_rubin_sky_image(
                    raw_img, visit_id=raw_v, opacity=opacity
                )
                plot_sky_image_to(
                    api_url,
                    stel_cam,
                    ra,
                    dec,
                    rsp,
                    script_img_name,
                    layer_id=layer_id,
                    max_bright=max_bright,
                    show_grid=show_grid,
                    update_mosaic_camera=is_last,
                )
            except Exception as exc:
                cprint(
                    f"[{raw_v}] Could not plot Rubin sky image ({exc}); falling back to detector grid.",
                    "bright_red",
                )
                slew_to(api_url, stel_cam, ra, dec, rsp, visible=True)
        else:
            slew_to(api_url, stel_cam, ra, dec, rsp, visible=True)

        last_info = (stel_cam, ra, dec, rsp, v_time)

    if last_info is not None:
        stel_cam, ra, dec, rsp, v_time = last_info
        # Set visit time and pause with timerate=0
        set_stellarium_time(api_url, v_time, timerate=0)
        if not no_follow:
            set_view_to_camera(api_url)
        print_state(api_url)


@cli.command()
@click.argument("day_obs_begin", type=int)
@click.argument("day_obs_end", type=int, required=False)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to replay. [default: LSSTCam or inferred from date]",
    default=None,
)
@click.option(
    "--rsp-token",
    type=str,
    help=f"Path to RSP token file. [default: {DEFAULT_RSP_TOKEN_PATH}]",
    default=DEFAULT_RSP_TOKEN_PATH,
)
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the replay with a Stellarium view change.",
)
@click.option(
    "--grid-only",
    is_flag=True,
    help="Plot only the blue/cyan detector wireframe grid instead of downloading/plotting the real Rubin images.",
)
@click.option(
    "--show-grid",
    is_flag=True,
    help="Overlay the blue/cyan detector wireframe grid on top of the real Rubin images.",
)
@click.option(
    "--cookies",
    type=str,
    default=None,
    help=f"Path to cookies.txt for RubinTV download. [default: {DEFAULT_COOKIES_PATH}]",
)
@click.option(
    "--max-bright",
    type=float,
    default=8.0,
    help="Stellarium surface brightness parameter for the plotted sky image. [default: 8.0]",
)
@click.option(
    "--opacity",
    type=float,
    default=DEFAULT_IMAGE_OPACITY,
    help=f"Opacity of the plotted Rubin sky image between 0.0 and 1.0. [default: {DEFAULT_IMAGE_OPACITY}]",
)
@click.pass_context
def replay(
    ctx,
    day_obs_begin,
    day_obs_end,
    camera,
    rsp_token,
    rsp_server,
    no_follow,
    grid_only,
    show_grid,
    cookies,
    max_bright,
    opacity,
):
    """Replay a range of days.

    DAY_OBS_BEGIN and DAY_OBS_END form a range of days (inclusive) to replay.  If
    DAY_OBS_END is omitted, it is set equal to DAY_OBS_BEGIN.
    """
    api_url = ctx.obj["URL"]

    if day_obs_end is None:
        day_obs_end = day_obs_begin
    if day_obs_end < day_obs_begin:
        raise ValueError("DAY_OBS_END must be greater than or equal to DAY_OBS_BEGIN")

    # Try to infer the camera
    if camera is None:
        if day_obs_end < 20241023:
            print("Inferring camera is LATISS")
            camera = "LATISS"
        elif day_obs_end < 20241212 and day_obs_begin > 20241023:
            print("Inferring camera is ComCam")
            camera = "ComCam"
        elif day_obs_end < 20250415 and day_obs_begin > 20241212:
            print("Inferring camera is LATISS")
            camera = "LATISS"
        else:
            print("Inferring camera is LSSTCam")
            camera = "LSSTCam"

    token = get_access_token(rsp_token)
    consdb = ConsDB(rsp_server, token)

    if camera == "ComCam":
        database = "cdb_lsstcomcam"
    elif camera == "LSSTCam":
        database = "cdb_lsstcam"
    elif camera == "LATISS":
        database = "cdb_latiss"
    else:
        raise ValueError(f"Unknown camera: {camera}")

    visits = consdb.query(
        f"select * from {database}.visit1 where day_obs >= {day_obs_begin}"
        f" and day_obs <= {day_obs_end}"
    )

    # Set stellarium time to first visit with can_see_sky True
    visit0 = visits[visits["can_see_sky"]][0]
    set_stellarium_time(
        api_url,
        Time(visit0["exp_midpt_mjd"], format="mjd"),
        timerate=100 / 86400,
    )

    stel_cam = "LSSTCam" if camera == "ComCam" else camera
    currentindex = None
    while True:
        time = get_stellarium_attributes(api_url)["time"]
        index = np.searchsorted(visits["exp_midpt_mjd"], time.mjd)
        if index == currentindex:
            continue
        currentindex = index
        try:
            visit = visits[index]
        except IndexError:
            continue
        print("\n\n\n")
        print(
            visit[
                [
                    "visit_id",
                    "day_obs",
                    "seq_num",
                    "band",
                    "exp_time",
                    "img_type",
                    "science_program",
                    "observation_reason",
                    "target_name",
                ]
            ]
        )
        print_state(api_url)
        ra = Angle(visit["s_ra"] * u.deg)
        dec = Angle(visit["s_dec"] * u.deg)
        rsp = Angle(visit["sky_rotation"] * u.deg)
        visit_id_val = int(visit["visit_id"])

        if not grid_only and camera == "LSSTCam":
            try:
                raw_img = download_rubin_mosaic(
                    visit_id_val, rsp_server=rsp_server, cookies_path=cookies
                )
                script_img_name, _ = prepare_rubin_sky_image(
                    raw_img, visit_id=visit_id_val, opacity=opacity
                )
                plot_sky_image_to(
                    api_url,
                    stel_cam,
                    ra,
                    dec,
                    rsp,
                    script_img_name,
                    max_bright=max_bright,
                    show_grid=show_grid,
                )
            except Exception as exc:
                cprint(
                    f"Could not plot Rubin sky image ({exc}); falling back to detector grid.",
                    "bright_red",
                )
                slew_to(api_url, stel_cam, ra, dec, rsp)
        else:
            clear_sky_image(api_url)
            slew_to(api_url, stel_cam, ra, dec, rsp)

        if not no_follow:
            set_view_to_camera(api_url)


@cli.command(name="clear-image")
@click.option(
    "--show-grid",
    is_flag=True,
    help="Re-enable the MosaicCamera detector wireframe grid after removing the sky image.",
)
@click.pass_context
def clear_image_cmd(ctx, show_grid):
    """Remove the plotted Rubin focal plane sky image from Stellarium."""
    api_url = ctx.obj["URL"]
    clear_sky_image(api_url)
    if show_grid:
        requests.post(
            f"{api_url}/api/stelproperty/set",
            data={"id": "MosaicCamera.visible", "value": "True"},
        )
    print("Removed Rubin sky image from Stellarium.")


@cli.command(name="download-image")
@click.argument("image_id", type=str)
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--cookies",
    type=str,
    default=None,
    help=f"Path to cookies.txt for RubinTV download. [default: {DEFAULT_COOKIES_PATH}]",
)
@click.option(
    "--out-dir",
    type=click.Path(),
    default=DEFAULT_IMAGE_DIR,
    help=f"Directory to save downloaded image. [default: {DEFAULT_IMAGE_DIR}]",
)
def download_image_cmd(image_id, rsp_server, cookies, out_dir):
    """Download RubinTV focal_plane_mosaic JPEG by exposure/visit ID."""
    try:
        path = download_rubin_mosaic(
            image_id,
            rsp_server=rsp_server,
            cookies_path=cookies,
            out_dir=out_dir,
        )
        print(f"Downloaded successfully: {path}")
    except Exception as exc:
        cprint(f"Error downloading image: {exc}", "bright_red")
        sys.exit(1)


def query_efd_retry(efd, topic, time):
    """Query/poll EFD until a valid row is found.

    Parameters
    ----------
    efd : EFD
        EFD client.
    topic : str
        Topic to query.
    time : astropy.time.Time
        Time to query.

    Returns
    -------
    dict
        Mount status.
    """

    while True:
        try:
            out = efd.get_most_recent_row_before(topic, time)
        except (TypeError, ValueError):
            sleep(1.0)
            print("Waiting for efd...")
            continue
        else:
            break
    return out


@cli.command()
@click.argument(
    "camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    default="LSSTCam",
    metavar="CAMERA",
)
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--delay",
    type=float,
    help="Delay between updates in seconds. [default: 5.0]",
    default=5.0,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the replay with a Stellarium view change.",
)
@click.pass_context
def follow(ctx, camera, rsp_server, delay, no_follow):
    """Follow camera pointing in real time with Stellarium.

    CAMERA is the camera to follow (LSSTCam, ComCam, or LATISS).

    To follow both LSSTCam and LATISS, run this command twice with different cameras and
    most likely with --no-follow on one of them.
    """
    api_url = ctx.obj["URL"]

    efd = EFD(rsp_server)

    if camera in ["ComCam", "LSSTCam"]:
        topic = "lsst.sal.MTPtg.mountStatus"
    elif camera == "LATISS":
        topic = "lsst.sal.ATPtg.mountStatus"
    else:
        raise ValueError(f"Unknown camera: {camera}")

    while True:
        tnow = Time.now()
        set_stellarium_time(api_url, tnow, timerate=1 / 86400)
        mount_status = query_efd_retry(efd, topic, tnow)
        try:
            ra = Angle(mount_status["mountRA"] * 15 * u.deg)
            dec = Angle(mount_status["mountDec"] * u.deg)
            rtp = Angle(mount_status["mountRot"] * u.deg)
            tefd = Time(mount_status["timestamp"], format="unix")
        except ValueError:
            print("Waiting for EFD...")
            sleep(delay)
            continue
        coord = SkyCoord(ra=ra, dec=dec)
        q = parallactic_angle(coord, tefd)
        rsp = q - rtp - 90 * u.deg

        slew_to(api_url, "LSSTCam" if camera == "ComCam" else camera, ra, dec, rsp)
        if not no_follow:
            set_view_to_camera(api_url)
        print()
        print_state(api_url)
        sleep(delay)


@cli.command()
@click.argument("camera", type=str, required=False, default="LSSTCam")
@click.pass_context
def status(ctx, camera):
    """Print current Stellarium status."""
    api_url = ctx.obj["URL"]
    set_camera(api_url, camera)
    print_state(api_url)


@cli.command()
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--delay",
    type=float,
    help="Delay between updates in seconds. [default: 5.0]",
    default=5.0,
)
@click.pass_context
def dome(ctx, rsp_server, delay):
    """Follow the dome in Stellarium."""
    api_url = ctx.obj["URL"]

    efd = EFD(DEFAULT_RSP_SERVER)

    topic = "lsst.sal.MTDome.azimuth"
    while True:
        tnow = Time.now()
        dome_az = query_efd_retry(efd, topic, tnow)
        az = Angle(dome_az["positionActual"] * u.deg)
        tefd = Time(dome_az["timestamp"], format="unix")
        alt = Angle("50d")
        rtp = Angle("0d")
        coord = SkyCoord(
            alt=alt, az=az, frame=AltAz(obstime=tnow, location=RUBIN_OBSERVATORY)
        )
        ra = coord.icrs.ra
        dec = coord.icrs.dec
        q = parallactic_angle(coord, tnow)
        rsp = q - rtp

        set_camera(api_url, "Dome")
        slew_to(api_url, "Dome", ra, dec, rsp)
        print(f"Time: {tnow.iso}")
        print(f"TEFD: {tefd.iso}")
        print(f"q: {q.deg:.2f} deg")
        print(f"Dome azimuth: {az.deg:.2f} deg")
        print()

        sleep(delay)


if __name__ == "__main__":
    cli()
