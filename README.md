# Track Vera C. Rubin Observatory in Stellarium

Script for using the *Mosaic Camera* Stellarium plugin with Vera C. Rubin Observatory.


Mosaic Camera Plugin
====================

To get the plugin, you will need a recent version of [Stellarium](https://stellarium.org/).  Until Version 25.2 is released in Summer 2025, that means downloading a [weekly snapshot](https://github.com/Stellarium/stellarium-data/releases/tag/weekly-snapshot).

To enable the Mosaic Camera plugin, first load the Configurations window (F2 by default), and locate the Plugins tab.  Scroll down to Mosaic Camera and check the "Load at startup" box.  Restart Stellarium.

Once you've restarted, you can navigate back to the Mosaic Camera plugin and click the "configure" button, which will bring up a dialog for different cameras and their current RA, dec, rotation, and visibility parameters.  (You can also right-click on the Mosaic Camera icon that appeared in the toolbar).  From the dialog you can also recenter the Stellarium view on the current camera, or set the camera position to the currently selected object or current view.

It's possible to associate many of these functions with keyboard shortcuts too.  (Press F7 to bring up the Keyboard Shortcuts editor.)


tvs.py
======
The Mosaic Camera plugin is also controllable over an http connection, which is where the tvs.py script comes in.  With this script you can:
- slew the camera mosaic overlay to a given RA/DEC/RotSkyPos or Az/Alt/RotTelPos
- slew to a target by name
- slew to a specific past visit and overlay the observed Rubin focal plane image directly on the sky
- replay a past night's observations
- follow the Rubin (and LATISS) cameras or dome in real time (accesses the Rubin Engineering Facilities Database)
- clear plotted sky images from Stellarium
- display detailed status of the current camera overlay

Note that for these functions to work optimally, you should set your Stellarium location to Cerro Pachón (Use F6).

All functions require `click`, `astropy`, and `Pillow`.  Additionally, you'll need `astroquery` to slew to a target by name.  If you have the LSST Science Pipelines `summit_utils` package available, that'll be used to access the Rubin Consolidated Database (but I've added a rudimentary fallback if this is unavailable.)  Similarly, if `lsst_efd_client` is available, that'll be used to access the Rubin Engineering Facilities Database, but a rudimentary fallback also exists.

There's no installation.  The script is self-contained.  You can either call it with python a la `python tvs.py ...` or add your own [shebang](https://en.wikipedia.org/wiki/Shebang_(Unix)) to the beginning and just use `tvs.py ...`.

For the http control to work, you must enable the Remote Control Stellarium plugin, and configure it with "Server enabled".  You may want to check the "Enable automatically on startup" button too.


Commands
========
tvs.py works with subcommands, kinda like `git`.

Running

`python tvs.py`

Will display the overall help message including listing available subcommands.


python tvs.py slew
------------------
The slew command will reposition a camera mosaic overlay (or sky image).

`python tvs.py slew 12:34:56.7 -12:34:56.7 45`

Slew LSSTCam to RA 12h 34m 56.7s, Dec -12d 34m 56.7s with RotSkyPos 45 deg.

`python tvs.py slew 12:34:56.7 -12:34:56.7 45 --horizon`

Use horizon coordinates (Az/Alt/RotTelPos).

Options: `--camera`, `--no-follow`, `--time`, `--horizon`, `--visit-image`, `--image-file`, `--opacity`, `--show-grid`.


python tvs.py target
--------------------
Slew to a target by lookup name (via `astroquery`).

`python tvs.py target M20`

`python tvs.py target "Trifid Nebula" 45deg --camera LATISS`

Options: `--camera`, `--horizon`, `--no-follow`, `--time`, `--visit-image`, `--image-file`, `--opacity`, `--show-grid`.


python tvs.py visit
-------------------
The `visit` subcommand looks up a past visit in the Rubin Consolidated Database (ConsDB) or local `visits.ecsv` file, slews Stellarium to the pointing, sets the observation time, and projects the observed focal plane sky image onto Stellarium's celestial sphere.

### Sky Image Projection & RubinTV Downloads
For LSSTCam visits, `visit` automatically downloads the corresponding RubinTV focal plane mosaic image using an authentication cookie file (`cookies.txt`). The image is processed (cropped, oriented, and masked so empty raft corners are transparent) and loaded directly onto the celestial sphere at the visit's exact RA, Dec, and RotSkyPos rotation angle. Downloaded images are cached in `images_rubin/`.

To access ConsDB, an RSP token is searched for at `~/.lsst/rsp_token` (or specified via `--rsp-token`). For downloading RubinTV images, place a valid `cookies.txt` file in your working directory or pass `--cookies`.

### Options & Examples

- **Basic visit lookup and sky image overlay:**
  `python tvs.py visit 2026041100613`

- **Show the detector wireframe grid on top of the real image:**
  `python tvs.py visit 2026041100613 --show-grid`

- **Plot only the detector wireframe grid (no image download):**
  `python tvs.py visit 2026041100613 --grid-only`

- **Use a local image file instead of downloading:**
  `python tvs.py visit 2025110200011 --image-file image02_11.jpg`

- **Specify custom cookies or RSP token paths:**
  `python tvs.py visit 2026041100613 --cookies /path/to/cookies.txt --rsp-token ~/.lsst/rsp_token`

- **Adjust image opacity (0.0 to 1.0) or Stellarium surface brightness:**
  `python tvs.py visit 2026041100613 --opacity 0.5 --max-bright 10.0`

- **Explicit camera selection or offset rotation:**
  `python tvs.py visit 2024121100608 --camera ComCam`
  `python tvs.py visit 2025040300825 --camera LATISS`
  `python tvs.py visit 2026041100613 --rot-offset 90`

- **Do not move Stellarium view focus:**
  `python tvs.py visit 2026041100613 --no-follow`


python tvs.py replay
--------------------
Replay observations for a range of dates. Automatically fetches visits from ConsDB and updates Stellarium's clock and sky images in sequence.

`python tvs.py replay 20250415`

`python tvs.py replay 20241210 20241211`

Options: `--camera`, `--grid-only`, `--show-grid`, `--cookies`, `--opacity`, `--no-follow`.


python tvs.py follow
--------------------
Polls the Rubin Engineering Facilities Database (EFD) to track camera pointing in real time.

`python tvs.py follow`

`python tvs.py follow LATISS`


python tvs.py dome
------------------
Follows the actual Rubin dome azimuth position in real time via EFD.

`python tvs.py dome`


python tvs.py clear-image
-------------------------
Removes any plotted Rubin focal plane sky image from Stellarium.

`python tvs.py clear-image`

`python tvs.py clear-image --show-grid`


python tvs.py status
--------------------
Prints current position (equatorial and horizon coordinates, rates, and LST) to stdout.

`python tvs.py status`

`python tvs.py status LATISS`
