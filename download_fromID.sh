#!/usr/bin/env bash

# Usage:
#   ./download-by-id.sh <exposure_id_or_visit_id> [id2 ...]
#
# Examples:
#   ./download-by-id.sh 2026110400264
#   ./download-by-id.sh 20261104000264
#   ./download-by-id.sh 2026-11-04_000264
#   ./download-by-id.sh 2026110400264 2026110400265 2026110400266
#
# Rubin visit/exposure IDs typically consist of:
#   YYYYMMDD (date) + sequence number (5 or 6 digits)
#
# This script extracts the date and zero-padded sequence number,
# downloads the corresponding focal_plane_mosaic JPEG from RubinTV,
# and saves it into images_<YYYY-MM-DD>/image_<NUM>.jpg

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exposure_id> [exposure_id_2 ...]"
    echo
    echo "Examples:"
    echo "  $0 2026110400264"
    echo "  $0 2026-11-04_000264"
    echo "  $0 2026110400264 2026110400265"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate cookies.txt
if [ -f "${SCRIPT_DIR}/cookies.txt" ]; then
    COOKIES="${SCRIPT_DIR}/cookies.txt"
elif [ -f "cookies.txt" ]; then
    COOKIES="cookies.txt"
else
    COOKIES="${SCRIPT_DIR}/cookies.txt"
fi

BASE_URL="https://usdf-rsp.slac.stanford.edu/rubintv/event_image/summit-usdf/lsstcam/focal_plane_mosaic"
ALT_BASE_URL="https://usdf-rsp.slac.stanford.edu/rubintv/streamer-s3/usdf_Butler/lsstcam/focal_plane_mosaic"
SUFFIX=".jpg"

for RAW_ID in "$@"; do
    # Extract only digits from the input ID
    CLEAN_ID=$(echo "$RAW_ID" | tr -cd '0-9')

    # Need at least 8 digits for date (YYYYMMDD) + 1 digit for sequence
    if [ ${#CLEAN_ID} -lt 9 ]; then
        echo "⚠️  Error: '$RAW_ID' is not a valid Rubin ID."
        echo "   Expected format: YYYYMMDD + sequence number (e.g. 2026110400264 or 2026-11-04_000264)"
        echo
        continue
    fi

    YEAR="${CLEAN_ID:0:4}"
    MONTH="${CLEAN_ID:4:2}"
    DAY="${CLEAN_ID:6:2}"
    SEQ="${CLEAN_ID:8}"

    DATE="${YEAR}-${MONTH}-${DAY}"
    # Format sequence number as 6 digits with leading zeros (base 10)
    NUM=$(printf "%06d" "$((10#$SEQ))")

    PREFIX="lsstcam_focal_plane_mosaic_${DATE}_"
    OUTDIR="${SCRIPT_DIR}/images_${DATE}"
    mkdir -p "$OUTDIR"

    OUTFILE="${OUTDIR}/image_${NUM}.jpg"
    URL="${BASE_URL}/${PREFIX}${NUM}${SUFFIX}"

    echo "----------------------------------------"
    echo "Processing ID: $RAW_ID"
    echo "Date:          $DATE"
    echo "Sequence:      $NUM"
    echo "Destination:   $OUTFILE"
    echo "URL:           $URL"
    echo "Downloading..."

    # Attempt download from primary URL
    curl -s -L -b "$COOKIES" -o "$OUTFILE" "$URL"

    # If downloaded file is HTML (e.g. 404 or auth page) or empty, try alternate URL
    if [ ! -s "$OUTFILE" ] || grep -qiE "<html|<!doctype" "$OUTFILE" 2>/dev/null; then
        ALT_URL="${ALT_BASE_URL}/${PREFIX}${NUM}${SUFFIX}"
        echo "Retrying with alternate endpoint: $ALT_URL"
        curl -s -L -b "$COOKIES" -o "$OUTFILE" "$ALT_URL"
    fi

    # Final verification
    if [ ! -s "$OUTFILE" ]; then
        echo "❌ Download failed: file is empty."
        rm -f "$OUTFILE"
    elif grep -qiE "<html|<!doctype" "$OUTFILE" 2>/dev/null; then
        echo "⚠️  Warning: $OUTFILE appears to be HTML (authentication failed or image not found on RubinTV)."
        echo "   Check cookies in: $COOKIES"
    else
        echo "✅ Saved successfully to: $OUTFILE"
    fi
    echo
done

echo "Done."
