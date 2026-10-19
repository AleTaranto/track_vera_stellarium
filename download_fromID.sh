#!/usr/bin/env bash
if [ $# -ne 1 ]; then
    echo "Usage: $0 <id_list.txt>"
    exit 1
fi

IDFILE="$1"

if [ ! -f "$IDFILE" ]; then
    echo "Error: file '$IDFILE' not found."
    exit 1
fi

BASE_URL="https://usdf-rsp.slac.stanford.edu/rubintv/event_image/summit-usdf/lsstcam/focal_plane_mosaic"
COOKIES="cookies.txt"

OUTDIR="images_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "Saving images to: $OUTDIR"
echo

while read -r ID || [ -n "$ID" ]; do

    # Salta righe vuote
    [[ -z "$ID" ]] && continue

    # Controlla il formato
    if ! [[ "$ID" =~ ^[0-9]{13}$ ]]; then
        echo "Skipping invalid ID: $ID"
        continue
    fi

    YEAR="${ID:0:4}"
    MONTH="${ID:4:2}"
    DAY="${ID:6:2}"

    DATE="${YEAR}-${MONTH}-${DAY}"

    IMGNUM=$((10#${ID:8}))
    NUM=$(printf "%06d" "$IMGNUM")

    PREFIX="lsstcam_focal_plane_mosaic_${DATE}_"
    URL="${BASE_URL}/${PREFIX}${NUM}.png"


    OUTFILE="${OUTDIR}/${ID}.png"

    echo "Downloading $ID"
    if [ -f "$COOKIES" ]; then
        curl -s -k -L -b "$COOKIES" -o "$OUTFILE" "$URL"
    else
        curl -s -k -L -o "$OUTFILE" "$URL"
    fi

    if file "$OUTFILE" | grep -q "HTML"; then
        echo "  ⚠ Authentication failed (received HTML)"
    fi

done < "$IDFILE"

echo
echo "Done."
