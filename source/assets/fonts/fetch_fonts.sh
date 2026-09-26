#!/usr/bin/env bash
# downloads the exact typefaces the reel renders with (OFL / Apache — see FONTS.md)
set -e
base="https://raw.githubusercontent.com/google/fonts/main"
declare -A F=(
 [Anton-Regular.ttf]="$base/ofl/anton/Anton-Regular.ttf"
 [ArchivoBlack-Regular.ttf]="$base/ofl/archivoblack/ArchivoBlack-Regular.ttf"
 [IBMPlexMono-Bold.ttf]="$base/ofl/ibmplexmono/IBMPlexMono-Bold.ttf"
 [IBMPlexMono-SemiBold.ttf]="$base/ofl/ibmplexmono/IBMPlexMono-SemiBold.ttf"
 [IBMPlexSansCondensed-Bold.ttf]="$base/ofl/ibmplexsanscondensed/IBMPlexSansCondensed-Bold.ttf"
 [SpaceGrotesk_wght.ttf]="$base/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf"
 [Syne_wght.ttf]="$base/ofl/syne/Syne%5Bwght%5D.ttf"
 [BebasNeue-Regular.ttf]="$base/ofl/bebasneue/BebasNeue-Regular.ttf"
 [Archivo_wdth_wght.ttf]="$base/ofl/archivo/Archivo%5Bwdth,wght%5D.ttf"
 [DMMono-Regular.ttf]="$base/ofl/dmmono/DMMono-Regular.ttf"
)
for fn in "${!F[@]}"; do curl -sfL -o "$fn" "${F[$fn]}" && echo "ok  $fn"; done
