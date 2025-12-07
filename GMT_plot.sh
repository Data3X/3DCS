#!/usr/bin/env bash
src_lon=-125.02
src_lat=40.371
adj_lat=39.6
lf=-126
rg=-114
up=41
dw=32
gmt begin inset-map_2 png
    a=-117.658/32.5616
    b=-116.143/33.31
    c=-117.997/35.89
    d=-119.54/35.12
    center=-117.8/34.23
    src=-125.02/40.37
    gmt grdimage @earth_relief_30s -R${lf}/${rg}/${dw}/${up} -JM12c -Baf -BWSne -I+d
    gmt colorbar -DJMR+w10c+o1.5c/0c+ml -Bxa1000f -By+l"m"
    gmt inset begin -DjBL+w4.2c/3.5c+o0.1c -F+gwhite+p1p
        gmt coast -R-135/-67/5/51 -JM? -W1/0.5p,black -Ggrey
gmt text -F+f10p,1 << EOF
-99 40 North America
EOF
gmt text -F+f10p+a-45,1 << EOF
-117 18 Pacific Ocean
EOF
        # Plot a rectangle region using -Sr+s
        echo $lf $dw $rg $up | gmt plot -Sr+s -W1p,blue
    gmt inset end
    gmt project -C${a} -E${b} -G0.1 | gmt plot -W2.5p,red -A
    gmt project -C${b} -E${c} -G0.1 | gmt plot -W2.5p,red -A
    gmt project -C${c} -E${d} -G0.1 | gmt plot -W2.5p,red -A
    gmt project -C${d} -E${a} -G0.1 | gmt plot -W2.5p,red -A
    gmt project -C${src} -E${center} -G0.1 | gmt plot -W4p,black -A
#gmt plot -R-125.02/40.37/-117.8/34.23 -S=0.5c+eA -W1.5p
gmt psmeca -Sa1c << EOF
$src_lon $src_lat 10 280 84 -179 7 
EOF
gmt text -F+f14p,1,white << EOF
$src_lon $adj_lat Mw 7.0
EOF
gmt plot -Gwhite -Sc0.5c -W1p  << EOF
-118.29 34.02
EOF
gmt text -F+f16p,1 << EOF
-118.29 33.4 Los Angeles
EOF
#gmt plot -St0.2c -W0.1p,black -Gcyan -l"Selected stations" < stations_latlon.txt
gmt end show

