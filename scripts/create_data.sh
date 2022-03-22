python tools/create_data.py inhouse \
--root-path /mnt/12T/nout/inhouse_unfiltered \
--out-dir /mnt/12T/nout/V3/inhouse_unfiltered \
--workers 48 \
--extra-tag inhouse
echo "FLAG: LIDAR DONE"


python tools/create_data.py inhouse \
--root-path /mnt/12T/nout/inhouse_unfiltered \
--out-dir /mnt/12T/nout/V3/inhouse_unfiltered_radar \
--workers 48 \
--extra-tag inhouse \
--version radar
echo "FLAG: RADAR DONE"