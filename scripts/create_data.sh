python tools/create_data.py inhouse \
--root-path /mnt/12T/nout/inhouse_unfiltered \
--out-dir /mnt/12T/nout/demo/data/lidar \
--workers 64 \
--extra-tag inhouse
echo "FLAG: LIDAR DONE"


python tools/create_data.py inhouse \
--root-path /mnt/12T/nout/inhouse_unfiltered \
--out-dir /mnt/12T/nout/demo/data/radar \
--workers 64 \
--extra-tag inhouse \
--version radar
echo "FLAG: RADAR DONE"