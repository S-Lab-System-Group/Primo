for i in $( seq 1 76800 )
do
   sudo dd if=/dev/urandom of=/dev/sdb count=8 bs=1M iflag=fullblock &
   sudo dd if=/dev/urandom of=/dev/sdc count=8 bs=1M iflag=fullblock &
   sudo dd if=/dev/urandom of=/dev/sdd count=8 bs=1M iflag=fullblock
done
